"""
Federated RAG Utilities — Multi-Framework & Isolated Vector Query Engine
------------------------------------------------------------------------
Enables querying isolated per-framework ChromaDB collections:
1. Single-Framework Retrieval (Zero Cross-Framework Contamination)
2. Multi-Framework Scatter-Gather Retrieval with Reciprocal Rank Fusion (RRF)
3. Dynamic Framework Collection Discovery & Auto-Matching
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer

try:
    import agents.config as config
except ImportError:
    import config

DEFAULT_FRAMEWORKS_DIR = getattr(config, "CHROMA_FRAMEWORKS_DIR", "chroma_db_frameworks")
_cached_client = None
_cached_manifest = None


def get_frameworks_client(chroma_dir: str = DEFAULT_FRAMEWORKS_DIR) -> chromadb.PersistentClient:
    global _cached_client
    if _cached_client is None:
        _cached_client = chromadb.PersistentClient(path=chroma_dir)
    return _cached_client


def load_manifest(chroma_dir: str = DEFAULT_FRAMEWORKS_DIR) -> Dict[str, Any]:
    """Loads framework collections manifest or discovers dynamically."""
    manifest_path = os.path.join(chroma_dir, "framework_collections_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Fallback to dynamic inspection
    client = get_frameworks_client(chroma_dir)
    colls = client.list_collections()
    manifest = {
        "total_frameworks": len(colls),
        "chroma_dir": chroma_dir,
        "collections": {}
    }
    for c in colls:
        manifest["collections"][c.name] = {
            "collection_name": c.name,
            "control_count": c.count()
        }
    return manifest


def list_available_frameworks(chroma_dir: str = DEFAULT_FRAMEWORKS_DIR) -> List[Dict[str, Any]]:
    """Returns a list of all indexed frameworks with their metadata."""
    manifest = load_manifest(chroma_dir)
    results = []
    for coll_name, meta in manifest.get("collections", {}).items():
        results.append({
            "collection_name": coll_name,
            "framework": meta.get("framework", coll_name.replace("controls__", "")),
            "jurisdiction": meta.get("jurisdiction", "general"),
            "control_count": meta.get("control_count", 0),
            "source_file": meta.get("source_file", "")
        })
    return sorted(results, key=lambda x: x["framework"])


def resolve_collection_name(framework_query: str, chroma_dir: str = DEFAULT_FRAMEWORKS_DIR) -> Optional[str]:
    """
    Fuzzy resolves a user string (e.g. 'gdpr', 'NIST CSF', 'iso27001') to the exact collection name.
    """
    manifest = load_manifest(chroma_dir)
    available_colls = list(manifest.get("collections", {}).keys())

    q = re.sub(r'[^a-zA-Z0-9]', '', framework_query.lower())
    if not q:
        return None

    # Exact match
    for c in available_colls:
        clean_c = re.sub(r'[^a-zA-Z0-9]', '', c.lower())
        if q == clean_c or q == clean_c.replace("controls", ""):
            return c

    # Substring / keyword match
    for c in available_colls:
        clean_c = re.sub(r'[^a-zA-Z0-9]', '', c.lower())
        if q in clean_c or clean_c in q:
            return c

    return None


def retrieve_single_framework(
    query: str,
    framework: str,
    k: int = 5,
    chroma_dir: str = DEFAULT_FRAMEWORKS_DIR,
    embedder: Optional[SentenceTransformer] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves Top-K relevant controls strictly from a single framework collection.
    Guarantees 0% cross-framework leakage.
    """
    coll_name = resolve_collection_name(framework, chroma_dir)
    if not coll_name:
        # Try direct name
        coll_name = framework

    client = get_frameworks_client(chroma_dir)
    try:
        coll = client.get_collection(coll_name)
    except Exception as e:
        print(f"Collection '{coll_name}' not found: {e}")
        return []

    if embedder is None:
        embedder = config.get_embedder()

    query_embedding = embedder.encode([query])[0].tolist()
    res = coll.query(
        query_embeddings=[query_embedding],
        n_results=min(k, coll.count()) if coll.count() > 0 else k,
        include=["documents", "metadatas", "distances"]
    )

    if not res["documents"] or not res["documents"][0]:
        return []

    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        score = max(0.0, 1.0 - (dist / 2.0)) if dist > 1.0 else max(0.0, 1.0 - dist)
        hits.append({
            "control_id": meta.get("control_id", ""),
            "title": meta.get("title", ""),
            "text": doc,
            "jurisdiction": meta.get("jurisdiction", ""),
            "framework": meta.get("framework", ""),
            "source_file": meta.get("source_file", ""),
            "collection_name": coll_name,
            "similarity_score": round(score, 4),
            "distance": round(dist, 4)
        })
    return hits


def retrieve_multi_framework(
    query: str,
    frameworks: List[str],
    k_per_framework: int = 3,
    fusion_method: str = "rrf",
    chroma_dir: str = DEFAULT_FRAMEWORKS_DIR,
    embedder: Optional[SentenceTransformer] = None
) -> List[Dict[str, Any]]:
    """
    Scatter-Gather Multi-Framework Retrieval:
    Queries each specified framework collection independently, then fuses and reranks results
    using Reciprocal Rank Fusion (RRF) with similarity scores.
    """
    if embedder is None:
        embedder = config.get_embedder()

    resolved_collections = []
    for fw in frameworks:
        coll = resolve_collection_name(fw, chroma_dir)
        if coll and coll not in resolved_collections:
            resolved_collections.append(coll)

    if not resolved_collections:
        return []

    all_hits_by_framework = {}
    for coll_name in resolved_collections:
        hits = retrieve_single_framework(
            query=query,
            framework=coll_name,
            k=k_per_framework,
            chroma_dir=chroma_dir,
            embedder=embedder
        )
        all_hits_by_framework[coll_name] = hits

    # Apply Reciprocal Rank Fusion (RRF)
    # RRF Score = 1 / (60 + rank) + (similarity_score * 0.5)
    fused_results = []
    for coll_name, hits in all_hits_by_framework.items():
        for rank, hit in enumerate(hits, start=1):
            rrf_score = (1.0 / (60.0 + rank)) + (hit["similarity_score"] * 0.5)
            hit_copy = dict(hit)
            hit_copy["rrf_score"] = round(rrf_score, 5)
            hit_copy["rank_within_framework"] = rank
            fused_results.append(hit_copy)

    # Sort globally by fused score descending
    fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused_results
