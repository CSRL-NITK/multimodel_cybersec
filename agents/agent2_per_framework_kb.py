"""
Agent 2 (Per-Framework Edition) — Isolated Vector Knowledge Base Builder
-------------------------------------------------------------------------
Loads structured controls from structured_controls/*.json and creates a
DEDICATED ChromaDB collection for each framework.

Features:
- 1 collection per framework (e.g. controls__nist__csf, controls__eu__gdpr, controls__us__hipaa)
- Clean Chroma-safe naming conventions
- Duplicate control ID resolution (__dup2, __dup3)
- Memory-safe batch encoding with GPU/CPU fallback
- JSON manifest generation tracking all framework collections and counts
"""

import os
import re
import json
import glob
from collections import Counter
from typing import Dict, List, Any, Optional
import torch
import chromadb
from sentence_transformers import SentenceTransformer

try:
    import agents.config as config
except ImportError:
    import config

DEFAULT_FRAMEWORKS_DIR = getattr(config, "CHROMA_FRAMEWORKS_DIR", "chroma_db_frameworks")


def sanitize_collection_name(name: str) -> str:
    """Ensure ChromaDB compliant collection name: 3-63 chars, [a-zA-Z0-9_-]."""
    clean = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower().strip())
    clean = re.sub(r'_+', '_', clean).strip('_')
    if len(clean) < 3:
        clean = f"coll_{clean}"
    return clean[:63]


def get_framework_collection_name(jurisdiction: str, framework: str) -> str:
    raw = f"controls__{jurisdiction}__{framework}"
    return sanitize_collection_name(raw)


def load_framework_file(file_path: str) -> Dict[str, Any]:
    """Loads a single framework JSON file and returns normalized controls."""
    filename = os.path.basename(file_path).replace(".json", "")
    parts = filename.split("__")
    default_jurisdiction = parts[0] if len(parts) > 0 else "general"
    default_framework = parts[1] if len(parts) > 1 else filename

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_controls = []
    if isinstance(data, list):
        raw_controls = data
    elif isinstance(data, dict):
        if "controls" in data and isinstance(data["controls"], list):
            raw_controls = data["controls"]
        else:
            # Check for top-level list-like values
            for v in data.values():
                if isinstance(v, list):
                    raw_controls.extend(v)

    normalized_controls = []
    for c in raw_controls:
        if not isinstance(c, dict):
            continue
        control_id = c.get("control_id") or c.get("id") or "UNKNOWN"
        title = c.get("title") or c.get("name") or control_id
        desc = c.get("description") or c.get("text") or ""
        jurisdiction = c.get("jurisdiction") or default_jurisdiction
        framework = c.get("framework") or default_framework
        source_file = c.get("source_file") or os.path.basename(file_path)

        normalized_controls.append({
            "control_id": str(control_id),
            "title": str(title),
            "description": str(desc),
            "jurisdiction": str(jurisdiction),
            "framework": str(framework),
            "source_file": str(source_file),
        })

    return {
        "file_path": file_path,
        "filename": filename,
        "jurisdiction": default_jurisdiction,
        "framework": default_framework,
        "controls": normalized_controls
    }


def generate_unique_ids(controls: List[Dict[str, Any]]) -> List[str]:
    """Generate collision-free IDs within a single framework collection."""
    base_ids = [c["control_id"] for c in controls]
    counter = Counter()
    unique_ids = []

    for base in base_ids:
        counter[base] += 1
        if counter[base] == 1:
            unique_ids.append(base)
        else:
            unique_ids.append(f"{base}__dup{counter[base]}")
    return unique_ids


def safe_encode(embedder: SentenceTransformer, texts: List[str], batch_size: int = 16) -> List[List[float]]:
    """Encodes texts using SentenceTransformer with CUDA safety and CPU fallback."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        embeddings = embedder.encode(texts, batch_size=batch_size, show_progress_bar=False)
        return embeddings.tolist()
    except Exception as e:
        print(f"  ⚠️ GPU encoding fallback to CPU due to: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        embedder_cpu = SentenceTransformer(config.EMBED_MODEL_NAME, device="cpu")
        embeddings = embedder_cpu.encode(texts, batch_size=32, show_progress_bar=False)
        return embeddings.tolist()


def build_per_framework_collections(
    structured_dir: str = config.STRUCTURED_CONTROLS_DIR,
    chroma_dir: str = DEFAULT_FRAMEWORKS_DIR,
    reset_all: bool = False
) -> Dict[str, Any]:
    """
    Iterates through all framework files and creates a separate Chroma collection for each.
    """
    os.makedirs(chroma_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir)
    embedder = config.get_embedder()

    json_files = sorted(glob.glob(os.path.join(structured_dir, "*.json")))
    if not json_files:
        print(f"❌ No structured control files found in {structured_dir}")
        return {}

    manifest = {
        "total_frameworks": 0,
        "total_controls": 0,
        "chroma_dir": chroma_dir,
        "collections": {}
    }

    print(f"🚀 Starting Per-Framework Vector DB Ingestion ({len(json_files)} framework files)...")
    print(f"📁 Target Chroma Directory: {chroma_dir}")

    for file_path in json_files:
        fw_data = load_framework_file(file_path)
        controls = fw_data["controls"]
        if not controls:
            print(f"⏩ Skipping {fw_data['filename']} (0 controls)")
            continue

        coll_name = get_framework_collection_name(fw_data["jurisdiction"], fw_data["framework"])

        if reset_all:
            try:
                client.delete_collection(coll_name)
            except Exception:
                pass

        collection = client.get_or_create_collection(coll_name, embedding_function=None)

        texts = [f"{c['title']}. {c['description']}" for c in controls]
        ids = generate_unique_ids(controls)
        metadatas = [
            {
                "control_id": c["control_id"],
                "title": c["title"][:200],
                "jurisdiction": c["jurisdiction"],
                "framework": c["framework"],
                "source_file": c["source_file"],
            }
            for c in controls
        ]

        # Batch encode and upsert
        BATCH_SIZE = 1000
        for i in range(0, len(controls), BATCH_SIZE):
            end_idx = min(i + BATCH_SIZE, len(controls))
            b_texts = texts[i:end_idx]
            b_ids = ids[i:end_idx]
            b_metas = metadatas[i:end_idx]
            b_embeddings = safe_encode(embedder, b_texts, batch_size=32)

            collection.upsert(
                ids=b_ids,
                embeddings=b_embeddings,
                documents=b_texts,
                metadatas=b_metas
            )

        manifest["total_frameworks"] += 1
        manifest["total_controls"] += len(controls)
        manifest["collections"][coll_name] = {
            "collection_name": coll_name,
            "framework": fw_data["framework"],
            "jurisdiction": fw_data["jurisdiction"],
            "source_file": os.path.basename(file_path),
            "control_count": len(controls),
            "sample_ids": ids[:5]
        }

        print(f"  ✅ [{manifest['total_frameworks']:02d}/{len(json_files)}] {coll_name:<36} -> {len(controls):>4} controls indexed")

    manifest_path = os.path.join(chroma_dir, "framework_collections_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✨ Completed! Indexed {manifest['total_controls']} controls across {manifest['total_frameworks']} isolated framework collections.")
    print(f"📜 Manifest saved to: {manifest_path}")

    return manifest


if __name__ == "__main__":
    build_per_framework_collections(reset_all=True)
