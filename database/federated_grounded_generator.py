"""
Federated Grounded Generator Pipeline
-------------------------------------
Integrates Per-Framework Vector DB retrieval with Qwen2.5 generation:
1. Retrieves verbatim statutory controls from isolated per-framework ChromaDB.
2. Injects exact statutory text into an auditor prompt.
3. Generates grounded compliance answers with verified citations.
"""

import os
import sys
import time
import re
from typing import Dict, List, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.federated_rag_utils import retrieve_single_framework, retrieve_multi_framework
import agents.config as agent_config


def build_grounded_context_block(hits: List[Dict[str, Any]]) -> str:
    """Formats retrieved vector hits into an unambiguous statutory context block."""
    if not hits:
        return "No specific regulatory controls found in knowledge base."
    
    blocks = []
    for idx, h in enumerate(hits, 1):
        cid = h.get("control_id", "UNKNOWN")
        title = h.get("title", "")
        text = h.get("text", "")
        fw = h.get("framework", "").upper()
        jur = h.get("jurisdiction", "").upper()
        
        blocks.append(
            f"--- CONTROL {idx} [{fw} / {jur}] ---\n"
            f"Control ID: {cid}\n"
            f"Title: {title}\n"
            f"Requirement Text: {text}"
        )
    return "\n\n".join(blocks)


def generate_grounded_compliance_answer(
    query: str,
    framework: str,
    k: int = 5,
    max_tokens: int = 350
) -> Dict[str, Any]:
    """
    Executes Per-Framework Grounded Generation:
    Retrieves from isolated framework vector store -> formats grounded context -> generates answer.
    """
    t0 = time.perf_counter()
    hits = retrieve_single_framework(query=query, framework=framework, k=k)
    retrieval_time_ms = (time.perf_counter() - t0) * 1000

    context_block = build_grounded_context_block(hits)

    prompt = f"""You are a certified regulatory compliance auditor. Answer the following audit question using ONLY the provided official regulatory controls.

OFFICIAL STATUTORY CONTROLS CONTEXT:
{context_block}

AUDIT QUESTION:
{query}

INSTRUCTIONS:
1. Cite the exact Control ID(s) or Article number(s) from the context provided above.
2. Explain the specific requirements clearly and concisely.
3. Do not invent or reference controls not present in the context.

AUDIT ASSESSMENT & REQUIREMENTS:"""

    t1 = time.perf_counter()
    answer = agent_config.generate(prompt, max_new_tokens=max_tokens)
    gen_time_ms = (time.perf_counter() - t1) * 1000
    total_time_ms = (time.perf_counter() - t0) * 1000

    # Extract cited control IDs
    cited_ids = []
    for h in hits:
        cid = h.get("control_id", "")
        if cid and cid.lower() in answer.lower():
            cited_ids.append(cid)

    return {
        "framework": framework,
        "query": query,
        "answer": answer,
        "retrieved_hits": hits,
        "retrieved_control_ids": [h["control_id"] for h in hits],
        "cited_control_ids": cited_ids,
        "retrieval_latency_ms": round(retrieval_time_ms, 2),
        "generation_latency_ms": round(gen_time_ms, 2),
        "total_latency_ms": round(total_time_ms, 2)
    }


def generate_ungrounded_answer(
    query: str,
    framework: str,
    max_tokens: int = 350
) -> Dict[str, Any]:
    """
    Baseline: Direct model generation WITHOUT vector DB grounding.
    """
    prompt = f"""You are a certified regulatory compliance auditor. Answer the following audit question regarding {framework.upper()} compliance:

AUDIT QUESTION:
{query}

INSTRUCTIONS:
Cite the specific official Control ID(s) / Article number(s) and detail the mandatory compliance requirements.

AUDIT ASSESSMENT & REQUIREMENTS:"""

    t0 = time.perf_counter()
    answer = agent_config.generate(prompt, max_new_tokens=max_tokens)
    total_time_ms = (time.perf_counter() - t0) * 1000

    return {
        "framework": framework,
        "query": query,
        "answer": answer,
        "total_latency_ms": round(total_time_ms, 2)
    }
