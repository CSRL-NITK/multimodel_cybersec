"""
Compliance & Cybersecurity Assistant - Streamlit Chat Interface
---------------------------------------------------------------
* LLM tokens stream in real time (TextIteratorStreamer + st.write_stream)
* Agents 3/4/5 run simultaneously with the LLM (ThreadPoolExecutor)
* NIST queries -> best LoRA adapter (auto-routed, name hidden from UI)
* EU/India/non-NIST queries -> compliance agents via ChromaDB
* No model names or adapter labels shown to the user

Run with:
    streamlit run app.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load environment variables from .env file (secrets, API keys, config)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Sync Streamlit Cloud secrets into os.environ
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, (str, int, float, bool)):
                os.environ[k] = str(v)
except Exception:
    pass


import re
import json
import glob
import time
import sys
import warnings
import asyncio

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*max_new_tokens.*")
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

# System Path Auto-Resolver for Restructured Subdirectories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in ["database", "governance", "utils", "ingestion", "evaluation", "agents"]:
    p = os.path.join(PROJECT_ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

# Configure HuggingFace cache directory (environment-aware with fallback)
hf_cache = os.getenv("HF_HOME", os.path.join(PROJECT_ROOT, "hf_cache"))
try:
    os.makedirs(hf_cache, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache
except Exception:
    pass

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
warnings.filterwarnings("ignore")


# Suppress Windows asyncio connection reset log noise
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

import concurrent.futures
from typing import Optional, Dict, List, Any
from threading import Thread
from datetime import datetime

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st
import torch
from core.system_settings import get_system_setting
from sentence_transformers import SentenceTransformer
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

try:
    import rag_utils
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

try:
    import database.self_healing_rag as self_healing_rag
    SELF_HEALING_AVAILABLE = True
except ImportError:
    try:
        import self_healing_rag
        SELF_HEALING_AVAILABLE = True
    except ImportError:
        SELF_HEALING_AVAILABLE = False

try:
    import compliance_jurisdictions
    JURISDICTION_REGISTRY_AVAILABLE = True
except ImportError:
    JURISDICTION_REGISTRY_AVAILABLE = False

try:
    import agentic_router
    AGENTIC_ROUTER_AVAILABLE = True
except ImportError:
    AGENTIC_ROUTER_AVAILABLE = False

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
BASE_MODEL_NAME              = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTERS_DIR                 = "adapters"
EMBED_MODEL_NAME             = "all-MiniLM-L6-v2"
SAMPLES_PER_DOMAIN           = 50
ROUTING_CONFIDENCE_THRESHOLD = 0.25

LENGTH_PRESETS = {
    "Short":  ("Provide a concise 2-3 sentence overview.", 30, 300),
    "Medium": ("Provide a clear, well-structured compliance answer in 3-4 paragraphs.", 80, 768),
    "Long":   ("Provide a detailed, multi-section compliance analysis organized under clear Markdown headings.", 150, 1536),
}

# -------------------------------------------------------------------------
# Page config & CSS
# -------------------------------------------------------------------------
st.set_page_config(page_title="Cybersecurity Compliance Platform", page_icon="shield", layout="wide")

from core.session_state import init_session_state, sync_user_session, auto_save_current_session, format_relative_time
from core.system_settings import get_system_setting, set_system_setting
init_session_state()
sync_user_session()
_auto_save_current_session = auto_save_current_session
_format_relative_time = format_relative_time

# -------------------------------------------------------------------------
# User Role & Session Info
# -------------------------------------------------------------------------
user_role = st.session_state.get("user_role", "guest")
username = st.session_state.get("username", "guest")

# Load external CSS stylesheet
THEME_CSS_PATH = os.path.join(PROJECT_ROOT, "styles", "theme.css")
if os.path.exists(THEME_CSS_PATH):
    with open(THEME_CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>\n{f.read()}\n</style>", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# High-Tech App Launch Intro Splash Screen Overlay
# -------------------------------------------------------------------------
try:
    import ui.intro_splash as ui_intro
    ui_intro.render_intro_splash_if_needed()
except Exception as _intro_exc:
    pass


# -------------------------------------------------------------------------
# Cached loaders & Device Resolution
# -------------------------------------------------------------------------
from core.model_loading import (
    get_device,
    load_model_and_tokenizer,
    check_and_load_new_adapters,
    load_router,
    get_domain_keywords
)


def route(embedder, centroids, query: str):
    q    = embedder.encode([query])[0]
    sims = {
        d: float(np.dot(q, c) / (np.linalg.norm(q) * np.linalg.norm(c)))
        for d, c in centroids.items()
    }
    
    ql = query.lower()

    # Disambiguate NIST vs NIS2: penalize NIS2 if query explicitly says 'nist' without 'nis2'
    if re.search(r'\bnist\b', ql) and not re.search(r'\bnis2\b', ql):
        if "qwen3-nis2-lora" in sims:
            sims["qwen3-nis2-lora"] -= 0.40

    domain_keywords = get_domain_keywords()

    # Exact keyword priority boost
    for domain, keywords in domain_keywords.items():
        if domain in sims:
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', ql):
                    sims[domain] += 0.30
                    break

    return max(sims, key=sims.get), sims


# -------------------------------------------------------------------------
# Streaming LLM answer
# -------------------------------------------------------------------------
def rag_retrieve(embedder, rag_index, centroids, query, top_k=3):
    """
    Retrieve the top-K most relevant Q&A pairs across ALL domains.
    Returns (top_hits, primary_domain, router_sims).
    """
    best_domain, sims = route(embedder, centroids, query)

    if not rag_index:
        return [], best_domain, sims

    q_emb  = embedder.encode([query])[0]
    q_norm = np.linalg.norm(q_emb)

    all_hits = []

    # Query ChromaDB controls collection for self-healed ground-truth entries
    try:
        import agent_config
        import chromadb
        client = chromadb.PersistentClient(path=agent_config.CHROMA_DB_DIR)
        collection = client.get_or_create_collection("controls", embedding_function=None)
        q_emb_list = q_emb.tolist()
        c_res = collection.query(
            query_embeddings=[q_emb_list],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        if c_res and c_res.get("documents") and c_res["documents"][0]:
            for doc, meta, dist in zip(c_res["documents"][0], c_res["metadatas"][0], c_res["distances"][0]):
                sim = max(0.0, 1.0 - (dist / 2.0)) if dist > 1.0 else max(0.0, 1.0 - dist)
                if sim >= 0.35:
                    all_hits.append({
                        "sim": float(sim),
                        "domain": meta.get("framework", "verified_ground_truth"),
                        "instruction": meta.get("title", f"Verified Answer: {query[:30]}"),
                        "output": doc,
                    })
    except Exception:
        pass

    for domain, idx_data in rag_index.items():
        embs  = idx_data["embeddings"]
        norms = np.linalg.norm(embs, axis=1) * q_norm
        cos_s = np.dot(embs, q_emb) / np.where(norms == 0, 1, norms)
        top_indices = np.argsort(cos_s)[-top_k:][::-1]
        for i in top_indices:
            all_hits.append({
                "sim": float(cos_s[i]),
                "domain": domain,
                "instruction": idx_data["examples"][i]["instruction"],
                "output": idx_data["examples"][i]["output"],
            })

    all_hits.sort(key=lambda x: x["sim"], reverse=True)
    top_hits = all_hits[:top_k]

    if top_hits:
        from collections import Counter
        domain_counts = Counter(h["domain"] for h in top_hits)
        primary_domain = domain_counts.most_common(1)[0][0]
    else:
        primary_domain = best_domain

    return top_hits, primary_domain, sims


def match_adapter_for_framework(target_framework: str, loaded_adapters: list[str]) -> Optional[str]:
    """Finds the adapter corresponding to target_framework (e.g. 'eu/gdpr' -> 'qwen3-gdpr-lora')."""
    if not target_framework or target_framework == "Auto-Detect (Smart Route)":
        return None
    fw_clean = target_framework.split("/")[-1].lower().replace("_", "").replace("-", "")
    for adapter in loaded_adapters:
        adapter_clean = adapter.lower().replace("_", "").replace("-", "")
        if fw_clean in adapter_clean:
            return adapter
    return None


def answer_hybrid(model, tokenizer, device, embedder, centroids,
                  query, length_label, rag_index=None, history=None,
                  use_self_healing=False, target_framework=None):
    """
    Unified compliance generation combining specialized LoRA adapters + RAG knowledge:
      - Automatically sets adapter for selected/routed framework.
      - Provides full system instructions & RAG reference knowledge so answers are complete & detailed.

    Returns (token_generator, metadata_dict, thread).
    """
    HIGH_CONFIDENCE = 0.50

    # 0. Delegate multi-agent tasks to LangGraph StateGraph Application
    if AGENTIC_ROUTER_AVAILABLE:
        query_l = query.lower()
        if any(k in query_l for k in ["compare", "versus", "v/s", "map", "mapping", "probe", "scan", "live test", "onboard", "fine-tune"]):
            agent_app = agentic_router.build_master_compliance_graph()
            classifier_state = {
                "query": query, "intent": "", "base_jurisdiction": "", "base_framework": target_framework or "",
                "compare_jurisdiction": "", "compare_framework": "", "file_path": "", "target_url": "", "repo_path": "",
                "controls": [], "mappings": [], "assessment": [], "discovered_endpoints": [], "probe_results": [],
                "report_path": "", "requires_approval": False, "approved": False, "execution_logs": [], "output": ""
            }
            final_res = agent_app.invoke(classifier_state)
            out_text = final_res.get("output", "")
            
            def _agentic_stream():
                yield out_text
                
            metadata = {
                "low_confidence": False,
                "domain": final_res.get("base_framework") or "Agentic Router",
                "active_adapter": f"LangGraph ({final_res.get('intent')})",
                "target_framework": target_framework,
                "sims": {},
                "generation_mode": f"LangGraph Multi-Agent ({final_res.get('intent')})",
                "router_confidence": 1.0,
                "rag_hits": [],
                "top_similarity": 1.0,
            }
            return _agentic_stream(), metadata, None

    # Check framework mismatch when target framework is locked
    if target_framework and target_framework != "Auto-Detect (Smart Route)":
        detected_intent_info = detect_intent(query)
        query_fw = detected_intent_info.get("framework")
        is_comparison = any(w in query.lower() for w in ["compare", "versus", "v/s", "vs", "difference", "differ"])
        
        # Mismatch detected if query mentions a specific different framework without asking for comparison
        if query_fw and query_fw.lower() != target_framework.lower() and not is_comparison:
            target_name = format_framework_display_name(target_framework)
            query_name = format_framework_display_name(query_fw)

            def _mismatch_stream():
                yield (
                    f"🎯 **Active Framework Focus Enforcement**\n\n"
                    f"The Compliance Assistant is currently locked to **{target_name}** (`{target_framework}`).\n\n"
                    f"Your query pertains to **{query_name}** (`{query_fw}`).\n\n"
                    f"To query **{query_name}**, please switch your focus in the **➕ Focus** menu at the bottom left, "
                    f"or select `Auto-Detect (Smart Route)`.\n\n"
                    f"*(Alternatively, ask a compliance question related to **{target_name}**).* "
                )

            metadata = {
                "low_confidence": False,
                "domain": target_framework,
                "active_adapter": None,
                "target_framework": target_framework,
                "sims": {},
                "generation_mode": "Framework Mismatch Guardrail",
                "router_confidence": 1.0,
                "rag_hits": [],
                "top_similarity": 0.0,
            }
            return _mismatch_stream(), metadata, None

    # Inject Target Framework Focus instruction into query if explicitly selected
    if target_framework and target_framework != "Auto-Detect (Smart Route)":
        query_effective = (
            f"[Active Framework Focus: {target_framework}]\n{query}\n\n"
            f"(Note: Prioritize principles, requirements, and compliance rules of {target_framework} while answering. "
            f"If the user asks about another framework, explain it and compare/relate it to {target_framework}.)"
        )
    else:
        query_effective = query

    top_hits, primary_domain, sims = rag_retrieve(
        embedder, rag_index, centroids, query_effective, top_k=3
    )

    # Suppress RAG context and source tracking for queries with low confidence / similarity scores
    RAG_CONFIDENCE_THRESHOLD = 0.35
    if top_hits and top_hits[0]["sim"] < RAG_CONFIDENCE_THRESHOLD:
        top_hits = []


    best_domain, _ = route(embedder, centroids, query_effective)
    router_confidence = sims.get(best_domain, 0.0)
    loaded_adapters = check_and_load_new_adapters(model, list(centroids.keys()))
    length_instruction, min_tok, max_tok = LENGTH_PRESETS[length_label]

    # Check if query is a general explanation/overview or multi-framework comparison question requiring RAG synthesis
    GENERAL_EXPLANATION_KEYWORDS = [
        "what is", "explain", "describe", "overview", "summary", "report",
        "assess", "principles", "scope", "requirements", "directive", "act",
        "guidelines", "framework", "introduction", "tell me about", "details",
        "difference", "different", "vs", "versus", "compare", "comparison", "contrast",
        "is ", "are ", "does ", "can ", "where ", "when ", "how ", "implemented", "applicable", "applies"
    ]
    is_general_query = any(k in query.lower() for k in GENERAL_EXPLANATION_KEYWORDS)

    # Match adapter: use Base Model + RAG for general overview queries (to generate rich multi-section reports), 
    # and use fine-tuned LoRA Adapter for specific control lookups / technical assessments.
    target_adapter = match_adapter_for_framework(target_framework, loaded_adapters)
    if target_adapter and not is_general_query:
        active_adapter = target_adapter
    elif best_domain in loaded_adapters and router_confidence >= HIGH_CONFIDENCE and not is_general_query:
        active_adapter = best_domain
    else:
        active_adapter = None

    if active_adapter:
        model.set_adapter(active_adapter)
        generation_mode = f"Adapter ({active_adapter})"
    else:
        generation_mode = "RAG + Base Model"

    dynamic_defs = []
    try:
        import compliance_jurisdictions as _cj
        for reg in _cj.get_registered_standards():
            sc = reg.get("short_code", reg["key"])
            title = reg.get("title", "")
            cat = reg.get("category", "")
            ver = reg.get("version", "")
            jur = reg.get("jurisdiction", "")
            gov = reg.get("governing_body", "")
            dynamic_defs.append(f"- {sc} ({ver}) [Jurisdiction: {jur} | Governing Body: {gov}]: {title} [{cat}]")
    except Exception:
        pass

    defs_block = "\n".join(dynamic_defs) if dynamic_defs else "- Dynamic Regulatory Compliance & Cybersecurity Standards"

    SYSTEM_INSTRUCTION = (
        "You are an expert compliance & cybersecurity AI assistant. "
        "Provide clear, accurate, and structured answers directly addressing the user's question. "
        "Never generate unrelated topics or lists of random items. "
        "Keep your output strictly focused on cybersecurity, AI governance, and data privacy regulations.\n\n"
        "Statutory Accuracy & Integrity Rules:\n"
        "1. Strictly adhere to statutory framework taxonomies. For NIST AI RMF (1.0), the 4 core functions are strictly GOVERN, MAP, MEASURE, and MANAGE. Never invent or hallucinate alternate taxonomies like 'Plan', 'Design', 'Verify'.\n"
        "2. When displaying comparison or mapping data, ALWAYS use clean Markdown pipe-delimited tables (e.g., | Aspect | NIS2 | DPDP |) with clear headers. "
        "Do NOT use raw tabs, commas, or unaligned text spaces for tables. Ensure all cells have meaningful content.\n"
        "3. Do NOT insert literal newlines or HTML line breaks (like <br> or <br/>) inside Markdown table cells. If you need to separate points in a cell, use semicolons or simple spaces.\n"
        "4. Do NOT print or mention raw vector database similarity scores or match percentages in your written response text.\n"
        "5. Ensure that the compliance report/comparison provides deep, analytical insights mapping the frameworks.\n"
        "6. Do NOT invent or hallucinate fake programming libraries or mock API packages.\n\n"
        f"Active Ingested Frameworks & Standards (Registry Metadata):\n{defs_block}"
    )

    # Always inject reference knowledge context if available
    if top_hits:
        context_lines = []
        for hit in top_hits:
            context_lines.append(f"Q: {hit['instruction']}\nA: {hit['output']}")
        context_block = "\n\n".join(context_lines)
        user_content = (
            f"Based on the following reference compliance knowledge, "
            f"{length_instruction.lower()}\n\n"
            f"--- Reference Knowledge ---\n{context_block}\n"
            f"--- End ---\n\nQuestion: {query}"
        )
        if length_label == "Short":
            user_content += (
                "\n\nFormatting Guidelines: Keep your response brief, concise, and direct (2-3 sentences max). Do not use multiple section headings."
            )
        elif length_label == "Medium":
            user_content += (
                "\n\nFormatting Guidelines: Provide a concise executive overview organized into 2-3 focused sections (e.g. ## Executive Summary, ## Key Requirements, ## Key Takeaways)."
            )
        elif length_label == "Long":
            # Check if this is a comparative report or assessment audit query
            REPORT_KEYWORDS = ["report", "compare", "comparison", "contrast", "versus", "vs", "difference", "map", "mapping", "assess", "assessment", "audit"]
            is_report_query = any(k in query.lower() for k in REPORT_KEYWORDS)
            
            if is_report_query:
                user_content += (
                    "\n\nFormatting Guidelines: Provide an exhaustive, deep-dive compliance audit report. "
                    "Organize your analysis using 5 detailed section headings:\n"
                    "1. ## Executive Summary & Regulatory Scope\n"
                    "2. ## Detailed Regulatory & Standard Comparison (Present comparison details in a clean Markdown table with analytical insights; do not include raw similarity scores or empty cells, and do not put newlines or HTML breaks like <br> in cells)\n"
                    "3. ## Technical Capabilities & Control Mapping (Highlight compliant/non-compliant statuses with auditor explanations)\n"
                    "4. ## Legal Obligations, Penalties & Enforcement\n"
                    "5. ## Enterprise Implementation Roadmap & Remediation Steps\n"
                    "Elaborate extensively under every section with maximum technical depth and clarity."
                )
            else:
                user_content += (
                    "\n\nFormatting Guidelines: Provide a detailed, exhaustive, and structured explanation of the topic. "
                    "Organize your response into logical sections using descriptive Markdown headings. "
                    "Provide complete technical details, definitions, and clear conceptual examples where relevant (do not generate fake code blocks or hallucinate non-existent libraries)."
                )

    else:
        user_content = query


    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_tok,
        min_new_tokens=20,
        do_sample=True,
        temperature=0.3,
        top_p=0.85,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    if active_adapter:
        t = Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
    else:
        def _generate_with_disabled_adapter(**kwargs):
            with model.disable_adapter():
                model.generate(**kwargs)
        t = Thread(target=_generate_with_disabled_adapter, kwargs=gen_kwargs, daemon=True)

    t.start()

    # Generator filter to strip any residual <think>...</think> tags
    def filtered_stream():
        in_think = False
        for tok in streamer:
            if "<think>" in tok:
                in_think = True
                if "</think>" in tok:
                    in_think = False
                    after = tok.split("</think>", 1)[1]
                    if after.strip():
                        yield after
                continue
            if in_think:
                if "</think>" in tok:
                    in_think = False
                    after = tok.split("</think>", 1)[1]
                    if after.strip():
                        yield after
                continue
            yield tok

    # Track which adapter was loaded so we can clean it up after generation
    _active_adapter_for_cleanup = active_adapter

    def cleanup_adapter_after_generation():
        """Unloads the LoRA adapter from GPU VRAM after generation to free memory for agent pipeline."""
        if _active_adapter_for_cleanup:
            try:
                import gc
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # Wrap filtered_stream to auto-cleanup adapter after streaming completes
    def streaming_with_cleanup():
        yield from filtered_stream()
        cleanup_adapter_after_generation()

    metadata = {
        "low_confidence": router_confidence < 0.40,
        "domain": active_adapter or primary_domain,
        "active_adapter": active_adapter,
        "target_framework": target_framework,
        "sims": sims,
        "generation_mode": generation_mode,
        "router_confidence": router_confidence,
        "rag_hits": top_hits,
        "top_similarity": top_hits[0]["sim"] if top_hits else 0.0,
    }

    try:
        import pipeline_logger
        pipeline_logger.log_info(
            "hybrid_inference",
            extra={"domain": active_adapter or primary_domain, "generation_mode": generation_mode, "confidence": router_confidence}
        )
    except Exception:
        pass
    return streaming_with_cleanup(), metadata, t


# -------------------------------------------------------------------------
# Compliance agents (3 / 4 / 5)
# -------------------------------------------------------------------------
import sys as _sys
_agents_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents")
if _agents_path not in _sys.path:
    _sys.path.insert(0, _agents_path)

try:
    import agent3_control_mapping       as _a3
    import agent4_compliance_assessment as _a4
    import agent5_report_generation     as _a5
    import config as _cfg
    COMPLIANCE_AGENTS_AVAILABLE = True
except ImportError:
    COMPLIANCE_AGENTS_AVAILABLE = False

_sc_dir = "structured_controls"
_available_fw: list[str] = []
if os.path.isdir(_sc_dir):
    for _fn in sorted(os.listdir(_sc_dir)):
        if _fn.endswith(".json"):
            parts = _fn.replace(".json", "").split("__")
            if len(parts) == 2:
                _available_fw.append(f"{parts[0]}/{parts[1]}")


def get_default_jurisdiction_framework() -> tuple[str, str]:
    """Dynamically resolves default framework from scanned structured_controls instead of hardcoded defaults."""
    if _available_fw:
        parts = _available_fw[0].split("/")
        if len(parts) == 2:
            return parts[0], parts[1]
    return "general", "compliance"


def format_framework_display_name(fw_str: str) -> str:
    """Dynamically formats 'jurisdiction/framework' into clean, readable display text (e.g. 'EU GDPR', 'US NIST AI RMF')."""
    if not fw_str or fw_str == "Auto-Detect (Smart Route)":
        return fw_str or ""
    try:
        import compliance_jurisdictions as _cj
        if fw_str in _cj.STANDARDS_VERSION_REGISTRY:
            meta = _cj.STANDARDS_VERSION_REGISTRY[fw_str]
            jur = meta.get("jurisdiction", "").upper()
            sc = meta.get("short_code", "")
            return f"{jur} {sc}".strip() if sc else f"{jur} {fw_str}"
    except Exception:
        pass
    parts = fw_str.split("/")
    if len(parts) == 2:
        jur, std = parts[0].upper(), parts[1].replace("_", " ").replace("-", " ").upper()
        return f"{jur} {std}"
    return fw_str.replace("_", " ").replace("-", " ").title()


def _format_pointwise_explanation(explanation: str) -> str:
    explanation = explanation.strip()
    if not explanation:
        return "\n  * **Assessment Status:** Assessed against evidence profile."

    explanation = re.sub(r"\s+", " ", explanation)
    
    # Split by section headers if present (e.g., Technical Requirements:, Potential Risks:, Evidence Details:)
    pattern = r'(?=\b(?:Technical Requirements|Potential Risks|Evidence Details|Mandatory restriction|Control Requirement|Audit Finding|Remediation Action|Skipped Test|Scan Findings):)'
    parts = [p.strip() for p in re.split(pattern, explanation, flags=re.IGNORECASE) if p.strip()]

    bullet_items = []
    if len(parts) > 1:
        for part in parts:
            if ":" in part[:40]:
                hdr, body = part.split(":", 1)
                bullet_items.append(f"  * **{hdr.strip().title()}:** {body.strip()}")
            else:
                bullet_items.append(f"  * {part}")
    else:
        # Split paragraph into clean sentence bullets
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', explanation) if s.strip()]
        for s in sentences:
            if len(s) < 10:
                continue
            s_lower = s.lower()
            if any(w in s_lower for w in ["require", "mandate", "must", "shall", "policy", "obligation"]):
                bullet_items.append(f"  * **Control Requirement:** {s}")
            elif any(w in s_lower for w in ["restriction", "aes", "encryption", "timeframe", "limit", "sync"]):
                bullet_items.append(f"  * **Technical Restrictions:** {s}")
            elif any(w in s_lower for w in ["test", "finding", "skipped", "file", "evidence", "zap"]):
                bullet_items.append(f"  * **Evidence & Scan Details:** {s}")
            elif any(w in s_lower for w in ["risk", "vulnerability", "exposed", "gap", "lack", "threat"]):
                bullet_items.append(f"  * **Potential Risks:** {s}")
            else:
                bullet_items.append(f"  * **Audit Finding:** {s}")

    return "\n" + "\n".join(bullet_items) if bullet_items else f"\n  * {explanation}"


def _format_pointwise_remediation(remediation: str) -> str:
    remediation = remediation.strip()
    if not remediation or remediation.lower() in ("none required.", "none"):
        return "None required."
    
    # Clean inline backticks to prevent horizontal line overflow
    remediation = re.sub(r"`", "", remediation)
    
    # Split numbered steps e.g. "1. **Step**: ... 2. **Step**: ..."
    if re.search(r'\d+\.\s+\*\*', remediation):
        steps = [s.strip() for s in re.split(r'(?=\d+\.\s+\*\*)', remediation) if s.strip()]
        return "\n" + "\n".join([f"  * {st}" for st in steps])
    elif re.search(r'\d+\.\s+', remediation):
        steps = [s.strip() for s in re.split(r'(?=\d+\.\s+)', remediation) if s.strip()]
        return "\n" + "\n".join([f"  * {st}" for st in steps])
    else:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', remediation) if s.strip()]
        if len(sentences) > 1:
            return "\n" + "\n".join([f"  * {s}" for s in sentences])
        return remediation


def _clean_report_list(items, default_ev="Vault Evidence", show_rationale=True):
    if not items:
        return "- None"
    formatted = []
    for r in items:
        cid = r.get('control_id', 'CTRL').strip()
        title = r.get('title', 'Control Requirement').strip()
        
        cid_clean = cid.strip().upper()
        if COMPLIANCE_AGENTS_AVAILABLE and hasattr(_a4, 'WSTG_TITLE_MAP') and cid_clean in _a4.WSTG_TITLE_MAP:
            title = _a4.WSTG_TITLE_MAP[cid_clean]
        else:
            # Strip PDF artifact noise & embedded keywords (headers, dates, Remediation, Summary, ports)
            title = re.sub(r"(Date|Content-Type|Content-Length|ETag|Connection|Server|X-Powered-By):[^\n]*", "", title, flags=re.IGNORECASE)
            title = re.sub(r"Web Security Testing Guide v\d+(\.\d+)?", "", title, flags=re.IGNORECASE)
            title = re.sub(r"^(\d+\.\d+(\.\d+)?|:\d+|Remediation|Summary|Description|Objective|Test\s+Objectives)\s*[-:]?\s*", "", title, flags=re.IGNORECASE)
            title = re.sub(r"^\s*[-:]\s*", "", title)
            title = re.sub(r"[\s,]+[a-zA-Z]{1,2}$", "", title)  # Strip stray trailing single letters/dangling fragments
            title = re.sub(r"\s+", " ", title).strip()
            if not title or len(title) <= 3:
                title = "Security Safeguard Requirement"
        
        explanation = r.get('explanation') or r.get('rationale') or 'Assessed against evidence profile.'
        explanation = re.sub(r"\s+", " ", explanation).strip()
        explanation = re.sub(r"^<[^>]+>\s*", "", explanation)
        explanation = re.sub(r"^\[[^\]]+\]\s*", "", explanation)
        if not explanation or "No organizational evidence file matched" in explanation:
            explanation = (
                f"Control safeguard '{title}' ({cid}) mandates verifiable technical controls and administrative policies. "
                f"No matching organizational evidence was identified in the vault with required confidence, "
                f"creating an unverified compliance gap that requires formal audit documentation and operational verification."
            )
        
        remediation = r.get('remediation', '')
        remediation = re.sub(r"\s+", " ", remediation).strip()
        remediation = re.sub(r"^<[^>]+>\s*", "", remediation)
        remediation = re.sub(r"^\[[^\]]+\]\s*", "", remediation)
        if not remediation or "Provide evidence documentation" in remediation:
            remediation = f"Develop and publish formal operational documentation, security logs, or architectural procedures proving active enforcement of control {cid} ({title})."
        
        ev = r.get('evidence_source', default_ev)
        ev_type = r.get('evidence_type', 'document_claim')
        if "Unified_Finding" in str(ev) or ev_type == "dynamic_scan":
            ev_tag = "⚡ **[Dynamic Scan Test]**"
        elif ev_type == "untested" or "Untested" in str(ev) or not ev:
            ev_tag = "⚪ **[Untested / No Data]**"
        elif any(str(ev).endswith(ext) for ext in [".py", ".js", ".ts", ".php", ".go", ".java", ".json", ".yaml", ".yml", ".dockerfile", ".toml", ".env"]) or "Extracted" in str(ev) or ":" in str(ev) or "/" in str(ev):
            ev_tag = "💻 **[Repository Code Inspection]**"
        else:
            ev_tag = "📄 **[Document Claim]**"
        
        item_str = f"#### 🔹 `{cid}` — {title}\n"
        item_str += f"- **Auditor Explanation:**{_format_pointwise_explanation(explanation)}\n"
        if show_rationale and remediation and remediation != "None required.":
            item_str += f"- **Client Remediation Plan:** {_format_pointwise_remediation(remediation)}\n"
        item_str += f"- **Evidence Source:** {ev_tag} `{ev}`"
        
        formatted.append(item_str)
    return "\n\n".join(formatted)



def get_dynamic_keyword_map() -> dict[str, str]:
    """Dynamically builds a keyword-to-framework dictionary from registered standards metadata and ingested frameworks on disk."""
    kw_map = {}
    
    # 1. From compliance_jurisdictions STANDARDS_VERSION_REGISTRY
    try:
        import compliance_jurisdictions as _cj
        for fw_key, meta in _cj.STANDARDS_VERSION_REGISTRY.items():
            kw_map[fw_key.lower()] = fw_key
            sc = meta.get("short_code", "").lower()
            title = meta.get("title", "").lower()
            if sc:
                kw_map[sc] = fw_key
                kw_map[sc.replace(" ", "_")] = fw_key
                kw_map[sc.replace(" ", "")] = fw_key
                kw_map[sc.replace("-", " ")] = fw_key
            if title:
                kw_map[title] = fw_key
    except Exception:
        pass

    # 2. From all available frameworks on disk (_available_fw)
    for fw in _available_fw:
        kw_map[fw.lower()] = fw
        parts = fw.split("/")
        if len(parts) == 2:
            jur, std = parts[0].lower(), parts[1].lower()
            kw_map[std] = fw
            kw_map[std.replace("_", "")] = fw
            kw_map[std.replace("-", "")] = fw
            kw_map[std.replace("_", " ")] = fw
            kw_map[std.replace("-", " ")] = fw
            
            sub_tokens = [t for t in re.split(r"[_-]", std) if t]
            if len(sub_tokens) > 1:
                joined_space = " ".join(sub_tokens)
                kw_map[joined_space] = fw
                
                acronym = "".join([t[0] for t in sub_tokens if t])
                if len(acronym) >= 2:
                    kw_map[acronym] = fw
                
                for i in range(len(sub_tokens)):
                    for j in range(i + 2, len(sub_tokens) + 1):
                        ngram = " ".join(sub_tokens[i:j])
                        if len(ngram) >= 3 and ngram not in kw_map:
                            kw_map[ngram] = fw

    return kw_map


def detect_intent(text: str) -> dict:
    """
    Returns: intent, framework, base, compare
    intent values: 'map' | 'assess' | 'report' | 'list_frameworks' | 'llm_chat'
    """
    tl    = text.lower()
    fw_re = re.compile(
        r"([a-z0-9_-]+)[\\/]([a-z0-9_-]+)",
        re.IGNORECASE,
    )
    matches = fw_re.findall(tl)
    
    # Filter matches against known frameworks
    valid_matches = []
    for m in matches:
        cand = f"{m[0]}/{m[1]}".lower()
        if cand in _available_fw:
            valid_matches.append(cand)

    framework = valid_matches[0] if valid_matches else None
    base      = valid_matches[0] if len(valid_matches) >= 1 else None
    compare   = valid_matches[1] if len(valid_matches) >= 2 else None

    # Dynamic framework keyword search
    kw_map = get_dynamic_keyword_map()
    found_frameworks = []
    for kw, fwp in kw_map.items():
        if re.search(r'\b' + re.escape(kw) + r'\b', tl):
            if fwp not in found_frameworks:
                found_frameworks.append(fwp)

    if valid_matches:
        framework = valid_matches[0]
        base = framework
        compare = valid_matches[1] if len(valid_matches) >= 2 else (found_frameworks[1] if len(found_frameworks) >= 2 else None)
    elif found_frameworks:
        framework = found_frameworks[0]
        base = found_frameworks[0]
        compare = found_frameworks[1] if len(found_frameworks) >= 2 else None
    else:
        framework = base = compare = None

    INFORMATIONAL_PREFIXES = (
        "what ", "what's", "whats", "how ", "why ", "who ",
        "explain", "describe", "tell me", "give me", "define",
        "meaning of", "overview", "introduction", "summary", "is there",
    )
    # Explicit intent detection keywords: only trigger agent workflows on explicit command keywords
    if any(w in tl for w in ["assess", "audit", "evaluate compliance", "check compliance"]):
        intent = "assess" if framework else "llm_chat"
    elif any(w in tl for w in ["map controls", "control mapping", "cross map"]):
        intent = "map" if (base and compare and base != compare) else "llm_chat"
    elif any(w in tl for w in ["report", "generate report", "build report", "show report"]):
        intent = "report" if framework else "llm_chat"
    elif any(w in tl for w in ["what frameworks", "which frameworks", "available frameworks", "ingested frameworks"]):
        intent = "list_frameworks"
    else:
        intent = "llm_chat"

    return {
        "intent": intent, "framework": framework,
        "base": base, "compare": compare,
    }


# -------------------------------------------------------------------------
# Parallel agent execution (runs agents 3/4/5 concurrently)
# -------------------------------------------------------------------------
def run_agents_parallel(intent_info: dict) -> dict:
    """
    Submits all applicable agents to a thread pool simultaneously.
    Returns: {assessment, mappings, report_md, report_path, errors}
    """
    results = {
        "assessment": None, "mappings": None,
        "report_md": None, "report_path": None, "errors": [],
    }
    if not COMPLIANCE_AGENTS_AVAILABLE:
        return results

    fw      = intent_info.get("framework")
    base    = intent_info.get("base")
    compare = intent_info.get("compare")
    intent  = intent_info.get("intent")
    futures = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:

        # Agent 4: compliance assessment
        if fw:
            j, f = fw.split("/")
            futures["assessment"] = pool.submit(_a4.assess_compliance, j, f)

        # Agent 3: control mapping (needs two distinct frameworks)
        if base and compare and base != compare:
            bj, bf = base.split("/")
            cj, cf = compare.split("/")
            futures["mappings"] = pool.submit(_a3.map_controls, bj, bf, cj, cf, False)

        # Agent 5: report (runs agent 4 internally then agent 5)
        if intent == "report" and fw:
            def _gen_report(fw=fw):
                j2, f2 = fw.split("/")
                asmt   = _a4.assess_compliance(j2, f2)
                rmd    = _a5.build_report(j2, f2, asmt, with_remediation=True)
                os.makedirs("reports", exist_ok=True)
                from datetime import datetime as _dt
                ts    = _dt.now().strftime("%Y%m%d_%H%M%S")
                rpath = f"reports/{j2}__{f2}_report_{ts}.md"
                with open(rpath, "w", encoding="utf-8-sig") as fh:
                    fh.write(rmd)
                return rmd, rpath
            futures["report"] = pool.submit(_gen_report)

        # Collect results as they complete
        for key, fut in futures.items():
            try:
                val = fut.result(timeout=180)
                if key == "report":
                    results["report_md"], results["report_path"] = val
                else:
                    results[key] = val
            except Exception as exc:
                results["errors"].append(f"{key}: {exc}")

    return results


# -------------------------------------------------------------------------
# Render expandable agent detail panels
# -------------------------------------------------------------------------
def render_agent_details(d: dict, key_suffix: str = ""):
    if d.get("type") == "mappings":
        with st.expander(f"Control Mapping: {d.get('base','?')} to {d.get('compare','?')}"):
            rows = []
            for m in d["mappings"]:
                src, tgt = m["source_control"], m["target_control"]
                sid = src.get("id", "?")
                tid = tgt.get("id", "?")
                rows.append({
                    "Source":       sid if sid != "UNKNOWN" else (src.get("title") or "?")[:60],
                    "Relationship": m["relationship"],
                    "Target":       tid if tid != "UNKNOWN" else (tgt.get("title") or "?")[:60],
                    "Similarity":   m["similarity"],
                })
            st.dataframe(rows, width="stretch", hide_index=True)

    elif d.get("type") == "assessment":
        with st.expander("Compliance Assessment Details"):
            rows = [
                {
                    "Control":  i.get("control_id", "?"),
                    "Title":    (i.get("title") or "")[:55],
                    "Status":   i["status"],
                    "Evidence": i.get("evidence_source", "N/A"),
                    "Sim":      round(i.get("evidence_similarity", 0), 3),
                }
                for i in d["assessment"]
            ]
            st.dataframe(rows, width="stretch", hide_index=True)

    elif d.get("type") == "report":
        with st.expander("📄 Compliance Report Preview (Markdown)", expanded=True):
            st.markdown(d["report_md"])
        
        st.markdown("#### 📥 **Export Report (Select Preferred Format)**")
        
        import importlib
        import utils.report_exporter as report_exporter
        importlib.reload(report_exporter)

        target_entity = report_exporter.extract_target_entity_name(d["report_md"], fallback="Application")
        fw_slug = d.get("framework", "Compliance").replace("/", "_").replace(" ", "_")
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_name_base = f"The_Audit_Report_of_{target_entity}_{fw_slug}_{ts_str}"

        docx_out_path = d["path"].replace(".md", ".docx")
        pdf_out_path = os.path.splitext(d["path"])[0] + ".pdf"
        pdf_bytes = None
        pdf_error = None
        docx_bytes = None
        docx_error = None

        try:
            pdf_bytes = report_exporter.export_pdf(
                md_content=d["report_md"],
                jurisdiction=d.get("jurisdiction", "nist"),
                framework=d.get("framework", "csf"),
                output_path=pdf_out_path
            )
        except Exception as p_exc:
            pdf_error = str(p_exc)

        try:
            docx_bytes = report_exporter.export_docx(
                md_content=d["report_md"],
                jurisdiction=d.get("jurisdiction", "nist"),
                framework=d.get("framework", "csf"),
                output_path=docx_out_path
            )
        except Exception as exc:
            docx_error = str(exc)

        # Primary PDF Download Button
        if pdf_bytes:
            st.download_button(
                "📕 **Download Full Audit Report (.pdf)** — Executive Publication Format",
                data=pdf_bytes,
                file_name=f"{download_name_base}.pdf",
                mime="application/pdf",
                key=f"dl_pdf_hero_{key_suffix}",
                type="primary",
                use_container_width=True,
            )
        elif pdf_error:
            st.error(f"PDF Export Note: {pdf_error}")

        if docx_bytes:
            st.download_button(
                "📘 **Download Word Document (.docx)** — Standardized Control Matrix",
                data=docx_bytes,
                file_name=f"{download_name_base}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=f"dl_docx_hero_{key_suffix}",
                use_container_width=True,
            )
        elif docx_error:
            st.error(f"DOCX Export Note: {docx_error}")

        st.caption("Alternative Export Formats:")
        c_dl1, c_dl2, c_dl3, c_dl4 = st.columns(4)
        
        c_dl1.download_button(
            "Markdown (.md)",
            data=d["report_md"].encode("utf-8"),
            file_name=f"{download_name_base}.md",
            mime="text/markdown",
            key=f"dl_md_{key_suffix}",
            use_container_width=True,
        )
        try:
            import utils.report_exporter as report_exporter
            html_content = report_exporter.HTML_TEMPLATE.format(
                title="Compliance Report",
                body=report_exporter.markdown_to_html(d["report_md"]),
                export_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            c_dl2.download_button(
                "HTML (.html)",
                data=html_content.encode("utf-8"),
                file_name=f"{download_name_base}.html",
                mime="text/html",
                key=f"dl_html_{key_suffix}",
                use_container_width=True,
            )
            txt_content = report_exporter.markdown_to_text(d["report_md"])
            c_dl3.download_button(
                "Plain Text (.txt)",
                data=txt_content.encode("utf-8"),
                file_name=f"{download_name_base}.txt",
                mime="text/plain",
                key=f"dl_txt_{key_suffix}",
                use_container_width=True,
            )
            jur_val = d.get("jurisdiction", "nist")
            fw_val = d.get("framework", "csf")
            json_data = report_exporter.markdown_to_json(d["report_md"], jur_val, fw_val)
            c_dl4.download_button(
                "JSON (.json)",
                data=json.dumps(json_data, indent=2).encode("utf-8"),
                file_name=f"{download_name_base}.json",
                mime="application/json",
                key=f"dl_json_{key_suffix}",
                use_container_width=True,
            )
        except Exception:
            pass


def render_rag_and_explainability(rag_hits: list | None, exp_report: dict | None):
    """
    Renders top RAG sources and explainability report in Streamlit expanders.
    """
    if rag_hits:
        with st.expander("Sources (matched training knowledge)"):
            for i, hit in enumerate(rag_hits, 1):
                instr = hit.get("instruction", hit.get("q", ""))
                st.markdown(f"**{i}.** _{instr}_ (domain: `{hit['domain']}`, similarity: `{hit.get('sim', 0.0)*100:.1f}%`) ")
    
    if exp_report:
        with st.expander("Answer Explainability & Confidence"):
            st.markdown(f"**Confidence Level:** `{exp_report.get('confidence_label', 'N/A')}` ({exp_report.get('confidence_score', 0.0) * 100:.1f}%)")
            st.markdown(f"**Avg Retrieval Similarity:** `{exp_report.get('avg_retrieval_similarity', 0.0)}`")
            if exp_report.get("grounding_passed") is not None:
                st.markdown(f"**Grounding Verified:** `{'Yes' if exp_report['grounding_passed'] else 'No'}`")


# -------------------------------------------------------------------------
# Session state — canonical init is in core/session_state.init_session_state()
# called at app startup (line 112). Aliases set at lines 113-114.
# -------------------------------------------------------------------------


# Initialize active retrieval engine in session state
if "active_db_engine" not in st.session_state:
    try:
        import neo4j_utils
        st.session_state.active_db_engine = "Neo4j Aura (Graph + Vector)" if neo4j_utils.is_neo4j_available() else "ChromaDB (Local Vector)"
    except Exception:
        st.session_state.active_db_engine = "ChromaDB (Local Vector)"

# Synchronize neo4j_utils state with session_state active engine
try:
    import neo4j_utils
    if "ChromaDB" in st.session_state.active_db_engine:
        neo4j_utils.force_disable(True)
    else:
        neo4j_utils.force_disable(False)
    neo4j_active = neo4j_utils.is_neo4j_available()
except Exception:
    neo4j_active = False

def _on_db_engine_change():
    chosen = st.session_state.get("db_engine_radio")
    if chosen:
        st.session_state.active_db_engine = chosen
        try:
            import neo4j_utils
            if "ChromaDB" in chosen:
                neo4j_utils.force_disable(True)
            else:
                neo4j_utils.force_disable(False)
                neo4j_utils.close_driver()  # Reset connection cache to test Neo4j server
        except Exception:
            pass

# -------------------------------------------------------------------------
# Auto-Scan & Ingest New Standards on Startup
# -------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _auto_ingest_on_startup():
    try:
        import ingest_incremental
        ingest_incremental.run_incremental(force=False, dry_run=False)
    except Exception as exc:
        print(f"Startup auto-ingestion note: {exc}")

_auto_ingest_on_startup()

# -------------------------------------------------------------------------
# Load model (cached)
# -------------------------------------------------------------------------
try:
    model, tokenizer, domains, domain_dirs, device = load_model_and_tokenizer()
    embedder, centroids, rag_index = load_router(domains, domain_dirs)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# -------------------------------------------------------------------------
# Header
# -------------------------------------------------------------------------
import importlib
import ui.header as ui_header
importlib.reload(ui_header)
ui_header.render_header_and_metrics(len(_available_fw) if _available_fw else 0, device, neo4j_active, user_role=user_role, username=username)

# -------------------------------------------------------------------------
# Full-Screen Claude-Style "Chats" Management Dashboard View
# -------------------------------------------------------------------------
if st.session_state.get("active_view", "audit") == "chats_dashboard":
    import ui.chats_dashboard as ui_chats_dash
    importlib.reload(ui_chats_dash)
    ui_chats_dash.render_chats_dashboard()
    st.stop()

# -------------------------------------------------------------------------
# Full-Screen Human Auditor Control Review Cockpit View
# -------------------------------------------------------------------------
if st.session_state.get("active_view", "audit") == "hitl_cockpit":
    import ui.hitl_verification_panel as ui_hitl
    importlib.reload(ui_hitl)
    ui_hitl.render_hitl_audit_reports_view()
    st.stop()

# -------------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------------
with st.sidebar:
    # ------------------------------------------------------------------
    # Real-Time Chat Sidebar (New Chat + Recents with Pin, Rename, Delete)
    # ------------------------------------------------------------------
    import ui.sidebar as ui_sidebar
    importlib.reload(ui_sidebar)
    ui_sidebar.render_sidebar_recents(neo4j_active, neo4j_utils, _on_db_engine_change, user_role=user_role)

    import ui.settings_panel as ui_settings
    importlib.reload(ui_settings)
    ui_settings.render_settings_panel(_available_fw, domains if "domains" in locals() else None, neo4j_active, neo4j_utils, _on_db_engine_change, user_role=user_role)
    if user_role == "admin":
        ui_settings.render_mlops_registry_and_active_learning()

    # ------------------------------------------------------------------
    # Advanced Platform Modules (Gated: Registered & Admin Users Only)
    # ------------------------------------------------------------------
    if user_role in ("registered", "admin"):
        # Manage & Remove Adapters (Admin Only)
        if user_role == "admin":
            with st.expander("Manage / Remove Adapters"):
                st.caption("Remove an active or unwanted LoRA adapter from disk.")
                all_disk_adapters = sorted([
                    os.path.basename(d) for d in glob.glob(f"{ADAPTERS_DIR}/*") 
                    if os.path.isdir(d) and not os.path.basename(d).startswith("_")
                ])
                
                if all_disk_adapters:
                    selected_adapter = st.selectbox("Select Adapter to Remove", options=all_disk_adapters)
                    delete_controls = st.checkbox("Also delete structured_controls file if found", value=False)
                    
                    if st.button("Delete Adapter", width="stretch", type="primary"):
                        target_adapter_dir = os.path.join(ADAPTERS_DIR, selected_adapter)
                        try:
                            import shutil
                            if os.path.exists(target_adapter_dir):
                                shutil.rmtree(target_adapter_dir, ignore_errors=True)
                            
                            if delete_controls:
                                slug_part = selected_adapter.replace("qwen3-", "").replace("-lora", "")
                                for sc_file in glob.glob("structured_controls/*.json"):
                                    if slug_part in os.path.basename(sc_file):
                                        os.remove(sc_file)
                            
                            st.success(f"Adapter `{selected_adapter}` deleted successfully!")
                            time.sleep(1.5)
                            st.cache_resource.clear()
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Error removing adapter: {exc}")
                else:
                    st.info("No custom adapters available to remove.")

        # Multi-Jurisdiction & Standards Registry Section
        if JURISDICTION_REGISTRY_AVAILABLE:
            with st.expander("Multi-Jurisdiction & Version Registry", expanded=False):
                st.caption("Auto-detect compliance coverage by country/sector & track standard versions.")

                registry_tab1, registry_tab2, registry_tab3, registry_tab4 = st.tabs([
                    "Onboarding", "Standards Registry", "Freshness Check", "Platform Trust Posture"
                ])

                with registry_tab1:
                    st.markdown("#### Multi-Axis Adapter Classification & Selection Engine")
                    op_countries = st.multiselect(
                        "Operating / Registration Countries",
                        options=["India", "EU", "US", "UK", "Germany", "France", "Global"],
                        default=["India", "EU", "US"]
                    )
                    ind_sector = st.selectbox(
                        "Industry Vertical",
                        options=["fintech", "healthcare", "e-commerce", "saas", "general_saas", "critical_infrastructure", "iot_hardware", "ai_ml"]
                    )
                    app_type = st.selectbox(
                        "Application Type",
                        options=["web_app", "cloud_native", "api_service", "mobile_app", "enterprise_software", "ai_ml_system"]
                    )
                    req_domains = st.multiselect(
                        "Target Control Domains",
                        options=["access_control", "data_protection_and_privacy", "incident_response", "cloud_and_infrastructure_security", "cryptography_and_encryption", "application_security_and_devsecops", "identity_and_authentication"],
                        default=["access_control", "data_protection_and_privacy"]
                    )

                    all_fw_options = sorted(_available_fw) if _available_fw else ["eu/gdpr", "eu/nis2", "india/dpdp", "nist/csf", "international/iso27001"]
                    manual_overrides = st.multiselect(
                        "🔧 Manual Framework Overrides (Force Include)",
                        options=all_fw_options,
                        help="Select any adapter/framework to override auto-recommendations"
                    )

                    if st.button("Recommend & Auto-Select Adapters", type="primary", width="stretch"):
                        res = compliance_jurisdictions.detect_company_jurisdiction(
                            operating_countries=op_countries,
                            industry_sector=ind_sector,
                            application_type=app_type,
                            control_domains=req_domains,
                            manual_overrides=manual_overrides,
                        )
                        st.success(f"Detected Jurisdictions: {', '.join(res['detected_jurisdictions'])}")
                        st.markdown("**🎯 Recommended Adapters & Frameworks:**")
                        for item in res.get("multi_axis_recommendations", []):
                            override_tag = " [🔧 MANUAL OVERRIDE]" if item.get("is_manual_override") else ""
                            st.write(f"• **{item['short_code']}** (`{item['framework']}` / `{item['adapter_name']}`){override_tag} — Score: `{item['score']}`")
                            st.caption(f"  *Reasons:* {', '.join(item.get('match_reasons', []))}")

                        st.session_state["selected_consolidated_frameworks"] = res.get("auto_selected_frameworks", [])

                    if "selected_consolidated_frameworks" in st.session_state and st.session_state["selected_consolidated_frameworks"]:
                        st.divider()
                        st.markdown(f"**Consolidated Audit Target ({len(st.session_state['selected_consolidated_frameworks'])} Frameworks):**")
                        st.caption(", ".join(st.session_state["selected_consolidated_frameworks"]))
                        if st.button("📄 Generate Consolidated Multi-Framework Audit Report (Agent 5)", width="stretch"):
                            with st.spinner("Generating unified cross-framework report..."):
                                try:
                                    import agents.agent5_report_generation as a5
                                    report_md, report_path = a5.generate_consolidated_multi_framework_report(
                                        client_name="Client_App",
                                        framework_list=st.session_state["selected_consolidated_frameworks"],
                                        with_remediation=True
                                    )
                                    st.session_state.messages.append({"role": "assistant", "content": report_md})
                                    st.success("🎉 Consolidated Multi-Framework Audit Report generated!")
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Consolidated Report Generation Error: {exc}")

                with registry_tab2:
                    st.markdown("#### Standards Version Registry")
                    all_stds = compliance_jurisdictions.get_registered_standards()
                    
                    col_reg1, col_reg2, col_reg3 = st.columns(3)
                    col_reg1.metric("Registered Standards", len(all_stds))
                    col_reg2.metric("Jurisdictions Covered", len(set(s['jurisdiction'] for s in all_stds)))
                    col_reg3.metric("Version Spec Alignment", "Active & Ingested")
                    st.divider()

                    st.markdown("##### Active Ingested Framework Version Matrix")
                    reg_df_data = []
                    for std in all_stds:
                        reg_df_data.append({
                            "Framework": std["short_code"],
                            "Title": std["title"],
                            "Jurisdiction": std["jurisdiction"],
                            "Version / Spec": std["version"],
                            "Governing Body": std["governing_body"],
                            "Last Amended": std["last_amendment_date"],
                            "Ingestion Status": "Active on Disk"
                        })
                    st.dataframe(reg_df_data, width="stretch")

                with registry_tab3:
                    st.markdown("#### Assessment Version Freshness Check")
                    fw_opt = [s["key"] for s in compliance_jurisdictions.get_registered_standards()]
                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        sel_fw = st.selectbox("Select Target Framework", options=fw_opt, key="freshness_fw_select")
                    with col_f2:
                        run_date = st.date_input("Audit Assessment Run Date", key="freshness_date_input")

                    if st.button("Check Version Freshness & Compliance Delta", width="stretch"):
                        fresh = compliance_jurisdictions.check_assessment_freshness(
                            framework_key=sel_fw,
                            assessment_run_date=run_date.strftime("%Y-%m-%d")
                        )
                        
                        std_info = compliance_jurisdictions.STANDARDS_VERSION_REGISTRY.get(sel_fw, {})
                        last_amend_str = std_info.get("last_amendment_date", "2024-01-01")
                        
                        try:
                            from datetime import datetime as _dt
                            r_dt = _dt.strptime(run_date.strftime("%Y-%m-%d"), "%Y-%m-%d")
                            a_dt = _dt.strptime(last_amend_str, "%Y-%m-%d")
                            days_diff = (a_dt - r_dt).days
                        except Exception:
                            days_diff = 0

                        col_m1, col_m2, col_m3 = st.columns(3)
                        col_m1.metric("Active Standard Version", std_info.get("version", "v1.0"))
                        col_m2.metric("Last Standard Amendment", last_amend_str)
                        col_m3.metric("Freshness Status", "OUTDATED" if fresh["needs_rerun"] else "UP TO DATE", f"{abs(days_diff)} days delta" if days_diff != 0 else "Latest")

                        if fresh["needs_rerun"]:
                            st.warning(f"**Re-assessment Recommended!**\n\n{fresh['reason']}")
                        else:
                            st.success(f"**Audit Assessment Current!**\n\n{fresh['reason']}")

                with registry_tab4:
                    st.markdown("#### Dynamic Platform Trust Posture (SOC 2 / ISO 27001)")
                    st.caption("Live evaluation of platform access isolation, encryption, audit logs, and legal gates.")
                    import application_security_trust as ast
                    posture = ast.get_dynamic_platform_security_posture()
                    st.metric("Platform Security Score", f"{posture['posture_score_pct']}%", f"{posture['controls_passed']}/{posture['controls_evaluated']} Passed")
                    for c in posture["live_controls"]:
                        st.write(f"• **[{c['status']}] {c['control_id']}** — {c['domain']}")
                        st.caption(f"  *Evidence:* {c['live_evidence']}")
                        st.divider()

        # Remediation State Tracking Dashboard
        with st.expander("Remediation Tracking & State Management"):
            st.caption("Track, update, and manage remediation lifecycle states across assessment runs.")
            client_rem_id = st.text_input("Client ID for Remediation Tracker", value="Client_App", key="rem_client_id_input").strip()
            import remediation_tracker_engine as rte
            rem_data = rte.load_client_remediations(client_rem_id)
            rem_items = rem_data.get("items", {})

            if not rem_items:
                st.info("No active remediation gaps logged yet. Run a compliance assessment or generate a consolidated report.")
            else:
                st.write(f"**Total Tracked Items:** {len(rem_items)}")
                rem_keys = list(rem_items.keys())
                sel_item_key = st.selectbox("Select Control Gap to Manage", options=rem_keys)
                if sel_item_key:
                    cur_item = rem_items[sel_item_key]
                    st.markdown(f"**Control:** `{cur_item['control_id']}` ({cur_item['framework']}) — *{cur_item['title']}*")
                    st.write(f"**Current Verdict Status:** `{cur_item['assessment_status']}`")
                    st.write(f"**Evidence Confidence Indicator:** {cur_item.get('evidence_strength', 'N/A')}")
                    
                    col_rem1, col_rem2 = st.columns(2)
                    with col_rem1:
                        new_state = st.selectbox(
                            "Remediation State",
                            options=["open", "in_progress", "resolved", "accepted_risk"],
                            index=["open", "in_progress", "resolved", "accepted_risk"].index(cur_item.get("remediation_state", "open"))
                        )
                    with col_rem2:
                        owner_name = st.text_input("Assigned Owner", value=cur_item.get("owner", "Unassigned"))
                    
                    rem_notes = st.text_area("Remediation Progress Notes / Auditor Rationale", value=cur_item.get("notes", ""))
                    
                    if st.button("💾 Save Remediation State", width="stretch"):
                        updated = rte.update_remediation_status(
                            client_id=client_rem_id,
                            item_key=sel_item_key,
                            new_state=new_state,
                            owner=owner_name,
                            notes=rem_notes
                        )
                        st.success(f"Updated '{sel_item_key}' state to '{new_state.upper()}'!")
                        st.rerun()

        # Compliance Governance, Review Gate & Benchmarks Dashboard
        with st.expander("Compliance Governance & Expert Review Gate"):
            st.caption("Human-in-the-Loop review gates, hallucination benchmarks, and standards licensing.")
            import robustness_governance as rg

            gov_tab1, gov_tab2, gov_tab3 = st.tabs(["Human Review Gate", "Hallucination Benchmark", "Standards Licensing"])

            with gov_tab1:
                st.markdown("#### 🛡️ Human Compliance Review & Control Verification Cockpit")
                st.caption("Examine all generated reports, give comments for each mapped control, modify verdicts, and issue certified audits.")
                
                col_gw1, col_gw2 = st.columns([3, 1])
                with col_gw2:
                    if st.button("🖥️ Open Full-Screen Cockpit", width="stretch", type="primary", key="open_fullscreen_cockpit_btn"):
                        st.session_state.active_view = "hitl_cockpit"
                        st.rerun()

                import ui.hitl_verification_panel as ui_hitl
                importlib.reload(ui_hitl)
                ui_hitl.render_hitl_audit_reports_view()

                with st.expander("Legacy Sign-Off Archive Queue", expanded=False):
                    pending_files = glob.glob(os.path.join(rg.HUMAN_SIGN_OFF_DIR, "*.json"))
                    if not pending_files:
                        st.info("No legacy reports in sign-off queue.")
                    else:
                        p_options = [os.path.basename(f).replace(".json", "") for f in pending_files]
                        sel_report_id = st.selectbox("Select Report to Audit", options=p_options, key="legacy_sel_report")
                        if sel_report_id:
                            with open(os.path.join(rg.HUMAN_SIGN_OFF_DIR, f"{sel_report_id}.json"), "r", encoding="utf-8") as f:
                                rec = json.load(f)
                            st.write(f"**Client:** `{rec['client_id']}` | **Status:** `{rec['human_review_status']}`")
                            st.write(f"**Auto Verdict:** `{rec['auto_verdict']}` | **Submitted At:** `{rec['submitted_at']}`")
                            auditor_name = st.text_input("Auditor Name / License ID", value="Certifying Auditor CISA-9821", key="leg_aud_name")
                            audit_notes = st.text_area("Auditor Review Notes", value="Verified control evidence against evidence vault.", key="leg_aud_notes")
                            
                            col_sign1, col_sign2 = st.columns(2)
                            with col_sign1:
                                if st.button("✅ Approve & Certify Report", width="stretch", key="leg_appr_btn"):
                                    rg.execute_human_sign_off(sel_report_id, auditor_name, approved=True, expert_notes=audit_notes)
                                    st.success(f"Report {sel_report_id} APPROVED by {auditor_name}!")
                                    st.rerun()
                            with col_sign2:
                                if st.button("❌ Reject & Request Revision", width="stretch", key="leg_rej_btn"):
                                    rg.execute_human_sign_off(sel_report_id, auditor_name, approved=False, expert_notes=audit_notes)
                                    st.warning(f"Report {sel_report_id} REJECTED by {auditor_name}.")
                                    st.rerun()

            with gov_tab2:
                st.markdown("#### 🧪 Hallucination & Error-Rate Evaluation Benchmark")
                st.caption("Benchmark LLM accuracy against ground-truth labeled control evidence pairs.")
                if st.button("Run Hallucination Benchmark Test", width="stretch"):
                    eval_res = rg.evaluate_hallucination_and_error_rate()
                    st.metric("Verdict Accuracy", f"{eval_res['verdict_accuracy_pct']}%", f"Error Rate: {eval_res['hallucination_error_rate_pct']}%")
                    st.json(eval_res["detailed_results"])

            with gov_tab3:
                st.markdown("#### 📜 Proprietary vs Public Standards Licensing Catalog")
                for fw_k, info in rg.LICENSING_CATALOG.items():
                    st.write(f"• **{fw_k.upper()}**: `{info['license_type']}` — Status: `[{info['status']}]`")

        # Incremental Ingestion Control (Admin Only)
        if user_role == "admin":
            with st.expander("Incremental Standards Ingestion"):
                st.caption("Scan standards/ folder and ingest only missing/new PDF/TXT documents.")
                force_ingest = st.checkbox("Force re-ingest all documents", value=False)
                if st.button("Run Incremental Ingestion", width="stretch"):
                    try:
                        import ingest_incremental
                        with st.spinner("Ingesting new standards..."):
                            ingest_incremental.run_incremental(force=force_ingest)
                        st.success("Ingestion complete!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Ingestion failed: {exc}")

        # Pipeline Logging & Monitoring (Admin Only)
        if user_role == "admin":
            with st.expander("Pipeline Logs & Monitoring"):
                st.caption("Live execution metrics and structured system logs.")
                try:
                    import pipeline_logger as plog
                    logs = plog.get_recent_logs(n=10)
                    summary = plog.get_stage_summary()

                    if summary:
                        st.markdown("**Stage Execution Summary**")
                        for stage_name, metrics in summary.items():
                            duration_str = f" | {metrics['total_duration']:.2f}s" if metrics['total_duration'] > 0 else ""
                            st.markdown(
                                f"""
                                <div class="stage-metric-box">
                                    <span class="log-stage-tag">⚙️ {stage_name}</span>
                                    <span style="float: right; color: #94a3b8; font-size: 0.8rem;">
                                        Calls: <strong>{metrics['count']}</strong>{duration_str}
                                    </span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    if logs:
                        st.markdown("**Recent Logs Stream**")
                        for entry in reversed(logs):
                            lvl = entry.get("level", "INFO").upper()
                            badge_class = "log-badge-info"
                            if lvl == "WARNING":
                                badge_class = "log-badge-warn"
                            elif lvl == "ERROR":
                                badge_class = "log-badge-error"

                            ts = entry.get("timestamp", "")
                            if "T" in ts:
                                ts = ts.split("T")[1].split(".")[0]

                            dur = f" ({entry['duration_seconds']:.2f}s)" if "duration_seconds" in entry else ""

                            st.markdown(
                                f"""
                                <div class="log-card">
                                    <div>
                                        <span class="{badge_class}">{lvl}</span>
                                        <span class="log-stage-tag">[{entry.get('stage','?')}]</span>
                                        <span style="color: #64748b; float: right; font-size: 0.75rem;">{ts}</span>
                                    </div>
                                    <div class="log-msg-text">{entry.get('message','')}{dur}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info("No log entries captured yet.")
                except Exception as exc:
                    st.error(f"Log error: {exc}")

        # RAG Evaluation Runner (Admin Only)
        if user_role == "admin":
            with st.expander("Run RAG Evaluation"):
                st.caption("Evaluate RAG retrieval precision and quality across all domains.")
                eval_k = st.slider("Retrieval K", 1, 10, 5)
                eval_samples = st.slider("Eval samples per domain", 5, 30, 10)
                if st.button("Run RAG Evaluation", width="stretch"):
                    try:
                        import evaluate_rag
                        with st.spinner("Evaluating RAG quality..."):
                            import agents.config as agent_config
                            import rag_utils
                            emb = agent_config.get_embedder()
                            col = rag_utils.get_collection()
                            dirs = sorted([d for d in glob.glob(f"{ADAPTERS_DIR}/*") if os.path.isdir(d)])
                            results = []
                            for ddir in dirs:
                                aname = os.path.basename(ddir)
                                fw_slug = evaluate_rag.extract_framework_from_adapter_name(aname)
                                qs = evaluate_rag.load_eval_questions(ddir, start=30, count=eval_samples)
                                if qs:
                                    res = evaluate_rag.evaluate_retrieval(emb, col, qs, fw_slug, k=eval_k)
                                    results.append({"domain": aname, "precision": res["retrieval_precision_at_k"], "avg_sim": res["avg_similarity"]})
                            if results:
                                st.dataframe(results, width="stretch")
                            else:
                                st.warning("No held-out samples available for evaluation.")
                    except Exception as exc:
                        st.error(f"Evaluation error: {exc}")

        st.divider()

        # Agent 0: Master Automation Panel (Admin Only)
        if user_role == "admin":
            with st.expander("Automate New Standard (Agent 0)"):
                st.caption("Upload any PDF/TXT standard to automate ingestion, mapping, synthesis & LoRA fine-tuning.")
                
                uploaded_file = st.file_uploader("Upload Standard PDF/TXT", type=["pdf", "txt"])
                j_input = st.text_input("Jurisdiction", value="us", help="e.g. us, eu, india, nist, international")
                f_input = st.text_input("Framework Slug", value="hipaa", help="e.g. hipaa, pci_dss, soc2")
                fn_input = st.text_input("Full Name", value="Health Insurance Portability and Accountability Act")
                desc_input = st.text_area("Description", value="US healthcare data privacy and security regulation.", height=68)
                epochs_input = st.number_input("Fine-tuning Epochs", min_value=1, max_value=10, value=3)

                if st.button("Run Agent 0 Automation", width="stretch"):
                    if not uploaded_file:
                        st.error("Please upload a PDF or TXT file first.")
                    elif not j_input.strip() or not f_input.strip():
                        st.error("Please provide both Jurisdiction and Framework slug.")
                    else:
                        temp_dir = os.path.join(ADAPTERS_DIR, "_temp_uploads")
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        progress_bar = st.progress(0.0)
                        status_text = st.empty()
                        log_container = st.empty()
                        logs_list = []

                        def ui_callback(stage_msg, pct):
                            status_text.markdown(f"**Status:** {stage_msg}")
                            progress_bar.progress(min(pct, 1.0))
                            logs_list.append(stage_msg)
                            log_container.code("\n".join(logs_list[-6:]), language="text")

                        try:
                            import agents.agent0_master_orchestrator as a0
                            res = a0.run_agent0_pipeline(
                                file_path=temp_path,
                                jurisdiction=j_input.strip(),
                                framework=f_input.strip(),
                                full_name=fn_input.strip(),
                                description=desc_input.strip(),
                                epochs=int(epochs_input),
                                progress_callback=ui_callback,
                            )
                            
                            adapter_name = f"qwen3-{res['slug']}-lora"
                            adapter_path = res['adapter_dir']
                            
                            st.success(f"Agent 0 Pipeline Completed in {res['total_time_seconds']:.1f}s!")
                            st.markdown(
                                f"""
                                ### New LoRA Adapter Created & Registered
                                | Property | Value |
                                |---|---|
                                | **Adapter Name** | `{adapter_name}` |
                                | **Adapter Path** | `{adapter_path}` |
                                | **Base Model** | `{BASE_MODEL_NAME}` |
                                | **Ingested Controls** | `{res['controls_count']}` controls |
                                | **Training Examples** | `{res['training_examples']}` Q&A pairs |
                                | **Status** | `Ready for Routing & Live Inference` |
                                """
                            )
                            time.sleep(3)
                            st.cache_resource.clear()
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Agent 0 Automation failed: {exc}")

# -------------------------------------------------------------------------
# Render existing chat history or Welcome Hero Banner
# -------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown(
        """
        <div class="hero-wrapper">
            <div class="hero-icon">🛡️</div>
            <div class="hero-title">Cybersecurity Compliance Platform</div>
            <div class="hero-subtitle">
                Autonomous Multi-Agent Audit, Controls Cross-Mapping & Regulatory Intelligence Engine
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<p style='text-align: center; font-weight: 700; color: #94a3b8; font-size: 0.92rem; margin-bottom: 12px; letter-spacing: 0.04em; text-transform: uppercase;'>Quick-Action Compliance Workflows</p>", unsafe_allow_html=True)
    
    # 4 Clean, Prominent Smart Action Chips
    chip_c1, chip_c2 = st.columns(2)
    with chip_c1:
        if st.button("🛡️ **Assess NIST CSF 2.0**\n\nFull safeguard and controls compliance assessment against NIST CSF 2.0.", width="stretch", key="hero_chip_nist"):
            st.session_state.active_chat_framework = "nist/csf"
            st.session_state.preset_prompt = "assess nist/csf"
            st.rerun()
        if st.button("🔒 **GDPR vs DPDP Cross-Map**\n\nCompare European GDPR provisions against India DPDP 2023 controls.", width="stretch", key="hero_chip_gdpr_dpdp"):
            st.session_state.preset_prompt = "map eu/gdpr to india/dpdp"
            st.rerun()
    with chip_c2:
        if st.button("🌐 **ISO 27001 ISMS Audit**\n\nAudit technical and organizational measures against ISO/IEC 27001 Annex A.", width="stretch", key="hero_chip_iso"):
            st.session_state.active_chat_framework = "international/iso27001"
            st.session_state.preset_prompt = "assess international/iso27001"
            st.rerun()
        if st.button("⚡ **Scan Web App Security (WSTG)**\n\nEvaluate OWASP Web Security Testing Guide vulnerabilities and findings.", width="stretch", key="hero_chip_wstg"):
            st.session_state.preset_prompt = "assess owasp/top10_web"
            st.rerun()
    
    st.divider()

for idx, msg in enumerate(st.session_state.messages):
    # Show compressed memory summaries as a visible context indicator (not a chat bubble)
    if msg.get("role") == "system" and msg.get("source") == "Memory Compression":
        with st.expander("Prior Conversation Summary (Compressed Memory)", expanded=False):
            st.markdown(msg["content"])
        continue
    # Skip other system messages from rendering as chat bubbles
    if msg.get("role") == "system":
        continue
    # Direct answer mode: Skip rendering query bubble for preset quick action triggers
    if msg.get("role") == "user" and msg.get("is_preset"):
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            src = msg.get("source", "")
            gen = msg.get("gen_seconds")
            # Strip any residual match percentages or internal lora names from legacy history captions
            clean_src = re.sub(r'\s*\(`?qwen3-[^,]+`?,\s*match:\s*[\d\.%]+\)', '', src).strip()
            clean_src = re.sub(r'\s*\(match:\s*[\d\.%]+\)', '', clean_src).strip()
            parts = ([clean_src] if clean_src else []) + ([f"{gen:.1f}s"] if gen else [])
            if parts:
                st.caption(" | ".join(parts))
            if msg.get("self_healing_trace"):
                with st.expander("Self-healing trace"):
                    for step in msg["self_healing_trace"]:
                        st.json(step)
            if msg.get("rag_hits") or msg.get("explainability_report"):
                render_rag_and_explainability(msg.get("rag_hits"), msg.get("explainability_report"))
            for d in msg.get("agent_details", []):
                render_agent_details(d, key_suffix=f"{idx}_{msg.get('ts', '')}")
            
            # Active Learning Auditor Feedback Action (Checked against DB & Persisted)
            msg_id = f"{idx}_{msg.get('ts') or hash(msg.get('content', ''))}"
            prev_q = st.session_state.messages[idx - 1].get("content", "") if idx > 0 else ""

            import core.feedback_collector as fb_col
            from core.session_state import auto_save_current_session
            
            # Check DB if status not in session state
            current_status = msg.get("feedback_status")
            if not current_status and prev_q and msg.get("content"):
                db_status = fb_col.get_feedback_status_for_message(prev_q, msg.get("content", ""))
                if db_status:
                    msg["feedback_status"] = db_status
                    current_status = db_status

            content_text = msg.get("content", "")
            f_col1, f_col2, f_col3, _ = st.columns([0.05, 0.05, 0.06, 0.84])

            if not current_status:
                with f_col1:
                    if st.button("👍", key=f"up_{msg_id}", help="Accurate & grounded response"):
                        try:
                            fb_col.record_feedback(
                                query=prev_q,
                                response=content_text,
                                rating=1,
                                username=st.session_state.get("username", "guest"),
                                framework=st.session_state.get("active_chat_framework", "Auto-Detect")
                            )
                            msg["feedback_status"] = "positive"
                            auto_save_current_session()
                            st.toast("Feedback saved! Logged as verified benchmark ground truth.")
                            st.rerun()
                        except Exception:
                            pass
                with f_col2:
                    if st.button("👎", key=f"dn_{msg_id}", help="Flag inaccurate response for Active Learning fine-tuning remediation"):
                        try:
                            fb_col.record_feedback(
                                query=prev_q,
                                response=content_text,
                                rating=-1,
                                username=st.session_state.get("username", "guest"),
                                framework=st.session_state.get("active_chat_framework", "Auto-Detect")
                            )
                            msg["feedback_status"] = "negative"
                            auto_save_current_session()
                            st.toast("Response flagged for Active Learning fine-tuning remediation.")
                            st.rerun()
                        except Exception:
                            pass
            elif current_status == "positive":
                with f_col1:
                    st.caption("👍")
            elif current_status == "negative":
                with f_col2:
                    st.caption("👎")

            # Only show 📥 Report Download button for actual Compliance Audit Reports (hide for normal conversational chats)
            is_audit_report = any(k in content_text for k in [
                "Compliance Audit Report",
                "Compliance Breakdown",
                "Fully Compliant",
                "Not Compliant",
                "Agent 0 — Mode B",
                "Dynamic Control Inspection Matrix",
                "Auditor Explanation",
                "Client Remediation Plan"
            ])

            if is_audit_report and len(content_text.strip()) > 50:
                with f_col3:
                    with st.popover("📥", help="Download Report (.docx, .md, .html, .txt)"):
                        st.markdown("##### 📄 Export Compliance Report")
                        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        import importlib
                        import utils.report_exporter as report_exporter
                        importlib.reload(report_exporter)

                        # Intelligently extract framework title and target entity from report content
                        detected_fw_display = report_exporter.resolve_canonical_framework_name(
                            md_content=content_text,
                            framework=st.session_state.get("active_chat_framework", "")
                        )
                        target_entity = report_exporter.extract_target_entity_name(content_text, fallback="Application")
                        fw_slug = re.sub(r'[^A-Za-z0-9_-]', '_', detected_fw_display).strip('_')
                        if not fw_slug or "auto_detect" in fw_slug.lower():
                            fw_slug = "Compliance_Audit"
                        file_prefix = f"The_Audit_Report_of_{target_entity}_{fw_slug}_{ts_str}"

                        # 1. PDF Document (.pdf) — Publication Quality
                        try:
                            pdf_data = report_exporter.render_pdf_report_bytes(
                                md_content=content_text,
                                framework=detected_fw_display
                            )
                            st.download_button(
                                "📕 PDF Report (.pdf)",
                                data=pdf_data,
                                file_name=f"{file_prefix}.pdf",
                                mime="application/pdf",
                                key=f"dl_pdf_btn_{msg_id}",
                                type="primary",
                                use_container_width=True
                            )
                        except Exception as p_exc:
                            st.caption(f"PDF generator notice: {p_exc}")

                        # 2. Word Document (.docx) — Standardized Matrix
                        try:
                            docx_data = report_exporter.render_template_docx_bytes(
                                md_content=content_text,
                                framework=detected_fw_display
                            )
                            st.download_button(
                                "📘 Word Document (.docx)",
                                data=docx_data,
                                file_name=f"{file_prefix}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                key=f"dl_docx_btn_{msg_id}",
                                use_container_width=True
                            )
                        except Exception as d_exc:
                            st.caption(f"DOCX generator notice: {d_exc}")

                        # 3. Markdown (.md)
                        st.download_button(
                            "📄 Markdown Report (.md)",
                            data=content_text.encode("utf-8"),
                            file_name=f"{file_prefix}.md",
                            mime="text/markdown",
                            key=f"dl_md_btn_{msg_id}",
                            use_container_width=True
                        )

                        # 4. HTML (.html)
                        try:
                            html_data = report_exporter.HTML_TEMPLATE.format(
                                title=f"The Audit Report of {target_entity} — {detected_fw_display}",
                                body=report_exporter.markdown_to_html(content_text),
                                export_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                            st.download_button(
                                "📑 Styled HTML (.html)",
                                data=html_data.encode("utf-8"),
                                file_name=f"{file_prefix}.html",
                                mime="text/html",
                                key=f"dl_html_btn_{msg_id}",
                                use_container_width=True
                            )
                        except Exception as h_exc:
                            st.caption(f"HTML generator notice: {h_exc}")

                        # 4. Plain Text (.txt)
                        try:
                            txt_data = report_exporter.markdown_to_text(content_text)
                            st.download_button(
                                "📝 Plain Text (.txt)",
                                data=txt_data.encode("utf-8"),
                                file_name=f"{file_prefix}.txt",
                                mime="text/plain",
                                key=f"dl_txt_btn_{msg_id}",
                                use_container_width=True
                            )
                        except Exception:
                            pass

# -------------------------------------------------------------------------
# Chat input -- unified real-time handler with ChatGPT/Claude '+' selector
# -------------------------------------------------------------------------
# Chat input -- permanently fixed bottom position (ChatGPT / Claude style)
# -------------------------------------------------------------------------
def scan_available_frameworks_realtime() -> list[str]:
    """Scans structured_controls, adapters, and standards directories in real-time."""
    fws = set()
    sc_dir = "structured_controls"
    if os.path.isdir(sc_dir):
        for fn in os.listdir(sc_dir):
            if fn.endswith(".json"):
                parts = fn.replace(".json", "").split("__")
                if len(parts) == 2:
                    fws.add(f"{parts[0]}/{parts[1]}")

    standards_dir = "standards"
    if os.path.isdir(standards_dir):
        for jur in os.listdir(standards_dir):
            jur_path = os.path.join(standards_dir, jur)
            if os.path.isdir(jur_path):
                for std in os.listdir(jur_path):
                    std_path = os.path.join(standards_dir, jur, std)
                    if os.path.isdir(std_path):
                        fws.add(f"{jur}/{std}")
    return sorted(list(fws))


def get_country_flag_and_label(jur_code: str) -> str:
    jur = jur_code.lower().strip()
    FLAG_MAP = {
        "eu": ("🇪🇺", "European Union (EU)"),
        "india": ("🇮🇳", "India"),
        "in": ("🇮🇳", "India"),
        "us": ("🇺🇸", "United States (US)"),
        "nist": ("🇺🇸", "United States (NIST)"),
        "uk": ("🇬🇧", "United Kingdom (UK)"),
        "gb": ("🇬🇧", "United Kingdom (UK)"),
        "ca": ("🇨🇦", "Canada"),
        "au": ("🇦🇺", "Australia"),
        "de": ("🇩🇪", "Germany"),
        "fr": ("🇫🇷", "France"),
        "jp": ("🇯🇵", "Japan"),
        "sg": ("🇸🇬", "Singapore"),
        "international": ("🌐", "International Standards (ISO)"),
        "global": ("🌐", "Global Frameworks"),
    }
    if jur in FLAG_MAP:
        flag, name = FLAG_MAP[jur]
        return f"{flag} {name}"
    return f"🏳️ {jur.upper()}"


def get_country_jurisdictions_realtime() -> dict[str, list[str]]:
    fws = scan_available_frameworks_realtime()
    groups = {"All Countries / Global": fws}
    for fw in fws:
        parts = fw.split("/")
        jur = parts[0] if len(parts) == 2 else "global"
        label = get_country_flag_and_label(jur)
        if label not in groups:
            groups[label] = []
        groups[label].append(fw)
    return groups



# Root-level chat input automatically pins fixed at the bottom of the viewport
preset_query = st.session_state.pop("preset_prompt", None)
user_query = st.chat_input("Ask about NIST, EU, India, or any compliance framework...")
query = preset_query or user_query

if query:
    is_preset = bool(preset_query)
    if user_role == "guest":
        guest_user_msgs = [m for m in st.session_state.get("messages", []) if m.get("role") == "user"]
        if len(guest_user_msgs) >= 5:
            st.warning("🔒 **Guest message limit reached (5 messages/session)**. Please sign in or register for an account to enjoy unlimited audit questions.")
            st.stop()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.messages.append({"role": "user", "content": query, "ts": now_str, "is_preset": is_preset})
    if not is_preset:
        with st.chat_message("user"):
            st.markdown(query)

    intent_info = detect_intent(query)
    # Apply selected framework focus override if set via '+' selector or sidebar
    active_fw = st.session_state.get("active_chat_framework", "Auto-Detect (Smart Route)")
    if active_fw != "Auto-Detect (Smart Route)":
        intent_info["framework"] = active_fw

    intent      = intent_info["intent"]
    fw          = intent_info.get("framework")

    use_memory = st.session_state.get("use_memory_toggle", True)
    memory_turns = st.session_state.get("memory_turns_slider", 2)
    length_label = st.session_state.get("length_preset_radio", "Medium")
    use_self_healing = get_system_setting("self_healing_rag_enabled", False)

    history = None
    if use_memory:
        prior   = st.session_state.messages[:-1]
        trimmed = prior[-(memory_turns * 2):]
        history = [{"role": m["role"], "content": m["content"]} for m in trimmed]

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Path A: List frameworks
    # ------------------------------------------------------------------
    if intent == "list_frameworks":
        q_lower = query.lower()
        filtered_fws = _available_fw
        j_label = "all ingested"
        
        if any(k in q_lower for k in ["us", "usa", "united states", "america"]):
            j_label = "US"
            filtered_fws = [f for f in _available_fw if f.startswith("us/") or f.startswith("nist/")]
        elif any(k in q_lower for k in ["india", "indian"]):
            j_label = "India"
            filtered_fws = [f for f in _available_fw if f.startswith("india/")]
        elif any(k in q_lower for k in ["eu", "europe", "european"]):
            j_label = "EU"
            filtered_fws = [f for f in _available_fw if f.startswith("eu/")]
            
        if filtered_fws:
            fw_list = "\n".join(f"- `{f}`" for f in filtered_fws)
            resp = f"I have these **{j_label} Jurisdiction** frameworks ingested and ready:\n\n{fw_list}"
        else:
            resp = f"No frameworks found for {j_label} jurisdiction. Run Agent 1 + Agent 2 to ingest standards."
        with st.chat_message("assistant"):
            st.markdown(resp)
            st.caption("Knowledge Base Registry")
        st.session_state.messages.append({
            "role": "assistant", "content": resp,
            "source": "Knowledge Base Registry", "agent_details": [], "ts": now_str,
        })

    # ------------------------------------------------------------------
    # Path B: Compliance / non-NIST -- LLM streams + Agents run in parallel
    # ------------------------------------------------------------------
    elif intent in ["assess", "map", "report"]:
        with st.chat_message("assistant"):
            t_start = time.time()

            # Launch agents into thread pool immediately (non-blocking)
            pool         = None
            agent_future = None
            if COMPLIANCE_AGENTS_AVAILABLE:
                pool         = concurrent.futures.ThreadPoolExecutor(max_workers=4)
                agent_future = pool.submit(run_agents_parallel, intent_info)

            # Hybrid RAG+LLM answer + agents running simultaneously
            try:
                token_gen, llm_meta, gen_thread = answer_hybrid(
                    model, tokenizer, device, embedder, centroids,
                    query, length_label, rag_index=rag_index, history=history,
                    use_self_healing=use_self_healing,
                    target_framework=st.session_state.get("active_chat_framework"),
                )
                response_text = st.write_stream(token_gen)
                if gen_thread is not None:
                    gen_thread.join(timeout=60)
            except Exception as exc:
                response_text = f"Error: {exc}"
                st.markdown(response_text)
                llm_meta = {"domain": "error", "sims": {}, "rag_hits": []}

            gen_seconds = time.time() - t_start

            domain = llm_meta.get("domain", "unknown")
            top_sim = llm_meta.get("top_similarity", 0.0)
            gen_mode = llm_meta.get("generation_mode", "rag")
            mode_label = "Adapter-only" if gen_mode == "adapter" else "RAG + Base Model"

            # Collect agent results (likely finished by now)
            agent_result  = {}
            agent_details = []
            if agent_future is not None:
                try:
                    agent_result = agent_future.result(timeout=180)
                    if pool:
                        pool.shutdown(wait=False)
                except Exception as exc:
                    st.warning(f"Agent error: {exc}")

            assessment  = agent_result.get("assessment")
            mappings    = agent_result.get("mappings")
            report_md   = agent_result.get("report_md")
            report_path = agent_result.get("report_path")
            errors      = agent_result.get("errors", [])

            # Only show "Agents 3/4/5 ran" if agents actually produced results
            has_agent_data = bool(assessment or mappings or report_md)
            agent_suffix = " | Agents 3/4/5 ran simultaneously" if has_agent_data else ""
            # Clean caption: strip internal adapter/domain names from user-facing output
            clean_domain = re.sub(r'qwen3-[a-z0-9]+-lora', 'Specialized Engine', str(domain))
            caption_parts = [mode_label]
            if top_sim > 0:
                caption_parts.append(f"match: {top_sim*100:.1f}%")
            caption_parts.append(f"{gen_seconds:.1f}s")
            st.caption(" | ".join(caption_parts) + agent_suffix)

            if errors:
                st.warning("Agent errors: " + " | ".join(errors))

            # Show assessment results
            if assessment:
                if fw:
                    st.session_state.last_assessment[fw] = assessment
                total  = len(assessment)
                counts = {
                    "Compliant": 0, "Partially Compliant": 0,
                    "Not Compliant": 0, "No Evidence Found": 0,
                }
                for item in assessment:
                    counts[item["status"]] = counts.get(item["status"], 0) + 1
                st.markdown(
                    f"\n\n**Agent 4 - Compliance Assessment: `{fw or 'unknown'}`** "
                    f"({total} controls evaluated)\n\n"
                    f"| Compliant | Partial | Not Compliant | No Evidence |\n"
                    f"|---|---|---|---|\n"
                    f"| {counts['Compliant']} | {counts['Partially Compliant']} "
                    f"| {counts['Not Compliant']} | {counts['No Evidence Found']} |"
                )
                dd = {"type": "assessment", "assessment": assessment}
                render_agent_details(dd, key_suffix=now_str + "_a4")
                agent_details.append(dd)

            # Show mapping results
            if mappings:
                base_fw    = intent_info.get("base", "?")
                compare_fw = intent_info.get("compare", "?")
                ck         = f"{base_fw}_vs_{compare_fw}"
                st.session_state.last_mappings[ck] = mappings
                equiv   = sum(1 for m in mappings if m["relationship"] == "Equivalent")
                partial = sum(1 for m in mappings if m["relationship"] == "Partially Overlapping")
                st.markdown(
                    f"\n\n**Agent 3 - Control Mapping** "
                    f"({len(mappings)} mappings | {equiv} equivalent | {partial} partial)"
                )
                dd = {
                    "type": "mappings", "mappings": mappings,
                    "base": base_fw, "compare": compare_fw,
                }
                render_agent_details(dd, key_suffix=now_str + "_a3")
                agent_details.append(dd)

            # Show report
            if report_md and report_path:
                st.markdown(f"\n\n**Agent 5 - Report generated** -- `{report_path}`")
                dd = {
                    "type": "report",
                    "report_md": report_md,
                    "path": report_path,
                    "jurisdiction": jur or "nist",
                    "framework": fw or "csf"
                }
                render_agent_details(dd, key_suffix=now_str + "_a5")
                agent_details.append(dd)

            if not assessment and not mappings and not report_md and fw and not errors:
                st.info(
                    f"No controls found for **{fw}**. "
                    "Run Agent 1 + Agent 2 to ingest that standard first."
                )

            # Compute explainability report and render RAG hits and explainability
            rag_hits = llm_meta.get("rag_hits", [])
            exp_report = None
            if rag_hits:
                try:
                    import explainability
                    exp_hits = [{
                        "text": h.get("instruction", h.get("q", "")),
                        "score": h.get("sim", 0.0),
                        "jurisdiction": "standard",
                        "framework": h.get("domain", "unknown"),
                        "source_file": h.get("domain", "unknown")
                    } for h in rag_hits]
                    exp_report = explainability.build_explainability_report(
                        query=query,
                        answer=response_text,
                        hits=exp_hits,
                        trace=llm_meta.get("self_healing_trace")
                    )
                except Exception:
                    pass

            render_rag_and_explainability(rag_hits, exp_report)

        st.session_state.messages.append({
            "role": "assistant", "content": response_text,
            "source": "AI + Agents 3/4/5 (simultaneous real-time)",
            "gen_seconds": gen_seconds,
            "self_healing_trace": llm_meta.get("self_healing_trace"),
            "rag_hits": rag_hits,
            "explainability_report": exp_report,
            "agent_details": agent_details,
            "ts": now_str,
        })

    # ------------------------------------------------------------------
    # Path C: General chat -- RAG + LLM synthesis
    # ------------------------------------------------------------------
    else:
        with st.chat_message("assistant"):
            t_start = time.time()
            res_box = st.empty()
            try:
                token_gen, llm_meta, gen_thread = answer_hybrid(
                    model, tokenizer, device, embedder, centroids,
                    query, length_label, rag_index=rag_index, history=history,
                    use_self_healing=use_self_healing,
                    target_framework=st.session_state.get("active_chat_framework"),
                )
                response_text = res_box.write_stream(token_gen)
                if gen_thread is not None:
                    gen_thread.join(timeout=60)
            except Exception as exc:
                response_text = f"Error: {exc}"
                llm_meta = {"domain": "error", "sims": {}, "rag_hits": []}
                res_box.markdown(response_text)
            gen_seconds = time.time() - t_start

            # Real-Time Processing Layer Interceptor (OpenRouter Nemotron / Gemini API)
            import verification.realtime_verifier as verifier_mod
            response_text, verification_badge = verifier_mod.run_realtime_verification(query, response_text, res_box, target_length=length_label)

            domain = llm_meta.get("domain", "unknown")
            top_sim = llm_meta.get("top_similarity", 0.0)
            gen_mode = llm_meta.get("generation_mode", "rag")
            raw_active_fw = st.session_state.get("active_chat_framework", "Auto-Detect (Smart Route)")
            active_fw_display = format_framework_display_name(raw_active_fw) if raw_active_fw != "Auto-Detect (Smart Route)" else "Auto-Detect (Smart Route)"
            adapter_display = llm_meta.get("active_adapter") or "Base Model + RAG"
            st.caption(f"🎯 Target Framework: **{active_fw_display}** | 🤖 Active Adapter: `{adapter_display}` | {verification_badge} | ⏱️ {gen_seconds:.1f}s")

            # Compute explainability report and render RAG hits and explainability
            rag_hits = llm_meta.get("rag_hits", [])
            exp_report = None
            if rag_hits:
                try:
                    import explainability
                    exp_hits = [{
                        "text": h.get("instruction", h.get("q", "")),
                        "score": h.get("sim", 0.0),
                        "jurisdiction": "standard",
                        "framework": h.get("domain", "unknown"),
                        "source_file": h.get("domain", "unknown")
                    } for h in rag_hits]
                    exp_report = explainability.build_explainability_report(
                        query=query,
                        answer=response_text,
                        hits=exp_hits,
                        trace=llm_meta.get("self_healing_trace")
                    )
                except Exception:
                    pass

            render_rag_and_explainability(rag_hits, exp_report)

            if llm_meta.get("self_healing_trace"):
                with st.expander("Self-healing trace"):
                    for step in llm_meta["self_healing_trace"]:
                        st.json(step)

        st.session_state.messages.append({
            "role": "assistant", "content": response_text,
            "source": f"{active_fw_display} ({adapter_display})",
            "gen_seconds": gen_seconds,
            "top_similarity": top_sim,
            "self_healing_trace": llm_meta.get("self_healing_trace"),
            "rag_hits": rag_hits,
            "explainability_report": exp_report,
            "agent_details": [], "ts": now_str,
        })
        _auto_save_current_session()


# -------------------------------------------------------------------------
# Focus controls and Active Framework pill (always rendered at the very bottom)
# -------------------------------------------------------------------------
import ui.focus_bar as focus_bar_ui
importlib.reload(focus_bar_ui)
focus_bar_ui.render_focus_bar(_available_fw, user_role=user_role, clean_report_list_fn=_clean_report_list)


