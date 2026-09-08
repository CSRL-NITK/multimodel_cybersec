"""
Grounded Generation Accuracy Evaluation Benchmark
--------------------------------------------------
Compares:
1. Ungrounded Base Model Generation (Direct LLM Zero-Shot)
2. Per-Framework Grounded Generation (Method 1: Isolated Vector DB + LLM)

Computes quantitative metrics:
- Valid Control ID Citation Precision (%)
- Hallucination / Fabrication Rate (%)
- Ground-Truth Target Control Recall (%)
- Statutory Groundedness / Faithfulness Score (%)
- End-to-End Generation Latency (ms)
"""

import os
import sys
import time
import json
import re
import glob
import numpy as np
from typing import List, Dict, Any, Set

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.federated_grounded_generator import (
    generate_grounded_compliance_answer,
    generate_ungrounded_answer
)


# Load all official valid control IDs from structured_controls
def load_all_valid_framework_control_ids() -> Dict[str, Set[str]]:
    valid_ids_by_framework = {}
    for path in glob.glob(os.path.join(PROJECT_ROOT, "structured_controls", "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                controls = data if isinstance(data, list) else data.get("controls", [])
                fn = os.path.basename(path).replace(".json", "")
                parts = fn.split("__")
                fw_key = parts[1] if len(parts) > 1 else fn
                
                cids = set()
                for c in controls:
                    cid = c.get("control_id") or c.get("id")
                    if cid:
                        cids.add(str(cid).lower().strip())
                valid_ids_by_framework[fw_key.lower()] = cids
            except Exception:
                pass
    return valid_ids_by_framework


VALID_CONTROL_CATALOG = load_all_valid_framework_control_ids()


EVAL_CASES = [
    {
        "framework": "gdpr",
        "query": "Under EU GDPR, what are the mandatory conditions and timelines for notifying supervisory authorities about a personal data breach?",
        "expected_control_ids": ["article 33", "33.1", "article 34"],
        "required_concepts": ["72 hours", "supervisory authority", "undue delay", "personal data breach"]
    },
    {
        "framework": "hipaa",
        "query": "What are the specific technical safeguard specifications for automatic logoff and ePHI encryption?",
        "expected_control_ids": ["164.312(a)(2)(iii)", "164.312(a)(2)(iv)", "164.312(a).118"],
        "required_concepts": ["automatic logoff", "encryption", "decryption", "ephi"]
    },
    {
        "framework": "csf",
        "query": "Under NIST CSF 2.0, what controls define identity management, credentials, and access control under the Protect function?",
        "expected_control_ids": ["pr.aa-01", "pr.aa-02", "pr.aa-05"],
        "required_concepts": ["identities", "credentials", "access control", "authentication"]
    },
    {
        "framework": "iso27001",
        "query": "What are the ISO/IEC 27001:2022 Annex A controls for logging, monitoring, and information security for cloud services?",
        "expected_control_ids": ["a.8.15", "a.8.16", "a.5.23"],
        "required_concepts": ["logging", "monitoring", "cloud services", "audit"]
    },
    {
        "framework": "dpdp",
        "query": "Under India's Digital Personal Data Protection (DPDP) Act 2023, what are the obligations for obtaining verifiable parental consent for processing children's data?",
        "expected_control_ids": ["section 9", "9(1)", "9(2)", "9(3)"],
        "required_concepts": ["child", "parental consent", "tracking", "behavioral monitoring"]
    },
    {
        "framework": "soc2",
        "query": "In SOC 2 Trust Services Criteria, what are the Common Criteria controls governing logical access security and credentials?",
        "expected_control_ids": ["cc6.1", "cc6.2", "cc6.3"],
        "required_concepts": ["logical access", "credentials", "unauthorized access", "infrastructure"]
    },
    {
        "framework": "asvs",
        "query": "In OWASP ASVS v5.0, what are the verification requirements for password security, salting, and hashing algorithms?",
        "expected_control_ids": ["v2.1.1", "v2.1.2", "v2.4.1", "v2.4.2"],
        "required_concepts": ["password", "salt", "hashing", "bcrypt", "pbkdf2", "argon2"]
    },
    {
        "framework": "dora",
        "query": "Under EU DORA, what are the key requirements for financial entities regarding ICT third-party risk management and contractual arrangements?",
        "expected_control_ids": ["article 28", "article 29", "article 30"],
        "required_concepts": ["ict third-party", "contractual arrangements", "subcontracting", "audit"]
    }
]


def extract_potential_control_mentions(text: str, framework: str) -> List[str]:
    """Extracts candidate control IDs or article numbers cited in generated text."""
    patterns = [
        r'(?:article|art\.?|section|sec\.?)\s*([0-9]+(?:\([0-9a-zA-Z]+\))*)',
        r'164\.[0-9]+(?:\([a-z0-9]+\))*(?:\.[0-9]+)*',
        r'(?:[A-Z]{2}\.[A-Z]{2}-[0-9]{2})',
        r'(?:A\.[0-9]+\.[0-9]+)',
        r'(?:CC[0-9]+\.[0-9]+)',
        r'(?:V[0-9]+\.[0-9]+(?:\.[0-9]+)*)',
    ]
    
    found = set()
    for p in patterns:
        for match in re.finditer(p, text, re.IGNORECASE):
            found.add(match.group(0).lower().strip())
    return list(found)


def evaluate_single_answer(
    answer: str,
    framework: str,
    expected_ids: List[str],
    required_concepts: List[str]
) -> Dict[str, Any]:
    """Evaluates citation validity, hallucination presence, and concept recall."""
    valid_catalog = VALID_CONTROL_CATALOG.get(framework.lower(), set())
    extracted_citations = extract_potential_control_mentions(answer, framework)
    
    valid_citations = 0
    hallucinated_citations = 0

    for cite in extracted_citations:
        clean_cite = cite.replace("article", "").replace("section", "").replace("control", "").strip()
        
        # Check if cite matches any valid control in catalog
        is_valid = any(
            clean_cite in valid_id or valid_id in cite or cite in valid_id 
            for valid_id in valid_catalog
        ) if valid_catalog else True

        if is_valid:
            valid_citations += 1
        else:
            hallucinated_citations += 1

    total_citations = len(extracted_citations)
    citation_precision = (valid_citations / total_citations) * 100 if total_citations > 0 else 100.0
    has_hallucination = hallucinated_citations > 0

    # Ground-truth expected control recall
    target_hits = sum(1 for exp in expected_ids if exp.lower() in answer.lower())
    target_recall = (target_hits / len(expected_ids)) * 100 if expected_ids else 0.0

    # Required concept recall
    concept_hits = sum(1 for c in required_concepts if c.lower() in answer.lower())
    concept_recall = (concept_hits / len(required_concepts)) * 100 if required_concepts else 0.0

    return {
        "total_citations_found": total_citations,
        "valid_citations": valid_citations,
        "hallucinated_citations": hallucinated_citations,
        "citation_precision_pct": round(citation_precision, 1),
        "has_hallucination": has_hallucination,
        "target_control_recall_pct": round(target_recall, 1),
        "concept_recall_pct": round(concept_recall, 1),
        "citations_extracted": extracted_citations
    }


def run_grounded_vs_ungrounded_evaluation() -> Dict[str, Any]:
    """Runs systematic comparison between Ungrounded and Grounded Generation."""
    print("=" * 90)
    print("🔬 RUNNING GROUNDED GENERATION ACCURACY EVALUATION")
    print(f"📊 Benchmarking {len(EVAL_CASES)} High-Stakes Audit Scenarios across 8 Frameworks")
    print("=" * 90)

    ungrounded_runs = []
    grounded_runs = []

    for idx, case in enumerate(EVAL_CASES, 1):
        fw = case["framework"]
        query = case["query"]
        expected_ids = case["expected_control_ids"]
        concepts = case["required_concepts"]

        print(f"\n[{idx:02d}/{len(EVAL_CASES)}] Evaluating Framework: {fw.upper()}...")

        # 1. Run Grounded Generation (Method 1)
        print("  🟢 Running Grounded Generation (Per-Framework Vector DB)...")
        grounded_res = generate_grounded_compliance_answer(query, fw, k=4, max_tokens=250)
        grounded_eval = evaluate_single_answer(grounded_res["answer"], fw, expected_ids, concepts)
        grounded_eval["latency_ms"] = grounded_res["total_latency_ms"]
        grounded_eval["retrieval_ms"] = grounded_res["retrieval_latency_ms"]
        grounded_eval["framework"] = fw.upper()
        grounded_runs.append(grounded_eval)

        # 2. Run Ungrounded Baseline Generation
        print("  ⚪ Running Ungrounded Baseline (Direct LLM)...")
        ungrounded_res = generate_ungrounded_answer(query, fw, max_tokens=250)
        ungrounded_eval = evaluate_single_answer(ungrounded_res["answer"], fw, expected_ids, concepts)
        ungrounded_eval["latency_ms"] = ungrounded_res["total_latency_ms"]
        ungrounded_eval["framework"] = fw.upper()
        ungrounded_runs.append(ungrounded_eval)

        print(f"     Grounded  -> Precision: {grounded_eval['citation_precision_pct']}% | Recall: {grounded_eval['target_control_recall_pct']}% | Hallucination: {grounded_eval['has_hallucination']}")
        print(f"     Baseline  -> Precision: {ungrounded_eval['citation_precision_pct']}% | Recall: {ungrounded_eval['target_control_recall_pct']}% | Hallucination: {ungrounded_eval['has_hallucination']}")

    # Aggregate Metrics
    n = len(EVAL_CASES)
    summary = {
        "scenarios_evaluated": n,
        "ungrounded_baseline": {
            "mean_citation_precision_pct": round(float(np.mean([r["citation_precision_pct"] for r in ungrounded_runs])), 2),
            "hallucination_rate_pct": round(float(sum(1 for r in ungrounded_runs if r["has_hallucination"]) / n) * 100, 2),
            "target_control_recall_pct": round(float(np.mean([r["target_control_recall_pct"] for r in ungrounded_runs])), 2),
            "statutory_concept_recall_pct": round(float(np.mean([r["concept_recall_pct"] for r in ungrounded_runs])), 2),
            "mean_latency_ms": round(float(np.mean([r["latency_ms"] for r in ungrounded_runs])), 2)
        },
        "per_framework_grounded": {
            "mean_citation_precision_pct": round(float(np.mean([r["citation_precision_pct"] for r in grounded_runs])), 2),
            "hallucination_rate_pct": round(float(sum(1 for r in grounded_runs if r["has_hallucination"]) / n) * 100, 2),
            "target_control_recall_pct": round(float(np.mean([r["target_control_recall_pct"] for r in grounded_runs])), 2),
            "statutory_concept_recall_pct": round(float(np.mean([r["concept_recall_pct"] for r in grounded_runs])), 2),
            "mean_latency_ms": round(float(np.mean([r["latency_ms"] for r in grounded_runs])), 2),
            "mean_retrieval_latency_ms": round(float(np.mean([r["retrieval_ms"] for r in grounded_runs])), 2)
        },
        "grounded_runs": grounded_runs,
        "ungrounded_runs": ungrounded_runs
    }

    return summary


def print_comparison_table(summary: Dict[str, Any]):
    u = summary["ungrounded_baseline"]
    g = summary["per_framework_grounded"]

    print("\n" + "=" * 90)
    print("📊 GENERATION ACCURACY EVALUATION RESULTS (GROUNDED vs UNGROUNDED)")
    print("=" * 90)
    print(f"""
| Metric / Evaluation Dimension         | Ungrounded Baseline (Direct LLM) | Per-Framework Grounded (Method 1) | Relative Improvement |
|---------------------------------------|:--------------------------------:|:--------------------------------:|:--------------------:|
| **Valid Citation Precision Rate**     | {u['mean_citation_precision_pct']:>6.1f}%                          | 🌟 {g['mean_citation_precision_pct']:>6.1f}%                       | **+{g['mean_citation_precision_pct'] - u['mean_citation_precision_pct']:.1f}%** |
| **Statutory Hallucination Rate**      | {u['hallucination_rate_pct']:>6.1f}%                          | 🌟 {g['hallucination_rate_pct']:>6.1f}%                       | **-{u['hallucination_rate_pct'] - g['hallucination_rate_pct']:.1f}% reduction** |
| **Target Control ID Recall**          | {u['target_control_recall_pct']:>6.1f}%                          | 🌟 {g['target_control_recall_pct']:>6.1f}%                       | **+{g['target_control_recall_pct'] - u['target_control_recall_pct']:.1f}%** |
| **Statutory Concept Recall**          | {u['statutory_concept_recall_pct']:>6.1f}%                          | 🌟 {g['statutory_concept_recall_pct']:>6.1f}%                       | **+{g['statutory_concept_recall_pct'] - u['statutory_concept_recall_pct']:.1f}%** |
| **Average End-to-End Latency**        | {u['mean_latency_ms']:>6.1f} ms                       | {g['mean_latency_ms']:>6.1f} ms                       | *(Includes {g['mean_retrieval_latency_ms']:.1f}ms RAG)* |
""")
    print("=" * 90)


def main():
    summary = run_grounded_vs_ungrounded_evaluation()
    print_comparison_table(summary)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grounded_accuracy_evaluation_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"💾 Detailed generation accuracy report saved to: {output_path}")


if __name__ == "__main__":
    main()
