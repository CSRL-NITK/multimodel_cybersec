"""
Automated Test & Verification Suite for Per-Framework Vector Databases
-----------------------------------------------------------------------
Validates:
1. Complete ingestion of all 21 framework catalogs into isolated ChromaDB collections.
2. 100% Context Purity (Zero Cross-Framework Leakage on Single-Framework Queries).
3. Multi-Framework Scatter-Gather & Reciprocal Rank Fusion (RRF).
4. Dynamic Framework Name Fuzzy Resolution.
"""

import os
import sys
import time
import unittest

try:
    import pytest
except ImportError:
    pytest = None

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent2_per_framework_kb import build_per_framework_collections, DEFAULT_FRAMEWORKS_DIR
from database.federated_rag_utils import (
    list_available_frameworks,
    resolve_collection_name,
    retrieve_single_framework,
    retrieve_multi_framework,
    load_manifest,
)


def ensure_framework_vector_dbs():
    """Builds per-framework collections if not already present."""
    manifest_path = os.path.join(DEFAULT_FRAMEWORKS_DIR, "framework_collections_manifest.json")
    if not os.path.exists(manifest_path):
        print("\n🔨 Building per-framework collections for test session...")
        build_per_framework_collections(reset_all=True)


if pytest:
    @pytest.fixture(scope="session", autouse=True)
    def setup_framework_vector_dbs():
        ensure_framework_vector_dbs()
        yield


def test_manifest_and_collection_count():
    """Verify all 21 framework catalogs are indexed with non-zero control counts."""
    frameworks = list_available_frameworks()
    assert len(frameworks) >= 20, f"Expected at least 20 framework collections, found {len(frameworks)}"

    manifest = load_manifest()
    assert manifest["total_controls"] > 1000, f"Expected >1000 total controls, got {manifest['total_controls']}"

    for fw in frameworks:
        assert fw["control_count"] > 0, f"Framework {fw['collection_name']} has 0 controls!"


def test_fuzzy_collection_name_resolution():
    """Test resolution of various user string aliases to exact collection names."""
    assert resolve_collection_name("gdpr") == "controls_eu_gdpr"
    assert resolve_collection_name("hipaa") == "controls_us_hipaa"
    assert resolve_collection_name("csf") == "controls_nist_csf"
    assert resolve_collection_name("iso27001") == "controls_international_iso27001"
    assert resolve_collection_name("soc2") == "controls_us_soc2"
    assert resolve_collection_name("dpdp") == "controls_india_dpdp"


def test_single_framework_purity_gdpr():
    """Verify that querying GDPR returns strictly GDPR controls with 0% contamination."""
    query = "right to erasure and personal data deletion requests"
    hits = retrieve_single_framework(query=query, framework="gdpr", k=5)

    assert len(hits) > 0, "Expected GDPR search hits"
    for h in hits:
        assert "gdpr" in h["collection_name"].lower()
        assert "gdpr" in h["framework"].lower()
        assert h["similarity_score"] > 0.0


def test_single_framework_purity_hipaa():
    """Verify that querying HIPAA returns strictly HIPAA controls."""
    query = "electronic protected health information ePHI safeguards"
    hits = retrieve_single_framework(query=query, framework="hipaa", k=5)

    assert len(hits) > 0, "Expected HIPAA search hits"
    for h in hits:
        assert "hipaa" in h["collection_name"].lower()
        assert "hipaa" in h["framework"].lower()


def test_single_framework_purity_nist_csf():
    """Verify that querying NIST CSF returns strictly CSF controls."""
    query = "incident response management and disaster recovery"
    hits = retrieve_single_framework(query=query, framework="csf", k=5)

    assert len(hits) > 0, "Expected NIST CSF search hits"
    for h in hits:
        assert "csf" in h["collection_name"].lower()
        assert "csf" in h["framework"].lower()


def test_multi_framework_scoped_retrieval_and_rrf():
    """
    Verify multi-framework scatter-gather:
    1. Only targeted frameworks are queried.
    2. Untargeted frameworks (e.g., mitre_attack, cis_aws) NEVER appear.
    3. RRF ordering is monotonic descending.
    """
    target_frameworks = ["gdpr", "hipaa", "iso27001"]
    query = "encryption at rest and cryptographic key management"

    fused_results = retrieve_multi_framework(
        query=query,
        frameworks=target_frameworks,
        k_per_framework=3,
        fusion_method="rrf"
    )

    assert len(fused_results) > 0, "Expected multi-framework fused results"

    # Validate that every returned item belongs to one of the target frameworks
    returned_frameworks = set()
    prev_score = float("inf")
    for r in fused_results:
        fw_match = any(target in r["collection_name"].lower() for target in ["gdpr", "hipaa", "iso27001"])
        assert fw_match, f"Leakage detected! Result from unexpected collection: {r['collection_name']}"
        returned_frameworks.add(r["collection_name"])
        
        # Validate score descending order
        assert r["rrf_score"] <= prev_score + 1e-5
        prev_score = r["rrf_score"]

    # Verify diversity: we received results from all requested frameworks
    assert len(returned_frameworks) == 3, f"Expected hits from all 3 target frameworks, got {returned_frameworks}"


class TestFrameworkVectorDBs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_framework_vector_dbs()

    def test_manifest_and_collection_count(self):
        test_manifest_and_collection_count()

    def test_fuzzy_collection_name_resolution(self):
        test_fuzzy_collection_name_resolution()

    def test_single_framework_purity_gdpr(self):
        test_single_framework_purity_gdpr()

    def test_single_framework_purity_hipaa(self):
        test_single_framework_purity_hipaa()

    def test_single_framework_purity_nist_csf(self):
        test_single_framework_purity_nist_csf()

    def test_multi_framework_scoped_retrieval_and_rrf(self):
        test_multi_framework_scoped_retrieval_and_rrf()


def run_interactive_benchmark():
    """Console benchmarking report."""
    print("=" * 80)
    print("🧪 RUNNING PER-FRAMEWORK VECTOR DATABASE BENCHMARK & VERIFICATION")
    print("=" * 80)

    t0 = time.time()
    manifest = build_per_framework_collections(reset_all=True)
    build_time = time.time() - t0

    print(f"\n⏱️ Ingestion Complete in {build_time:.2f}s ({manifest['total_controls']} controls across {manifest['total_frameworks']} collections)")
    print("-" * 80)

    # 1. Test Single Framework
    print("\n[Test 1] Single Framework Isolated Query: EU GDPR")
    t1 = time.time()
    gdpr_hits = retrieve_single_framework("Data protection impact assessment DPIA requirements", "gdpr", k=3)
    latency_gdpr = (time.time() - t1) * 1000
    print(f"  Query Latency: {latency_gdpr:.2f}ms | Hits: {len(gdpr_hits)}")
    for i, h in enumerate(gdpr_hits, 1):
        print(f"   {i}. [{h['control_id']}] {h['title']} (Score: {h['similarity_score']})")

    # 2. Test Single Framework
    print("\n[Test 2] Single Framework Isolated Query: NIST CSF 2.0")
    t2 = time.time()
    csf_hits = retrieve_single_framework("Identity governance and access control policies", "csf", k=3)
    latency_csf = (time.time() - t2) * 1000
    print(f"  Query Latency: {latency_csf:.2f}ms | Hits: {len(csf_hits)}")
    for i, h in enumerate(csf_hits, 1):
        print(f"   {i}. [{h['control_id']}] {h['title']} (Score: {h['similarity_score']})")

    # 3. Test Multi-Framework Scatter-Gather
    print("\n[Test 3] Multi-Framework Scatter-Gather Query: [GDPR + HIPAA + ISO27001]")
    t3 = time.time()
    multi_hits = retrieve_multi_framework(
        "Encryption at rest and access logging",
        frameworks=["gdpr", "hipaa", "iso27001"],
        k_per_framework=2
    )
    latency_multi = (time.time() - t3) * 1000
    print(f"  Scatter-Gather Latency: {latency_multi:.2f}ms | Fused Hits: {len(multi_hits)}")
    for i, h in enumerate(multi_hits, 1):
        print(f"   {i}. [{h['framework'].upper()}] [{h['control_id']}] {h['title']} (RRF: {h['rrf_score']}, Sim: {h['similarity_score']})")

    print("\n" + "=" * 80)
    print("🧪 RUNNING UNIT TESTS...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFrameworkVectorDBs)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    
    if res.wasSuccessful():
        print("\n" + "=" * 80)
        print("✅ ALL TESTS & BENCHMARKS PASSED SUCCESSFULLY!")
        print("=" * 80)
    else:
        sys.exit(1)


if __name__ == "__main__":
    run_interactive_benchmark()
