"""
Automated Test Suite for Human-in-the-Loop (HITL) Verification & Re-Examination
--------------------------------------------------------------------------------
Validates:
1. "Verify the Data, Not the File" structured JSON session initialization.
2. Per-control human approval vs. dispute/modification workflows.
3. LLM self-healing re-generation using auditor comments.
4. S2Score dynamic recalculation (300-850 index).
5. Immutable DPO training triplet logging (logs/dpo_human_feedback.jsonl).
6. Dual-mode export (Raw AI Draft vs. Certified Verified Report with attestation banner).
"""

import os
import sys
import json
import unittest
from datetime import datetime, timezone

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from governance.hitl_schema import (
    create_hitl_review_session,
    load_hitl_review_session,
    save_hitl_review_session,
    list_hitl_review_sessions
)
from governance.hitl_regenerator import (
    reconcile_and_recalculate_session,
    DPO_LOG_FILE
)
from utils.hitl_report_formatter import (
    format_hitl_session_to_markdown,
    export_hitl_session
)


SAMPLE_AI_FINDINGS = [
    {
        "control_id": "PR.AA-01",
        "title": "Identities and credentials are managed",
        "description": "Identities and credentials for authorized users, services, and hardware are managed.",
        "status": "Non-Compliant",
        "rationale": "No local user password hashing function detected in auth controller.",
        "evidence": ["controllers/auth.py:45"],
        "confidence": 0.85
    },
    {
        "control_id": "PR.DS-01",
        "title": "Data-at-rest is protected",
        "description": "Data-at-rest is protected using approved cryptographic algorithms.",
        "status": "Compliant",
        "rationale": "Found AES-256-GCM database column encryption in models/vault.py.",
        "evidence": ["models/vault.py:112"],
        "confidence": 0.95
    },
    {
        "control_id": "PR.DS-02",
        "title": "Data-in-transit is protected",
        "description": "Data-in-transit is protected using modern TLS protocols.",
        "status": "Non-Compliant",
        "rationale": "HTTP endpoint does not enforce strict HSTS header.",
        "evidence": ["nginx/conf.d/default.conf:12"],
        "confidence": 0.78
    },
    {
        "control_id": "DE.CM-01",
        "title": "Networks and environments are monitored",
        "description": "Networks and physical environments are monitored to detect potential cybersecurity events.",
        "status": "Partially Compliant",
        "rationale": "Application logs to local files but lacks centralized SIEM log shipping.",
        "evidence": ["utils/logger.py:34"],
        "confidence": 0.88
    }
]


class TestHITLVerificationLoop(unittest.TestCase):

    def setUp(self):
        self.session = create_hitl_review_session(
            findings=SAMPLE_AI_FINDINGS,
            framework="NIST_CSF",
            client_id="TestCorp_Fintech",
            assessment_id=f"test_audit_{int(datetime.now().timestamp())}"
        )
        self.session_id = self.session["assessment_id"]

    def test_session_initialization_schema(self):
        """Verify session is initialized with valid structured JSON schema."""
        self.assertIsNotNone(self.session_id)
        self.assertEqual(self.session["review_status"], "DRAFT")
        self.assertEqual(self.session["total_controls"], 4)
        
        # Initial score: Compliant (1) + Partial (0.5) / 4 = 1.5/4 * 550 + 300 = 506
        self.assertEqual(self.session["draft_s2score"], 506)
        self.assertEqual(len(self.session["findings"]), 4)

        # Check first finding structure
        f1 = self.session["findings"][0]
        self.assertEqual(f1["control_id"], "PR.AA-01")
        self.assertEqual(f1["ai_output"]["status"], "Non-Compliant")
        self.assertEqual(f1["human_review"]["verdict"], "unreviewed")
        self.assertFalse(f1["final_effective"]["is_human_overridden"])

    def test_human_approval_workflow(self):
        """Verify auditor approval maintains AI verdict and updates review count."""
        verifications = {
            "PR.DS-01": {"verdict": "approve", "comment": "Verified database encryption"}
        }

        reconciled = reconcile_and_recalculate_session(
            session_id=self.session_id,
            verifications=verifications,
            reviewer_id="lead_auditor@cert.org",
            auto_regenerate=False
        )

        f_ds01 = next(f for f in reconciled["findings"] if f["control_id"] == "PR.DS-01")
        self.assertEqual(f_ds01["human_review"]["verdict"], "approve")
        self.assertEqual(f_ds01["final_effective"]["status"], "Compliant")
        self.assertFalse(f_ds01["final_effective"]["is_human_overridden"])
        self.assertEqual(reconciled["reviewed_count"], 1)
        self.assertEqual(reconciled["overridden_count"], 0)

    def test_human_dispute_and_score_recalculation(self):
        """Verify disputing non-compliant controls dynamically increases S2Score."""
        initial_score = self.session["draft_s2score"]

        # Auditor disputes PR.AA-01 and PR.DS-02, marking them as Compliant
        verifications = {
            "PR.AA-01": {
                "verdict": "modify",
                "corrected_status": "Compliant",
                "comment": "Authentication is federated via Okta SAML 2.0 at API Gateway."
            },
            "PR.DS-01": {
                "verdict": "approve",
                "comment": "Confirmed AES-256."
            },
            "PR.DS-02": {
                "verdict": "modify",
                "corrected_status": "Compliant",
                "comment": "TLS termination and HSTS are enforced by AWS CloudFront distribution."
            },
            "DE.CM-01": {
                "verdict": "approve",
                "comment": "Confirmed partial SIEM coverage."
            }
        }

        reconciled = reconcile_and_recalculate_session(
            session_id=self.session_id,
            verifications=verifications,
            reviewer_id="ciso@testcorp.com",
            auto_regenerate=True
        )

        # 3.5 / 4 * 550 + 300 = 781
        new_score = reconciled["certified_s2score"]
        self.assertGreater(new_score, initial_score)
        self.assertEqual(new_score, 781)
        self.assertEqual(reconciled["review_status"], "HUMAN_CERTIFIED")
        self.assertEqual(reconciled["overridden_count"], 2)

        # Verify PR.AA-01 is marked as human overridden
        f_aa01 = next(f for f in reconciled["findings"] if f["control_id"] == "PR.AA-01")
        self.assertEqual(f_aa01["final_effective"]["status"], "Compliant")
        self.assertTrue(f_aa01["final_effective"]["is_human_overridden"])

    def test_dpo_triplet_logging(self):
        """Verify rejected/modified findings write DPO training triplets to disk."""
        if os.path.exists(DPO_LOG_FILE):
            os.remove(DPO_LOG_FILE)

        verifications = {
            "PR.AA-01": {
                "verdict": "modify",
                "corrected_status": "Compliant",
                "comment": "MFA and password policies are offloaded to Okta SSO."
            }
        }

        reconcile_and_recalculate_session(
            session_id=self.session_id,
            verifications=verifications,
            reviewer_id="auditor@test.com",
            auto_regenerate=False
        )

        self.assertTrue(os.path.exists(DPO_LOG_FILE))
        with open(DPO_LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertGreaterEqual(len(lines), 1)
            last_entry = json.loads(lines[-1])
            self.assertEqual(last_entry["control_id"], "PR.AA-01")
            self.assertIn("Okta SSO", last_entry["reviewer_comment"])
            self.assertIn("Non-Compliant", last_entry["rejected"])

    def test_dual_mode_report_export(self):
        """Verify generation of Option A (Raw AI Draft) vs Option B (Certified Verified Report)."""
        # 1. Draft Mode
        draft_md = format_hitl_session_to_markdown(self.session, certified_mode=False)
        self.assertIn("PRELIMINARY AI DRAFT", draft_md)
        self.assertNotIn("CERTIFIED HUMAN AUDITOR ATTESTATION", draft_md)

        # Reconcile session for certification
        verifications = {
            "PR.AA-01": {"verdict": "modify", "corrected_status": "Compliant", "comment": "Handled via Okta SSO"},
            "PR.DS-01": {"verdict": "approve"},
            "PR.DS-02": {"verdict": "approve"},
            "DE.CM-01": {"verdict": "approve"}
        }
        reconciled = reconcile_and_recalculate_session(
            self.session_id, verifications, reviewer_id="auditor_lead@enterprise.com", auto_regenerate=False
        )

        # 2. Certified Mode
        cert_md = format_hitl_session_to_markdown(reconciled, certified_mode=True)
        self.assertIn("CERTIFIED HUMAN AUDITOR ATTESTATION", cert_md)
        self.assertIn("auditor_lead@enterprise.com", cert_md)
        self.assertIn("Auditor Verification Provenance & Change Log", cert_md)
        self.assertIn("Handled via Okta SSO", cert_md)

        # Test export to file
        out_path = export_hitl_session(self.session_id, certified_mode=True, output_format="md")
        self.assertTrue(os.path.exists(out_path))
        if os.path.exists(out_path):
            os.remove(out_path)


def main():
    print("=" * 80)
    print("🧪 RUNNING HUMAN-IN-THE-LOOP (HITL) VERIFICATION & RE-EXAMINATION TESTS")
    print("=" * 80)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHITLVerificationLoop)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    if not res.wasSuccessful():
        sys.exit(1)
    print("\n✅ ALL HITL VERIFICATION & RE-EXAMINATION TESTS PASSED!")


if __name__ == "__main__":
    main()
