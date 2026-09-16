"""
HITL Self-Healing Re-Examination & Reconciliation Engine
---------------------------------------------------------
Executes:
1. LLM Self-Healing Re-Generation on human rejection or modification.
2. Re-calculation of FISASCORE / S2Score (300-850 index).
3. Logging of (prompt, chosen, rejected) DPO training triplets.
4. Final certification and provenance metadata tracking.
"""

import os
import sys
import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from governance.hitl_schema import load_hitl_review_session, save_hitl_review_session
import agents.config as agent_config

DPO_LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "dpo_human_feedback.jsonl")
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)


def log_dpo_training_triplet(
    control_id: str,
    framework: str,
    ai_output: Dict[str, Any],
    reviewer_comment: str,
    corrected_output: Dict[str, Any],
    reviewer_id: str = "auditor"
):
    """
    Appends an immutable DPO pair:
    Chosen: Corrected Human/Regenerated Output
    Rejected: Flawed AI Output
    """
    prompt = (
        f"Assess compliance for {framework.upper()} control {control_id}. "
        f"Evidence: {json.dumps(ai_output.get('evidence', []))}"
    )
    
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "framework": framework.upper(),
        "control_id": control_id,
        "reviewer_id": reviewer_id,
        "reviewer_comment": reviewer_comment,
        "prompt": prompt,
        "chosen": f"Status: {corrected_output.get('status')}\nExplanation: {corrected_output.get('narrative')}",
        "rejected": f"Status: {ai_output.get('status')}\nExplanation: {ai_output.get('narrative')}"
    }

    try:
        with open(DPO_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"Warning: Failed to log DPO pair: {e}")


def regenerate_finding_with_feedback(
    control_id: str,
    framework: str,
    title: str,
    ai_output: Dict[str, Any],
    reviewer_comment: str,
    target_verdict: Optional[str] = None
) -> Dict[str, Any]:
    """
    Prompts Qwen 2.5 to re-examine the finding taking human domain feedback into account.
    """
    initial_status = ai_output.get("status", "Non-Compliant")
    initial_narrative = ai_output.get("narrative", "")
    evidence = ai_output.get("evidence", [])

    prompt = f"""You are an expert regulatory compliance auditor. An initial automated assessment was reviewed and modified by a certified human compliance officer.

REGULATORY CONTROL: [{framework.upper()}] {control_id} - {title}
EVIDENCE ON RECORD: {evidence}

INITIAL AI ASSESSMENT:
- Status: {initial_status}
- Rationale: {initial_narrative}

HUMAN AUDITOR CORRECTION & NOTES:
"{reviewer_comment}"
{f"MANDATED TARGET STATUS: {target_verdict}" if target_verdict else ""}

TASK:
Regenerate a professional, audit-ready compliance narrative that incorporates the human auditor's domain guidance while preserving technical rigor.

Format your response as:
Status: [Compliant | Partially Compliant | Non-Compliant | Risk Accepted]
Rationale: [2-3 sentences explaining the verified compliance posture incorporating the auditor guidance]"""

    try:
        raw_response = agent_config.generate(prompt, max_new_tokens=200)
        
        # Parse status
        status_match = re.search(r"Status:\s*(Compliant|Partially Compliant|Non-Compliant|Risk Accepted)", raw_response, re.IGNORECASE)
        status = target_verdict if target_verdict else (status_match.group(1).title() if status_match else initial_status)
        
        # Parse rationale
        if "Rationale:" in raw_response:
            narrative = raw_response.split("Rationale:", 1)[-1].strip()
        elif "Explanation:" in raw_response:
            narrative = raw_response.split("Explanation:", 1)[-1].strip()
        else:
            narrative = raw_response.strip()

        if len(narrative) < 15:
            narrative = f"Verified by certified auditor: {reviewer_comment}. Status reconciled to {status}."

        return {
            "status": status,
            "narrative": narrative,
            "regenerated": True
        }
    except Exception as e:
        print(f"Notice: Fallback generation during re-examination: {e}")
        status = target_verdict if target_verdict else initial_status
        return {
            "status": status,
            "narrative": f"Auditor Verified: {reviewer_comment}",
            "regenerated": False
        }


def reconcile_and_recalculate_session(
    session_id: str,
    verifications: Dict[str, Dict[str, Any]],
    reviewer_id: str = "auditor@enterprise.com",
    auto_regenerate: bool = True
) -> Dict[str, Any]:
    """
    Main HITL reconciliation loop:
    - Ingests human feedback
    - Re-generates disputed findings
    - Recalculates S2Score (300-850)
    - Logs DPO training pairs
    - Produces certified structured session
    """
    session = load_hitl_review_session(session_id)
    if not session:
        raise FileNotFoundError(f"Review session '{session_id}' not found.")

    findings = session.get("findings", [])
    framework = session.get("framework", "NIST_CSF")
    
    reviewed_count = 0
    overridden_count = 0
    compliant_count = 0
    partial_count = 0

    reconciled_findings = []

    for f in findings:
        cid = f["control_id"]
        v = verifications.get(cid, {})
        verdict = v.get("verdict", "unreviewed").lower()  # approve | reject | modify | unreviewed

        if verdict in ["approve", "approved", "true"]:
            f["human_review"]["verdict"] = "approve"
            f["human_review"]["reviewer_comment"] = v.get("comment", "")
            f["human_review"]["corrected_status"] = f["ai_output"]["status"]
            f["human_review"]["corrected_narrative"] = f["ai_output"]["narrative"]
            f["human_review"]["reviewer_id"] = reviewer_id
            f["human_review"]["timestamp"] = datetime.now(timezone.utc).isoformat()
            
            f["final_effective"]["status"] = f["ai_output"]["status"]
            f["final_effective"]["narrative"] = f["ai_output"]["narrative"]
            f["final_effective"]["is_human_overridden"] = False
            reviewed_count += 1

        elif verdict in ["reject", "rejected", "modify", "modified", "false"]:
            reviewer_comment = v.get("comment", "")
            target_status = v.get("corrected_status") or ("Compliant" if f["ai_output"]["status"] == "Non-Compliant" else "Non-Compliant")

            if auto_regenerate and reviewer_comment:
                regen_res = regenerate_finding_with_feedback(
                    control_id=cid,
                    framework=framework,
                    title=f["title"],
                    ai_output=f["ai_output"],
                    reviewer_comment=reviewer_comment,
                    target_verdict=target_status
                )
                corrected_status = regen_res["status"]
                corrected_narrative = regen_res["narrative"]
            else:
                corrected_status = target_status
                corrected_narrative = f"Auditor Overridden: {reviewer_comment}" if reviewer_comment else f"Human Override: Status set to {target_status}."

            f["human_review"]["verdict"] = verdict
            f["human_review"]["reviewer_comment"] = reviewer_comment
            f["human_review"]["corrected_status"] = corrected_status
            f["human_review"]["corrected_narrative"] = corrected_narrative
            f["human_review"]["reviewer_id"] = reviewer_id
            f["human_review"]["timestamp"] = datetime.now(timezone.utc).isoformat()

            f["final_effective"]["status"] = corrected_status
            f["final_effective"]["narrative"] = corrected_narrative
            f["final_effective"]["is_human_overridden"] = True

            # Log to DPO training loop
            log_dpo_training_triplet(
                control_id=cid,
                framework=framework,
                ai_output=f["ai_output"],
                reviewer_comment=reviewer_comment,
                corrected_output={"status": corrected_status, "narrative": corrected_narrative},
                reviewer_id=reviewer_id
            )

            reviewed_count += 1
            overridden_count += 1

        else:
            # Keep existing or draft state
            f["final_effective"]["status"] = f["ai_output"]["status"]
            f["final_effective"]["narrative"] = f["ai_output"]["narrative"]
            f["final_effective"]["is_human_overridden"] = False

        # Score tally
        eff_status = f["final_effective"]["status"].lower()
        if eff_status == "compliant":
            compliant_count += 1
        elif "partially" in eff_status or "partial" in eff_status:
            partial_count += 0.5

        reconciled_findings.append(f)

    total = len(reconciled_findings)
    certified_s2score = int(300 + ((compliant_count + partial_count) / max(1, total)) * 550)

    session["findings"] = reconciled_findings
    session["reviewed_count"] = reviewed_count
    session["overridden_count"] = overridden_count
    session["certified_s2score"] = certified_s2score
    session["review_status"] = "HUMAN_CERTIFIED" if reviewed_count == total else "IN_REVIEW"
    session["certified_by"] = reviewer_id
    session["certified_at"] = datetime.now(timezone.utc).isoformat()

    save_hitl_review_session(session)
    return session
