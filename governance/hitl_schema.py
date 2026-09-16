"""
HITL Schema & Review Session Manager
-------------------------------------
Defines the standard data model for Human-in-the-Loop compliance review sessions:
"Verify the Data, Not the File" architecture.

Tracks:
- AI Assessment findings (initial status, narrative, evidence citations)
- Human Review verdicts (approve, reject, modify, reviewer notes, timestamps)
- Final Reconciled state (effective status, narrative, human override flags)
- Session metadata and lifecycle status (DRAFT, IN_REVIEW, HUMAN_CERTIFIED)
"""

import os
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REVIEW_SESSIONS_DIR = os.path.join(PROJECT_ROOT, "governance", "review_sessions")
os.makedirs(REVIEW_SESSIONS_DIR, exist_ok=True)


def create_hitl_review_session(
    findings: List[Dict[str, Any]],
    framework: str,
    client_id: str = "default_client",
    assessment_id: Optional[str] = None,
    source_type: str = "Mode B (Codebase AST & Probes)"
) -> Dict[str, Any]:
    """
    Initializes a structured HITL review session from raw assessment findings.
    """
    if not assessment_id:
        assessment_id = f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    normalized_findings = []
    compliant_count = 0
    partial_count = 0

    for idx, f in enumerate(findings):
        cid = str(f.get("control_id") or f.get("id") or f"CTRL-{idx+1}")
        title = str(f.get("title") or f.get("name") or cid)
        desc = str(f.get("description") or f.get("text") or "")
        
        # Raw AI evaluation values
        ai_status = str(f.get("status") or f.get("verdict") or "Non-Compliant")
        ai_narrative = str(f.get("explanation") or f.get("rationale") or f.get("finding") or "Evaluated against ingested evidence.")
        evidence = f.get("evidence") or f.get("evidence_citations") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        confidence = float(f.get("confidence") or f.get("confidence_score") or 0.85)

        # Count initial score
        if ai_status.lower() == "compliant":
            compliant_count += 1
        elif "partially" in ai_status.lower() or "partial" in ai_status.lower():
            partial_count += 0.5

        finding_item = {
            "control_id": cid,
            "title": title,
            "description": desc,
            "framework": framework.upper(),
            "ai_output": {
                "status": ai_status,
                "narrative": ai_narrative,
                "evidence": evidence,
                "confidence": confidence
            },
            "human_review": {
                "verdict": "unreviewed",  # "approve" | "reject" | "modify" | "unreviewed"
                "reviewer_comment": "",
                "corrected_status": ai_status,
                "corrected_narrative": ai_narrative,
                "reviewer_id": None,
                "timestamp": None
            },
            "final_effective": {
                "status": ai_status,
                "narrative": ai_narrative,
                "is_human_overridden": False
            }
        }
        normalized_findings.append(finding_item)

    total = len(normalized_findings)
    draft_s2score = int(300 + ((compliant_count + partial_count) / max(1, total)) * 550)

    session = {
        "assessment_id": assessment_id,
        "framework": framework.upper(),
        "client_id": client_id,
        "source_type": source_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_updated_at": datetime.now(timezone.utc).isoformat(),
        "review_status": "DRAFT",  # "DRAFT" | "IN_REVIEW" | "HUMAN_CERTIFIED"
        "total_controls": total,
        "draft_s2score": draft_s2score,
        "certified_s2score": draft_s2score,
        "reviewed_count": 0,
        "overridden_count": 0,
        "certified_by": None,
        "certified_at": None,
        "findings": normalized_findings
    }

    save_hitl_review_session(session)
    return session


def save_hitl_review_session(session: Dict[str, Any]) -> str:
    """Persists review session to disk."""
    session_id = session["assessment_id"]
    path = os.path.join(REVIEW_SESSIONS_DIR, f"{session_id}.json")
    session["last_updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2)
    return path


def load_hitl_review_session(assessment_id: str) -> Optional[Dict[str, Any]]:
    """Loads a review session by assessment ID."""
    path = os.path.join(REVIEW_SESSIONS_DIR, f"{assessment_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_hitl_review_sessions() -> List[Dict[str, Any]]:
    """Lists all saved review sessions."""
    sessions = []
    for path in sorted(os.listdir(REVIEW_SESSIONS_DIR), reverse=True):
        if path.endswith(".json"):
            try:
                with open(os.path.join(REVIEW_SESSIONS_DIR, path), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sessions.append({
                        "assessment_id": data.get("assessment_id"),
                        "framework": data.get("framework"),
                        "client_id": data.get("client_id"),
                        "review_status": data.get("review_status"),
                        "draft_s2score": data.get("draft_s2score"),
                        "certified_s2score": data.get("certified_s2score"),
                        "total_controls": data.get("total_controls"),
                        "created_at": data.get("created_at"),
                        "certified_at": data.get("certified_at")
                    })
            except Exception:
                pass
    return sessions
