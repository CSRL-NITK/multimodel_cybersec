"""
HITL Report Formatter & Dual-Mode Exporter
------------------------------------------
Renders structured HITL Review Sessions (JSON) into full Markdown, HTML, and Word DOCX reports.

Supports:
1. Option A: Raw AI Draft Mode (Watermarked as "Draft — Pending Human Attestation")
2. Option B: Certified Verified Mode (Contains "Auditor Attestation Badge" + "Verification Provenance Table")
"""

import os
import sys
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from governance.hitl_schema import load_hitl_review_session
from utils.report_exporter import export_docx, export_report, REPORTS_DIR


def format_hitl_session_to_markdown(session: Dict[str, Any], certified_mode: bool = True) -> str:
    """
    Renders the structured review session JSON into audit-grade Markdown.
    """
    assessment_id = session.get("assessment_id", "AUDIT-001")
    framework = session.get("framework", "NIST_CSF")
    client_id = session.get("client_id", "Enterprise Client")
    total_controls = session.get("total_controls", len(session.get("findings", [])))
    findings = session.get("findings", [])
    
    is_certified = session.get("review_status") == "HUMAN_CERTIFIED" and certified_mode
    s2score = session.get("certified_s2score" if is_certified else "draft_s2score", 300)
    
    # Calculate compliant / partial / non-compliant counts
    compliant_count = 0
    partial_count = 0
    non_compliant_count = 0
    overridden_count = 0

    for f in findings:
        status = f["final_effective"]["status"] if is_certified else f["ai_output"]["status"]
        if status.lower() == "compliant":
            compliant_count += 1
        elif "partially" in status.lower() or "partial" in status.lower():
            partial_count += 1
        else:
            non_compliant_count += 1
            
        if f["final_effective"].get("is_human_overridden"):
            overridden_count += 1

    compliance_pct = round(((compliant_count + (0.5 * partial_count)) / max(1, total_controls)) * 100, 1)

    # Markdown Header
    status_badge = f"🛡️ CERTIFIED AUDIT REPORT — {framework}" if is_certified else f"⚠️ RAW AI DRAFT REPORT — {framework} (UNVERIFIED)"
    
    md = []
    md.append(f"# {status_badge}\n")
    md.append(f"**Target Entity / Client:** `{client_id}`  ")
    md.append(f"**Regulatory Framework:** `{framework}`  ")
    md.append(f"**Assessment UUID:** `{assessment_id}`  ")
    md.append(f"**Generated Date:** `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`  \n")

    # Attestation Stamp Box
    if is_certified:
        certified_by = session.get("certified_by", "auditor@enterprise.com")
        certified_at = session.get("certified_at", datetime.now(timezone.utc).isoformat())
        md.append("> [!IMPORTANT]")
        md.append(f"> **CERTIFIED HUMAN AUDITOR ATTESTATION**  ")
        md.append(f"> This compliance evaluation has been independently verified, re-examined, and signed off by certified compliance officer **`{certified_by}`** at **`{certified_at[:19]}`**.  ")
        md.append(f"> Total Controls Verified: **{total_controls}** | Human Overrides Applied: **{overridden_count}**\n")
    else:
        md.append("> [!WARNING]")
        md.append("> **PRELIMINARY AI DRAFT**  ")
        md.append("> This report contains unverified AI hypotheses. Formal audit certification requires human compliance review in the Verification Cockpit.\n")

    # Executive Scorecard Table
    md.append("## 1. Executive Summary & Benchmark S2Score\n")
    md.append(f"The target system achieved an overall **NIST S2Score Index of {s2score} / 850** ({compliance_pct}% statutory compliance rate).\n")
    md.append("| Compliance Metric | Result Value | Benchmark Rating |")
    md.append("|---|---|---|")
    md.append(f"| **FISASCORE / S2Score Index** | **{s2score} / 850** | {'🟢 Robust' if s2score >= 700 else ('🟡 Moderate' if s2score >= 500 else '🔴 Critical Gaps')} |")
    md.append(f"| **Total Evaluated Controls** | **{total_controls}** | 100% Ingested Catalog |")
    md.append(f"| **Compliant Controls** | **{compliant_count}** | {round((compliant_count/max(1,total_controls))*100, 1)}% |")
    md.append(f"| **Partially Compliant Controls** | **{partial_count}** | {round((partial_count/max(1,total_controls))*100, 1)}% |")
    md.append(f"| **Non-Compliant Posture Gaps** | **{non_compliant_count}** | {round((non_compliant_count/max(1,total_controls))*100, 1)}% |")
    md.append(f"| **Human Auditor Overrides** | **{overridden_count}** | Verification Provenance |")
    md.append("\n---\n")

    # Verification Provenance Table (Only in Certified Mode or if overrides exist)
    if is_certified and overridden_count > 0:
        md.append("## 2. Auditor Verification Provenance & Change Log\n")
        md.append("The following table documents each control where human auditor domain guidance modified or corrected the initial automated AI verdict:\n")
        md.append("| Control ID | Title | Initial AI Verdict | Final Certified Verdict | Auditor Rationale / Evidence Note |")
        md.append("|---|---|---|---|---|")
        for f in findings:
            if f["final_effective"].get("is_human_overridden"):
                cid = f["control_id"]
                title = f["title"][:35] + ("..." if len(f["title"]) > 35 else "")
                ai_stat = f["ai_output"]["status"]
                final_stat = f["final_effective"]["status"]
                note = f["human_review"].get("reviewer_comment", "Auditor Override")
                md.append(f"| `{cid}` | {title} | `{ai_stat}` | **`{final_stat}`** | {note} |")
        md.append("\n---\n")

    # Detailed Findings Section
    md.append(f"## {'3' if is_certified and overridden_count > 0 else '2'}. Detailed Control Findings\n")
    for idx, f in enumerate(findings, 1):
        cid = f["control_id"]
        title = f["title"]
        desc = f.get("description", "")
        eff_stat = f["final_effective"]["status"] if is_certified else f["ai_output"]["status"]
        narrative = f["final_effective"]["narrative"] if is_certified else f["ai_output"]["narrative"]
        evidence = f["ai_output"].get("evidence", [])
        
        status_icon = "🟢" if eff_stat.lower() == "compliant" else ("🟡" if "partial" in eff_stat.lower() else "🔴")
        overridden_tag = " *(Auditor Verified Override)*" if f["final_effective"].get("is_human_overridden") and is_certified else ""

        md.append(f"### {status_icon} [{cid}] {title}{overridden_tag}")
        if desc:
            md.append(f"> *Requirement: {desc}*\n")
        md.append(f"**Compliance Status:** `{eff_stat}`  ")
        md.append(f"**Assessment Rationale:** {narrative}  ")
        if evidence:
            md.append(f"**Evidence Citations:** `{evidence}`  ")
        md.append("")

    return "\n".join(md)


def export_hitl_session(
    session_id: str,
    certified_mode: bool = True,
    output_format: str = "docx",
    output_dir: str = REPORTS_DIR
) -> str:
    """
    Exports a HITL review session directly to DOCX, Markdown, HTML, or JSON.
    """
    session = load_hitl_review_session(session_id)
    if not session:
        raise FileNotFoundError(f"Review session '{session_id}' not found.")

    md_content = format_hitl_session_to_markdown(session, certified_mode=certified_mode)
    framework = session.get("framework", "csf").lower()
    client_id = session.get("client_id", "client")
    
    os.makedirs(output_dir, exist_ok=True)
    mode_prefix = "Certified_Verified" if certified_mode else "Raw_AI_Draft"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"Audit_Report_{mode_prefix}_{client_id}_{framework}_{timestamp}"

    if output_format == "md":
        out_path = os.path.join(output_dir, f"{base_name}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        return out_path

    elif output_format == "json":
        out_path = os.path.join(output_dir, f"{base_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2)
        return out_path

    elif output_format == "docx":
        out_path = os.path.join(output_dir, f"{base_name}.docx")
        export_docx(md_content, "general", framework, out_path)
        return out_path

    elif output_format == "html":
        out_path = os.path.join(output_dir, f"{base_name}.html")
        export_report(md_content, "general", framework, "html", output_dir)
        return out_path

    else:
        out_path = os.path.join(output_dir, f"{base_name}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        return out_path
