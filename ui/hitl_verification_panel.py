"""
HITL Verification Panel — Human Auditor Control Review Cockpit
--------------------------------------------------------------
Interactive Streamlit interface for reviewing, confirming, disputing, and
re-examining AI compliance assessment findings across all previously generated reports.

"Verify the Data, Not the File" architecture.
"""

import os
import glob
import json
import re
import streamlit as st
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from governance.hitl_schema import (
    create_hitl_review_session,
    load_hitl_review_session,
    save_hitl_review_session,
    list_hitl_review_sessions
)
from governance.hitl_regenerator import reconcile_and_recalculate_session
from utils.hitl_report_formatter import format_hitl_session_to_markdown, export_hitl_session

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSESSMENTS_DIR = os.path.join(PROJECT_ROOT, "assessments")
HUMAN_SIGNOFFS_DIR = os.path.join(PROJECT_ROOT, "governance", "human_sign_offs")


def discover_all_audit_reports() -> List[Dict[str, Any]]:
    """
    Discovers all previously generated compliance audit reports and assessment files.
    Ensures a corresponding HITL review session exists for each.
    """
    discovered = []

    # 1. Check existing HITL review sessions
    existing_sessions = list_hitl_review_sessions()
    existing_ids = {s.get("assessment_id") for s in existing_sessions}
    for s in existing_sessions:
        fw = s.get("framework", "Unknown").upper()
        sid = s.get("assessment_id", "")
        status = s.get("review_status", "DRAFT")
        cnt = s.get("total_controls", 0)
        s2 = s.get("certified_s2score" if status == "HUMAN_CERTIFIED" else "draft_s2score", 300)
        status_icon = "🛡️ Certified" if status == "HUMAN_CERTIFIED" else ("✏️ In Review" if status == "IN_REVIEW" else "📄 Draft")
        
        discovered.append({
            "session_id": sid,
            "display_name": f"{status_icon} [{fw}] {s.get('client_id', 'Audit')} ({cnt} controls, S2: {s2}/850) — ID: {sid[:24]}",
            "framework": fw,
            "total_controls": cnt,
            "status": status,
            "source": "Saved HITL Session"
        })

    # 2. Check assessments/ directory and auto-initialize review sessions if not present
    if os.path.exists(ASSESSMENTS_DIR):
        for p in sorted(glob.glob(os.path.join(ASSESSMENTS_DIR, "*.json"))):
            fname = os.path.basename(p)
            clean_fw_name = fname.replace("_assessment.json", "").replace("__", "/").upper()
            sid = f"assessment_{fname.replace('.json', '')}"
            
            if sid not in existing_ids:
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        findings = json.load(f)
                    if isinstance(findings, list) and len(findings) > 0:
                        sess = create_hitl_review_session(
                            findings=findings,
                            framework=clean_fw_name,
                            client_id="Local Vault Assessment",
                            assessment_id=sid,
                            source_type="Core Multi-Model Assessment Suite"
                        )
                        existing_ids.add(sid)
                        discovered.append({
                            "session_id": sid,
                            "display_name": f"📄 Draft [{clean_fw_name}] ({len(findings)} controls) — {fname}",
                            "framework": clean_fw_name,
                            "total_controls": len(findings),
                            "status": "DRAFT",
                            "source": "Pre-Generated Assessment"
                        })
                except Exception:
                    pass

    return discovered


def save_user_verifications_to_disk(session_id: str, user_verifs: Dict[str, Any]):
    """Persists current in-memory edits to the session JSON file."""
    session = load_hitl_review_session(session_id)
    if not session:
        return
    
    for f in session.get("findings", []):
        cid = f.get("control_id")
        if cid in user_verifs:
            uv = user_verifs[cid]
            f["human_review"]["verdict"] = uv["verdict"]
            f["human_review"]["reviewer_comment"] = uv.get("comment", "")
            f["human_review"]["corrected_status"] = uv["corrected_status"]
            f["final_effective"]["status"] = uv["corrected_status"]
            if uv.get("comment"):
                f["final_effective"]["narrative"] = f"Auditor Notes: {uv['comment']}"
            if uv["verdict"] in ["approve", "reject", "modify"]:
                f["human_review"]["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    if session.get("review_status") == "DRAFT":
        session["review_status"] = "IN_REVIEW"
        
    save_hitl_review_session(session)


def render_hitl_audit_reports_view(default_session_id: Optional[str] = None):
    """
    Main View: Dropdown selector of all previously generated reports + Full Control Review Cockpit.
    """
    st.markdown("""
    <div style="margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h2 style="margin: 0; color: #f8fafc; font-family: 'Outfit', sans-serif;">🛡️ Compliance Audit Reports & Control Review Cockpit</h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.9rem;">
                Examine generated audit reports, give comments for each mapped control, override verdicts, and issue certified reports.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_nav1, col_nav2 = st.columns([1, 4])
    with col_nav1:
        if st.button("💬 Return to Chat / Audit", width="stretch", type="secondary"):
            st.session_state.active_view = "audit"
            st.rerun()

    # Discover all available reports
    all_reports = discover_all_audit_reports()
    if not all_reports:
        st.warning("No generated audit reports or framework assessments found on disk.")
        return

    report_map = {r["display_name"]: r["session_id"] for r in all_reports}
    display_options = list(report_map.keys())

    # Determine default index
    default_idx = 0
    if default_session_id:
        for idx, r in enumerate(all_reports):
            if r["session_id"] == default_session_id:
                default_idx = idx
                break
    elif "hitl_selected_session_id" in st.session_state:
        saved_id = st.session_state["hitl_selected_session_id"]
        for idx, r in enumerate(all_reports):
            if r["session_id"] == saved_id:
                default_idx = idx
                break

    selected_display = st.selectbox(
        "Select an Audit Report to Examine & Review Controls:",
        options=display_options,
        index=default_idx,
        key="hitl_report_selector"
    )

    selected_sid = report_map[selected_display]
    st.session_state["hitl_selected_session_id"] = selected_sid

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Render Cockpit for chosen report
    render_hitl_verification_cockpit(selected_sid)


def render_hitl_verification_cockpit(session_id: str):
    """
    Renders the complete interactive control-by-control verification interface.
    """
    session = load_hitl_review_session(session_id)
    if not session:
        st.error(f"Audit Review Session `{session_id}` not found.")
        return

    findings = session.get("findings", [])
    total_controls = len(findings)
    if total_controls == 0:
        st.warning("No control findings available to verify in this report.")
        return

    # Initialize local UI state for this session
    state_key = f"hitl_state_{session_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            f["control_id"]: {
                "verdict": f["human_review"].get("verdict", "unreviewed"),
                "comment": f["human_review"].get("reviewer_comment", ""),
                "corrected_status": f["final_effective"].get("status", f["ai_output"]["status"])
            }
            for f in findings
        }

    user_verifs = st.session_state[state_key]

    # Calculate real-time stats for the cockpit bar
    reviewed_count = sum(1 for v in user_verifs.values() if v["verdict"] in ["approve", "reject", "modify"])
    overridden_count = sum(1 for v in user_verifs.values() if v["verdict"] in ["reject", "modify"])
    commented_count = sum(1 for v in user_verifs.values() if bool(v.get("comment", "").strip()))
    
    # Calculate projected S2Score
    compliant_proj = sum(1 for cid, v in user_verifs.items() if v["corrected_status"].lower() == "compliant")
    partial_proj = sum(0.5 for cid, v in user_verifs.items() if "partial" in v["corrected_status"].lower())
    projected_s2score = int(300 + ((compliant_proj + partial_proj) / max(1, total_controls)) * 550)
    score_delta = projected_s2score - session.get("draft_s2score", 300)

    # -------------------------------------------------------------------------
    # 1. Header & Live Scorecard Bar
    # -------------------------------------------------------------------------
    fw_name = session.get("framework", "").upper()
    client_name = session.get("client_id", "Default Target")
    rev_status = session.get("review_status", "DRAFT")

    status_badge = "🛡️ HUMAN CERTIFIED" if rev_status == "HUMAN_CERTIFIED" else ("✏️ IN REVIEW" if rev_status == "IN_REVIEW" else "📄 PRELIMINARY DRAFT")

    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 14px 18px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 0.8rem; color: #38bdf8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">FRAMEWORK BENCHMARK</span>
                <h3 style="margin: 2px 0 0 0; color: #f8fafc; font-size: 1.3rem;">{fw_name} — {client_name}</h3>
            </div>
            <div style="text-align: right;">
                <span style="background: rgba(30, 41, 59, 0.8); border: 1px solid #475569; padding: 5px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; color: #38bdf8;">
                    {status_badge}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    bar_col1, bar_col2, bar_col3, bar_col4, bar_col5 = st.columns(5)
    with bar_col1:
        st.metric("Review Progress", f"{reviewed_count} / {total_controls}", f"{int((reviewed_count/total_controls)*100)}% Reviewed")
    with bar_col2:
        st.metric("Auditor Comments", f"{commented_count}", f"{int((commented_count/max(1, reviewed_count))*100)}% of Reviewed" if reviewed_count else "0%")
    with bar_col3:
        st.metric("Draft S2Score", f"{session.get('draft_s2score', 300)}/850", "Initial AI")
    with bar_col4:
        st.metric("Projected Score", f"{projected_s2score}/850", f"{score_delta:+d} pts" if score_delta != 0 else "No Change")
    with bar_col5:
        st.metric("Overrides", f"{overridden_count}", f"{int((overridden_count/max(1, reviewed_count))*100)}% of Reviewed" if reviewed_count else "0%")

    st.progress(reviewed_count / total_controls)

    # -------------------------------------------------------------------------
    # 2. Controls Toolbar & Filter
    # -------------------------------------------------------------------------
    col_tools1, col_tools2, col_tools3 = st.columns([2, 2, 1])

    with col_tools1:
        search_query = st.text_input(
            "🔍 Search Controls (by ID, Title, or Keyword)",
            placeholder="e.g. PR.AA-01, password, encryption, access...",
            key=f"search_in_{session_id}"
        ).strip().lower()

    with col_tools2:
        filter_opt = st.radio(
            "Filter Controls View",
            options=["All", "🔴 Non-Compliant", "🟡 Partial", "⏳ Pending Review", "✏️ Overridden", "💬 With Comments"],
            horizontal=True,
            key=f"filt_opt_{session_id}"
        )

    with col_tools3:
        st.write("")
        st.write("")
        if st.button("⚡ Approve All Confident", help="Approve all unreviewed controls with confidence >= 80%"):
            for f in findings:
                cid = f["control_id"]
                if user_verifs[cid]["verdict"] == "unreviewed" and f["ai_output"].get("confidence", 0) >= 0.80:
                    user_verifs[cid]["verdict"] = "approve"
                    user_verifs[cid]["corrected_status"] = f["ai_output"]["status"]
            save_user_verifications_to_disk(session_id, user_verifs)
            st.toast("Approved all remaining high-confidence controls!")
            st.rerun()

    # Apply search and category filter
    filtered_indices = []
    for idx, f in enumerate(findings):
        cid = f["control_id"].lower()
        title = f["title"].lower()
        ai_stat = f["ai_output"]["status"].lower()
        current_v = user_verifs[f["control_id"]]
        comment = current_v.get("comment", "").lower()

        # Search matching
        if search_query and not (search_query in cid or search_query in title or search_query in f["ai_output"].get("narrative", "").lower() or search_query in comment):
            continue

        # Filter option
        if filter_opt == "🔴 Non-Compliant" and not ("non-compliant" in ai_stat or "no evidence" in ai_stat):
            continue
        if filter_opt == "🟡 Partial" and "partial" not in ai_stat:
            continue
        if filter_opt == "⏳ Pending Review" and current_v["verdict"] != "unreviewed":
            continue
        if filter_opt == "✏️ Overridden" and current_v["verdict"] not in ["reject", "modify"]:
            continue
        if filter_opt == "💬 With Comments" and not current_v.get("comment", "").strip():
            continue

        filtered_indices.append(idx)

    st.markdown(f"<span style='color: #94a3b8; font-size: 0.85rem;'>Showing <b>{len(filtered_indices)}</b> of <b>{total_controls}</b> controls</span>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Pagination Controls for fast and responsive rendering
    # -------------------------------------------------------------------------
    PAGE_SIZE = 15
    total_pages = max(1, (len(filtered_indices) + PAGE_SIZE - 1) // PAGE_SIZE)
    page_key = f"page_num_{session_id}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    curr_page = st.session_state[page_key]
    if curr_page > total_pages:
        curr_page = total_pages
        st.session_state[page_key] = total_pages

    if total_pages > 1:
        p_col1, p_col2, p_col3 = st.columns([1, 3, 1])
        with p_col1:
            if st.button("⬅️ Previous", disabled=(curr_page <= 1), key=f"prev_btn_{session_id}"):
                st.session_state[page_key] = max(1, curr_page - 1)
                st.rerun()
        with p_col2:
            st.markdown(f"<p style='text-align: center; color: #94a3b8; margin-top: 6px; font-size: 0.9rem;'>Page <b>{curr_page}</b> of <b>{total_pages}</b> ({len(filtered_indices)} controls matched)</p>", unsafe_allow_html=True)
        with p_col3:
            if st.button("Next ➡️", disabled=(curr_page >= total_pages), key=f"next_btn_{session_id}"):
                st.session_state[page_key] = min(total_pages, curr_page + 1)
                st.rerun()

    start_idx = (curr_page - 1) * PAGE_SIZE
    end_idx = start_idx + PAGE_SIZE
    active_indices = filtered_indices[start_idx:end_idx]

    # -------------------------------------------------------------------------
    # 4. Control Card Deck with Individual Comment Boxes
    # -------------------------------------------------------------------------
    for rank_idx, idx in enumerate(active_indices):
        f = findings[idx]
        cid = f["control_id"]
        title = f["title"]
        ai_out = f["ai_output"]
        ai_stat = ai_out["status"]
        current_v = user_verifs[cid]

        # Dynamic styling based on auditor state
        status_color = "#22c55e" if "compliant" in ai_stat.lower() and "non" not in ai_stat.lower() and "partial" not in ai_stat.lower() else ("#eab308" if "partial" in ai_stat.lower() else "#ef4444")
        verdict_badge = "✅ Approved" if current_v["verdict"] == "approve" else ("✏️ Overridden / Disputed" if current_v["verdict"] in ["reject", "modify"] else "⏳ Unreviewed")

        comment_indicator = f" &nbsp;•&nbsp; 💬 <i>'{current_v['comment'][:30]}...'</i>" if current_v.get("comment", "").strip() else ""

        with st.expander(f"[{cid}] {title} — Effective: {current_v['corrected_status']} [{verdict_badge}]", expanded=(current_v["verdict"] == "unreviewed" and rank_idx < 2)):
            st.markdown(f"""
            <div style="padding: 10px 14px; background: rgba(30, 41, 59, 0.45); border-radius: 8px; border-left: 4px solid {status_color}; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <span style="font-weight: 600; color: #94a3b8;">AI Assessment:</span> 
                        <b style="color: {status_color}; font-size: 1.05rem;">{ai_stat}</b> &nbsp;|&nbsp; 
                        <span style="font-weight: 600; color: #94a3b8;">Confidence:</span> <b>{int(ai_out.get('confidence', 0.85)*100)}%</b>
                    </div>
                    <div>
                        <span style="font-size: 0.8rem; color: #94a3b8;">Auditor Status: <b>{verdict_badge}</b>{comment_indicator}</span>
                    </div>
                </div>
                <p style="margin: 8px 0 4px 0; color: #cbd5e1; font-size: 0.92rem; line-height: 1.5;"><b>AI Finding / Rationale:</b> {ai_out.get('narrative')}</p>
                <p style="margin: 4px 0 0 0; color: #64748b; font-size: 0.82rem;"><b>Evidence Cited:</b> <code>{ai_out.get('evidence', ['None'])}</code></p>
            </div>
            """, unsafe_allow_html=True)

            c_col1, c_col2 = st.columns([1, 2])

            with c_col1:
                st.markdown("**Auditor Verdict**")
                verdict_radio = st.radio(
                    f"Auditor Decision for {cid}",
                    options=["✅ Approve (Agree)", "❌ Dispute / Override", "⏳ Unreviewed"],
                    index=0 if current_v["verdict"] == "approve" else (1 if current_v["verdict"] in ["reject", "modify"] else 2),
                    key=f"rad_{session_id}_{cid}",
                    label_visibility="collapsed"
                )

                if "Approve" in verdict_radio:
                    current_v["verdict"] = "approve"
                    current_v["corrected_status"] = ai_stat
                elif "Dispute" in verdict_radio:
                    current_v["verdict"] = "modify"
                else:
                    current_v["verdict"] = "unreviewed"

            with c_col2:
                if current_v["verdict"] in ["reject", "modify"]:
                    st.markdown("**Auditor Correction & Comments**")
                    sub_a, sub_b = st.columns([1, 2])
                    with sub_a:
                        new_stat = st.selectbox(
                            "Corrected Status",
                            options=["Compliant", "Partially Compliant", "Non-Compliant", "Risk Accepted", "Not Applicable"],
                            index=0 if "compliant" not in ai_stat.lower() else 2,
                            key=f"stat_sel_{session_id}_{cid}"
                        )
                        current_v["corrected_status"] = new_stat
                    with sub_b:
                        auditor_note = st.text_input(
                            "Auditor Comments / Evidence Guidance",
                            value=current_v.get("comment", ""),
                            placeholder="e.g. Mitigated via Okta SSO & TLS 1.3 reverse proxy",
                            key=f"note_in_{session_id}_{cid}"
                        )
                        current_v["comment"] = auditor_note
                else:
                    st.markdown("**Auditor Comments (Optional)**")
                    auditor_note = st.text_input(
                        "Leave a comment or evidence note for this control:",
                        value=current_v.get("comment", ""),
                        placeholder="e.g. Verified against production configuration on staging",
                        key=f"note_in_appr_{session_id}_{cid}"
                    )
                    current_v["comment"] = auditor_note

    # -------------------------------------------------------------------------
    # 5. Bottom Actions: Save Progress, Re-Examine & Exports
    # -------------------------------------------------------------------------
    st.markdown("---")
    act_col1, act_col2, act_col3 = st.columns([2, 1, 1])

    with act_col1:
        st.markdown(f"**Session ID:** `{session.get('assessment_id')}` &nbsp;|&nbsp; **Review Status:** `{session.get('review_status')}`")
        if session.get("certified_by"):
            st.success(f"🛡️ Certified by `{session.get('certified_by')}` (Certified S2Score: **{session.get('certified_s2score')}/850**)")

    with act_col2:
        if st.button("💾 Save Comments & Progress", width="stretch", type="secondary", key=f"save_prog_btn_{session_id}"):
            save_user_verifications_to_disk(session_id, user_verifs)
            st.toast("Auditor comments and verdicts saved to disk!", icon="💾")

    with act_col3:
        if st.button("🚀 Re-Examine & Certify", type="primary", width="stretch", key=f"cert_act_btn_{session_id}"):
            with st.spinner("Reconciling auditor feedback, recalculating S2Score, and certifying report..."):
                save_user_verifications_to_disk(session_id, user_verifs)
                reconciled_session = reconcile_and_recalculate_session(
                    session_id=session_id,
                    verifications=user_verifs,
                    reviewer_id="certifying_auditor@enterprise.com",
                    auto_regenerate=True
                )
                st.toast("✅ Re-examination complete! Certified audit report generated.", icon="🎉")
                st.rerun()

    # Dual Export Section
    st.markdown("#### 📥 Export Audit Report")
    render_dual_export_buttons(session_id)


def render_dual_export_buttons(session_id: str):
    """
    Renders Option A (Raw AI Draft) and Option B (Certified Verified Report) downloads.
    """
    session = load_hitl_review_session(session_id)
    if not session:
        return

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("##### 📄 Option A: Raw AI Draft Report")
        st.caption("Export preliminary assessment findings (Watermarked as Draft).")
        try:
            draft_md = format_hitl_session_to_markdown(session, certified_mode=False)
            dl_prefix = re.sub(r'[^a-zA-Z0-9_-]', '_', f"Draft_Report_{session.get('framework')}_{session_id[:16]}")
            st.download_button(
                "📥 Download AI Draft (.md)",
                data=draft_md.encode("utf-8"),
                file_name=f"{dl_prefix}.md",
                mime="text/markdown",
                key=f"dl_draft_md_{session_id}",
                use_container_width=True
            )
        except Exception as exc:
            st.caption(f"Export generator notice: {exc}")

    with exp_col2:
        st.markdown("##### 🛡️ Option B: Certified Verified Report")
        st.caption("Final report with human attestation, score delta, and auditor comments.")
        if session.get("review_status") == "HUMAN_CERTIFIED":
            try:
                cert_md = format_hitl_session_to_markdown(session, certified_mode=True)
                dl_prefix = re.sub(r'[^a-zA-Z0-9_-]', '_', f"Certified_Audit_Report_{session.get('framework')}_{session_id[:16]}")
                st.download_button(
                    "📥 Download Certified Report (.md)",
                    data=cert_md.encode("utf-8"),
                    file_name=f"{dl_prefix}.md",
                    mime="text/markdown",
                    type="primary",
                    key=f"dl_cert_md_{session_id}",
                    use_container_width=True
                )
            except Exception as exc:
                st.caption(f"Export generator notice: {exc}")
        else:
            st.button("📥 Download Certified Report (Review & Certify First)", disabled=True, key=f"cert_dis_{session_id}", use_container_width=True)
            st.caption("⚠️ Click **'🚀 Re-Examine & Certify'** above to unlock the Certified publication report.")
