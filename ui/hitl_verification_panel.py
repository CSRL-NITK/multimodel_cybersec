"""
HITL Verification Panel — Human Auditor Control Review Cockpit
--------------------------------------------------------------
Interactive Streamlit interface for reviewing, confirming, disputing, and
re-examining AI compliance assessment findings before final report export.

"Verify the Data, Not the File" architecture.
"""

import os
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
from utils.report_exporter import export_all_formats


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
        st.warning("No control findings available to verify.")
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
    
    # Calculate projected S2Score
    compliant_proj = sum(1 for cid, v in user_verifs.items() if v["corrected_status"].lower() == "compliant")
    partial_proj = sum(0.5 for cid, v in user_verifs.items() if "partial" in v["corrected_status"].lower())
    projected_s2score = int(300 + ((compliant_proj + partial_proj) / max(1, total_controls)) * 550)
    score_delta = projected_s2score - session.get("draft_s2score", 300)

    # -------------------------------------------------------------------------
    # 1. Header & Live Scorecard Bar
    # -------------------------------------------------------------------------
    st.markdown("### 🛡️ Human Auditor Verification & Sign-Off Cockpit")
    st.caption("Verify each AI-evaluated control. Approve, dispute, or modify findings. Re-examine to generate the certified audit report.")

    bar_col1, bar_col2, bar_col3, bar_col4 = st.columns(4)
    with bar_col1:
        st.metric("Review Progress", f"{reviewed_count} / {total_controls}", f"{int((reviewed_count/total_controls)*100)}% Reviewed")
    with bar_col2:
        st.metric("Draft S2Score", f"{session.get('draft_s2score')}/850", "Initial AI")
    with bar_col3:
        st.metric("Projected Certified Score", f"{projected_s2score}/850", f"{score_delta:+d} pts" if score_delta != 0 else "No Change")
    with bar_col4:
        st.metric("Human Overrides", f"{overridden_count}", f"{int((overridden_count/max(1, reviewed_count))*100)}% of Reviewed" if reviewed_count else "0%")

    st.progress(reviewed_count / total_controls)

    # -------------------------------------------------------------------------
    # 2. Filter Toolbar
    # -------------------------------------------------------------------------
    filt_col1, filt_col2 = st.columns([3, 1])
    with filt_col1:
        filter_opt = st.radio(
            "Filter Controls View",
            options=["All", "🔴 Non-Compliant", "🟡 Partial", "⏳ Pending Review", "✏️ Overridden"],
            horizontal=True,
            label_visibility="collapsed"
        )
    with filt_col2:
        if st.button("⚡ Approve All High-Confidence", help="Approve all remaining controls with confidence > 80%"):
            for f in findings:
                cid = f["control_id"]
                if user_verifs[cid]["verdict"] == "unreviewed" and f["ai_output"].get("confidence", 0) >= 0.80:
                    user_verifs[cid]["verdict"] = "approve"
                    user_verifs[cid]["corrected_status"] = f["ai_output"]["status"]
            st.rerun()

    # -------------------------------------------------------------------------
    # 3. Control Card Deck
    # -------------------------------------------------------------------------
    for idx, f in enumerate(findings):
        cid = f["control_id"]
        title = f["title"]
        ai_out = f["ai_output"]
        ai_stat = ai_out["status"]
        current_v = user_verifs[cid]

        # Filter check
        if filter_opt == "🔴 Non-Compliant" and ai_stat != "Non-Compliant": continue
        if filter_opt == "🟡 Partial" and "partial" not in ai_stat.lower(): continue
        if filter_opt == "⏳ Pending Review" and current_v["verdict"] != "unreviewed": continue
        if filter_opt == "✏️ Overridden" and current_v["verdict"] not in ["reject", "modify"]: continue

        # Card container with dynamic styling
        border_color = "#22c55e" if current_v["verdict"] == "approve" else ("#f59e0b" if current_v["verdict"] in ["reject", "modify"] else "#334155")
        status_color = "#16a34a" if ai_stat == "Compliant" else ("#ca8a04" if "partial" in ai_stat.lower() else "#dc2626")

        with st.expander(f"[{cid}] {title} — Current: {current_v['corrected_status']}", expanded=(current_v["verdict"] == "unreviewed" and idx < 3)):
            st.markdown(f"""
            <div style="padding: 10px; background: rgba(30, 41, 59, 0.4); border-radius: 6px; border-left: 4px solid {status_color}; margin-bottom: 10px;">
                <span style="font-weight: 600; color: #94a3b8;">AI Verdict:</span> <b style="color: {status_color};">{ai_stat}</b> &nbsp;|&nbsp; 
                <span style="font-weight: 600; color: #94a3b8;">Confidence:</span> <b>{int(ai_out.get('confidence', 0.85)*100)}%</b>
                <p style="margin: 6px 0 0 0; color: #cbd5e1; font-size: 0.9rem;"><b>AI Narrative:</b> {ai_out.get('narrative')}</p>
                <p style="margin: 4px 0 0 0; color: #64748b; font-size: 0.82rem;"><b>Evidence Cited:</b> <code>{ai_out.get('evidence', ['None'])}</code></p>
            </div>
            """, unsafe_allow_html=True)

            c_col1, c_col2 = st.columns([1, 2])

            with c_col1:
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
                    sub_a, sub_b = st.columns([1, 2])
                    with sub_a:
                        new_stat = st.selectbox(
                            "Correct Verdict",
                            options=["Compliant", "Partially Compliant", "Non-Compliant", "Risk Accepted", "Not Applicable"],
                            index=0 if ai_stat != "Compliant" else 2,
                            key=f"stat_sel_{session_id}_{cid}"
                        )
                        current_v["corrected_status"] = new_stat
                    with sub_b:
                        auditor_note = st.text_input(
                            "Auditor Rationale / Evidence Guidance",
                            value=current_v.get("comment", ""),
                            placeholder="e.g. Offloaded to Okta SSO at API Gateway layer",
                            key=f"note_in_{session_id}_{cid}"
                        )
                        current_v["comment"] = auditor_note

    # -------------------------------------------------------------------------
    # 4. Action Bar & Final Re-Examination
    # -------------------------------------------------------------------------
    st.markdown("---")
    act_col1, act_col2 = st.columns([2, 1])

    with act_col1:
        st.markdown(f"**Session State:** `{session.get('review_status')}` &nbsp;|&nbsp; **Last Updated:** `{session.get('last_updated_at', '')[:19]}`")
        if session.get("certified_by"):
            st.success(f"🛡️ Certified by `{session.get('certified_by')}` at `{session.get('certified_at', '')[:19]}` (S2Score: **{session.get('certified_s2score')}/850**)")

    with act_col2:
        if st.button("🚀 Re-Examine & Generate Certified Report", type="primary", use_container_width=True):
            with st.spinner("Executing LLM self-healing re-examination and recalculating S2Score..."):
                reconciled_session = reconcile_and_recalculate_session(
                    session_id=session_id,
                    verifications=user_verifs,
                    reviewer_id="auditor@enterprise.com",
                    auto_regenerate=True
                )
                st.toast("✅ Re-examination complete! Certified report generated.", icon="🎉")
                st.rerun()


def render_dual_export_buttons(session_id: str):
    """
    Renders Option A (Raw AI Draft) and Option B (Certified Verified Report) downloads.
    """
    session = load_hitl_review_session(session_id)
    if not session:
        return

    st.markdown("#### 📥 Dual-Mode Report Downloads")
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("##### 📄 Option A: Raw AI Draft Report")
        st.caption("Instant export of unverified AI findings.")
        draft_btn = st.button("Export Raw AI Draft (.docx / .md)", key=f"draft_exp_{session_id}")
        if draft_btn:
            st.info(f"Raw AI Draft ready for download (Draft S2Score: {session.get('draft_s2score')}/850)")

    with exp_col2:
        st.markdown("##### 🛡️ Option B: Certified Verified Report")
        st.caption("Final report with human attestation and changelog.")
        if session.get("review_status") == "HUMAN_CERTIFIED":
            cert_btn = st.button("Export Certified Audit Report (.docx / .md)", type="primary", key=f"cert_exp_{session_id}")
            if cert_btn:
                st.success(f"Certified Report ready (Certified S2Score: {session.get('certified_s2score')}/850)")
        else:
            st.button("Export Certified Audit Report (Review Required)", disabled=True, key=f"cert_dis_{session_id}")
            st.caption("⚠️ Complete control review to unlock Certified export.")
