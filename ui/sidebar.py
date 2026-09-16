"""
UI Component: Sidebar Navigation & History Controls
--------------------------------------------------
Renders clean sidebar navigation and history controls without emojis.
"""

import os
from datetime import datetime
import streamlit as st
import conversation_memory_manager as cmm
from core.session_state import auto_save_current_session, format_relative_time

def render_sidebar_recents(neo4j_active: bool, neo4j_utils, on_db_engine_change_fn, user_role: str = "guest"):
    """Renders clean sidebar navigation and role-gated history controls."""
    st.markdown(
        """
        <div style="margin-bottom: 8px;">
            <h3 style="margin: 0; font-size: 1.15rem; color: #f8fafc; font-family: 'Outfit', sans-serif;">Navigation</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. New Chat & View Switcher Buttons
    curr_user = st.session_state.get("username", "guest")
    
    if user_role in ("registered", "admin"):
        col_sb1, col_sb2 = st.columns(2)
        with col_sb1:
            if st.button("**+ New Chat**", width="stretch", type="primary", key="sb_new_chat_btn"):
                auto_save_current_session()
                new_id = f"Audit_Run_{datetime.now().strftime('%b%d_%H%M%S')}"
                st.session_state.current_session_id = new_id
                st.session_state.messages = []
                st.session_state.active_view = "audit"
                cmm.save_session(new_id, [], username=curr_user)
                st.rerun()
        with col_sb2:
            curr_v = st.session_state.get("active_view", "audit")
            v_btn_txt = "**Audit View**" if curr_v == "chats_dashboard" else "**Chats List**"
            if st.button(v_btn_txt, width="stretch", key="sb_view_switch_btn"):
                st.session_state.active_view = "audit" if curr_v == "chats_dashboard" else "chats_dashboard"
                st.rerun()
        
        curr_v = st.session_state.get("active_view", "audit")
        cockpit_btn_txt = "💬 **Return to Audit Chat**" if curr_v == "hitl_cockpit" else "🛡️ **Auditor Control Cockpit**"
        if st.button(cockpit_btn_txt, width="stretch", type="secondary" if curr_v == "hitl_cockpit" else "primary", key="sb_hitl_cockpit_btn"):
            st.session_state.active_view = "audit" if curr_v == "hitl_cockpit" else "hitl_cockpit"
            st.rerun()
    else:
        # Guest Mode Prominent Action Buttons
        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            if st.button("**+ New Audit**", width="stretch", type="primary", key="sb_guest_new_chat_btn"):
                auto_save_current_session()
                new_id = f"Audit_Run_{datetime.now().strftime('%b%d_%H%M%S')}"
                st.session_state.current_session_id = new_id
                st.session_state.messages = []
                st.session_state.active_view = "audit"
                st.rerun()
        with col_g2:
            curr_v = st.session_state.get("active_view", "audit")
            cockpit_btn_txt = "💬 **Chat**" if curr_v == "hitl_cockpit" else "🛡️ **Cockpit**"
            if st.button(cockpit_btn_txt, width="stretch", key="sb_guest_hitl_btn"):
                st.session_state.active_view = "audit" if curr_v == "hitl_cockpit" else "hitl_cockpit"
                st.rerun()

    # Guest users get compact glass notification card
    if user_role == "guest":
        st.markdown(
            """
            <div style="background: rgba(13, 18, 34, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 12px; margin: 12px 0; font-size: 0.82rem; color: #94a3b8; line-height: 1.45;">
                <span style="color: #fbbf24; font-weight: 600;">Guest Mode:</span> Guest chat sessions automatically expire and are purged after a 24-hour retention window. Sign in to permanently save and pin your audits.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.divider()
        return

    saved_chats = cmm.list_sessions(username=curr_user)
    curr_id = st.session_state.get("current_session_id", "Default_Session")

    pinned_chats = [s for s in saved_chats if s.get("is_pinned")]
    recent_chats = [s for s in saved_chats if not s.get("is_pinned")]

    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

    # 2. Audit History & Recents Expander
    with st.expander("**Audit History & Recents**", expanded=True):
        if not recent_chats and not pinned_chats:
            st.caption("No saved audit history for this account.")
        else:
            if pinned_chats:
                st.markdown("<p style='font-size: 0.78rem; font-weight: 700; color: #fbbf24; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;'>Pinned Audits</p>", unsafe_allow_html=True)
                for s in pinned_chats:
                    sid = s["session_id"]
                    sname = s["session_name"]
                    is_active = (sid == curr_id)
                    btn_type = "primary" if is_active else "secondary"
                    rel_time = format_relative_time(s.get("saved_at", ""))
                    display_title = (sname[:18] + "...") if len(sname) > 18 else sname
                    
                    if st.button(f"[Pinned] {display_title}", key=f"sb_btn_{sid}", width="stretch", type=btn_type, help=f"{sname} ({rel_time})"):
                        auto_save_current_session()
                        st.session_state.current_session_id = sid
                        st.session_state.messages = cmm.load_session(sid)
                        st.session_state.active_view = "audit"
                        st.rerun()

            if recent_chats:
                st.markdown("<p style='font-size: 0.78rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 10px; margin-bottom: 6px;'>Recent Audits</p>", unsafe_allow_html=True)
                for s in recent_chats[:10]:
                    sid = s["session_id"]
                    sname = s["session_name"]
                    is_active = (sid == curr_id)
                    btn_type = "primary" if is_active else "secondary"
                    rel_time = format_relative_time(s.get("saved_at", ""))
                    display_title = (sname[:20] + "...") if len(sname) > 20 else sname
                    
                    if st.button(f"{display_title}", key=f"sb_btn_{sid}", width="stretch", type=btn_type, help=f"{sname} ({rel_time})"):
                        auto_save_current_session()
                        st.session_state.current_session_id = sid
                        st.session_state.messages = cmm.load_session(sid)
                        st.session_state.active_view = "audit"
                        st.rerun()

    # 3. Actions Popover for Active Chat (Pin, Rename, Delete)
    active_session_obj = next((s for s in saved_chats if s["session_id"] == curr_id), None)
    active_title = active_session_obj["session_name"] if active_session_obj else "Active Audit Session"
    active_is_pinned = active_session_obj.get("is_pinned", False) if active_session_obj else False

    with st.popover("**Manage Active Chat**", width="stretch"):
        st.markdown(f"**{active_title}**")
        
        new_name_val = st.text_input("Rename Chat", value=active_title, key="rename_chat_input")
        if st.button("Apply Rename", width="stretch", key="popover_rename_btn"):
            if new_name_val.strip():
                cmm.rename_session(curr_id, new_name_val.strip())
                st.success("Renamed!")
                st.rerun()

        col_act1, col_act2 = st.columns(2)
        with col_act1:
            pin_btn_label = "Unpin" if active_is_pinned else "Pin to Top"
            if st.button(pin_btn_label, width="stretch", key="popover_pin_btn"):
                cmm.toggle_pin_session(curr_id)
                st.rerun()

        with col_act2:
            if st.button("Delete Chat", width="stretch", type="primary", key="popover_del_btn"):
                cmm.delete_session(curr_id)
                st.session_state.messages = []
                st.session_state.current_session_id = f"Audit_Run_{datetime.now().strftime('%b%d_%H%M%S')}"
                st.rerun()

    st.divider()
