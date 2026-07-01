from __future__ import annotations

import streamlit as st

from services.auth_service import auth_service
from utils.session import clear_token, get_token, set_token
from utils.ui import section_title


def render() -> None:
    st.title("Attendance Management System")
    st.caption("Secure login for administrators and employees")

    tab_login, tab_forgot, tab_reset = st.tabs(["Login", "Forgot Password", "Reset Password"])

    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username or Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", width="stretch")
        if submitted:
            user, token = auth_service.authenticate(username=username.strip(), password=password)
            if user and token:
                set_token(token)
                st.session_state["current_user"] = {
                    "id": user.id,
                    "username": user.username,
                    "role": user.role,
                    "full_name": user.full_name or user.username,
                }
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab_forgot:
        with st.form("forgot_form"):
            identity = st.text_input("Username or Email", key="forgot_identity")
            submitted = st.form_submit_button("Generate Reset Token", width="stretch")
        if submitted:
            ok, message, token = auth_service.request_password_reset(identity.strip())
            if ok and token:
                st.success(message)
                st.code(token, language="text")
                st.caption("Use this token on the Reset Password tab.")
            else:
                st.info(message)

    with tab_reset:
        with st.form("reset_form"):
            identity = st.text_input("Username or Email", key="reset_identity")
            token = st.text_input("Reset Token")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            submitted = st.form_submit_button("Reset Password", width="stretch")
        if submitted:
            if new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                ok, message = auth_service.reset_password(identity.strip(), token.strip(), new_password)
                if ok:
                    st.success(message)
                else:
                    st.error(message)

    if get_token():
        st.info("You are already signed in. Use the sidebar navigation.")
