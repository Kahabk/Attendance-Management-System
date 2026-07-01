from __future__ import annotations

import streamlit as st

from config import settings
from database import init_database
from pages import Admin, Attendance, Dashboard, Employees, Login, RegisterFace, Reports, Settings as SettingsPage
from services.auth_service import auth_service
from utils.session import clear_token, get_token, set_token
from utils.ui import inject_global_styles


st.set_page_config(page_title=settings.app_name, page_icon=":busts_in_silhouette:", layout="wide")
inject_global_styles()
init_database()


def _load_user() -> dict[str, object] | None:
    token = get_token()
    if not token:
        return None
    user = auth_service.validate_token(token)
    if user is None:
        clear_token()
        return None
    return {
        "id": user.id,
        "username": user.username,
        "role": user.role,
        "full_name": user.full_name or user.username,
        "email": user.email,
    }


current_user = st.session_state.get("current_user") or _load_user()
if current_user is None:
    Login.render()
    st.stop()

st.session_state["current_user"] = current_user

with st.sidebar:
    st.markdown(f"### {settings.app_name}")
    st.caption(f"Signed in as {current_user['full_name']} ({current_user['role']})")
    page_options = ["Dashboard", "Attendance", "Reports", "Settings"]
    if current_user["role"] == "Admin":
        page_options = ["Dashboard", "Employees", "Register Face", "Attendance", "Reports", "Settings", "Admin"]
    page = st.radio("Navigation", page_options, index=0)
    if st.button("Logout", width="stretch"):
        token = get_token()
        if token:
            auth_service.logout(token)
        clear_token()
        st.success("Logged out")
        st.rerun()


if page == "Dashboard":
    Dashboard.render(current_user)
elif page == "Employees":
    Employees.render(current_user)
elif page == "Register Face":
    RegisterFace.render(current_user)
elif page == "Attendance":
    Attendance.render(current_user)
elif page == "Reports":
    Reports.render(current_user)
elif page == "Settings":
    SettingsPage.render(current_user)
elif page == "Admin":
    Admin.render(current_user)
