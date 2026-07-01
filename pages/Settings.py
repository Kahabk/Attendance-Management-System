from __future__ import annotations

from datetime import time

import streamlit as st

from config import settings
from services.admin_service import admin_service
from utils.ui import section_title


def render(current_user: dict[str, object]) -> None:
    section_title("Settings")

    can_edit = current_user.get("role") == "Admin"
    threshold = float(admin_service.get_setting("recognition_threshold", str(settings.threshold)))
    cooldown = int(admin_service.get_setting("attendance_cooldown_minutes", str(settings.attendance_cooldown_minutes)))
    reentry_gap = int(admin_service.get_setting("attendance_reentry_gap_seconds", str(settings.attendance_reentry_gap_seconds)))
    work_start_time = admin_service.get_setting("work_start_time", settings.work_start_time)
    theme = admin_service.get_setting("theme", settings.theme)
    language = admin_service.get_setting("language", settings.language)
    camera_id = admin_service.get_setting("camera_id", settings.camera_id)

    with st.form("settings_form"):
        new_threshold = st.slider("Recognition Threshold", 0.0, 1.0, value=threshold, step=0.01, disabled=not can_edit)
        new_cooldown = st.number_input("Attendance Cooldown (minutes)", min_value=1, max_value=240, value=cooldown, step=1, disabled=not can_edit)
        new_reentry_gap = st.number_input(
            "Face Re-entry Gap (seconds)",
            min_value=5,
            max_value=600,
            value=reentry_gap,
            step=5,
            help="How long the face must be absent before a later recognition can mark OUT.",
            disabled=not can_edit,
        )
        parsed_hour, parsed_minute = [int(v) for v in work_start_time.split(":", 1)]
        new_work_start = st.time_input("Work Start Time", value=time(parsed_hour, parsed_minute), disabled=not can_edit)
        new_theme = st.selectbox("Theme", ["dark", "light"], index=0 if theme == "dark" else 1, disabled=not can_edit)
        new_language = st.text_input("Language", value=language, disabled=not can_edit)
        new_camera = st.text_input("Default Camera ID", value=camera_id, disabled=not can_edit)
        save = st.form_submit_button("Save Settings", width="stretch", disabled=not can_edit)

    if save and can_edit:
        admin_service.set_setting("recognition_threshold", str(new_threshold))
        admin_service.set_setting("attendance_cooldown_minutes", str(int(new_cooldown)))
        admin_service.set_setting("attendance_reentry_gap_seconds", str(int(new_reentry_gap)))
        admin_service.set_setting("work_start_time", new_work_start.strftime("%H:%M"))
        admin_service.set_setting("theme", new_theme)
        admin_service.set_setting("language", new_language.strip())
        admin_service.set_setting("camera_id", new_camera.strip())
        st.success("Settings saved")
        st.rerun()

    st.divider()
    st.subheader("Database Backup")
    if st.button("Create Backup", disabled=not can_edit, width="stretch"):
        path = admin_service.backup_db()
        st.success(f"Backup created: {path}")
        st.download_button("Download Backup", data=path.read_bytes(), file_name=path.name, mime="application/octet-stream")

    restore_file = st.file_uploader("Restore Database", type=["db", "sqlite"], disabled=not can_edit)
    if restore_file is not None and st.button("Restore Selected Backup", disabled=not can_edit, width="stretch"):
        temp_path = settings.database_dir / f"restore_{restore_file.name}"
        temp_path.write_bytes(restore_file.read())
        admin_service.restore_db(temp_path)
        st.success("Database restored. Please restart the application.")
