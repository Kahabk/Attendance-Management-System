from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from services.admin_service import admin_service
from services.attendance_service import attendance_service
from services.export_service import export_service
from utils.ui import section_title


def render(current_user: dict[str, object]) -> None:
    if current_user.get("role") != "Admin":
        st.warning("Admin access required.")
        return

    section_title("Admin Panel")
    tab_users, tab_departments, tab_cameras, tab_attendance, tab_data_control, tab_logs = st.tabs(
        ["Users", "Departments", "Cameras", "Attendance", "Data Control", "Logs"]
    )

    with tab_users:
        users = admin_service.list_users()
        st.dataframe(users, hide_index=True, width="stretch")
        col1, col2 = st.columns(2)
        with col1:
            with st.form("create_user_admin"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                role = st.selectbox("Role", ["Admin", "Employee"])
                full_name = st.text_input("Full Name")
                email = st.text_input("Email")
                employee_id = st.number_input("Employee Link (optional)", min_value=0, value=0)
                submit = st.form_submit_button("Create User", width="stretch")
            if submit:
                ok, msg = admin_service.create_user(
                    username=username,
                    password=password,
                    role=role,
                    full_name=full_name or None,
                    email=email or None,
                    employee_id=int(employee_id) or None,
                )
                (st.success if ok else st.error)(msg)
                if ok:
                    st.rerun()
        with col2:
            if not users.empty:
                selected = st.selectbox("Select User", users["id"].tolist(), format_func=lambda x: f'{users.loc[users["id"] == x, "username"].iloc[0]} ({users.loc[users["id"] == x, "role"].iloc[0]})')
                with st.form("update_user_admin"):
                    row = users.loc[users["id"] == selected].iloc[0]
                    upd_role = st.selectbox("Role", ["Admin", "Employee"], index=0 if row["role"] == "Admin" else 1)
                    upd_active = st.checkbox("Active", value=bool(row["is_active"]))
                    upd_password = st.text_input("New Password (optional)", type="password")
                    save = st.form_submit_button("Update User", width="stretch")
                if save:
                    fields = {"role": upd_role, "is_active": upd_active}
                    if upd_password:
                        fields["password"] = upd_password
                    ok, msg = admin_service.update_user(int(selected), **fields)
                    (st.success if ok else st.error)(msg)
                    if ok:
                        st.rerun()
                if st.button("Delete User", width="stretch"):
                    ok, msg = admin_service.delete_user(int(selected))
                    (st.success if ok else st.error)(msg)
                    if ok:
                        st.rerun()

    with tab_departments:
        deps = admin_service.list_departments()
        st.dataframe(deps, hide_index=True, width="stretch")
        with st.form("department_form"):
            name = st.text_input("Department Name")
            description = st.text_area("Description")
            add = st.form_submit_button("Add Department", width="stretch")
        if add:
            ok, msg = admin_service.create_department(name, description or None)
            (st.success if ok else st.error)(msg)
            if ok:
                st.rerun()

    with tab_cameras:
        cams = admin_service.list_cameras()
        st.dataframe(cams, hide_index=True, width="stretch")
        with st.form("camera_form"):
            name = st.text_input("Camera Name")
            source = st.text_input("Source", value="0")
            camera_type = st.selectbox("Camera Type", ["webcam", "usb", "ip", "rtsp"])
            notes = st.text_area("Notes")
            add = st.form_submit_button("Add Camera", width="stretch")
        if add:
            ok, msg = admin_service.create_camera(name, source, camera_type=camera_type, notes=notes or None)
            (st.success if ok else st.error)(msg)
            if ok:
                st.rerun()

    with tab_attendance:
        section_title("Attendance Administration")
        selected_date = st.date_input("Select Attendance Date", value=date.today())
        day_df = attendance_service.attendance_history(start_date=selected_date, end_date=selected_date)
        st.dataframe(day_df, hide_index=True, width="stretch")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Delete Selected Date Attendance", type="primary", width="stretch"):
                deleted = attendance_service.reset_attendance(selected_date, selected_date)
                st.success(f"Deleted {deleted} attendance rows for {selected_date}")
                st.rerun()
        with col2:
            start_date = st.date_input("Reset Start Date", value=date.today().replace(day=1), key="attendance_reset_start")
            end_date = st.date_input("Reset End Date", value=date.today(), key="attendance_reset_end")
            if st.button("Delete Attendance Range", width="stretch"):
                deleted = attendance_service.reset_attendance(start_date, end_date)
                st.success(f"Deleted {deleted} attendance rows")
                st.rerun()

    with tab_data_control:
        section_title("Data Control")
        st.warning("These actions are destructive. Use them only when you are sure.")

        confirm_reset = st.text_input("Type DELETE ALL to confirm database reset")
        if st.button("Delete All Database Data", type="primary", width="stretch"):
            if confirm_reset.strip().upper() != "DELETE ALL":
                st.error("Please type DELETE ALL to confirm.")
            else:
                admin_service.reset_database()
                st.success("Database reset complete. Defaults were restored.")
                st.rerun()

        st.divider()
        st.subheader("CSV Exports")
        csv_exports = export_service.list_exports("csv")
        if csv_exports:
            selected_csv = st.selectbox("Select CSV export", csv_exports, format_func=lambda path: path.name)
            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    "Download Selected CSV",
                    data=selected_csv.read_bytes(),
                    file_name=selected_csv.name,
                    mime="text/csv",
                    width="stretch",
                )
            with col_b:
                if st.button("Delete Selected CSV", width="stretch"):
                    if export_service.delete_export(selected_csv):
                        st.success(f"Deleted {selected_csv.name}")
                        st.rerun()
                    else:
                        st.error("CSV file not found.")
        else:
            st.info("No CSV exports found.")

    with tab_logs:
        logs = admin_service.list_logs()
        audits = admin_service.list_audit_logs()
        st.subheader("System Logs")
        st.dataframe(logs, hide_index=True, width="stretch")
        if not logs.empty:
            st.download_button("Download Logs CSV", logs.to_csv(index=False).encode("utf-8"), "logs.csv", "text/csv")
        st.subheader("Audit Logs")
        st.dataframe(audits, hide_index=True, width="stretch")
        if not audits.empty:
            st.download_button("Download Audit CSV", audits.to_csv(index=False).encode("utf-8"), "audit_logs.csv", "text/csv")
