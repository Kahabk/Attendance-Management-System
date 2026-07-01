from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from sqlalchemy.orm import joinedload

from config import settings
from database import get_db_session
from models import Department, Employee, FaceEmbedding
from services.admin_service import admin_service
from services.face_service import FaceService
from utils.serialization import array_to_blob
from utils.ui import section_title


def _save_upload(employee_code: str, uploaded_file) -> Path:
    employee_dir = settings.upload_dir / "employees" / employee_code
    employee_dir.mkdir(parents=True, exist_ok=True)
    path = employee_dir / uploaded_file.name
    image = Image.open(uploaded_file)
    image.save(path)
    return path


def _register_face_embeddings(employee_id: int, employee_code: str, image_paths: list[Path]) -> tuple[int, int]:
    face_service = FaceService()
    saved = 0
    rejected = 0
    embeddings: list[np.ndarray] = []

    with get_db_session() as session:
        for image_path in image_paths:
            try:
                image = Image.open(image_path)
                rgb = np.array(image.convert("RGB"))
                result = face_service.extract_face_embedding(rgb)
                if result is None:
                    rejected += 1
                    continue

                session.add(
                    FaceEmbedding(
                        employee_id=employee_id,
                        embedding=array_to_blob(result.embedding),
                        normalized_embedding=array_to_blob(face_service.normalize_embedding(result.embedding)),
                        embedding_dim=int(result.embedding.shape[0]),
                        source_image_path=str(image_path),
                        quality_score=float(result.quality_score),
                        is_primary=False,
                    )
                )
                embeddings.append(result.embedding)
                saved += 1
            except Exception:
                rejected += 1

        if embeddings:
            avg_embedding = face_service.average_embedding(embeddings)
            session.add(
                FaceEmbedding(
                    employee_id=employee_id,
                    embedding=array_to_blob(avg_embedding),
                    normalized_embedding=array_to_blob(face_service.normalize_embedding(avg_embedding)),
                    embedding_dim=int(avg_embedding.shape[0]),
                    source_image_path=str(image_paths[0]) if image_paths else None,
                    quality_score=1.0,
                    is_primary=True,
                )
            )

    return saved, rejected


def render(current_user: dict[str, object]) -> None:
    if current_user.get("role") != "Admin":
        st.warning("Only administrators can manage employees.")
        return

    section_title("Employee Management")
    search = st.text_input("Search employees", placeholder="Search by ID, name, or email")
    df = admin_service.list_employees(search.strip() or None)
    st.dataframe(df, hide_index=True, width="stretch")

    col_left, col_right = st.columns(2)
    with col_left:
        section_title("Add Employee")
        departments = admin_service.list_departments()
        dept_map = {row["name"]: int(row["id"]) for _, row in departments.iterrows()} if not departments.empty else {}
        with st.form("add_employee_form", clear_on_submit=True):
            employee_id = st.text_input("Employee ID")
            name = st.text_input("Name")
            department_name = st.selectbox("Department", options=list(dept_map.keys()) or ["General"])
            email = st.text_input("Email")
            phone = st.text_input("Phone")
            designation = st.text_input("Designation")
            joining_date = st.date_input("Joining Date", value=date.today())
            is_active = st.checkbox("Active", value=True)
            files = st.file_uploader("Upload employee photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
            create_user = st.checkbox("Create login account", value=False)
            username = st.text_input("Login Username", disabled=not create_user)
            password = st.text_input("Login Password", type="password", disabled=not create_user)
            submitted = st.form_submit_button("Create Employee", width="stretch")

        if submitted:
            if not employee_id or not name:
                st.error("Employee ID and Name are required.")
            else:
                dept_id = dept_map.get(department_name)
                ok, msg, new_id = admin_service.create_employee(
                    employee_id=employee_id,
                    name=name,
                    department_id=dept_id,
                    email=email or None,
                    phone=phone or None,
                    designation=designation or None,
                    joining_date=joining_date,
                    is_active=is_active,
                )
                if ok and new_id:
                    saved_paths: list[Path] = []
                    for file in files or []:
                        saved_paths.append(_save_upload(employee_id, file))
                    if saved_paths:
                        admin_service.update_employee(new_id, photo_path=str(saved_paths[0]))
                        saved_count, rejected_count = _register_face_embeddings(new_id, employee_id, saved_paths)
                    if create_user and username and password:
                        admin_service.create_user(username=username, password=password, role="Employee", full_name=name, email=email or None, employee_id=new_id)
                    st.success(msg)
                    if saved_paths:
                        st.info(f"Registered {saved_count} face embeddings. Rejected samples: {rejected_count}.")
                    st.rerun()
                else:
                    st.error(msg)

    with col_right:
        section_title("Edit / Delete")
        if df.empty:
            st.info("No employees found.")
            return
        selected = st.selectbox("Select Employee", options=df["id"].tolist(), format_func=lambda x: f'{df.loc[df["id"] == x, "employee_id"].iloc[0]} - {df.loc[df["id"] == x, "name"].iloc[0]}')
        with get_db_session() as session:
            employee = session.get(Employee, int(selected), options=[joinedload(Employee.department)])
            departments_rows = session.query(Department).order_by(Department.name.asc()).all()
            dept_options = {dept.name: dept.id for dept in departments_rows}

        if employee:
            with st.form("edit_employee_form"):
                upd_employee_id = st.text_input("Employee ID", value=employee.employee_id)
                upd_name = st.text_input("Name", value=employee.name)
                dept_names = list(dept_options.keys()) or ["General"]
                current_department = employee.department.name if employee.department else None
                current_index = dept_names.index(current_department) if current_department in dept_names else 0
                upd_dept = st.selectbox("Department", options=dept_names, index=current_index)
                upd_email = st.text_input("Email", value=employee.email or "")
                upd_phone = st.text_input("Phone", value=employee.phone or "")
                upd_designation = st.text_input("Designation", value=employee.designation or "")
                upd_joining = st.date_input("Joining Date", value=employee.joining_date or date.today())
                upd_active = st.checkbox("Active", value=employee.is_active)
                upd_photo = st.file_uploader("Upload new photo", type=["jpg", "jpeg", "png"])
                save = st.form_submit_button("Save Changes", width="stretch")
            if save:
                photo_path = None
                if upd_photo is not None:
                    photo_path = str(_save_upload(upd_employee_id, upd_photo))
                admin_service.update_employee(
                    employee.id,
                    employee_id=upd_employee_id,
                    name=upd_name,
                    department_id=dept_options.get(upd_dept),
                    email=upd_email or None,
                    phone=upd_phone or None,
                    designation=upd_designation or None,
                    joining_date=upd_joining,
                    is_active=upd_active,
                    photo_path=photo_path,
                )
                if upd_photo is not None:
                    saved_count, rejected_count = _register_face_embeddings(employee.id, upd_employee_id, [Path(photo_path)])
                st.success("Employee updated")
                if upd_photo is not None:
                    st.info(f"Registered {saved_count} face embeddings. Rejected samples: {rejected_count}.")
                st.rerun()

            if st.button("Delete Employee", type="primary", width="stretch"):
                admin_service.delete_employee(employee.id)
                st.success("Employee deleted")
                st.rerun()
