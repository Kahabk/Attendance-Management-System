from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, or_

from config import settings
from database import backup_database, get_db_session, reset_database, restore_database
from models import AuditLog, Camera, Department, Employee, LogEntry, Setting, User
from utils.logger import logger
from utils.security import hash_password


class AdminService:
    def list_users(self) -> pd.DataFrame:
        with get_db_session() as session:
            rows = session.query(User.id, User.username, User.role, User.full_name, User.email, User.is_active).order_by(User.id.desc()).all()
        return pd.DataFrame(rows, columns=["id", "username", "role", "full_name", "email", "is_active"])

    def list_departments(self) -> pd.DataFrame:
        with get_db_session() as session:
            rows = session.query(Department.id, Department.name, Department.description).order_by(Department.name.asc()).all()
        return pd.DataFrame(rows, columns=["id", "name", "description"])

    def list_cameras(self) -> pd.DataFrame:
        with get_db_session() as session:
            rows = session.query(Camera.id, Camera.name, Camera.source, Camera.camera_type, Camera.is_active, Camera.notes).order_by(Camera.id.desc()).all()
        return pd.DataFrame(rows, columns=["id", "name", "source", "camera_type", "is_active", "notes"])

    def list_employees(self, search: str | None = None) -> pd.DataFrame:
        with get_db_session() as session:
            query = session.query(
                Employee.id,
                Employee.employee_id,
                Employee.name,
                Department.name.label("department"),
                Employee.email,
                Employee.phone,
                Employee.designation,
                Employee.joining_date,
                Employee.is_active,
            ).outerjoin(Department, Employee.department_id == Department.id)
            if search:
                like = f"%{search}%"
                query = query.filter(or_(Employee.name.ilike(like), Employee.employee_id.ilike(like), Employee.email.ilike(like)))
            rows = query.order_by(Employee.id.desc()).all()
        return pd.DataFrame(rows)

    def list_logs(self, limit: int = 500) -> pd.DataFrame:
        with get_db_session() as session:
            rows = session.query(LogEntry.id, LogEntry.level, LogEntry.message, LogEntry.source, LogEntry.created_at).order_by(LogEntry.created_at.desc()).limit(limit).all()
        return pd.DataFrame(rows, columns=["id", "level", "message", "source", "created_at"])

    def list_audit_logs(self, limit: int = 500) -> pd.DataFrame:
        with get_db_session() as session:
            rows = session.query(AuditLog.id, AuditLog.action, AuditLog.entity_type, AuditLog.entity_id, AuditLog.severity, AuditLog.created_at).order_by(AuditLog.created_at.desc()).limit(limit).all()
        return pd.DataFrame(rows, columns=["id", "action", "entity_type", "entity_id", "severity", "created_at"])

    def create_department(self, name: str, description: str | None = None) -> tuple[bool, str]:
        with get_db_session() as session:
            if session.query(Department).filter(func.lower(Department.name) == name.lower()).first():
                return False, "Department already exists"
            session.add(Department(name=name.strip(), description=description))
            return True, "Department created"

    def create_camera(self, name: str, source: str, camera_type: str = "webcam", notes: str | None = None) -> tuple[bool, str]:
        with get_db_session() as session:
            if session.query(Camera).filter(Camera.name == name).first():
                return False, "Camera already exists"
            session.add(Camera(name=name.strip(), source=source.strip(), camera_type=camera_type, notes=notes))
            return True, "Camera created"

    def update_camera(self, camera_id: int, **fields: Any) -> tuple[bool, str]:
        with get_db_session() as session:
            camera = session.get(Camera, camera_id)
            if camera is None:
                return False, "Camera not found"
            for key, value in fields.items():
                if hasattr(camera, key) and value is not None:
                    setattr(camera, key, value)
            return True, "Camera updated"

    def delete_camera(self, camera_id: int) -> tuple[bool, str]:
        with get_db_session() as session:
            camera = session.get(Camera, camera_id)
            if camera is None:
                return False, "Camera not found"
            session.delete(camera)
            return True, "Camera deleted"

    def create_employee(self, **fields: Any) -> tuple[bool, str, int | None]:
        with get_db_session() as session:
            employee = Employee(
                employee_id=str(fields["employee_id"]).strip(),
                name=str(fields["name"]).strip(),
                department_id=fields.get("department_id"),
                email=(fields.get("email") or None),
                phone=(fields.get("phone") or None),
                designation=(fields.get("designation") or None),
                joining_date=fields.get("joining_date"),
                is_active=bool(fields.get("is_active", True)),
            )
            session.add(employee)
            session.flush()
            return True, "Employee created successfully", employee.id

    def update_employee(self, employee_pk: int, **fields: Any) -> tuple[bool, str]:
        with get_db_session() as session:
            employee = session.get(Employee, employee_pk)
            if employee is None:
                return False, "Employee not found"
            for key, value in fields.items():
                if hasattr(employee, key) and value is not None:
                    setattr(employee, key, value)
            return True, "Employee updated successfully"

    def delete_employee(self, employee_id: int) -> tuple[bool, str]:
        with get_db_session() as session:
            employee = session.get(Employee, employee_id)
            if employee is None:
                return False, "Employee not found"
            session.delete(employee)
            return True, "Employee deleted"

    def set_setting(self, key: str, value: str) -> None:
        with get_db_session() as session:
            row = session.get(Setting, key)
            if row is None:
                row = Setting(key=key, value=value)
                session.add(row)
            else:
                row.value = value

    def get_setting(self, key: str, default: str = "") -> str:
        with get_db_session() as session:
            row = session.get(Setting, key)
            return row.value if row else default

    def update_user(self, user_id: int, **fields: Any) -> tuple[bool, str]:
        with get_db_session() as session:
            user = session.get(User, user_id)
            if user is None:
                return False, "User not found"
            for key, value in fields.items():
                if value is None:
                    continue
                if key == "password":
                    user.password_hash = hash_password(str(value))
                elif hasattr(user, key):
                    setattr(user, key, value)
            return True, "User updated successfully"

    def create_user(self, username: str, password: str, role: str, full_name: str | None = None, email: str | None = None, employee_id: int | None = None) -> tuple[bool, str]:
        with get_db_session() as session:
            query = session.query(User).filter(User.username == username)
            if email:
                query = query.union(session.query(User).filter(User.email == email))
            if query.first():
                return False, "User already exists"
            session.add(
                User(
                    username=username,
                    password_hash=hash_password(password),
                    role=role,
                    full_name=full_name,
                    email=email,
                    employee_id=employee_id,
                    is_active=True,
                )
            )
            return True, "User created"

    def delete_user(self, user_id: int) -> tuple[bool, str]:
        with get_db_session() as session:
            user = session.get(User, user_id)
            if user is None:
                return False, "User not found"
            session.delete(user)
            return True, "User deleted"

    def backup_db(self) -> Path:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        destination = settings.database_dir / f"attendance_backup_{stamp}.db"
        return backup_database(destination)

    def restore_db(self, source_path: Path) -> None:
        restore_database(source_path)

    def reset_database(self) -> None:
        reset_database()

    def log_message(self, level: str, message: str, source: str | None = None) -> None:
        logger.log(getattr(__import__("logging"), level.upper(), 20), message)


admin_service = AdminService()
