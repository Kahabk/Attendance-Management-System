from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import and_, desc, distinct, func
from sqlalchemy.orm import Session

from config import settings
from database import get_db_session
from models import Attendance, Department, Employee, FaceEmbedding, Setting
from utils.logger import logger
from utils.security import utcnow


class AttendanceService:
    @staticmethod
    def _ensure_aware_datetime(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _setting(self, session: Session, key: str, default: str) -> str:
        row = session.get(Setting, key)
        return row.value if row is not None else default

    def cooldown_minutes(self, session: Session | None = None) -> int:
        if session is None:
            with get_db_session() as temp:
                return int(self._setting(temp, "attendance_cooldown_minutes", str(settings.attendance_cooldown_minutes)))
        return int(self._setting(session, "attendance_cooldown_minutes", str(settings.attendance_cooldown_minutes)))

    def recognition_threshold(self, session: Session | None = None) -> float:
        if session is None:
            with get_db_session() as temp:
                return float(self._setting(temp, "recognition_threshold", str(settings.threshold)))
        return float(self._setting(session, "recognition_threshold", str(settings.threshold)))

    def work_start_time(self, session: Session | None = None) -> time:
        raw = settings.work_start_time
        if session is not None:
            raw = self._setting(session, "work_start_time", raw)
        hour, minute = [int(v) for v in raw.split(":", 1)]
        return time(hour=hour, minute=minute)

    def record_recognition(
        self,
        employee_id: int,
        confidence: float,
        face_embedding_id: int | None = None,
        camera_id: str | None = None,
        device: str | None = None,
        location: str | None = None,
        notes: str | None = None,
        at: datetime | None = None,
    ) -> tuple[bool, str]:
        now = self._ensure_aware_datetime(at or utcnow())
        if now is None:
            return False, "Invalid recognition time"
        with get_db_session() as session:
            employee = session.get(Employee, employee_id)
            if employee is None or not employee.is_active:
                return False, "Employee not found or inactive"

            today = now.date()
            cooldown = timedelta(minutes=self.cooldown_minutes(session))
            latest = (
                session.query(Attendance)
                .filter(and_(Attendance.employee_id == employee_id, Attendance.attendance_date == today))
                .order_by(desc(Attendance.check_in_at), desc(Attendance.created_at))
                .first()
            )

            if latest is not None:
                if latest.check_out_at is None:
                    latest.confidence = max(latest.confidence, confidence)
                    latest.face_embedding_id = face_embedding_id or latest.face_embedding_id
                    latest.camera_id = camera_id or latest.camera_id
                    latest.device = device or latest.device
                    latest.location = location or latest.location
                    latest.notes = notes or latest.notes
                    latest.check_out_at = now
                    latest.status = "checked_out"
                    return True, "Check-out marked successfully"

                last_moment = self._ensure_aware_datetime(latest.check_out_at or latest.check_in_at)
                if last_moment and now - last_moment < cooldown:
                    return False, "Attendance ignored due to cooldown"
                latest.confidence = max(latest.confidence, confidence)
                latest.face_embedding_id = face_embedding_id or latest.face_embedding_id
                latest.camera_id = camera_id or latest.camera_id
                latest.device = device or latest.device
                latest.location = location or latest.location
                latest.notes = notes or latest.notes
                return False, "Attendance already completed for today"

            work_start = self.work_start_time(session)
            status = "late" if now.time() > work_start else "checked_in"
            attendance = Attendance(
                employee_id=employee_id,
                attendance_date=today,
                check_in_at=now,
                status=status,
                confidence=confidence,
                face_embedding_id=face_embedding_id,
                camera_id=camera_id,
                device=device,
                location=location,
                notes=notes,
            )
            session.add(attendance)
            return True, "Check-in marked successfully"

    def get_dashboard_metrics(self) -> dict[str, Any]:
        with get_db_session() as session:
            total_employees = session.query(Employee).filter(Employee.is_active.is_(True)).count()
            present_today = (
                session.query(func.count(distinct(Attendance.employee_id)))
                .filter(Attendance.attendance_date == date.today())
                .scalar()
                or 0
            )
            late_arrivals = (
                session.query(Attendance)
                .filter(and_(Attendance.attendance_date == date.today(), Attendance.status == "late"))
                .count()
            )
            registered_faces = session.query(FaceEmbedding).count()
            absent_today = max(total_employees - present_today, 0)
            monthly_attendance_percentage = self._monthly_attendance_percentage(session)
            return {
                "total_employees": total_employees,
                "present_today": present_today,
                "absent_today": absent_today,
                "late_arrivals": late_arrivals,
                "registered_faces": registered_faces,
                "monthly_attendance_percentage": monthly_attendance_percentage,
            }

    def _monthly_attendance_percentage(self, session: Session) -> float:
        month_start = date.today().replace(day=1)
        total_active = session.query(Employee).filter(Employee.is_active.is_(True)).count()
        if total_active == 0:
            return 0.0
        present_days = (
            session.query(func.count(distinct(Attendance.attendance_date)))
            .filter(Attendance.attendance_date >= month_start)
            .scalar()
            or 0
        )
        days_elapsed = max((date.today() - month_start).days + 1, 1)
        return round((present_days / days_elapsed) * 100.0, 2)

    def get_daily_series(self, days: int = 14) -> pd.DataFrame:
        with get_db_session() as session:
            start = date.today() - timedelta(days=days - 1)
            rows = (
                session.query(
                    Attendance.attendance_date.label("date"),
                    func.count(distinct(Attendance.employee_id)).label("present"),
                )
                .filter(Attendance.attendance_date >= start)
                .group_by(Attendance.attendance_date)
                .order_by(Attendance.attendance_date.asc())
                .all()
            )
        frame = pd.DataFrame(rows, columns=["date", "present"])
        if frame.empty:
            return pd.DataFrame({"date": pd.date_range(start, periods=days), "present": [0] * days})
        frame["date"] = pd.to_datetime(frame["date"])
        return frame

    def get_weekly_series(self, weeks: int = 8) -> pd.DataFrame:
        with get_db_session() as session:
            start = date.today() - timedelta(days=weeks * 7)
            rows = (
                session.query(
                    func.strftime("%Y-%W", Attendance.attendance_date).label("week"),
                    func.count(distinct(Attendance.employee_id)).label("present"),
                )
                .filter(Attendance.attendance_date >= start)
                .group_by("week")
                .order_by("week")
                .all()
            )
        return pd.DataFrame(rows, columns=["week", "present"])

    def get_monthly_series(self, months: int = 12) -> pd.DataFrame:
        with get_db_session() as session:
            start = date.today().replace(day=1) - timedelta(days=months * 31)
            rows = (
                session.query(
                    func.strftime("%Y-%m", Attendance.attendance_date).label("month"),
                    func.count(distinct(Attendance.employee_id)).label("present"),
                )
                .filter(Attendance.attendance_date >= start)
                .group_by("month")
                .order_by("month")
                .all()
            )
        return pd.DataFrame(rows, columns=["month", "present"])

    def department_statistics(self) -> pd.DataFrame:
        with get_db_session() as session:
            rows = (
                session.query(
                    Department.name.label("department"),
                    func.count(distinct(Employee.id)).label("employees"),
                    func.count(distinct(Attendance.employee_id)).label("present_today"),
                )
                .outerjoin(Employee, Employee.department_id == Department.id)
                .outerjoin(Attendance, and_(Attendance.employee_id == Employee.id, Attendance.attendance_date == date.today()))
                .group_by(Department.id)
                .order_by(Department.name.asc())
                .all()
            )
        return pd.DataFrame(rows, columns=["department", "employees", "present_today"])

    def attendance_history(
        self,
        employee_id: int | None = None,
        department_id: int | None = None,
        status: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        with get_db_session() as session:
            query = (
                session.query(
                    Attendance.id,
                    Employee.employee_id.label("employee_id"),
                    Employee.name.label("name"),
                    Department.name.label("department"),
                    Attendance.attendance_date,
                    Attendance.check_in_at,
                    Attendance.check_out_at,
                    Attendance.status,
                    Attendance.confidence,
                    Attendance.face_embedding_id,
                    Attendance.camera_id,
                    Attendance.device,
                    Attendance.location,
                    FaceEmbedding.source_image_path.label("face_sample"),
                )
                .join(Employee, Attendance.employee_id == Employee.id)
                .outerjoin(Department, Employee.department_id == Department.id)
                .outerjoin(FaceEmbedding, Attendance.face_embedding_id == FaceEmbedding.id)
            )
            if employee_id:
                query = query.filter(Employee.id == employee_id)
            if department_id:
                query = query.filter(Employee.department_id == department_id)
            if status:
                query = query.filter(Attendance.status == status)
            if start_date:
                query = query.filter(Attendance.attendance_date >= start_date)
            if end_date:
                query = query.filter(Attendance.attendance_date <= end_date)
            rows = query.order_by(Attendance.attendance_date.desc(), Attendance.check_in_at.desc()).all()
        return pd.DataFrame(rows)

    def update_attendance(
        self,
        attendance_id: int,
        *,
        attendance_date: date | None = None,
        check_in_at: datetime | None = None,
        check_out_at: datetime | None = None,
        status: str | None = None,
        confidence: float | None = None,
        face_embedding_id: int | None = None,
        camera_id: str | None = None,
        device: str | None = None,
        location: str | None = None,
        notes: str | None = None,
    ) -> tuple[bool, str]:
        with get_db_session() as session:
            row = session.get(Attendance, attendance_id)
            if row is None:
                return False, "Attendance record not found"

            if attendance_date is not None:
                row.attendance_date = attendance_date
            if check_in_at is not None:
                row.check_in_at = self._ensure_aware_datetime(check_in_at)
            if check_out_at is not None:
                row.check_out_at = self._ensure_aware_datetime(check_out_at)
            if status is not None:
                row.status = status
            if confidence is not None:
                row.confidence = confidence
            if face_embedding_id is not None:
                row.face_embedding_id = face_embedding_id
            if camera_id is not None:
                row.camera_id = camera_id
            if device is not None:
                row.device = device
            if location is not None:
                row.location = location
            if notes is not None:
                row.notes = notes
            return True, "Attendance updated successfully"

    def todays_attendance(self) -> pd.DataFrame:
        df = self.attendance_history(start_date=date.today(), end_date=date.today())
        if df.empty:
            return df

        df = df.sort_values(by=["check_in_at", "check_out_at"], ascending=[False, False]).reset_index(drop=True)
        df["mark_state"] = df["status"].map(
            {
                "checked_in": "IN",
                "late": "IN (Late)",
                "checked_out": "OUT",
            }
        ).fillna(df["status"])

        def _date_value(value: object) -> str:
            if pd.isna(value) or value is None:
                return ""
            if hasattr(value, "date"):
                return str(value.date())
            return str(value)

        def _time_value(value: object) -> str:
            if pd.isna(value) or value is None:
                return ""
            if hasattr(value, "strftime"):
                return value.strftime("%H:%M:%S")
            return str(value)

        df["in_date"] = df["check_in_at"].map(_date_value)
        df["in_time"] = df["check_in_at"].map(_time_value)
        df["out_date"] = df["check_out_at"].map(_date_value)
        df["out_time"] = df["check_out_at"].map(_time_value)
        return df

    def employees_queryset(self) -> list[Employee]:
        with get_db_session() as session:
            return session.query(Employee).order_by(Employee.name.asc()).all()

    def departments_queryset(self) -> list[Department]:
        with get_db_session() as session:
            return session.query(Department).order_by(Department.name.asc()).all()

    def reset_attendance(self, start_date: date | None = None, end_date: date | None = None) -> int:
        with get_db_session() as session:
            query = session.query(Attendance)
            if start_date:
                query = query.filter(Attendance.attendance_date >= start_date)
            if end_date:
                query = query.filter(Attendance.attendance_date <= end_date)
            count = query.count()
            query.delete(synchronize_session=False)
            logger.info("Deleted %s attendance records", count)
            return count

    def delete_attendance(self, attendance_id: int) -> bool:
        with get_db_session() as session:
            row = session.get(Attendance, attendance_id)
            if row is None:
                return False
            session.delete(row)
            return True

    def mark_missing_absences(self) -> int:
        # Utility for bulk processing or scheduled jobs.
        with get_db_session() as session:
            total = 0
            today = date.today()
            active_employees = session.query(Employee).filter(Employee.is_active.is_(True)).all()
            present_ids = {
                row[0]
                for row in session.query(distinct(Attendance.employee_id))
                .filter(Attendance.attendance_date == today)
                .all()
            }
            for employee in active_employees:
                if employee.id not in present_ids:
                    total += 1
            return total


attendance_service = AttendanceService()
