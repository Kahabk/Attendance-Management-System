from __future__ import annotations

from datetime import timedelta

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from config import settings
from models import Camera, Department, Setting, User
from utils.security import hash_password, utcnow


def _column_exists(engine: Engine, table_name: str, column_name: str) -> bool:
    inspector = inspect(engine)
    return any(column["name"] == column_name for column in inspector.get_columns(table_name))


def upgrade_database(engine: Engine) -> None:
    with engine.begin() as connection:
        # Lightweight migrations for incremental schema evolution.
        migrations: list[tuple[str, str, str]] = [
            ("users", "reset_token_hash", "ALTER TABLE users ADD COLUMN reset_token_hash TEXT"),
            ("users", "reset_token_expires_at", "ALTER TABLE users ADD COLUMN reset_token_expires_at DATETIME"),
            ("users", "failed_login_attempts", "ALTER TABLE users ADD COLUMN failed_login_attempts INTEGER DEFAULT 0"),
            ("users", "locked_until", "ALTER TABLE users ADD COLUMN locked_until DATETIME"),
            ("employees", "photo_path", "ALTER TABLE employees ADD COLUMN photo_path TEXT"),
            ("attendance", "location", "ALTER TABLE attendance ADD COLUMN location TEXT"),
            ("attendance", "notes", "ALTER TABLE attendance ADD COLUMN notes TEXT"),
            ("attendance", "face_embedding_id", "ALTER TABLE attendance ADD COLUMN face_embedding_id INTEGER"),
            ("cameras", "notes", "ALTER TABLE cameras ADD COLUMN notes TEXT"),
            ("logs", "source", "ALTER TABLE logs ADD COLUMN source TEXT"),
        ]
        for table_name, column_name, ddl in migrations:
            try:
                if not _column_exists(engine, table_name, column_name):
                    connection.execute(text(ddl))
            except Exception:
                # SQLite "add column" migrations are best-effort and may already exist.
                continue


def seed_defaults(session: Session) -> None:
    if session.query(User).count() == 0:
        session.add(
            User(
                username=settings.default_admin_username,
                password_hash=hash_password(settings.default_admin_password),
                role="Admin",
                full_name="System Administrator",
                email=settings.default_admin_email,
                is_active=True,
            )
        )

    if session.query(Department).count() == 0:
        session.add_all(
            [
                Department(name="General", description="Default department"),
                Department(name="Operations", description="Operations team"),
                Department(name="HR", description="Human resources"),
                Department(name="IT", description="Information technology"),
            ]
        )

    defaults = {
        "recognition_threshold": str(settings.threshold),
        "attendance_cooldown_minutes": str(settings.attendance_cooldown_minutes),
        "attendance_reentry_gap_seconds": str(settings.attendance_reentry_gap_seconds),
        "theme": settings.theme,
        "language": settings.language,
        "camera_id": settings.camera_id,
        "work_start_time": settings.work_start_time,
    }
    for key, value in defaults.items():
        if session.get(Setting, key) is None:
            session.add(Setting(key=key, value=value))

    if session.query(Camera).count() == 0:
        session.add(
            Camera(
                name="Default Webcam",
                source="0",
                camera_type="webcam",
                is_active=True,
                notes="Initial local webcam source",
            )
        )
