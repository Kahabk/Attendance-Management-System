from __future__ import annotations

from datetime import timedelta
from sqlalchemy import or_
from sqlalchemy.orm import Session

from config import settings
from database import get_db_session
from models import AuditLog, SessionRecord, User
from utils.logger import logger
from utils.security import create_jwt, generate_token, hash_password, hash_token, utcnow, verify_password
from utils.serialization import json_dumps


class AuthService:
    def __init__(self) -> None:
        self.lockout_threshold = 5
        self.lockout_minutes = 15

    def authenticate(self, username: str, password: str, ip_address: str | None = None, user_agent: str | None = None) -> tuple[User | None, str | None]:
        with get_db_session() as session:
            user = session.query(User).filter(or_(User.username == username, User.email == username)).one_or_none()
            if user is None or not user.is_active:
                logger.warning("Authentication failed for unknown/inactive user: %s", username)
                return None, "Invalid credentials"

            if user.locked_until and user.locked_until > utcnow():
                return None, "Account locked temporarily. Please try again later."

            if not verify_password(password, user.password_hash):
                user.failed_login_attempts += 1
                if user.failed_login_attempts >= self.lockout_threshold:
                    user.locked_until = utcnow() + timedelta(minutes=self.lockout_minutes)
                self._audit(session, user.id, "LOGIN_FAILED", details={"username": username}, severity="WARNING", ip_address=ip_address)
                logger.warning("Authentication failed for user %s", username)
                return None, "Invalid credentials"

            user.failed_login_attempts = 0
            user.locked_until = None
            session.flush()
            jwt_id = generate_token(24)
            expires_at = utcnow() + timedelta(minutes=settings.jwt_exp_minutes)
            session_record = SessionRecord(
                user_id=user.id,
                jwt_id=jwt_id,
                expires_at=expires_at,
                ip_address=ip_address,
                user_agent=user_agent,
            )
            session.add(session_record)
            session.flush()
            token = create_jwt(user.id, user.role, session_record.id)
            session_record.jwt_id = hash_token(token)
            self._audit(session, user.id, "LOGIN_SUCCESS", details={"session_id": session_record.id}, ip_address=ip_address)
            session.flush()
            session.expunge(user)
            return user, token

    def validate_token(self, token: str) -> User | None:
        try:
            from utils.security import decode_jwt

            payload = decode_jwt(token)
            user_id = int(payload["sub"])
            with get_db_session() as session:
                user = session.get(User, user_id)
                if user is None or not user.is_active:
                    return None
                session_row = session.get(SessionRecord, int(payload["sid"]))
                if session_row is None or session_row.revoked_at is not None:
                    return None
                if session_row.expires_at < utcnow():
                    return None
                if session_row.jwt_id != hash_token(token):
                    return None
                return user
        except Exception:
            return None

    def logout(self, token: str) -> None:
        try:
            from utils.security import decode_jwt

            payload = decode_jwt(token)
            with get_db_session() as session:
                session_row = session.get(SessionRecord, int(payload["sid"]))
                if session_row:
                    session_row.revoked_at = utcnow()
        except Exception:
            return

    def create_user(
        self,
        username: str,
        password: str,
        role: str,
        full_name: str | None = None,
        email: str | None = None,
        employee_id: int | None = None,
    ) -> tuple[bool, str]:
        with get_db_session() as session:
            if session.query(User).filter(User.username == username).first():
                return False, "Username already exists"
            if email and session.query(User).filter(User.email == email).first():
                return False, "Email already exists"
            user = User(
                username=username,
                password_hash=hash_password(password),
                role=role,
                full_name=full_name,
                email=email,
                employee_id=employee_id,
                is_active=True,
            )
            session.add(user)
            self._audit(session, None, "USER_CREATED", details={"username": username, "role": role})
            return True, "User created successfully"

    def update_user(
        self,
        user_id: int,
        **fields: object,
    ) -> tuple[bool, str]:
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
            self._audit(session, user.id, "USER_UPDATED", details=fields)
            return True, "User updated successfully"

    def request_password_reset(self, username_or_email: str) -> tuple[bool, str, str | None]:
        reset_token = generate_token(32)
        reset_hash = hash_token(reset_token)
        with get_db_session() as session:
            user = session.query(User).filter(or_(User.username == username_or_email, User.email == username_or_email)).one_or_none()
            if user is None:
                return False, "If the account exists, a reset token has been created.", None
            user.reset_token_hash = reset_hash
            user.reset_token_expires_at = utcnow() + timedelta(minutes=settings.reset_token_exp_minutes)
            self._audit(session, user.id, "PASSWORD_RESET_REQUESTED", details={"reset": True})
            return True, "Password reset token generated", reset_token

    def reset_password(self, username_or_email: str, token: str, new_password: str) -> tuple[bool, str]:
        with get_db_session() as session:
            user = session.query(User).filter(or_(User.username == username_or_email, User.email == username_or_email)).one_or_none()
            if user is None:
                return False, "User not found"
            if not user.reset_token_hash or not user.reset_token_expires_at:
                return False, "No active reset token"
            if user.reset_token_expires_at < utcnow():
                return False, "Reset token expired"
            if hash_token(token) != user.reset_token_hash:
                return False, "Invalid reset token"
            user.password_hash = hash_password(new_password)
            user.reset_token_hash = None
            user.reset_token_expires_at = None
            user.failed_login_attempts = 0
            user.locked_until = None
            self._audit(session, user.id, "PASSWORD_RESET_COMPLETED", details={"username": user.username})
            return True, "Password reset successfully"

    def _audit(
        self,
        session: Session,
        user_id: int | None,
        action: str,
        *,
        details: object | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        severity: str = "INFO",
        ip_address: str | None = None,
    ) -> None:
        session.add(
            AuditLog(
                user_id=user_id,
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                details=json_dumps(details) if details is not None else None,
                severity=severity,
                ip_address=ip_address,
            )
        )


auth_service = AuthService()
