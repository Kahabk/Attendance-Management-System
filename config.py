from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_raspberry_pi() -> bool:
    machine = platform.machine().lower()
    return any(token in machine for token in ("arm", "aarch64", "armv7", "armv6"))


@dataclass(slots=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Attendance Management System")
    secret_key: str = os.getenv("SECRET_KEY", "change-me-in-production")
    jwt_secret: str = os.getenv("JWT_SECRET", "change-me-in-production")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_exp_minutes: int = int(os.getenv("JWT_EXP_MINUTES", "480"))
    reset_token_exp_minutes: int = int(os.getenv("RESET_TOKEN_EXP_MINUTES", "15"))
    database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'database' / 'attendance.db'}")
    threshold: float = float(os.getenv("RECOGNITION_THRESHOLD", "0.45"))
    attendance_cooldown_minutes: int = int(os.getenv("ATTENDANCE_COOLDOWN_MINUTES", "30"))
    work_start_time: str = os.getenv("WORK_START_TIME", "09:15")
    theme: str = os.getenv("APP_THEME", "dark")
    language: str = os.getenv("APP_LANGUAGE", "en")
    export_dir: Path = Path(os.getenv("EXPORT_DIR", str(BASE_DIR / "exports")))
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))
    embeddings_dir: Path = Path(os.getenv("EMBEDDINGS_DIR", str(BASE_DIR / "embeddings")))
    logs_dir: Path = Path(os.getenv("LOGS_DIR", str(BASE_DIR / "logs")))
    database_dir: Path = Path(os.getenv("DATABASE_DIR", str(BASE_DIR / "database")))
    camera_id: str = os.getenv("DEFAULT_CAMERA_ID", "default")
    default_admin_username: str = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
    default_admin_password: str = os.getenv("DEFAULT_ADMIN_PASSWORD", "Admin@1234!")
    default_admin_email: str = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@example.com")
    enable_streamlit_webrtc: bool = _env_bool("ENABLE_STREAMLIT_WEBRTC", True)
    max_upload_images: int = int(os.getenv("MAX_UPLOAD_IMAGES", "50"))
    recognition_fps_target: int = int(os.getenv("RECOGNITION_FPS_TARGET", "15"))
    use_gpu: bool = _env_bool("USE_GPU", False)
    low_power_mode: bool = _env_bool("LOW_POWER_MODE", _is_raspberry_pi())
    face_backend: str = os.getenv("FACE_BACKEND", "sface").lower()
    yunet_model_path: Path = Path(os.getenv("YUNET_MODEL_PATH", str(BASE_DIR / "assets" / "models" / "face_detection_yunet_2023mar.onnx")))
    sface_model_path: Path = Path(os.getenv("SFACE_MODEL_PATH", str(BASE_DIR / "assets" / "models" / "face_recognition_sface_2021dec.onnx")))
    recognition_margin: float = float(os.getenv("RECOGNITION_MARGIN", "0.08"))
    recognition_confirmations: int = max(1, int(os.getenv("RECOGNITION_CONFIRMATIONS", "3")))
    face_detection_confidence: float = float(os.getenv("FACE_DETECTION_CONFIDENCE", "0.9"))
    max_frame_width: int = int(os.getenv("MAX_FRAME_WIDTH", "640" if low_power_mode else "1280"))
    face_detector_min_size: int = int(os.getenv("FACE_DETECTOR_MIN_SIZE", "64" if low_power_mode else "80"))
    insightface_det_size: int = int(os.getenv("INSIGHTFACE_DET_SIZE", "320" if low_power_mode else "640"))
    live_recognition_interval_ms: int = int(
        os.getenv("LIVE_RECOGNITION_INTERVAL_MS", "250" if low_power_mode else "150")
    )
    attendance_refresh_seconds: int = int(os.getenv("ATTENDANCE_REFRESH_SECONDS", "4" if low_power_mode else "2"))
    camera_width: int = int(os.getenv("CAMERA_WIDTH", "640" if low_power_mode else "960"))
    camera_height: int = int(os.getenv("CAMERA_HEIGHT", "480" if low_power_mode else "720"))
    attendance_reentry_gap_seconds: int = int(
        os.getenv("ATTENDANCE_REENTRY_GAP_SECONDS", "90" if low_power_mode else "45")
    )

    def ensure_directories(self) -> None:
        for folder in [
            self.export_dir,
            self.upload_dir,
            self.embeddings_dir,
            self.logs_dir,
            self.database_dir,
        ]:
            folder.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
