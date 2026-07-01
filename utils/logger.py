from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import settings


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("attendance_system")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(settings.logs_dir / "app.log", maxBytes=2_000_000, backupCount=5)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    logger.propagate = False
    return logger


logger = configure_logging()

