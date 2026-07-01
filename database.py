from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from config import settings
from database_migrations import upgrade_database, seed_defaults
from models import Base


DATABASE_PATH = Path(settings.database_url.replace("sqlite:///", ""))
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)


def init_database() -> None:
    Base.metadata.create_all(bind=engine)
    upgrade_database(engine)
    with SessionLocal() as session:
        seed_defaults(session)
        session.commit()


@contextmanager
def get_db_session() -> Session:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def backup_database(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATABASE_PATH, destination)
    return destination


def restore_database(source: Path) -> None:
    shutil.copy2(source, DATABASE_PATH)
