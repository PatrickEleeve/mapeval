"""Database connection and session management.

Supports SQLite (development) and PostgreSQL (production) via SQLAlchemy.
Provides a DatabaseManager that handles engine creation, table creation,
and session lifecycle.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)

try:
    from sqlalchemy import create_engine, event, text
    from sqlalchemy.orm import Session, sessionmaker
    from db_models import Base, SQLALCHEMY_AVAILABLE
except ImportError:
    SQLALCHEMY_AVAILABLE = False

DEFAULT_SQLITE_PATH = "data/trading.db"


class DatabaseManager:
    """Manages database connections and sessions.

    Usage::

        db = DatabaseManager("sqlite:///data/trading.db")
        db.create_tables()

        with db.session() as session:
            session.add(record)
            session.commit()
    """

    def __init__(self, url: Optional[str] = None) -> None:
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("sqlalchemy is required for database support. Install with: pip install sqlalchemy>=2.0")

        if url is None:
            url = os.getenv("MAPEVAL_DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}")

        self._url = url
        self._is_sqlite = url.startswith("sqlite")

        # Ensure directory exists for SQLite
        if self._is_sqlite:
            db_path = url.replace("sqlite:///", "")
            if db_path:
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._engine = create_engine(
            url,
            echo=False,
            pool_pre_ping=True,
            **({"connect_args": {"check_same_thread": False}} if self._is_sqlite else {}),
        )

        # Enable WAL mode for SQLite for better concurrent read performance
        if self._is_sqlite:
            @event.listens_for(self._engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.close()

        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)
        logger.info("Database initialized: %s", url.split("@")[-1] if "@" in url else url)

    def create_tables(self) -> None:
        """Create all tables if they don't exist."""
        Base.metadata.create_all(self._engine)
        logger.info("Database tables created/verified")

    def drop_tables(self) -> None:
        """Drop all tables (use with caution)."""
        Base.metadata.drop_all(self._engine)
        logger.warning("All database tables dropped")

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Provide a transactional session scope."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_session(self) -> Session:
        """Create a new session (caller responsible for commit/close)."""
        return self._session_factory()

    @property
    def engine(self):
        return self._engine

    @property
    def url(self) -> str:
        return self._url

    def health_check(self) -> bool:
        """Test database connectivity."""
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.error("Database health check failed: %s", exc)
            return False
