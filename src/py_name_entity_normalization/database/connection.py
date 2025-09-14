"""Manages the database connection and session creation.

This module sets up the SQLAlchemy engine and provides a session factory
for interacting with the database. The connection string is read from the
application's configuration.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import settings

# Create the SQLAlchemy engine using the database URL from settings
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations.

    This context manager will create a new session, manage its lifecycle,
    and ensure it's properly closed.

    Yields
    ------
        The SQLAlchemy session.

    """
    db_session = SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()
