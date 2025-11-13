# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker


import config

Base = declarative_base()


@lru_cache(maxsize=1)
def get_database_url() -> str:
    """Construct the SQLAlchemy database URL from settings."""
    settings = config.get_settings()
    mode = settings.storage_mode
    if mode == "mysql":
        from urllib.parse import quote_plus

        password = quote_plus(settings.mysql_password or "")
        return (
            f"mysql+pymysql://{settings.mysql_user}:{password}"
            f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}"
            f"?charset={settings.mysql_charset}"
        )

    # Default to sqlite
    sqlite_path = os.path.expanduser(settings.sqlite_path)
    if sqlite_path.startswith(":"):
        return f"sqlite:///{sqlite_path}"
    sqlite_path = os.path.abspath(sqlite_path)
    directory = os.path.dirname(sqlite_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return f"sqlite:///{sqlite_path}"


@lru_cache(maxsize=1)
def get_engine(echo: bool = False, pool_size: Optional[int] = None) -> Engine:
    """Create or reuse the SQLAlchemy engine."""
    url = get_database_url()
    kwargs = {"echo": echo, "future": True}

    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        if pool_size is not None:
            kwargs["pool_size"] = pool_size
        kwargs.setdefault("pool_pre_ping", True)
        kwargs.setdefault("pool_recycle", 3600)

    return create_engine(url, **kwargs)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker:
    """Return the sessionmaker bound to the router engine."""
    engine = get_engine()
    return sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


@contextmanager
def session_scope() -> Generator:
    """Provide a transactional scope around a series of operations."""
    session = get_session_factory()()
    try:
        yield session
        if session.in_transaction():
            session.commit()
    except Exception:  # pragma: no cover - propagate after rollback
        if session.in_transaction():
            session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Generator:
    """FastAPI dependency that yields a session."""
    with session_scope() as session:
        yield session
