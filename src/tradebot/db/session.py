"""Async SQLite engine (WAL) and session helpers."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tradebot.db.models import AppState, Base
from tradebot.settings import DATA_DIR, ensure_dirs

_engine = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def db_path() -> Path:
    """Database location. `TRADEBOT_DB` lets tests use an isolated file."""
    import os

    ensure_dirs()
    override = os.getenv("TRADEBOT_DB")
    if override:
        p = Path(override)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return Path(DATA_DIR) / "tradebot.db"


def get_engine():
    global _engine
    if _engine is None:
        url = f"sqlite+aiosqlite:///{db_path().as_posix()}"
        _engine = create_async_engine(url, echo=False, future=True, pool_pre_ping=True)

        @event.listens_for(_engine.sync_engine, "connect")
        def _set_pragmas(dbapi_conn, _record):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=8000")
            cur.close()

    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _sessionmaker


async def init_db() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_db() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    sm = get_sessionmaker()
    async with sm() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ------------------------------------------------------------------ app state
async def state_get(key: str, default=None):
    async with session_scope() as s:
        row = await s.get(AppState, key)
        return row.value if row else default


async def state_set(key: str, value) -> None:
    from tradebot.core.clock import now_utc

    async with session_scope() as s:
        row = await s.get(AppState, key)
        if row is None:
            s.add(AppState(key=key, value=value, updated_at=now_utc()))
        else:
            row.value = value
            row.updated_at = now_utc()


async def state_all() -> dict:
    async with session_scope() as s:
        res = await s.execute(select(AppState))
        return {r.key: r.value for r in res.scalars().all()}
