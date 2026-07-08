"""Process-wide defaults for database and cache paths (overridable in tests)."""

from __future__ import annotations

from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

from config.config import get_settings
from db.database import Database

_tool_session_id: ContextVar[str | None] = ContextVar("tool_session_id", default=None)


def set_tool_session_id(session_id: str | None) -> None:
    """Bind the active chat session id for tool calls in the current async/task context."""
    _tool_session_id.set(session_id)


def get_tool_session_id() -> str | None:
    """Return the session id set by :func:`set_tool_session_id`, if any."""
    return _tool_session_id.get()


@lru_cache
def get_default_database() -> Database:
    """Return a cached ``Database`` using ``AppSettings.database_path``."""
    settings = get_settings()
    return Database(settings.database_path)


def get_cache_dir() -> Path:
    """Directory for PDF and API response caches."""
    base = get_settings().database_path.parent
    out = base / "cache"
    out.mkdir(parents=True, exist_ok=True)
    return out


def clear_tool_caches() -> None:
    """Test helper: reset cached database handle and session context."""
    get_default_database.cache_clear()
    _tool_session_id.set(None)
