"""Process-wide defaults for database and cache paths (overridable in tests)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from config.config import get_settings
from db.database import Database


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
    """Test helper: reset cached database handle."""
    get_default_database.cache_clear()
