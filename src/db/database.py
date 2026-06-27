"""Connection management and guarded execution primitives."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


class DatabaseError(RuntimeError):
    """Raised when SQLite operations fail in the application boundary layer."""


class Database:
    """
    Lightweight SQLite façade with explicit lifecycle (no implicit global state).

    The database file directory is created on demand when opening a connection.
    """

    def __init__(self, path: Path) -> None:
        self._path = path.expanduser().resolve()

    @property
    def path(self) -> Path:
        return self._path

    def ensure_directory(self) -> None:
        """Create parent folders for ``path`` when missing."""
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Yield a configured SQLite connection with foreign keys enforced.

        Commits on success; rolls back on exception.
        """
        self.ensure_directory()
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            yield conn
            conn.commit()
        except Exception as exc:  # noqa: BLE001 — uniform ``DatabaseError`` at call sites
            conn.rollback()
            raise DatabaseError(f"SQLite transaction failed for {self._path}") from exc
        finally:
            conn.close()

    def executemany(
        self,
        sql: str,
        seq_of_parameters: list[tuple[object, ...]],
    ) -> None:
        """Execute a batch statement inside a managed transaction."""
        try:
            with self.connection() as conn:
                conn.executemany(sql, seq_of_parameters)
        except sqlite3.Error as exc:
            raise DatabaseError(str(exc)) from exc
