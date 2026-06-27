"""SQLite access layer: schema bootstrap and typed queries."""

from db.database import Database, DatabaseError

__all__ = ["Database", "DatabaseError"]
