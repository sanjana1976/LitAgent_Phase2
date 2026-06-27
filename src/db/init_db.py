"""
Schema bootstrap for the research paper analyzer local store.

Run as a module or import ``initialize_schema`` from application startup code.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

from db.database import Database, DatabaseError

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 4

DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS papers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        authors TEXT NOT NULL,
        abstract TEXT,
        doi TEXT UNIQUE,
        url TEXT,
        api_source TEXT NOT NULL,
        metadata_json TEXT,
        added_date TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS reading_lists (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        created_date TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS paper_reading_list_mapping (
        paper_id INTEGER NOT NULL,
        list_id INTEGER NOT NULL,
        reading_status TEXT NOT NULL DEFAULT 'unread',
        PRIMARY KEY (paper_id, list_id),
        FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
        FOREIGN KEY (list_id) REFERENCES reading_lists(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        paper_id INTEGER NOT NULL,
        summary_text TEXT NOT NULL,
        analysis_depth TEXT NOT NULL DEFAULT 'standard',
        created_date TEXT NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_message TEXT NOT NULL,
        agent_response TEXT NOT NULL,
        timestamp TEXT NOT NULL DEFAULT (datetime('now')),
        reading_list_context TEXT,
        session_id TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS permission_audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL DEFAULT (datetime('now')),
        session_id TEXT,
        tool_name TEXT NOT NULL,
        action TEXT NOT NULL,
        allowed INTEGER NOT NULL,
        needs_confirmation INTEGER NOT NULL,
        user_decision TEXT,
        reason TEXT
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_summaries_paper_id ON summaries(paper_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_mapping_list_id ON paper_reading_list_mapping(list_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_conversation_history_ts ON conversation_history(timestamp DESC);
    """,
    # --- LitSynth (A4): synthesis cache -------------------------------------
    """
    CREATE TABLE IF NOT EXISTS synthesis_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        question TEXT NOT NULL,
        review_text TEXT NOT NULL,
        result_json TEXT NOT NULL,
        confidence_score REAL NOT NULL DEFAULT 0.0,
        contradictions_found INTEGER NOT NULL DEFAULT 0,
        hallucinated_count INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_synthesis_runs_session ON synthesis_runs(session_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_synthesis_runs_created ON synthesis_runs(created_at DESC);
    """,
)


def _apply_column_migrations(conn: sqlite3.Connection) -> None:
    """Lightweight migrations for existing deployments (CREATE IF NOT EXISTS may skip new columns)."""
    try:
        conn.execute(
            """
            ALTER TABLE paper_reading_list_mapping
            ADD COLUMN reading_status TEXT NOT NULL DEFAULT 'unread';
            """
        )
    except sqlite3.OperationalError as exc:
        if "duplicate column" not in str(exc).lower():
            raise
    try:
        conn.execute(
            """
            ALTER TABLE conversation_history
            ADD COLUMN session_id TEXT;
            """
        )
    except sqlite3.OperationalError as exc:
        if "duplicate column" not in str(exc).lower():
            raise


def initialize_schema(db: Database) -> None:
    """
    Apply DDL when missing and record migration version idempotently.

    Raises:
        DatabaseError: if SQLite cannot execute the migration batch.
    """
    try:
        with db.connection() as conn:
            for statement in DDL_STATEMENTS:
                conn.execute(statement)
            _apply_column_migrations(conn)
            conn.execute(
                """
                INSERT OR IGNORE INTO schema_migrations (version)
                VALUES (?);
                """,
                (SCHEMA_VERSION,),
            )
    except DatabaseError:
        logger.exception("Failed to initialize database schema at %s", db.path)
        raise


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point for offline schema creation."""
    parser = argparse.ArgumentParser(description="Initialize SQLite schema for papers and chats.")
    parser.add_argument(
        "database_path",
        type=Path,
        nargs="?",
        default=Path.cwd() / "data" / "papers.sqlite3",
        help="Target SQLite path (directories are created automatically).",
    )
    parsed = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    db = Database(parsed.database_path)
    initialize_schema(db)
    logger.info("Schema ready at %s (version=%s)", db.path, SCHEMA_VERSION)
    return 0


if __name__ == "__main__":
    sys.exit(main())
