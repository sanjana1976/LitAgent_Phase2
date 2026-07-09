"""Typed persistence helpers — extend as tools gain behavior."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from db.database import Database, DatabaseError
from models.conversation_turn import ConversationTurnIn


def insert_conversation_turn(db: Database, turn: ConversationTurnIn) -> int:
    """
    Persist a completed chat turn.

    Returns:
        The SQLite row id assigned to ``conversation_history.id``.
    """
    sql = """
        INSERT INTO conversation_history (
            user_message,
            agent_response,
            reading_list_context,
            session_id
        ) VALUES (?, ?, ?, ?);
    """
    try:
        with db.connection() as conn:
            cursor = conn.execute(
                sql,
                (
                    turn.user_message,
                    turn.agent_response,
                    turn.reading_list_context,
                    getattr(turn, "session_id", None),
                ),
            )
            return int(cursor.lastrowid)
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not persist conversation history: {exc}") from exc


def fetch_recent_conversation_summaries(db: Database, limit: int = 50) -> list[tuple[int, str, str]]:
    """
    Lightweight helper for reloading recent chat summaries (id, user, assistant).

    Note:
        Intended for conversational context previews; callers may map into LangChain messages.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    sql = """
        SELECT id, user_message, agent_response
        FROM conversation_history
        ORDER BY datetime(timestamp) DESC
        LIMIT ?;
    """
    try:
        with db.connection() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
        return [(int(r["id"]), str(r["user_message"]), str(r["agent_response"])) for r in rows]
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not load conversation history: {exc}") from exc


def list_recent_sessions(
    db: Database,
    *,
    limit: int = 20,
) -> list[tuple[str, int, str, str]]:
    """
    Return recent sessions with aggregate stats.

    Output tuples: (session_id, turn_count, started_at, last_at).
    """
    sql = """
        SELECT
            session_id,
            COUNT(*) AS turn_count,
            MIN(timestamp) AS started_at,
            MAX(timestamp) AS last_at
        FROM conversation_history
        WHERE session_id IS NOT NULL AND session_id != ''
        GROUP BY session_id
        ORDER BY datetime(last_at) DESC
        LIMIT ?;
    """
    try:
        with db.connection() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
        return [
            (
                str(r["session_id"]),
                int(r["turn_count"]),
                str(r["started_at"]),
                str(r["last_at"]),
            )
            for r in rows
        ]
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not list recent sessions: {exc}") from exc


def fetch_conversation_by_session(
    db: Database,
    session_id: str,
    *,
    limit: int = 200,
) -> list[tuple[str, str]]:
    """Return ordered (user_message, agent_response) pairs for one session."""
    sql = """
        SELECT user_message, agent_response
        FROM conversation_history
        WHERE session_id = ?
        ORDER BY id ASC
        LIMIT ?;
    """
    try:
        with db.connection() as conn:
            rows = conn.execute(sql, (session_id, limit)).fetchall()
        return [(str(r["user_message"]), str(r["agent_response"])) for r in rows]
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not load conversation session: {exc}") from exc


def clear_conversation_session(db: Database, session_id: str) -> int:
    """Delete all conversation turns for a session and return deleted count."""
    sql = "DELETE FROM conversation_history WHERE session_id = ?;"
    try:
        with db.connection() as conn:
            cur = conn.execute(sql, (session_id,))
            return int(cur.rowcount)
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not clear conversation session: {exc}") from exc


def insert_permission_audit(
    db: Database,
    *,
    session_id: str | None,
    tool_name: str,
    action: str,
    allowed: bool,
    needs_confirmation: bool,
    user_decision: str | None,
    reason: str | None,
) -> int:
    """Persist one permission/confirmation decision for auditing."""
    sql = """
        INSERT INTO permission_audit (
            session_id,
            tool_name,
            action,
            allowed,
            needs_confirmation,
            user_decision,
            reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
    """
    try:
        with db.connection() as conn:
            cur = conn.execute(
                sql,
                (
                    session_id,
                    tool_name,
                    action,
                    int(allowed),
                    int(needs_confirmation),
                    user_decision,
                    reason,
                ),
            )
            return int(cur.lastrowid)
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not write permission audit: {exc}") from exc


# --- Reading lists, papers, summaries (agent tools) ---


def insert_reading_list(db: Database, name: str, description: str | None) -> int:
    """Insert a reading list row and return its id."""
    sql = """
        INSERT INTO reading_lists (name, description) VALUES (?, ?);
    """
    try:
        with db.connection() as conn:
            cur = conn.execute(sql, (name, description))
            return int(cur.lastrowid)
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not create reading list: {exc}") from exc


def list_reading_list_rows(db: Database) -> list[sqlite3.Row]:
    """Return all reading list rows ordered by id."""
    sql = "SELECT id, name, description, created_date FROM reading_lists ORDER BY id;"
    try:
        with db.connection() as conn:
            rows = conn.execute(sql).fetchall()
        return list(rows)
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not list reading lists: {exc}") from exc


def add_paper_to_list(
    db: Database,
    *,
    paper_db_id: int,
    list_id: int,
    reading_status: str,
) -> None:
    """Insert or update mapping row including ``reading_status``."""
    sql = """
        INSERT INTO paper_reading_list_mapping (paper_id, list_id, reading_status)
        VALUES (?, ?, ?)
        ON CONFLICT(paper_id, list_id) DO UPDATE SET reading_status = excluded.reading_status;
    """
    try:
        with db.connection() as conn:
            conn.execute(sql, (paper_db_id, list_id, reading_status))
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not add paper to list: {exc}") from exc


def remove_paper_from_list(db: Database, *, paper_db_id: int, list_id: int) -> bool:
    """Delete mapping row; returns whether a row was removed."""
    sql = """
        DELETE FROM paper_reading_list_mapping WHERE paper_id = ? AND list_id = ?;
    """
    try:
        with db.connection() as conn:
            cur = conn.execute(sql, (paper_db_id, list_id))
            return cur.rowcount > 0
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not remove paper from list: {exc}") from exc


def fetch_papers_for_list(db: Database, list_id: int) -> list[dict[str, Any]]:
    """
    Join papers with mapping for a list.

    Returns rows with paper columns plus ``reading_status``.
    """
    sql = """
        SELECT p.id,
               p.title,
               p.authors,
               p.abstract,
               p.doi,
               p.url,
               p.api_source,
               p.metadata_json,
               m.reading_status
        FROM paper_reading_list_mapping m
        JOIN papers p ON p.id = m.paper_id
        WHERE m.list_id = ?
        ORDER BY p.title;
    """
    try:
        with db.connection() as conn:
            rows = conn.execute(sql, (list_id,)).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not load list papers: {exc}") from exc


def get_paper_by_id(db: Database, paper_db_id: int) -> dict[str, Any] | None:
    """Return a single paper row as dict or None."""
    sql = """
        SELECT id, title, authors, abstract, doi, url, api_source, metadata_json, added_date
        FROM papers WHERE id = ?;
    """
    try:
        with db.connection() as conn:
            row = conn.execute(sql, (paper_db_id,)).fetchone()
        return dict(row) if row else None
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not fetch paper: {exc}") from exc


def list_summaries_for_paper(db: Database, paper_db_id: int) -> list[dict[str, Any]]:
    """Return summary rows for a paper."""
    sql = """
        SELECT id, paper_id, summary_text, analysis_depth, created_date
        FROM summaries WHERE paper_id = ? ORDER BY datetime(created_date) DESC;
    """
    try:
        with db.connection() as conn:
            rows = conn.execute(sql, (paper_db_id,)).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not list summaries: {exc}") from exc


def insert_summary_row(
    db: Database,
    *,
    paper_db_id: int,
    summary_text: str,
    depth: str,
) -> int:
    """Insert a summary and return its id."""
    sql = """
        INSERT INTO summaries (paper_id, summary_text, analysis_depth)
        VALUES (?, ?, ?);
    """
    try:
        with db.connection() as conn:
            cur = conn.execute(sql, (paper_db_id, summary_text, depth))
            return int(cur.lastrowid)
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not save summary: {exc}") from exc


def row_to_metadata_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Parse ``metadata_json`` if present."""
    raw = row.get("metadata_json")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


# --- LitSynth (A4) synthesis cache ----------------------------------------


def insert_synthesis_run(
    db: Database,
    *,
    session_id: str | None,
    question: str,
    review_text: str,
    result_json: str,
    confidence_score: float,
    contradictions_found: int,
    hallucinated_count: int,
) -> int:
    """Persist one completed synthesis run and return the row id."""
    sql = """
        INSERT INTO synthesis_runs (
            session_id, question, review_text, result_json,
            confidence_score, contradictions_found, hallucinated_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
    """
    try:
        with db.connection() as conn:
            cur = conn.execute(
                sql,
                (
                    session_id,
                    question,
                    review_text,
                    result_json,
                    float(confidence_score),
                    int(contradictions_found),
                    int(hallucinated_count),
                ),
            )
            return int(cur.lastrowid)
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not persist synthesis run: {exc}") from exc


def list_recent_synthesis_runs(
    db: Database,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return ``limit`` most recent synthesis runs (newest first)."""
    sql = """
        SELECT id, session_id, question, confidence_score, contradictions_found,
               hallucinated_count, created_at
        FROM synthesis_runs
        ORDER BY datetime(created_at) DESC
        LIMIT ?;
    """
    try:
        with db.connection() as conn:
            rows = conn.execute(sql, (max(1, limit),)).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not list synthesis runs: {exc}") from exc


def get_synthesis_run_result_json(db: Database, run_id: int) -> str | None:
    """Return the persisted ``result_json`` for one synthesis run id."""
    sql = "SELECT result_json FROM synthesis_runs WHERE id = ?;"
    try:
        with db.connection() as conn:
            row = conn.execute(sql, (int(run_id),)).fetchone()
        return str(row["result_json"]) if row is not None else None
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not load synthesis run {run_id}: {exc}") from exc


def get_latest_synthesis_run_for_session(
    db: Database,
    session_id: str | None,
) -> dict[str, Any] | None:
    """
    Return the newest synthesis run for ``session_id`` (or overall when None).

    Backs the assistant's review-context recall: after a literature review is
    generated in chat, follow-up turns can reload the paper set and topic.
    """
    if session_id:
        sql = """
            SELECT id, session_id, question, created_at, result_json
            FROM synthesis_runs
            WHERE session_id = ?
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT 1;
        """
        params: tuple[Any, ...] = (session_id,)
    else:
        sql = """
            SELECT id, session_id, question, created_at, result_json
            FROM synthesis_runs
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT 1;
        """
        params = ()
    try:
        with db.connection() as conn:
            row = conn.execute(sql, params).fetchone()
        return dict(row) if row is not None else None
    except sqlite3.Error as exc:
        raise DatabaseError(
            f"Could not load latest synthesis run for session: {exc}"
        ) from exc


def get_latest_synthesis_result_json_for_question(
    db: Database,
    question: str,
) -> str | None:
    """Return the newest persisted ``result_json`` for an exact research question."""
    sql = """
        SELECT result_json
        FROM synthesis_runs
        WHERE question = ?
        ORDER BY datetime(created_at) DESC, id DESC
        LIMIT 1;
    """
    try:
        with db.connection() as conn:
            row = conn.execute(sql, (question,)).fetchone()
        return str(row["result_json"]) if row is not None else None
    except sqlite3.Error as exc:
        raise DatabaseError(
            f"Could not load latest synthesis run for question: {exc}"
        ) from exc
