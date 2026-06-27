from __future__ import annotations

from pathlib import Path

from db.database import Database
from db.init_db import initialize_schema
from db.queries import (
    clear_conversation_session,
    fetch_conversation_by_session,
    fetch_recent_conversation_summaries,
    insert_conversation_turn,
    insert_permission_audit,
    insert_reading_list,
    list_recent_sessions,
)
from models.conversation_turn import ConversationTurnIn


def test_schema_and_conversation_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "store.sqlite3"
    db = Database(db_path)
    initialize_schema(db)

    turn = ConversationTurnIn(
        user_message="Hello",
        agent_response="Hi there",
        reading_list_context="cs-189",
        session_id="sess-a",
    )
    row_id = insert_conversation_turn(db, turn)
    assert row_id > 0

    rows = fetch_recent_conversation_summaries(db, limit=5)
    assert len(rows) == 1
    assert rows[0][1] == "Hello"
    assert rows[0][2] == "Hi there"

    by_session = fetch_conversation_by_session(db, "sess-a")
    assert len(by_session) == 1
    assert by_session[0][0] == "Hello"

    sessions = list_recent_sessions(db, limit=5)
    assert len(sessions) == 1
    assert sessions[0][0] == "sess-a"
    assert sessions[0][1] == 1

    deleted = clear_conversation_session(db, "sess-a")
    assert deleted == 1
    assert fetch_conversation_by_session(db, "sess-a") == []


def test_permission_audit_insert(tmp_path: Path) -> None:
    db = Database(tmp_path / "audit.sqlite3")
    initialize_schema(db)
    rid = insert_permission_audit(
        db,
        session_id="s1",
        tool_name="tool_search_arxiv",
        action="execute",
        allowed=True,
        needs_confirmation=False,
        user_decision=None,
        reason=None,
    )
    assert rid > 0


def test_reading_list_insert(tmp_path: Path) -> None:
    db = Database(tmp_path / "lists.sqlite3")
    initialize_schema(db)
    lid = insert_reading_list(db, "Review list", "For demo")
    assert lid > 0
