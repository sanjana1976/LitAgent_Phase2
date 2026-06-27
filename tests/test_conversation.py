from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from agent.conversation import ConversationManager
from db.database import Database
from db.init_db import initialize_schema
from db.queries import insert_reading_list


class _StubAgent:
    """Minimal agent double for deterministic unit tests."""

    def generate_reply(self, history):  # type: ignore[no-untyped-def]
        return AIMessage(content="stub-reply")


def test_conversation_manager_appends_turns(tmp_path: Path) -> None:
    db = Database(tmp_path / "c.sqlite")
    initialize_schema(db)
    manager = ConversationManager(_StubAgent(), database=db)

    reply = manager.append_turn("hi")
    assert reply.content == "stub-reply"
    assert len(manager.messages) == 2  # Human + assistant


def test_conversation_load_session_and_clear(tmp_path: Path) -> None:
    db = Database(tmp_path / "sess.sqlite")
    initialize_schema(db)
    manager = ConversationManager(None, database=db, session_id="load-me")
    manager.append_completed_turn(
        user_message="q1",
        assistant_message="a1",
        persist=True,
    )
    fresh = ConversationManager(None, database=db)
    loaded = fresh.load_session("load-me")
    assert loaded == 1
    assert len(fresh.messages) == 2

    deleted = fresh.clear_history(clear_persisted=True)
    assert deleted == 1
    assert len(fresh.messages) == 0


def test_conversation_context_snapshot_includes_lists(tmp_path: Path) -> None:
    db = Database(tmp_path / "ctx.sqlite")
    initialize_schema(db)
    insert_reading_list(db, "Paper stack", None)
    manager = ConversationManager(None, database=db, reading_list_context="thesis")
    snap = manager.context_snapshot()
    assert "thesis" in snap
    assert "reading_lists=" in snap
