"""Tests for chat session context propagation into synthesis tools."""

from __future__ import annotations

import json
from unittest.mock import patch

from tools.context import clear_tool_caches, get_tool_session_id, set_tool_session_id
from tools.synthesis_tools import tool_synthesize_literature_review


def test_tool_session_context_round_trip() -> None:
    clear_tool_caches()
    assert get_tool_session_id() is None
    set_tool_session_id("chat-session-42")
    assert get_tool_session_id() == "chat-session-42"
    clear_tool_caches()
    assert get_tool_session_id() is None


def test_synthesis_tool_passes_session_id_to_controller() -> None:
    clear_tool_caches()
    set_tool_session_id("chat-session-99")

    captured: dict[str, str | None] = {}

    def _fake_run(question: str, *, config, database, session_id, progress=None):
        captured["session_id"] = session_id
        from synthesis.schemas import SynthesisResult

        return SynthesisResult(
            question=question,
            review_text="ok",
            confidence_score=1.0,
        )

    with patch("tools.synthesis_tools.run_graph_synthesis", side_effect=_fake_run):
        with patch("tools.synthesis_tools.get_default_database", side_effect=RuntimeError("no db")):
            payload = json.loads(tool_synthesize_literature_review("What is RAG?"))

    assert payload["question"] == "What is RAG?"
    assert captured["session_id"] == "chat-session-99"
    clear_tool_caches()
