"""Tests for the agent-callable synthesis tool and registry/db wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from db.database import Database
from db.init_db import initialize_schema
from synthesis.schemas import ScoredPaper, SynthesisResult
from tools import synthesis_tools
from tools.context import clear_tool_caches
from tools.synthesis_tools import tool_synthesize_literature_review
from tools.tools_registry import TOOL_SPECS, list_tool_signatures


@pytest.fixture(autouse=True)
def _isolate_database(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the default database at a temp file and reset cached settings."""
    clear_tool_caches()
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "synth.sqlite3"))
    import config.config as cfg

    cfg.get_settings(reload=True)


def test_tool_is_registered_and_has_signature() -> None:
    names = {spec.name for spec in TOOL_SPECS}
    assert "tool_synthesize_literature_review" in names
    sigs = list_tool_signatures()
    assert "tool_synthesize_literature_review" in sigs
    assert "question" in sigs["tool_synthesize_literature_review"]


def test_tool_rejects_empty_question() -> None:
    payload = json.loads(tool_synthesize_literature_review("   "))
    assert "error" in payload


def test_tool_returns_compact_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """The synthesis tool runs the agentic controller and returns compact JSON."""

    fake_result = SynthesisResult(
        question="q",
        review_text="The literature [Smith et al. 2023] is split.",
        citations_used=["arxiv:1"],
        hallucinated_citations=[],
        contradictions_found=0,
        confidence_score=0.84,
        papers=[
            ScoredPaper(
                paper_id="arxiv:1",
                title="A long-context survey",
                authors=["Alice Smith"],
                abstract="x",
                year=2023,
                venue="ACL",
                url="https://example.org",
                api_source="arxiv",
                sections={},
                full_text="x",
                has_pdf=True,
                relevance_score=0.91,
            )
        ],
        claims=[],
        contradictions=[],
        citation_checks=[],
    )

    monkeypatch.setattr(
        synthesis_tools, "run_graph_synthesis", lambda q, **kw: fake_result
    )

    raw = tool_synthesize_literature_review("retrieval", word_budget=300, top_n=4)
    payload = json.loads(raw)
    assert payload["review_text"].startswith("The literature")
    assert payload["confidence_score"] == 0.84
    assert payload["papers"][0]["citation_key"] == "[Smith 2023]"
    assert payload["papers"][0]["has_pdf"] is True
    assert payload["citations_used"] == ["arxiv:1"]


def _persist_fake_run(db: Database, session_id: str | None, question: str) -> None:
    from db.queries import insert_synthesis_run

    result = SynthesisResult(
        question=question,
        review_text="A grounded review [Smith 2023].",
        citations_used=["arxiv:1"],
        hallucinated_citations=[],
        contradictions_found=0,
        confidence_score=0.9,
        papers=[
            ScoredPaper(
                paper_id="arxiv:1",
                title="A long-context survey",
                authors=["Alice Smith"],
                abstract="x",
                year=2023,
                api_source="arxiv",
            )
        ],
        claims=[],
        contradictions=[],
        citation_checks=[],
    )
    insert_synthesis_run(
        db,
        session_id=session_id,
        question=question,
        review_text=result.review_text,
        result_json=json.dumps(result.model_dump(mode="json")),
        confidence_score=result.confidence_score,
        contradictions_found=0,
        hallucinated_count=0,
    )


def test_review_context_tool_recalls_session_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from config.config import get_settings
    from tools.context import set_tool_session_id
    from tools.synthesis_tools import tool_get_review_context

    db = Database(get_settings().database_path)
    initialize_schema(db)
    _persist_fake_run(db, "other-session", "an older unrelated question")
    _persist_fake_run(db, "session-42", "long context retrieval")

    set_tool_session_id("session-42")
    payload = json.loads(tool_get_review_context())

    assert payload["question"] == "long context retrieval"
    assert payload["from_session"] == "session-42"
    assert payload["papers"][0]["paper_id"] == "arxiv:1"
    assert payload["papers"][0]["citation_key"] == "[Smith 2023]"


def test_review_context_tool_falls_back_to_latest_overall() -> None:
    from config.config import get_settings
    from tools.context import set_tool_session_id
    from tools.synthesis_tools import tool_get_review_context

    db = Database(get_settings().database_path)
    initialize_schema(db)
    _persist_fake_run(db, "someone-else", "the only review anywhere")

    set_tool_session_id("brand-new-session")
    payload = json.loads(tool_get_review_context())

    assert payload["question"] == "the only review anywhere"
    assert payload["from_session"] == "someone-else"


def test_review_context_tool_reports_missing_review() -> None:
    from config.config import get_settings
    from tools.context import set_tool_session_id
    from tools.synthesis_tools import tool_get_review_context

    db = Database(get_settings().database_path)
    initialize_schema(db)

    set_tool_session_id(None)
    payload = json.loads(tool_get_review_context())

    assert "error" in payload


def test_synthesis_runs_table_exists_after_init_db(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.sqlite3")
    initialize_schema(db)
    with db.connection() as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='synthesis_runs';"
        ).fetchone()
    assert row is not None
