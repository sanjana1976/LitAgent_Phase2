"""Tests for synthesis run persistence with decision traces."""

from __future__ import annotations

import json

from db.database import Database
from db.init_db import initialize_schema
from db.queries import get_synthesis_run_result_json
from synthesis.persistence import (
    build_persist_payload,
    load_synthesis_state_from_json,
    persist_synthesis_state,
    synthesis_result_from_json,
)
from synthesis.schemas import CitationCheck, ClaimRecord, ScoredPaper
from synthesis.state import SynthesisState
from synthesis.trace import DecisionStep, SearchParams


def test_build_persist_payload_includes_trace() -> None:
    state = SynthesisState(question="What is RAG?")
    step = DecisionStep.start(
        action="search",
        trigger="test",
        rationale="demo",
        params=SearchParams(kind="search", query="rag", sources=["arxiv"]),
    )
    step.complete(result="ok")
    state.log(step)
    state.review_text = "Review body."
    state.claims.append(
        ClaimRecord(
            paper_id="arxiv:1",
            claim="RAG helps.",
            evidence_quote="RAG helps grounding.",
            confidence=0.9,
            grounded=True,
        )
    )
    state.citation_checks.append(
        CitationCheck(citation_key="[A 2024]", resolved_paper_id="arxiv:1", is_valid=True)
    )
    state.papers.append(
        ScoredPaper(
            paper_id="arxiv:1",
            title="Demo",
            authors=["A"],
            year=2024,
            venue="Demo",
            url="https://example.org/1",
            relevance_score=0.9,
            has_pdf=True,
        )
    )
    state.terminal_reason = "synthesized"

    payload = build_persist_payload(state)

    assert len(payload["trace"]) == 1
    assert payload["trace"][0]["action"] == "search"
    assert payload["terminal_reason"] == "synthesized"


def test_persist_and_reload_round_trip(tmp_path) -> None:
    db = Database(tmp_path / "test.sqlite3")
    initialize_schema(db)

    state = SynthesisState(question="Trace round trip?")
    step = DecisionStep.start(
        action="search",
        trigger="start",
        rationale="split query",
        params=SearchParams(kind="search", query="trace", sources=["arxiv"]),
    )
    step.complete(result="ok")
    state.log(step)
    state.review_text = "Done."
    state.terminal_reason = "synthesized"

    persist_synthesis_state(db, "sess-1", state)
    raw = get_synthesis_run_result_json(db, 1)
    assert raw is not None

    loaded = load_synthesis_state_from_json(raw)
    assert loaded is not None
    assert loaded.question == "Trace round trip?"
    assert len(loaded.trace) == 1
    assert loaded.trace[0].action == "search"

    flat = synthesis_result_from_json(raw)
    assert flat.review_text == "Done."


def test_legacy_result_without_trace_returns_none() -> None:
    legacy = json.dumps(
        {
            "question": "legacy",
            "review_text": "old",
            "citations_used": [],
            "hallucinated_citations": [],
            "contradictions_found": 0,
            "confidence_score": 0.5,
            "papers": [],
            "claims": [],
            "contradictions": [],
            "citation_checks": [],
        }
    )
    assert load_synthesis_state_from_json(legacy) is None
    assert synthesis_result_from_json(legacy).question == "legacy"
