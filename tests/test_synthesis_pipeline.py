"""End-to-end tests for the LitSynth pipeline orchestrator with stubbed stages."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from db.database import Database
from db.init_db import initialize_schema
from db.queries import list_recent_synthesis_runs
from synthesis.pipeline import PipelineConfig, PipelineHooks, run_synthesis
from synthesis.schemas import (
    ClaimRecord,
    ContradictionPair,
    ResearchQuestion,
    ScoredPaper,
    SynthesisResult,
)
from tools.schemas import Paper


def _paper(pid: str, title: str = "T", year: int = 2023) -> Paper:
    return Paper(
        paper_id=pid,
        title=title,
        authors=["A. Author"],
        abstract="abs",
        api_source="arxiv",
        publication_date=date(year, 1, 1),
    )


def _scored(pid: str, score: float = 0.5) -> ScoredPaper:
    return ScoredPaper(
        paper_id=pid,
        title=f"Title {pid}",
        authors=["Alice Smith"],
        abstract="Short abstract.",
        year=2023,
        venue="ACL",
        url="https://example.org",
        api_source="arxiv",
        sections={},
        full_text="Short body.",
        has_pdf=False,
        relevance_score=score,
    )


def _stub_hooks(
    *,
    papers: list[Paper] | None = None,
    ranked: list[ScoredPaper] | None = None,
    claims: list[ClaimRecord] | None = None,
    contradictions: list[ContradictionPair] | None = None,
    review_text: str = "Sample [Smith et al. 2023] review.",
    citation_score: float = 1.0,
    citations_used: list[str] | None = None,
    hallucinated: list[str] | None = None,
) -> PipelineHooks:
    return PipelineHooks(
        decompose=lambda q, n=4: ResearchQuestion(question=q if len(q) >= 3 else f"{q}xx", sub_queries=[q]),
        retrieve=lambda rq, **kw: list(papers or []),
        fetch_parse=lambda raw, **kw: [_scored(p.paper_id) for p in (papers or [])],
        rank=lambda parsed, **kw: list(ranked if ranked is not None else parsed),
        extract_claims=lambda papers_, **kw: list(claims or []),
        detect_contradictions=lambda c, p, **kw: list(contradictions or []),
        build_prompt=lambda **kw: __import__("synthesis.prompt", fromlist=["SynthesisPrompt"]).SynthesisPrompt(
            system="s", user="u", expected_citations=["[Smith et al. 2023]"]
        ),
        generate=lambda prompt, **kw: review_text,
        validate_cites=lambda text, papers_: ([], list(citations_used or []), list(hallucinated or []), citation_score),
    )


def test_pipeline_empty_retrieval_returns_empty_result() -> None:
    hooks = _stub_hooks(papers=[])
    result = run_synthesis("anything", hooks=hooks)
    assert isinstance(result, SynthesisResult)
    assert result.papers == []
    assert result.confidence_score == 0.0
    assert "no papers" in result.review_text.lower()


def test_pipeline_full_run_threads_outputs() -> None:
    papers = [_paper("arxiv:1"), _paper("arxiv:2")]
    ranked = [_scored("arxiv:1", 0.9), _scored("arxiv:2", 0.7)]
    claims = [
        ClaimRecord(paper_id="arxiv:1", claim="claim one", evidence_quote="quote one", confidence=0.8, grounded=True),
        ClaimRecord(paper_id="arxiv:2", claim="claim two", evidence_quote="quote two", confidence=0.6, grounded=False),
    ]
    contradictions = [
        ContradictionPair(
            paper_a="arxiv:1",
            paper_b="arxiv:2",
            claim_a="claim one",
            claim_b="claim two",
            tension_type="contradiction",
            explanation="they disagree on scope",
        )
    ]
    hooks = _stub_hooks(
        papers=papers,
        ranked=ranked,
        claims=claims,
        contradictions=contradictions,
        review_text="The literature splits [Smith et al. 2023].",
        citation_score=1.0,
        citations_used=["arxiv:1"],
        hallucinated=[],
    )

    result = run_synthesis("question?", hooks=hooks)

    assert len(result.papers) == 2
    assert len(result.claims) == 2
    assert result.contradictions_found == 1
    assert result.citations_used == ["arxiv:1"]
    assert result.hallucinated_citations == []
    # confidence = citation_score(1.0) * grounded_fraction(0.5) = 0.5
    assert result.confidence_score == 0.5


def test_pipeline_confidence_penalizes_hallucinations() -> None:
    papers = [_paper("arxiv:1")]
    ranked = [_scored("arxiv:1")]
    claims = [
        ClaimRecord(paper_id="arxiv:1", claim="key claim", evidence_quote="key quote", confidence=0.9, grounded=True)
    ]
    hooks = _stub_hooks(
        papers=papers,
        ranked=ranked,
        claims=claims,
        contradictions=[],
        review_text="bla bla",
        citation_score=0.0,
        citations_used=[],
        hallucinated=["[Ghost 2024]"],
    )
    result = run_synthesis("question text", hooks=hooks)
    assert result.confidence_score == 0.0
    assert result.hallucinated_citations == ["[Ghost 2024]"]


def test_pipeline_persists_to_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "synth.sqlite3"
    monkeypatch.setenv("DATABASE_PATH", str(db_path))
    import config.config as cfg

    cfg.get_settings(reload=True)
    db = Database(db_path)
    initialize_schema(db)

    hooks = _stub_hooks(
        papers=[_paper("arxiv:1")],
        ranked=[_scored("arxiv:1")],
        claims=[
            ClaimRecord(
                paper_id="arxiv:1",
                claim="this is a grounded claim",
                evidence_quote="verbatim quote text",
                confidence=0.7,
                grounded=True,
            )
        ],
        contradictions=[],
        review_text="Body [Smith et al. 2023].",
        citation_score=1.0,
        citations_used=["arxiv:1"],
    )

    run_synthesis(
        "do GPUs help?",
        hooks=hooks,
        database=db,
        session_id="sess-1",
        config=PipelineConfig(word_budget=200, top_n_ranked=3),
    )

    rows = list_recent_synthesis_runs(db, limit=5)
    assert len(rows) == 1
    assert rows[0]["session_id"] == "sess-1"
    assert rows[0]["question"] == "do GPUs help?"
    assert rows[0]["contradictions_found"] == 0


def test_pipeline_progress_callback_invoked_per_stage() -> None:
    seen: list[str] = []
    hooks = _stub_hooks(
        papers=[_paper("arxiv:1")],
        ranked=[_scored("arxiv:1")],
    )
    run_synthesis("some question", hooks=hooks, progress=seen.append)
    # 9 progress steps fire on the happy path (decompose..validate).
    assert len(seen) >= 8
    assert any("decompos" in s.lower() for s in seen)
    assert any("validating" in s.lower() for s in seen)


def test_default_sources_is_arxiv_only() -> None:
    """Semantic Scholar is opt-in (rate-limit issues); arxiv is the default."""
    cfg = PipelineConfig()
    assert cfg.sources == ("arxiv",)


def test_pipeline_progress_callback_exceptions_are_swallowed() -> None:
    def boom(_label: str) -> None:
        raise RuntimeError("ui glitch")

    hooks = _stub_hooks(
        papers=[_paper("arxiv:1")],
        ranked=[_scored("arxiv:1")],
    )
    # Must not propagate.
    run_synthesis("some question", hooks=hooks, progress=boom)
