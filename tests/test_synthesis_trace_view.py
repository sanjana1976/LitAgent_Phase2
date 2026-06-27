"""Tests for Streamlit decision-trace formatting helpers."""

from __future__ import annotations

from synthesis.schemas import CitationCheck, ClaimRecord, ContradictionPair, ScoredPaper
from synthesis.state import Gap, SynthesisState
from synthesis.trace import DecisionStep, SearchParams, StepEffect
from synthesis.trace_view import (
    claim_rows,
    contradiction_rows,
    gap_rows,
    paper_rows,
    state_metrics,
    trace_rows,
)


def _claim(paper_id: str, *, grounded: bool = True) -> ClaimRecord:
    return ClaimRecord(
        paper_id=paper_id,
        claim="a sufficiently specific claim",
        evidence_quote="supporting quote",
        grounded=grounded,
        grounding_tier="abstract" if grounded else "none",
        supporting_paper_id=paper_id if grounded else None,
    )


def test_trace_rows_include_causal_and_effect_details() -> None:
    state = SynthesisState(question="long context retrieval")
    step = DecisionStep.start(
        action="search",
        params=SearchParams(query="long context retrieval", sources=["arxiv"]),
        trigger="working set has 0 parsed paper(s)",
        rationale="search for papers",
    )
    step.complete(
        result="ok",
        result_note="added one paper",
        effect=StepEffect(added_paper_ids=["arxiv:1"]),
        llm_calls=0,
        duration_ms=12,
    )
    state.log(step)

    rows = trace_rows(state)

    assert rows == [
        {
            "#": 1,
            "action": "search",
            "result": "ok",
            "parent": "",
            "trigger": "working set has 0 parsed paper(s)",
            "rationale": "search for papers",
            "result_note": "added one paper",
            "effect": "added papers: arxiv:1",
            "duration_ms": 12,
            "llm_calls": 0,
        }
    ]


def test_state_metrics_surface_dashboard_counts() -> None:
    state = SynthesisState(
        question="long context retrieval",
        papers=[ScoredPaper(paper_id="arxiv:1", title="A", text_tier="abstract")],
        claims=[_claim("arxiv:1", grounded=True), _claim("arxiv:1", grounded=False)],
        contradictions=[ContradictionPair(paper_a="arxiv:1", paper_b="arxiv:2", claim_a="a", claim_b="b")],
        citation_checks=[
            CitationCheck(citation_key="[A 2024]", resolved_paper_id="arxiv:1", is_valid=True),
            CitationCheck(citation_key="[Ghost 2099]", is_valid=False),
        ],
        hallucinated_citations=["[Ghost 2099]"],
    )
    state.gaps.append(Gap(kind="ungrounded_claim", description="missing support", status="open"))

    metrics = state_metrics(state)

    assert metrics["papers"] == 1
    assert metrics["claims"] == 2
    assert metrics["grounded_fraction"] == 0.5
    assert metrics["citation_validity"] == 0.5
    assert metrics["contradictions"] == 1
    assert metrics["open_gaps"] == 1
    assert metrics["hallucinated_citations"] == 1


def test_evidence_rows_include_grounding_and_stable_ids() -> None:
    paper = ScoredPaper(
        paper_id="arxiv:1",
        title="A",
        authors=["Ada Lovelace"],
        year=2024,
        abstract="abstract",
        text_tier="abstract",
        has_pdf=False,
        relevance_score=0.87654,
    )
    claim = _claim("arxiv:1")
    contradiction = ContradictionPair(
        paper_a="arxiv:1",
        paper_b="arxiv:2",
        claim_a="scaling helps",
        claim_b="scaling hurts",
    )
    gap = Gap(kind="ungrounded_claim", description="missing support", origin_claim_ref=claim.claim_id)
    state = SynthesisState(
        question="long context retrieval",
        papers=[paper],
        claims=[claim],
        contradictions=[contradiction],
        gaps=[gap],
    )

    assert paper_rows(state)[0]["text_tier"] == "abstract"
    assert paper_rows(state)[0]["citation_key"] == "[Lovelace 2024]"
    assert paper_rows(state)[0]["relevance_score"] == 0.8765
    assert claim_rows(state)[0]["grounding_tier"] == "abstract"
    assert claim_rows(state)[0]["supporting_paper_id"] == "arxiv:1"
    assert contradiction_rows(state)[0]["contradiction_id"] == contradiction.contradiction_id
    assert gap_rows(state)[0]["origin_claim_ref"] == claim.claim_id
