"""Tests for the controller working state (synthesis.state)."""

from __future__ import annotations

from synthesis.schemas import CitationCheck, ClaimRecord, ContradictionPair, ScoredPaper
from synthesis.state import Gap, SynthesisState
from synthesis.trace import DecisionStep, ReformulateParams, SearchParams


def _claim(paper_id: str, *, grounded: bool, tier: str) -> ClaimRecord:
    return ClaimRecord(
        paper_id=paper_id,
        claim="a sufficiently long claim about the method",
        evidence_quote="a verbatim quote from the paper text",
        grounded=grounded,
        grounding_tier=tier,
        supporting_paper_id=paper_id if grounded else None,
    )


def test_claim_id_is_auto_and_unique() -> None:
    a = _claim("arxiv:1", grounded=True, tier="full_text")
    b = _claim("arxiv:1", grounded=True, tier="full_text")
    assert a.claim_id and b.claim_id
    assert a.claim_id != b.claim_id


def test_derived_step_status_and_reformulation_count() -> None:
    state = SynthesisState(question="long context retrieval")
    assert state.current_step == 0
    assert state.status == "init"

    state.log(
        DecisionStep.start(
            action="search",
            params=SearchParams(query="long context retrieval", sources=["arxiv"]),
            trigger="initial question",
            rationale="first pass",
        )
    )
    state.log(
        DecisionStep.start(
            action="reformulate",
            params=ReformulateParams(original_query="long context retrieval", new_query="x"),
            trigger="only 2 papers",
            rationale="broaden",
        )
    )
    assert state.current_step == 2
    assert state.status == "reformulate"
    assert state.reformulation_count == 1

    state.terminal_reason = "synthesized"
    assert state.status == "done"


def test_grounded_fraction_and_tier_breakdown() -> None:
    state = SynthesisState(
        question="q",
        claims=[
            _claim("arxiv:1", grounded=True, tier="full_text"),
            _claim("arxiv:2", grounded=True, tier="abstract"),
            _claim("arxiv:3", grounded=False, tier="none"),
            _claim("arxiv:4", grounded=False, tier="none"),
        ],
    )
    assert state.grounded_fraction() == 0.5
    assert state.grounded_by_tier() == {"full_text": 1, "abstract": 1, "none": 2}


def test_grounded_fraction_empty_is_zero() -> None:
    assert SynthesisState(question="q").grounded_fraction() == 0.0


def test_citation_validity_and_citations_used_dedup_in_order() -> None:
    state = SynthesisState(
        question="q",
        citation_checks=[
            CitationCheck(citation_key="[Smith 2023]", resolved_paper_id="arxiv:1", is_valid=True),
            CitationCheck(citation_key="[Jones 2022]", resolved_paper_id="arxiv:2", is_valid=True),
            CitationCheck(citation_key="[Smith 2023]", resolved_paper_id="arxiv:1", is_valid=True),
            CitationCheck(citation_key="[Ghost 2099]", resolved_paper_id=None, is_valid=False),
        ],
    )
    assert state.citation_validity() == 0.75
    assert state.citations_used() == ["arxiv:1", "arxiv:2"]


def test_open_gaps_filters_by_lifecycle() -> None:
    state = SynthesisState(
        question="q",
        gaps=[
            Gap(kind="ungrounded_claim", description="c1 unsupported", origin_claim_ref="c1", status="open"),
            Gap(kind="coverage", description="missing theme", status="hunting"),
            Gap(kind="ungrounded_claim", description="c2 fixed", origin_claim_ref="c2", status="resolved"),
            Gap(kind="ungrounded_claim", description="c3 failed", origin_claim_ref="c3", status="flagged_unverified"),
        ],
    )
    assert {g.status for g in state.open_gaps} == {"open", "hunting"}
    assert len(state.open_gaps) == 2


def test_get_claim_and_paper_lookup() -> None:
    claim = _claim("arxiv:1", grounded=True, tier="full_text")
    paper = ScoredPaper(paper_id="arxiv:1", title="A", text_tier="full_text")
    state = SynthesisState(question="q", claims=[claim], papers=[paper])
    assert state.get_claim(claim.claim_id) is claim
    assert state.get_claim("missing") is None
    assert state.get_paper("arxiv:1") is paper
    assert state.get_paper("nope") is None


def test_to_result_builds_immutable_artifact_with_blended_legacy_score() -> None:
    state = SynthesisState(
        question="q",
        review_text="A review with [Smith 2023].",
        papers=[ScoredPaper(paper_id="arxiv:1", title="A", text_tier="full_text")],
        claims=[
            _claim("arxiv:1", grounded=True, tier="full_text"),
            _claim("arxiv:1", grounded=False, tier="none"),
        ],
        contradictions=[
            ContradictionPair(
                paper_a="arxiv:1",
                paper_b="arxiv:2",
                claim_a="claim a text long enough",
                claim_b="claim b text long enough",
            )
        ],
        citation_checks=[
            CitationCheck(citation_key="[Smith 2023]", resolved_paper_id="arxiv:1", is_valid=True),
        ],
        hallucinated_citations=["[Ghost 2099]"],
    )
    result = state.to_result()
    assert result.question == "q"
    assert result.review_text == "A review with [Smith 2023]."
    assert result.citations_used == ["arxiv:1"]
    assert result.hallucinated_citations == ["[Ghost 2099]"]
    assert result.contradictions_found == 1
    # citation_validity=1.0 * grounded_fraction=0.5 -> 0.5
    assert result.confidence_score == 0.5
