"""Tests for the critic stage (:mod:`synthesis.critic`)."""

from __future__ import annotations

from typing import Any

from synthesis.critic import critique_review
from synthesis.llm import SynthesisLLMError
from synthesis.schemas import ClaimRecord, ScoredPaper


_REVIEW = (
    "Long-context retrieval has advanced rapidly [Smith 2023]. "
    "Reportedly, attention mechanisms double throughput on all benchmarks. "
    "The field remains contested."
)


def _paper(pid: str = "arxiv:1") -> ScoredPaper:
    return ScoredPaper(
        paper_id=pid,
        title="A Survey",
        authors=["Alice Smith"],
        abstract="x",
        year=2023,
        text_tier="abstract",
    )


def _claim(paper_id: str = "arxiv:1") -> ClaimRecord:
    return ClaimRecord(
        paper_id=paper_id,
        claim="a sufficiently long paper-specific claim",
        evidence_quote="a verbatim evidence quote",
        grounded=True,
        grounding_tier="abstract",
    )


def _stub_llm(payload: dict[str, Any]):
    def _call(**_kwargs: Any) -> dict[str, Any]:
        return payload

    return _call


def test_valid_excerpt_objection_is_kept() -> None:
    payload = {
        "objections": [
            {
                "excerpt": "attention mechanisms double throughput on all benchmarks",
                "problem": "No extracted claim supports this number.",
            }
        ]
    }
    objections, used_llm = critique_review(
        "q", _REVIEW, [_paper()], [_claim()], [], [], llm_call=_stub_llm(payload)
    )
    assert used_llm is True
    assert len(objections) == 1
    assert "attention mechanisms double throughput" in objections[0]


def test_fabricated_excerpt_is_dropped() -> None:
    payload = {
        "objections": [
            {
                "excerpt": "this sentence never appears in the draft anywhere",
                "problem": "Fabricated by the critic.",
            }
        ]
    }
    objections, used_llm = critique_review(
        "q", _REVIEW, [_paper()], [_claim()], [], [], llm_call=_stub_llm(payload)
    )
    assert used_llm is True
    assert objections == []


def test_hallucinated_citations_become_auto_objections_even_when_llm_fails() -> None:
    def _boom(**_kw: Any) -> dict[str, Any]:
        raise SynthesisLLMError("offline")

    objections, used_llm = critique_review(
        "q",
        _REVIEW,
        [_paper()],
        [_claim()],
        [],
        ["[Ghost et al. 2020]"],
        llm_call=_boom,
    )
    assert used_llm is True  # the deterministic signal still counts
    assert len(objections) == 1
    assert "[Ghost et al. 2020]" in objections[0]


def test_llm_failure_with_clean_citations_is_noop() -> None:
    def _boom(**_kw: Any) -> dict[str, Any]:
        raise SynthesisLLMError("offline")

    objections, used_llm = critique_review(
        "q", _REVIEW, [_paper()], [_claim()], [], [], llm_call=_boom
    )
    assert objections == []
    assert used_llm is False


def test_empty_review_short_circuits() -> None:
    objections, used_llm = critique_review(
        "q", "  ", [_paper()], [_claim()], [], ["[Ghost 2020]"], llm_call=_stub_llm({})
    )
    assert objections == []
    assert used_llm is False


def test_garbage_payload_yields_no_objections() -> None:
    objections, used_llm = critique_review(
        "q",
        _REVIEW,
        [_paper()],
        [_claim()],
        [],
        [],
        llm_call=_stub_llm({"objections": "not-a-list"}),
    )
    assert objections == []
    assert used_llm is True
