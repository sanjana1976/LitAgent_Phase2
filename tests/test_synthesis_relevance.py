"""Tests for the LLM relevance gate (:mod:`synthesis.relevance`).

Graph-level gate behavior (traced step, active-set updates, config threshold
plumbing) is covered in ``test_synthesis_graph.py``.
"""

from __future__ import annotations

from typing import Any

from synthesis.llm import SynthesisLLMError
from synthesis.relevance import llm_relevance_filter
from synthesis.schemas import ScoredPaper


def _scored(pid: str, title: str = "Paper", abstract: str = "An abstract.") -> ScoredPaper:
    return ScoredPaper(
        paper_id=pid,
        title=title,
        abstract=abstract,
        text_tier="abstract",
    )


def _stub_llm(payload: dict[str, Any]):
    def _call(**_kwargs: Any) -> dict[str, Any]:
        return payload

    return _call


def test_relevance_filter_drops_low_scoring_papers() -> None:
    papers = [_scored("arxiv:on-topic"), _scored("arxiv:off-topic")]
    payload = {
        "scores": [
            {"paper_id": "arxiv:on-topic", "score": 9},
            {"paper_id": "arxiv:off-topic", "score": 2},
        ]
    }

    kept, scores, used_llm = llm_relevance_filter(
        "long context retrieval", papers, llm_call=_stub_llm(payload)
    )

    assert used_llm is True
    assert [p.paper_id for p in kept] == ["arxiv:on-topic"]
    assert scores == {"arxiv:on-topic": 9, "arxiv:off-topic": 2}


def test_relevance_filter_keeps_unscored_papers() -> None:
    papers = [_scored("arxiv:scored"), _scored("arxiv:omitted")]
    payload = {"scores": [{"paper_id": "arxiv:scored", "score": 8}]}

    kept, _scores, used_llm = llm_relevance_filter(
        "q", papers, llm_call=_stub_llm(payload)
    )

    assert used_llm is True
    assert {p.paper_id for p in kept} == {"arxiv:scored", "arxiv:omitted"}


def test_relevance_filter_fail_soft_on_llm_error() -> None:
    papers = [_scored("arxiv:1"), _scored("arxiv:2")]

    def _boom(**_kwargs: Any) -> dict[str, Any]:
        raise SynthesisLLMError("offline")

    kept, scores, used_llm = llm_relevance_filter("q", papers, llm_call=_boom)

    assert used_llm is False
    assert scores == {}
    assert [p.paper_id for p in kept] == ["arxiv:1", "arxiv:2"]


def test_relevance_filter_fail_soft_on_garbage_payload() -> None:
    papers = [_scored("arxiv:1")]

    kept, _scores, used_llm = llm_relevance_filter(
        "q", papers, llm_call=_stub_llm({"scores": "not-a-list"})
    )

    assert used_llm is False
    assert [p.paper_id for p in kept] == ["arxiv:1"]


def test_relevance_filter_clamps_scores_and_threshold() -> None:
    papers = [_scored("arxiv:huge"), _scored("arxiv:negative")]
    payload = {
        "scores": [
            {"paper_id": "arxiv:huge", "score": 42},
            {"paper_id": "arxiv:negative", "score": -3},
        ]
    }

    kept, scores, _used = llm_relevance_filter(
        "q", papers, keep_threshold=6, llm_call=_stub_llm(payload)
    )

    assert scores == {"arxiv:huge": 10, "arxiv:negative": 0}
    assert [p.paper_id for p in kept] == ["arxiv:huge"]


def test_relevance_filter_empty_input() -> None:
    kept, scores, used_llm = llm_relevance_filter("q", [], llm_call=_stub_llm({}))
    assert kept == []
    assert scores == {}
    assert used_llm is False
