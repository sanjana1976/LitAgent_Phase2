"""Tests for the LLM relevance gate (:mod:`synthesis.relevance`) and its controller loop."""

from __future__ import annotations

from typing import Any

from synthesis.controller import ControllerConfig, ControllerHooks, SynthesisController
from synthesis.llm import SynthesisLLMError
from synthesis.relevance import llm_relevance_filter
from synthesis.schemas import ScoredPaper
from synthesis.state import SynthesisState


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


# --------------------------------------------------------------------------- #
# Controller gate loop
# --------------------------------------------------------------------------- #


def test_gate_loop_drops_papers_and_logs_traced_step() -> None:
    def gate(question: str, papers: list[ScoredPaper], **_kw: Any):
        kept = [p for p in papers if p.paper_id != "arxiv:off-topic"]
        return kept, {"arxiv:off-topic": 1}, True

    controller = SynthesisController(hooks=ControllerHooks(relevance_filter=gate))
    state = SynthesisState(
        question="q",
        papers=[_scored("arxiv:kept"), _scored("arxiv:off-topic")],
    )

    controller.run_relevance_gate_loop(state)

    assert [p.paper_id for p in state.papers] == ["arxiv:kept"]
    step = state.trace[-1]
    assert step.action == "filter_relevance"
    assert step.result == "ok"
    assert "arxiv:off-topic" in step.result_note
    assert step.llm_calls == 1


def test_gate_loop_noop_when_llm_unavailable() -> None:
    def gate(question: str, papers: list[ScoredPaper], **_kw: Any):
        return list(papers), {}, False

    controller = SynthesisController(hooks=ControllerHooks(relevance_filter=gate))
    state = SynthesisState(question="q", papers=[_scored("arxiv:1")])

    controller.run_relevance_gate_loop(state)

    assert [p.paper_id for p in state.papers] == ["arxiv:1"]
    assert state.trace[-1].action == "filter_relevance"
    assert state.trace[-1].result == "noop"


def test_gate_loop_failure_is_logged_not_raised() -> None:
    def gate(question: str, papers: list[ScoredPaper], **_kw: Any):
        raise RuntimeError("gate exploded")

    controller = SynthesisController(hooks=ControllerHooks(relevance_filter=gate))
    state = SynthesisState(question="q", papers=[_scored("arxiv:1")])

    controller.run_relevance_gate_loop(state)

    assert [p.paper_id for p in state.papers] == ["arxiv:1"]
    assert state.trace[-1].result == "failed"
    assert "gate exploded" in state.trace[-1].result_note


def test_gate_loop_skips_when_no_papers() -> None:
    controller = SynthesisController(
        hooks=ControllerHooks(relevance_filter=lambda q, p, **kw: (p, {}, True))
    )
    state = SynthesisState(question="q")

    controller.run_relevance_gate_loop(state)

    assert state.trace == []


def test_gate_loop_threshold_comes_from_config() -> None:
    seen: dict[str, Any] = {}

    def gate(question: str, papers: list[ScoredPaper], **kw: Any):
        seen.update(kw)
        return list(papers), {}, True

    controller = SynthesisController(
        config=ControllerConfig(relevance_keep_threshold=8),
        hooks=ControllerHooks(relevance_filter=gate),
    )
    state = SynthesisState(question="q", papers=[_scored("arxiv:1")])

    controller.run_relevance_gate_loop(state)

    assert seen.get("keep_threshold") == 8
    assert state.trace[-1].params.keep_threshold == 8
