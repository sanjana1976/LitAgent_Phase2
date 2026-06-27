"""Unit tests for :mod:`synthesis.decompose` (stage 1 of the LitSynth pipeline)."""

from __future__ import annotations

from typing import Any

import pytest

from synthesis.decompose import decompose_question
from synthesis.llm import SynthesisLLMError
from synthesis.schemas import ResearchQuestion


def _make_llm_stub(payload: dict[str, Any]):
    """Build an ``llm_call`` replacement that returns a fixed JSON payload."""

    def _stub(**_kwargs: Any) -> dict[str, Any]:
        return payload

    return _stub


def test_decompose_happy_path_returns_sub_queries() -> None:
    payload = {
        "sub_queries": [
            "retrieval augmented generation methods",
            "RAG benchmarks and datasets",
            "limitations of RAG systems",
            "recent advances in RAG architectures",
        ]
    }
    rq = decompose_question(
        "How does retrieval augmented generation work?",
        llm_call=_make_llm_stub(payload),
    )

    assert isinstance(rq, ResearchQuestion)
    assert rq.question == "How does retrieval augmented generation work?"
    assert rq.sub_queries == payload["sub_queries"]


def test_decompose_falls_back_when_llm_raises() -> None:
    def _boom(**_kwargs: Any) -> dict[str, Any]:
        raise SynthesisLLMError("offline")

    rq = decompose_question("graph neural networks", llm_call=_boom)

    assert isinstance(rq, ResearchQuestion)
    assert len(rq.sub_queries) >= 2
    assert "graph neural networks" in rq.sub_queries
    assert any("methods" in s for s in rq.sub_queries)
    assert any("benchmarks" in s for s in rq.sub_queries)


def test_decompose_falls_back_on_garbage_payload() -> None:
    rq = decompose_question(
        "diffusion models for audio",
        llm_call=_make_llm_stub({"unexpected": "shape", "sub_queries": "not-a-list"}),
    )

    assert len(rq.sub_queries) >= 2
    assert any(s.startswith("diffusion models for audio") for s in rq.sub_queries)


def test_decompose_falls_back_when_fewer_than_two_sub_queries() -> None:
    rq = decompose_question(
        "self supervised learning",
        llm_call=_make_llm_stub({"sub_queries": ["only one angle"]}),
    )

    assert len(rq.sub_queries) >= 2
    assert "only one angle" not in {s.lower() for s in rq.sub_queries}


def test_decompose_dedupes_case_variants() -> None:
    payload = {
        "sub_queries": [
            "Vector Databases",
            "vector databases",
            "VECTOR DATABASES",
            "vector databases for RAG",
            "scaling vector databases",
        ]
    }
    rq = decompose_question("vector databases", llm_call=_make_llm_stub(payload))

    lowered = [s.lower() for s in rq.sub_queries]
    assert len(set(lowered)) == len(rq.sub_queries)
    assert "vector databases" in lowered


def test_decompose_clamps_to_n() -> None:
    payload = {
        "sub_queries": [
            "angle one",
            "angle two",
            "angle three",
            "angle four",
            "angle five",
            "angle six",
        ]
    }
    rq = decompose_question("some question", n=3, llm_call=_make_llm_stub(payload))

    assert len(rq.sub_queries) == 3
    assert rq.sub_queries == ["angle one", "angle two", "angle three"]


def test_decompose_strips_whitespace_and_empties() -> None:
    payload = {
        "sub_queries": [
            "  padded angle  ",
            "",
            "   ",
            "second angle",
            None,
        ]
    }
    rq = decompose_question("query", llm_call=_make_llm_stub(payload))

    assert "padded angle" in rq.sub_queries
    assert "second angle" in rq.sub_queries
    assert all(s == s.strip() and s for s in rq.sub_queries)


def test_decompose_uses_default_llm_call_via_monkeypatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"sub_queries": ["alpha", "beta", "gamma"]}

    monkeypatch.setattr("synthesis.decompose._default_llm_call", _fake)
    rq = decompose_question("default-llm-path question")

    assert rq.sub_queries == ["alpha", "beta", "gamma"]
    assert "system" in captured and "user" in captured
    assert "default-llm-path question" in captured["user"]
