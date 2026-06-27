"""Unit tests for query reformulation helpers."""

from __future__ import annotations

from synthesis.reformulate import (
    default_reformulate,
    llm_reformulate_query,
    resolve_reformulated_query,
)
from synthesis.state import SynthesisState


def test_default_reformulate_appends_suffix_by_attempt() -> None:
    state = SynthesisState(
        question="token compression for LLMs",
        sub_queries=["token compression methods"],
    )
    assert "methods benchmarks limitations" in default_reformulate(state)


def test_llm_reformulate_query_returns_cleaned_query() -> None:
    state = SynthesisState(
        question="token compression for LLMs",
        sub_queries=["token compression methods"],
    )

    def _fake_llm(**kwargs: object) -> dict[str, str]:
        return {"new_query": "KV cache pruning long context LLM"}

    result = llm_reformulate_query(
        state, "token compression methods", llm_call=_fake_llm
    )
    assert result == "KV cache pruning long context LLM"


def test_llm_reformulate_query_rejects_duplicate_query() -> None:
    state = SynthesisState(question="q", sub_queries=["same query"])

    def _fake_llm(**kwargs: object) -> dict[str, str]:
        return {"new_query": "same query"}

    assert llm_reformulate_query(state, "same query", llm_call=_fake_llm) is None


def test_resolve_reformulated_query_falls_back_when_llm_unusable() -> None:
    state = SynthesisState(question="q", sub_queries=["initial query"])

    def _bad_llm(**kwargs: object) -> dict[str, str]:
        return {"new_query": "x"}

    resolved = resolve_reformulated_query(
        state, "initial query", llm_call=_bad_llm
    )
    assert "methods benchmarks limitations" in resolved
