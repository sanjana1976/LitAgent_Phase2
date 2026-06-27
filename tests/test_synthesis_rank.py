"""Unit tests for ``synthesis.rank`` (stage 4 of the LitSynth pipeline)."""

from __future__ import annotations

import pytest

from synthesis.rank import _cosine, _tokenize, rank_papers
from synthesis.schemas import ScoredPaper


def _paper(
    pid: str,
    *,
    title: str = "",
    abstract: str = "",
    full_text: str = "",
) -> ScoredPaper:
    return ScoredPaper(
        paper_id=pid,
        title=title,
        abstract=abstract,
        full_text=full_text,
    )


def test_tokenize_drops_stopwords_and_short_tokens() -> None:
    tokens = _tokenize("The Transformer is a model for NLP")
    assert "transformer" in tokens
    assert "model" in tokens
    assert "nlp" in tokens
    assert "the" not in tokens
    assert "a" not in tokens
    assert "is" not in tokens


def test_cosine_on_disjoint_vectors_is_zero() -> None:
    assert _cosine({"a": 1.0}, {"b": 1.0}) == 0.0


def test_cosine_on_identical_vectors_is_one() -> None:
    v = {"a": 1.0, "b": 2.0}
    assert _cosine(v, dict(v)) == pytest.approx(1.0)


def test_relevance_orders_question_terms_first() -> None:
    relevant = _paper(
        "rel",
        title="Transformer attention models for language understanding",
        abstract="We propose new transformer attention models for downstream language tasks.",
    )
    unrelated = _paper(
        "off",
        title="Medieval pottery analysis",
        abstract="Patterns in 12th century European pottery and trade routes.",
    )
    out = rank_papers([unrelated, relevant], question="transformer attention models")

    assert [p.paper_id for p in out] == ["rel", "off"]
    assert out[0].relevance_score > out[1].relevance_score
    assert out[0].relevance_score > 0.0


def test_identical_scores_preserve_input_order() -> None:
    a = _paper("a", title="alpha alpha", abstract="alpha")
    b = _paper("b", title="beta beta", abstract="beta")
    out = rank_papers([a, b], question="zeta zeta zeta")

    assert [p.paper_id for p in out] == ["a", "b"]
    assert out[0].relevance_score == 0.0
    assert out[1].relevance_score == 0.0


def test_empty_papers_returns_empty_list() -> None:
    assert rank_papers([], question="anything") == []


def test_top_n_truncates_returned_list() -> None:
    papers = [
        _paper(f"p{i}", title="transformer attention", abstract="transformer attention")
        for i in range(10)
    ]
    out = rank_papers(papers, question="transformer attention", top_n=3)
    assert len(out) == 3


def test_empty_question_returns_input_order_truncated_with_zero_score() -> None:
    papers = [_paper(f"p{i}", title="content", abstract="content") for i in range(5)]
    out = rank_papers(papers, question="", top_n=3)

    assert [p.paper_id for p in out] == ["p0", "p1", "p2"]
    assert all(p.relevance_score == 0.0 for p in out)


def test_question_with_only_stopwords_returns_input_order() -> None:
    papers = [_paper(f"p{i}", title="t", abstract="a") for i in range(4)]
    out = rank_papers(papers, question="the a an of in", top_n=2)
    assert [p.paper_id for p in out] == ["p0", "p1"]


def test_rank_does_not_mutate_inputs() -> None:
    papers = [
        _paper("a", title="transformer attention", abstract="transformer"),
        _paper("b", title="unrelated", abstract="unrelated"),
    ]
    original_scores = [p.relevance_score for p in papers]
    rank_papers(papers, question="transformer attention")
    assert [p.relevance_score for p in papers] == original_scores
