"""Unit tests for ``synthesis.contradictions.detect_contradictions``."""

from __future__ import annotations

from typing import Any

from synthesis.contradictions import detect_contradictions
from synthesis.llm import SynthesisLLMError
from synthesis.schemas import ClaimRecord, ScoredPaper


def _paper(paper_id: str) -> ScoredPaper:
    return ScoredPaper(
        paper_id=paper_id,
        title=f"Title {paper_id}",
        authors=["A. Test"],
        year=2024,
    )


def _claim(paper_id: str, text: str) -> ClaimRecord:
    return ClaimRecord(
        paper_id=paper_id,
        claim=text,
        evidence_quote="some quote here",
        confidence=0.8,
        grounded=True,
    )


def test_detect_contradictions_happy_path_returns_two_pairs() -> None:
    papers = [_paper("arxiv:1"), _paper("arxiv:2"), _paper("arxiv:3")]
    claim_text_1 = "FlubberNet achieves 92.4 percent accuracy on WidgetBench"
    claim_text_2 = "FlubberNet achieves only 78 percent accuracy on WidgetBench"
    claim_text_3 = "Dropout regularization significantly reduces overfitting in our setup"
    claim_text_4 = "Dropout regularization had no measurable effect on overfitting"
    claims = [
        _claim("arxiv:1", claim_text_1),
        _claim("arxiv:2", claim_text_2),
        _claim("arxiv:3", claim_text_3),
        _claim("arxiv:1", claim_text_4),
    ]

    def fake_llm(**_: Any) -> dict[str, Any]:
        return {
            "contradictions": [
                {
                    "paper_a": "arxiv:1",
                    "paper_b": "arxiv:2",
                    "claim_a": claim_text_1,
                    "claim_b": claim_text_2,
                    "tension_type": "contradiction",
                    "explanation": "Different accuracy figures reported on the same benchmark.",
                },
                {
                    "paper_a": "arxiv:3",
                    "paper_b": "arxiv:1",
                    "claim_a": claim_text_3,
                    "claim_b": claim_text_4,
                    "tension_type": "methodology",
                    "explanation": "Disagree on whether dropout helps.",
                },
            ]
        }

    pairs = detect_contradictions(claims, papers, llm_call=fake_llm)
    assert len(pairs) == 2
    pair_keys = {tuple(sorted((p.paper_a, p.paper_b))) for p in pairs}
    assert ("arxiv:1", "arxiv:2") in pair_keys
    assert ("arxiv:1", "arxiv:3") in pair_keys


def test_detect_contradictions_drops_unknown_paper_id() -> None:
    papers = [_paper("arxiv:1"), _paper("arxiv:2")]
    real_a = "FlubberNet achieves 92.4 percent accuracy on WidgetBench"
    real_b = "FlubberNet achieves only 78 percent on WidgetBench"
    claims = [_claim("arxiv:1", real_a), _claim("arxiv:2", real_b)]

    def fake_llm(**_: Any) -> dict[str, Any]:
        return {
            "contradictions": [
                {
                    "paper_a": "arxiv:1",
                    "paper_b": "arxiv:999-imaginary",
                    "claim_a": real_a,
                    "claim_b": real_b,
                    "tension_type": "contradiction",
                    "explanation": "hallucinated paper id",
                },
            ]
        }

    pairs = detect_contradictions(claims, papers, llm_call=fake_llm)
    assert pairs == []


def test_detect_contradictions_drops_self_contradiction() -> None:
    papers = [_paper("arxiv:1"), _paper("arxiv:2")]
    text_a = "FlubberNet achieves 92.4 percent accuracy on WidgetBench"
    text_b = "FlubberNet achieves 78 percent accuracy on WidgetBench"
    claims = [_claim("arxiv:1", text_a), _claim("arxiv:2", text_b)]

    def fake_llm(**_: Any) -> dict[str, Any]:
        return {
            "contradictions": [
                {
                    "paper_a": "arxiv:1",
                    "paper_b": "arxiv:1",
                    "claim_a": text_a,
                    "claim_b": text_a,
                    "tension_type": "contradiction",
                    "explanation": "self",
                }
            ]
        }

    pairs = detect_contradictions(claims, papers, llm_call=fake_llm)
    assert pairs == []


def test_detect_contradictions_deduplicates_symmetric_pairs() -> None:
    papers = [_paper("arxiv:1"), _paper("arxiv:2")]
    text_a = "FlubberNet achieves 92.4 percent accuracy on WidgetBench"
    text_b = "FlubberNet achieves only 78 percent accuracy on WidgetBench"
    claims = [_claim("arxiv:1", text_a), _claim("arxiv:2", text_b)]

    def fake_llm(**_: Any) -> dict[str, Any]:
        return {
            "contradictions": [
                {
                    "paper_a": "arxiv:1",
                    "paper_b": "arxiv:2",
                    "claim_a": text_a,
                    "claim_b": text_b,
                    "tension_type": "contradiction",
                    "explanation": "first ordering",
                },
                {
                    "paper_a": "arxiv:2",
                    "paper_b": "arxiv:1",
                    "claim_a": text_b,
                    "claim_b": text_a,
                    "tension_type": "contradiction",
                    "explanation": "reversed ordering, same pair",
                },
            ]
        }

    pairs = detect_contradictions(claims, papers, llm_call=fake_llm)
    assert len(pairs) == 1


def test_detect_contradictions_returns_empty_without_llm_call_when_one_paper() -> None:
    papers = [_paper("arxiv:1")]
    claims = [
        _claim("arxiv:1", "First claim with enough characters to satisfy matchers"),
        _claim("arxiv:1", "Second claim with enough characters to satisfy matchers"),
    ]
    calls: list[dict[str, Any]] = []

    def fake_llm(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"contradictions": []}

    pairs = detect_contradictions(claims, papers, llm_call=fake_llm)
    assert pairs == []
    assert calls == [], "LLM must not be called when fewer than 2 distinct papers have claims"


def test_detect_contradictions_swallows_llm_error_and_returns_empty() -> None:
    papers = [_paper("arxiv:1"), _paper("arxiv:2")]
    claims = [
        _claim("arxiv:1", "FlubberNet achieves 92.4 percent accuracy on WidgetBench"),
        _claim("arxiv:2", "FlubberNet achieves 78 percent accuracy on WidgetBench"),
    ]

    def boom(**_: Any) -> dict[str, Any]:
        raise SynthesisLLMError("simulated outage")

    pairs = detect_contradictions(claims, papers, llm_call=boom)
    assert pairs == []


def test_detect_contradictions_drops_pair_with_fabricated_claim_text() -> None:
    papers = [_paper("arxiv:1"), _paper("arxiv:2")]
    real_a = "FlubberNet achieves 92.4 percent accuracy on WidgetBench"
    real_b = "FlubberNet achieves 78 percent accuracy on WidgetBench"
    claims = [_claim("arxiv:1", real_a), _claim("arxiv:2", real_b)]

    def fake_llm(**_: Any) -> dict[str, Any]:
        return {
            "contradictions": [
                {
                    "paper_a": "arxiv:1",
                    "paper_b": "arxiv:2",
                    "claim_a": "totally invented sentence with nothing in common at all",
                    "claim_b": "another invented sentence completely unrelated to reality",
                    "tension_type": "contradiction",
                    "explanation": "hallucinated claims",
                }
            ]
        }

    pairs = detect_contradictions(claims, papers, llm_call=fake_llm)
    assert pairs == []


def test_detect_contradictions_truncates_to_max_pairs() -> None:
    papers = [_paper(f"arxiv:{i}") for i in range(5)]
    base = "FlubberNet achieves a specific accuracy figure on the WidgetBench dataset"
    claims = [_claim(f"arxiv:{i}", f"{base} number {i}") for i in range(5)]

    def fake_llm(**_: Any) -> dict[str, Any]:
        return {
            "contradictions": [
                {
                    "paper_a": "arxiv:0",
                    "paper_b": "arxiv:1",
                    "claim_a": f"{base} number 0",
                    "claim_b": f"{base} number 1",
                    "tension_type": "contradiction",
                    "explanation": "x",
                },
                {
                    "paper_a": "arxiv:0",
                    "paper_b": "arxiv:2",
                    "claim_a": f"{base} number 0",
                    "claim_b": f"{base} number 2",
                    "tension_type": "contradiction",
                    "explanation": "x",
                },
                {
                    "paper_a": "arxiv:0",
                    "paper_b": "arxiv:3",
                    "claim_a": f"{base} number 0",
                    "claim_b": f"{base} number 3",
                    "tension_type": "contradiction",
                    "explanation": "x",
                },
            ]
        }

    pairs = detect_contradictions(claims, papers, max_pairs=2, llm_call=fake_llm)
    assert len(pairs) == 2
