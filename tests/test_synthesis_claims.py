"""Unit tests for ``synthesis.claims.extract_claims`` (no live LLM calls)."""

from __future__ import annotations

from typing import Any

import pytest

from synthesis.claims import extract_claims
from synthesis.llm import SynthesisLLMError
from synthesis.schemas import ScoredPaper


def _make_paper(
    paper_id: str = "arxiv:1",
    *,
    title: str = "Flubber Nets",
    full_text: str | None = None,
    abstract: str | None = None,
) -> ScoredPaper:
    return ScoredPaper(
        paper_id=paper_id,
        title=title,
        authors=["A. Test"],
        year=2024,
        full_text=full_text,
        abstract=abstract,
    )


def test_extract_claims_empty_input_returns_empty() -> None:
    assert extract_claims([]) == []


def test_extract_claims_happy_path_all_grounded() -> None:
    full_text = (
        "We propose FlubberNet, a new architecture for widget recognition.\n"
        "On the WidgetBench dataset, FlubberNet achieves 92.4 percent accuracy.\n"
        "Training uses Adam with a learning rate of 1e-4 for 30 epochs.\n"
    )
    paper = _make_paper(full_text=full_text)

    def fake_llm(**_: Any) -> dict[str, Any]:
        return {
            "claims": [
                {
                    "claim": "FlubberNet is proposed for widget recognition.",
                    "evidence_quote": "We propose FlubberNet, a new architecture for widget recognition.",
                    "confidence": 0.9,
                },
                {
                    "claim": "Achieves 92.4 percent accuracy on WidgetBench.",
                    "evidence_quote": "FlubberNet achieves 92.4 percent accuracy",
                    "confidence": 0.85,
                },
                {
                    "claim": "Training uses Adam optimizer.",
                    "evidence_quote": "Training uses Adam with a learning rate of 1e-4",
                    "confidence": 0.8,
                },
            ]
        }

    records = extract_claims([paper], llm_call=fake_llm)
    assert len(records) == 3
    assert all(r.paper_id == "arxiv:1" for r in records)
    assert all(r.grounded for r in records)
    assert records[0].confidence == pytest.approx(0.9)
    assert records[1].confidence == pytest.approx(0.85)
    assert records[2].confidence == pytest.approx(0.8)


def test_extract_claims_ungrounded_quote_marks_grounded_false_and_halves_confidence() -> None:
    full_text = "FlubberNet achieves 92.4 percent accuracy on WidgetBench."
    paper = _make_paper(full_text=full_text)

    def fake_llm(**_: Any) -> dict[str, Any]:
        return {
            "claims": [
                {
                    "claim": "Something the paper never said.",
                    "evidence_quote": "completely fabricated quote not in the source at all",
                    "confidence": 0.8,
                },
            ]
        }

    records = extract_claims([paper], llm_call=fake_llm)
    assert len(records) == 1
    assert records[0].grounded is False
    assert records[0].confidence == pytest.approx(0.4)


def test_extract_claims_truncates_to_max_claims_per_paper() -> None:
    full_text = "alpha beta gamma delta epsilon zeta eta theta"
    paper = _make_paper(full_text=full_text)

    def fake_llm(**_: Any) -> dict[str, Any]:
        return {
            "claims": [
                {"claim": f"claim number {i}", "evidence_quote": tok, "confidence": 0.5}
                for i, tok in enumerate(
                    ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
                )
            ]
        }

    records = extract_claims([paper], max_claims_per_paper=2, llm_call=fake_llm)
    assert len(records) == 2


def test_extract_claims_llm_error_on_one_paper_does_not_abort_batch() -> None:
    paper_bad = _make_paper(paper_id="arxiv:bad", full_text="some text here")
    paper_good = _make_paper(
        paper_id="arxiv:good",
        full_text="The proposed model achieves state of the art performance.",
    )
    call_log: list[str] = []

    def fake_llm(**kwargs: Any) -> dict[str, Any]:
        user = kwargs.get("user", "")
        if "arxiv:bad" in user:
            call_log.append("bad")
            raise SynthesisLLMError("simulated failure")
        call_log.append("good")
        return {
            "claims": [
                {
                    "claim": "Model achieves SOTA.",
                    "evidence_quote": "The proposed model achieves state of the art performance.",
                    "confidence": 0.9,
                }
            ]
        }

    records = extract_claims([paper_bad, paper_good], llm_call=fake_llm)
    assert call_log == ["bad", "good"]
    assert len(records) == 1
    assert records[0].paper_id == "arxiv:good"
    assert records[0].grounded is True


def test_extract_claims_skips_paper_with_no_text_and_no_abstract() -> None:
    paper = _make_paper(paper_id="arxiv:empty", full_text=None, abstract=None)
    calls: list[dict[str, Any]] = []

    def fake_llm(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"claims": []}

    records = extract_claims([paper], llm_call=fake_llm)
    assert records == []
    assert calls == []


def test_extract_claims_falls_back_to_abstract_when_no_full_text() -> None:
    abstract = "We introduce QuxNet which improves accuracy by 5 percent on QuxBench."
    paper = _make_paper(paper_id="arxiv:q", full_text=None, abstract=abstract)

    def fake_llm(**_: Any) -> dict[str, Any]:
        return {
            "claims": [
                {
                    "claim": "QuxNet improves accuracy on QuxBench.",
                    "evidence_quote": "QuxNet which improves accuracy by 5 percent on QuxBench",
                    "confidence": 0.7,
                }
            ]
        }

    records = extract_claims([paper], llm_call=fake_llm)
    assert len(records) == 1
    assert records[0].grounded is True


def test_extract_claims_drops_empty_claim_or_quote() -> None:
    paper = _make_paper(full_text="Some text discussing flubbers and widgets.")

    def fake_llm(**_: Any) -> dict[str, Any]:
        return {
            "claims": [
                {"claim": "", "evidence_quote": "Some text", "confidence": 0.5},
                {"claim": "Valid claim about flubbers.", "evidence_quote": "   ", "confidence": 0.5},
                {
                    "claim": "Flubbers and widgets are discussed.",
                    "evidence_quote": "discussing flubbers and widgets",
                    "confidence": 0.5,
                },
            ]
        }

    records = extract_claims([paper], llm_call=fake_llm)
    assert len(records) == 1
    assert records[0].claim == "Flubbers and widgets are discussed."
