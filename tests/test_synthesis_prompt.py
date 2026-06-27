"""Unit tests for :mod:`synthesis.prompt` (synthesis prompt builder)."""

from __future__ import annotations

import pytest

from synthesis.prompt import SynthesisPrompt, build_synthesis_prompt
from synthesis.schemas import ClaimRecord, ContradictionPair, ScoredPaper


def _make_paper(
    *,
    paper_id: str,
    title: str,
    authors: list[str],
    year: int,
    abstract: str | None = "Concise abstract.",
    api_source: str = "arxiv",
) -> ScoredPaper:
    return ScoredPaper(
        paper_id=paper_id,
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        api_source=api_source,
    )


@pytest.fixture
def papers() -> list[ScoredPaper]:
    return [
        _make_paper(
            paper_id="arxiv:1",
            title="Flubber Foundations",
            authors=["Alice Smith", "Bob Jones"],
            year=2023,
        ),
        _make_paper(
            paper_id="arxiv:2",
            title="Widgets at Scale",
            authors=["Carol Doe"],
            year=2024,
            abstract=None,
        ),
        _make_paper(
            paper_id="arxiv:3",
            title="Survey of Flubber",
            authors=["Dan Lee", "Erin Park", "Frank Kim"],
            year=2022,
            abstract="x" * 800,
        ),
    ]


@pytest.fixture
def claims() -> list[ClaimRecord]:
    return [
        ClaimRecord(
            paper_id="arxiv:1",
            claim="Flubber improves accuracy by 5%.",
            evidence_quote="Our flubber improves accuracy by 5%.",
            grounded=True,
        ),
        ClaimRecord(
            paper_id="arxiv:2",
            claim="Widgets scale linearly to 1B items.",
            evidence_quote="Widgets scale linearly up to one billion items.",
            grounded=False,
        ),
        ClaimRecord(
            paper_id="arxiv:1",
            claim="Flubber needs only 2GB of memory.",
            evidence_quote="Flubber uses 2GB of memory at inference time.",
            grounded=True,
        ),
    ]


@pytest.fixture
def contradictions() -> list[ContradictionPair]:
    return [
        ContradictionPair(
            paper_a="arxiv:1",
            paper_b="arxiv:2",
            claim_a="Flubber improves accuracy by 5%.",
            claim_b="Widgets do not improve accuracy meaningfully.",
            tension_type="contradiction",
            explanation="Disagree on whether the technique materially helps accuracy.",
        ),
    ]


def test_expected_citations_dedup_and_input_order(
    papers: list[ScoredPaper],
    claims: list[ClaimRecord],
    contradictions: list[ContradictionPair],
) -> None:
    duplicated = papers + [papers[0]]
    prompt = build_synthesis_prompt(
        question="What is flubber?",
        papers=duplicated,
        claims=claims,
        contradictions=contradictions,
    )
    assert prompt.expected_citations == [
        "[Smith et al. 2023]",
        "[Doe 2024]",
        "[Lee et al. 2022]",
    ]


def test_colliding_author_year_keys_are_disambiguated(
    claims: list[ClaimRecord],
) -> None:
    colliding = [
        _make_paper(
            paper_id="arxiv:liu_a",
            title="Long context survey",
            authors=["Hao Liu", "Wei Chen"],
            year=2025,
        ),
        _make_paper(
            paper_id="arxiv:liu_b",
            title="Retrieval benchmark",
            authors=["Hao Liu", "Mei Wang"],
            year=2025,
        ),
    ]
    prompt = build_synthesis_prompt(
        question="What is long-context retrieval?",
        papers=colliding,
        claims=[],
        contradictions=[],
    )
    # Both papers survive with distinct, resolvable keys (no silent drop).
    assert prompt.expected_citations == ["[Liu et al. 2025a]", "[Liu et al. 2025b]"]
    assert "[Liu et al. 2025a]" in prompt.user
    assert "[Liu et al. 2025b]" in prompt.user


def test_contradictions_appear_in_user_prompt(
    papers: list[ScoredPaper],
    claims: list[ClaimRecord],
    contradictions: list[ContradictionPair],
) -> None:
    prompt = build_synthesis_prompt(
        question="What is flubber?",
        papers=papers,
        claims=claims,
        contradictions=contradictions,
    )
    assert "[Smith et al. 2023] vs [Doe 2024]" in prompt.user
    assert "Disagree on whether" in prompt.user
    assert "[contradiction]" in prompt.user


def test_grounded_and_ungrounded_claims_are_tagged_correctly(
    papers: list[ScoredPaper],
    claims: list[ClaimRecord],
    contradictions: list[ContradictionPair],
) -> None:
    prompt = build_synthesis_prompt(
        question="What is flubber?",
        papers=papers,
        claims=claims,
        contradictions=contradictions,
    )
    grounded_line = "- Flubber improves accuracy by 5%."
    assert grounded_line in prompt.user
    assert f"{grounded_line} (UNGROUNDED)" not in prompt.user

    assert "Widgets scale linearly to 1B items. (UNGROUNDED)" in prompt.user


def test_word_budget_mentioned_in_system_prompt(
    papers: list[ScoredPaper],
    claims: list[ClaimRecord],
    contradictions: list[ContradictionPair],
) -> None:
    prompt = build_synthesis_prompt(
        question="What is flubber?",
        papers=papers,
        claims=claims,
        contradictions=contradictions,
        word_budget=750,
    )
    assert "750" in prompt.system
    assert "750" in prompt.user


def test_abstract_is_truncated_to_400_chars(
    papers: list[ScoredPaper],
    claims: list[ClaimRecord],
    contradictions: list[ContradictionPair],
) -> None:
    prompt = build_synthesis_prompt(
        question="What is flubber?",
        papers=papers,
        claims=claims,
        contradictions=contradictions,
    )
    long_run = "x" * 401
    assert long_run not in prompt.user
    assert "\u2026" in prompt.user


def test_returns_synthesis_prompt_dataclass(
    papers: list[ScoredPaper],
    claims: list[ClaimRecord],
    contradictions: list[ContradictionPair],
) -> None:
    prompt = build_synthesis_prompt(
        question="q",
        papers=papers,
        claims=claims,
        contradictions=contradictions,
    )
    assert isinstance(prompt, SynthesisPrompt)
    with pytest.raises(Exception):
        prompt.system = "mutated"  # type: ignore[misc]


def test_handles_empty_inputs() -> None:
    prompt = build_synthesis_prompt(
        question="empty case",
        papers=[],
        claims=[],
        contradictions=[],
    )
    assert prompt.expected_citations == []
    assert "(no papers provided)" in prompt.user
    assert "(no claims extracted)" in prompt.user
    assert "(no contradictions detected)" in prompt.user
