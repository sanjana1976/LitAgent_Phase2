"""Unit tests for :mod:`synthesis.validate_cites` (citation validator)."""

from __future__ import annotations

import pytest

from synthesis.schemas import ScoredPaper
from synthesis.validate_cites import validate_citations


def _make_paper(
    *,
    paper_id: str,
    authors: list[str],
    year: int,
    title: str = "Untitled",
) -> ScoredPaper:
    return ScoredPaper(
        paper_id=paper_id,
        title=title,
        authors=authors,
        year=year,
        api_source="arxiv",
    )


@pytest.fixture
def papers() -> list[ScoredPaper]:
    return [
        _make_paper(
            paper_id="arxiv:smith23",
            authors=["Alice Smith", "Bob Jones"],
            year=2023,
        ),
        _make_paper(
            paper_id="arxiv:doe24",
            authors=["Carol Doe"],
            year=2024,
        ),
    ]


def test_valid_smith_etal_2023_resolves(papers: list[ScoredPaper]) -> None:
    review = "Flubber works well [Smith et al. 2023]."
    checks, used, hallucinated, score = validate_citations(review, papers)

    assert len(checks) == 1
    assert checks[0].citation_key == "[Smith et al. 2023]"
    assert checks[0].is_valid is True
    assert checks[0].resolved_paper_id == "arxiv:smith23"
    assert used == ["arxiv:smith23"]
    assert hallucinated == []
    assert score == pytest.approx(1.0)


def test_unknown_citation_marked_hallucinated(papers: list[ScoredPaper]) -> None:
    review = "Old result [Nobody et al. 1999]."
    checks, used, hallucinated, score = validate_citations(review, papers)

    assert len(checks) == 1
    assert checks[0].is_valid is False
    assert checks[0].resolved_paper_id is None
    assert hallucinated == ["[Nobody et al. 1999]"]
    assert used == []
    assert score == pytest.approx(0.0)


def test_confidence_zero_when_all_hallucinated(papers: list[ScoredPaper]) -> None:
    review = "Two bad cites [Nobody 1999] and [Ghost et al. 1800]."
    _, used, hallucinated, score = validate_citations(review, papers)
    assert used == []
    assert len(hallucinated) == 2
    assert score == pytest.approx(0.0)


def test_confidence_one_when_all_valid(papers: list[ScoredPaper]) -> None:
    review = "First [Smith et al. 2023]; second [Doe 2024]."
    _, used, hallucinated, score = validate_citations(review, papers)
    assert hallucinated == []
    assert used == ["arxiv:smith23", "arxiv:doe24"]
    assert score == pytest.approx(1.0)


def test_duplicates_collapse_in_unique_counting(papers: list[ScoredPaper]) -> None:
    review = (
        "Once [Smith et al. 2023], twice [Smith et al. 2023], "
        "and a bad one [Nobody 1999]."
    )
    checks, used, hallucinated, score = validate_citations(review, papers)

    assert len(checks) == 2
    assert used == ["arxiv:smith23"]
    assert hallucinated == ["[Nobody 1999]"]
    assert score == pytest.approx(0.5)


def test_no_brackets_returns_empty(papers: list[ScoredPaper]) -> None:
    review = "This prose contains no bracketed citations whatsoever."
    checks, used, hallucinated, score = validate_citations(review, papers)
    assert checks == []
    assert used == []
    assert hallucinated == []
    assert score == 0.0


def test_tolerant_of_single_author_no_et_al(papers: list[ScoredPaper]) -> None:
    review = "Bare single-author cite [Doe 2024]."
    checks, used, _, score = validate_citations(review, papers)

    assert len(checks) == 1
    assert checks[0].is_valid is True
    assert checks[0].resolved_paper_id == "arxiv:doe24"
    assert used == ["arxiv:doe24"]
    assert score == pytest.approx(1.0)


def test_empty_review_text_returns_empty(papers: list[ScoredPaper]) -> None:
    checks, used, hallucinated, score = validate_citations("", papers)
    assert checks == []
    assert used == []
    assert hallucinated == []
    assert score == 0.0


def test_comma_separated_form_is_tolerated(papers: list[ScoredPaper]) -> None:
    review = "Comma form [Doe, 2024]."
    checks, used, _, score = validate_citations(review, papers)

    assert len(checks) == 1
    assert checks[0].is_valid is True
    assert used == ["arxiv:doe24"]
    assert score == pytest.approx(1.0)


def test_same_author_year_papers_disambiguate_and_both_resolve() -> None:
    # Two distinct papers that collapse to the same [Liu et al. 2025] key.
    liu_a = _make_paper(
        paper_id="arxiv:liu_a",
        authors=["Hao Liu", "Wei Chen"],
        year=2025,
        title="Long context survey",
    )
    liu_b = _make_paper(
        paper_id="arxiv:liu_b",
        authors=["Hao Liu", "Mei Wang"],
        year=2025,
        title="Retrieval benchmark",
    )
    papers = [liu_a, liu_b]

    review = (
        "Context windows grew sharply [Liu et al. 2025a], while a separate "
        "benchmark stress-tested retrieval [Liu et al. 2025b]."
    )
    checks, used, hallucinated, score = validate_citations(review, papers)

    assert hallucinated == []
    assert used == ["arxiv:liu_a", "arxiv:liu_b"]
    assert {c.resolved_paper_id for c in checks} == {"arxiv:liu_a", "arxiv:liu_b"}
    assert all(c.is_valid for c in checks)
    assert score == pytest.approx(1.0)
