"""Unit tests for :mod:`synthesis.retrieve` (stage 2 of the LitSynth pipeline)."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from synthesis.retrieve import retrieve_papers
from synthesis.schemas import ResearchQuestion
from tools.schemas import Paper


def _paper(
    paper_id: str,
    *,
    title: str = "Untitled",
    api_source: str = "arxiv",
    doi: str | None = None,
    abstract: str | None = "An abstract.",
    citation_count: int | None = None,
    publication_date: date | None = None,
    authors: list[str] | None = None,
) -> Paper:
    """Tiny ``Paper`` factory shared across the retrieve tests."""
    return Paper(
        paper_id=paper_id,
        title=title,
        api_source=api_source,
        doi=doi,
        abstract=abstract,
        citation_count=citation_count,
        publication_date=publication_date,
        authors=authors or [],
    )


def _make_search(results_per_source: dict[str, list[Paper]]):
    """Return a dict of source->callable that yields canned ``Paper`` lists."""

    callables: dict[str, Any] = {}
    for source, results in results_per_source.items():

        def _fn(_q: str, _filters: dict[str, Any] | None = None, _r=results) -> list[Paper]:
            return list(_r)

        callables[source] = _fn
    return callables


def test_retrieve_dedupes_by_doi() -> None:
    rq = ResearchQuestion(question="rag systems", sub_queries=["rag systems"])
    duplicate_doi = "10.1234/ABC"
    fakes = _make_search(
        {
            "arxiv": [
                _paper("arxiv:1", title="A paper", doi=duplicate_doi.lower()),
            ],
            "semantic_scholar": [
                _paper("s2:1", title="Same Paper, Different Title", doi=duplicate_doi),
            ],
        }
    )

    papers = retrieve_papers(
        rq, sources=("arxiv", "semantic_scholar"), search_callables=fakes
    )

    assert len(papers) == 1
    assert papers[0].doi is not None
    assert papers[0].doi.lower() == "10.1234/abc"


def test_retrieve_dedupes_arxiv_versions() -> None:
    rq = ResearchQuestion(question="query", sub_queries=["q one", "q two"])
    fakes = _make_search(
        {
            "arxiv": [
                _paper("arxiv:1234.5678v1", title="Versioned Paper"),
                _paper("arxiv:1234.5678v2", title="Versioned Paper Revised"),
                _paper("arxiv:9999.0000v3", title="Other Paper"),
            ],
        }
    )

    papers = retrieve_papers(rq, sources=("arxiv",), search_callables=fakes)

    paper_ids = {p.paper_id for p in papers}
    assert len(papers) == 2
    assert "arxiv:9999.0000v3" in paper_ids
    assert sum(1 for p in papers if "1234.5678" in p.paper_id) == 1


def test_retrieve_dedupes_by_normalized_title() -> None:
    rq = ResearchQuestion(question="query", sub_queries=["query"])
    fakes = _make_search(
        {
            "dblp": [
                _paper(
                    "dblp:x",
                    title="A Survey of Vector Databases!",
                    api_source="dblp",
                ),
                _paper(
                    "dblp:y",
                    title="a survey of   vector databases",
                    api_source="dblp",
                ),
                _paper(
                    "dblp:z",
                    title="Different Title Entirely",
                    api_source="dblp",
                ),
            ],
        }
    )

    papers = retrieve_papers(rq, sources=("dblp",), search_callables=fakes)

    assert len(papers) == 2
    titles_lower = {(p.title or "").lower() for p in papers}
    assert any("different title entirely" in t for t in titles_lower)


def test_retrieve_continues_when_one_provider_raises() -> None:
    rq = ResearchQuestion(question="query", sub_queries=["query"])

    def _arxiv_ok(_q: str, _f: dict[str, Any] | None = None) -> list[Paper]:
        return [_paper("arxiv:ok", title="Survives")]

    def _s2_boom(_q: str, _f: dict[str, Any] | None = None) -> list[Paper]:
        raise RuntimeError("provider down")

    papers = retrieve_papers(
        rq,
        sources=("arxiv", "semantic_scholar"),
        search_callables={"arxiv": _arxiv_ok, "semantic_scholar": _s2_boom},
    )

    assert len(papers) == 1
    assert papers[0].paper_id == "arxiv:ok"


def test_retrieve_truncates_to_total_limit() -> None:
    rq = ResearchQuestion(question="query", sub_queries=["query"])
    many = [
        _paper(f"arxiv:p{idx}", title=f"Paper {idx}", citation_count=100 - idx)
        for idx in range(20)
    ]
    fakes = _make_search({"arxiv": many})

    papers = retrieve_papers(
        rq, sources=("arxiv",), total_limit=5, search_callables=fakes
    )

    assert len(papers) == 5


def test_retrieve_sort_prefers_abstract_then_citations_then_recency() -> None:
    rq = ResearchQuestion(question="query", sub_queries=["query"])
    fakes = _make_search(
        {
            "arxiv": [
                _paper(
                    "arxiv:no-abs",
                    title="No Abstract But Many Citations",
                    abstract=None,
                    citation_count=999,
                    publication_date=date(2024, 1, 1),
                ),
                _paper(
                    "arxiv:older-cited",
                    title="Older Highly Cited",
                    abstract="abs",
                    citation_count=200,
                    publication_date=date(2018, 1, 1),
                ),
                _paper(
                    "arxiv:newer-less-cited",
                    title="Newer Less Cited",
                    abstract="abs",
                    citation_count=50,
                    publication_date=date(2024, 6, 1),
                ),
                _paper(
                    "arxiv:tiebreak-b",
                    title="b alphabetical",
                    abstract="abs",
                    citation_count=10,
                    publication_date=date(2020, 1, 1),
                ),
                _paper(
                    "arxiv:tiebreak-a",
                    title="a alphabetical",
                    abstract="abs",
                    citation_count=10,
                    publication_date=date(2020, 1, 1),
                ),
            ],
        }
    )

    papers = retrieve_papers(rq, sources=("arxiv",), search_callables=fakes)
    ordered_ids = [p.paper_id for p in papers]

    assert ordered_ids[-1] == "arxiv:no-abs"
    assert ordered_ids.index("arxiv:older-cited") < ordered_ids.index(
        "arxiv:newer-less-cited"
    )
    assert ordered_ids.index("arxiv:tiebreak-a") < ordered_ids.index(
        "arxiv:tiebreak-b"
    )


def test_retrieve_passes_per_query_limit_into_filters() -> None:
    rq = ResearchQuestion(question="query", sub_queries=["alpha", "beta"])
    seen: list[dict[str, Any] | None] = []

    def _capture(_q: str, filters: dict[str, Any] | None = None) -> list[Paper]:
        seen.append(filters)
        return [_paper(f"arxiv:{_q}", title=_q.title())]

    papers = retrieve_papers(
        rq,
        per_query_limit=7,
        sources=("arxiv",),
        search_callables={"arxiv": _capture},
    )

    assert len(seen) == 2
    assert all(f == {"max_results": 7} for f in seen)
    assert len(papers) == 2


def test_retrieve_falls_back_to_question_when_no_sub_queries() -> None:
    rq = ResearchQuestion(question="root question", sub_queries=[])
    calls: list[str] = []

    def _capture(q: str, _f: dict[str, Any] | None = None) -> list[Paper]:
        calls.append(q)
        return [_paper("arxiv:1", title="Only")]

    papers = retrieve_papers(
        rq, sources=("arxiv",), search_callables={"arxiv": _capture}
    )

    assert calls == ["root question"]
    assert len(papers) == 1


def test_retrieve_skips_unknown_sources(caplog: pytest.LogCaptureFixture) -> None:
    rq = ResearchQuestion(question="query", sub_queries=["query"])
    fakes = _make_search({"arxiv": [_paper("arxiv:1", title="Kept")]})

    with caplog.at_level("WARNING"):
        papers = retrieve_papers(
            rq, sources=("arxiv", "nonexistent"), search_callables=fakes
        )

    assert len(papers) == 1
    assert any("nonexistent" in record.message for record in caplog.records)
