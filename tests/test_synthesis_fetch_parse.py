"""Unit tests for ``synthesis.fetch_parse`` (stage 3 of the LitSynth pipeline)."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from synthesis.fetch_parse import fetch_and_parse
from tools.pdf_tools import PDFToolError
from tools.schemas import Paper


def _make_paper(
    *,
    paper_id: str = "arxiv:1234.5678",
    title: str = "A Sample Title",
    abstract: str | None = "Sample abstract paragraph.",
    url: str | None = "https://arxiv.org/abs/1234.5678",
    publication_date: date | None = date(2023, 5, 1),
    venue: str | None = "ICLR",
    api_source: str = "arxiv",
) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=title,
        authors=["Alice Example", "Bob Sample"],
        abstract=abstract,
        url=url,
        publication_date=publication_date,
        venue=venue,
        api_source=api_source,
    )


def test_arxiv_abs_url_is_transformed_to_pdf_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_pdf_tool(paper_id: str, url: str) -> dict[str, Any]:
        captured["paper_id"] = paper_id
        captured["url"] = url
        return {"full_text": "extracted body text", "sections": {"abstract": "x"}}

    paper = _make_paper(url="https://arxiv.org/abs/2401.99999")
    out = fetch_and_parse([paper], pdf_tool=fake_pdf_tool)

    assert captured["paper_id"] == paper.paper_id
    assert captured["url"] == "https://arxiv.org/pdf/2401.99999"
    assert len(out) == 1
    assert out[0].has_pdf is True
    assert out[0].full_text == "extracted body text"
    assert out[0].sections == {"abstract": "x"}


def test_pdf_tool_error_falls_back_to_abstract_only() -> None:
    def fake_pdf_tool(paper_id: str, url: str) -> dict[str, Any]:
        raise PDFToolError("network down")

    paper = _make_paper(abstract="the abstract survives", url="https://example.com/foo.pdf")
    out = fetch_and_parse([paper], pdf_tool=fake_pdf_tool)

    assert len(out) == 1
    sp = out[0]
    assert sp.has_pdf is False
    assert sp.full_text == "the abstract survives"
    assert sp.sections == {}
    assert sp.relevance_score == 0.0


def test_successful_parse_truncates_full_text_to_max_chars() -> None:
    paragraph = "x" * 4_000 + "\n\n"
    long_text = paragraph * 10  # ~40k chars across 10 paragraph boundaries

    def fake_pdf_tool(paper_id: str, url: str) -> dict[str, Any]:
        return {"full_text": long_text, "sections": {}}

    paper = _make_paper(url="https://example.com/big.pdf")
    out = fetch_and_parse([paper], pdf_tool=fake_pdf_tool, max_full_text_chars=10_000)

    assert out[0].has_pdf is True
    assert len(out[0].full_text) <= 10_000
    assert out[0].full_text.endswith("x" * 100)  # still made of the source paragraphs


def test_missing_abstract_and_failed_pdf_yields_empty_full_text() -> None:
    def fake_pdf_tool(paper_id: str, url: str) -> dict[str, Any]:
        raise RuntimeError("offline")

    paper = _make_paper(abstract=None, url="https://example.com/x.pdf")
    out = fetch_and_parse([paper], pdf_tool=fake_pdf_tool)

    assert out[0].has_pdf is False
    assert out[0].full_text == ""
    assert out[0].sections == {}


def test_year_is_derived_from_publication_date(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_pdf_tool(paper_id: str, url: str) -> dict[str, Any]:
        return {"full_text": "body", "sections": {}}

    paper = _make_paper(publication_date=date(2021, 3, 15))
    out = fetch_and_parse([paper], pdf_tool=fake_pdf_tool)

    assert out[0].year == 2021


def test_non_pdf_url_skips_tool_and_uses_abstract() -> None:
    calls: list[str] = []

    def fake_pdf_tool(paper_id: str, url: str) -> dict[str, Any]:
        calls.append(url)
        return {"full_text": "should not be reached", "sections": {}}

    paper = _make_paper(url="https://example.com/landing-page.html", abstract="abs")
    out = fetch_and_parse([paper], pdf_tool=fake_pdf_tool)

    assert calls == []
    assert out[0].has_pdf is False
    assert out[0].full_text == "abs"
