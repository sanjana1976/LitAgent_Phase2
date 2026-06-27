from __future__ import annotations

import pytest

from tools.search_tools import (
    SearchToolError,
    tool_search_arxiv,
    tool_search_crossref,
    tool_search_dblp,
    tool_search_semantic_scholar,
)


def test_search_dblp_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    xml = """<?xml version="1.0"?>
    <result>
      <hits total="1" sent="1">
        <hit>
          <info>
            <title>Flubber Nets</title>
            <authors><author>A. Author</author></authors>
            <year>2023</year>
            <key>conf/flubber</key>
            <ee>https://doi.org/10.1000/flubber</ee>
            <venue>NeurIPS</venue>
          </info>
        </hit>
      </hits>
    </result>"""

    class FakeResp:
        text = xml

    monkeypatch.setattr("tools.search_tools.rate_limited_get", lambda *a, **k: FakeResp())
    papers = tool_search_dblp("flubber", filters={"max_results": 5})
    assert len(papers) == 1
    assert papers[0].api_source == "dblp"
    assert "Flubber" in papers[0].title


def test_search_semantic_scholar_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "data": [
            {
                "paperId": "s2id",
                "title": "Graph RAG",
                "authors": [{"name": "A. Smith"}],
                "year": 2024,
                "abstract": "We study RAG.",
                "url": "https://example.com",
                "citationCount": 10,
                "influentialCitationCount": 2,
                "publicationDate": "2024-06-01",
                "externalIds": {"DOI": "10.1000/rag"},
                "venue": "ACL",
            }
        ]
    }

    class FakeResp:
        def json(self) -> dict:
            return payload

    monkeypatch.setattr("tools.search_tools.rate_limited_get", lambda *a, **k: FakeResp())
    papers = tool_search_semantic_scholar("graph rag", filters={"max_results": 3})
    assert len(papers) == 1
    assert papers[0].paper_id == "s2:s2id"
    assert papers[0].citation_count == 10


def test_search_crossref_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "message": {
            "items": [
                {
                    "title": ["Crossref Paper"],
                    "author": [{"given": "Jane", "family": "Doe"}],
                    "issued": {"date-parts": [[2022, 3, 1]]},
                    "DOI": "10.5555/cross",
                    "URL": "https://example.org/paper",
                    "container-title": ["Journal of Tests"],
                    "type": "journal-article",
                }
            ]
        }
    }

    class FakeResp:
        def json(self) -> dict:
            return payload

    monkeypatch.setattr("tools.search_tools.rate_limited_get", lambda *a, **k: FakeResp())
    papers = tool_search_crossref("crossref paper", filters={"max_results": 2})
    assert len(papers) == 1
    assert papers[0].doi == "10.5555/cross"
    assert papers[0].api_source == "crossref"


def test_search_crossref_doi_lookup_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "message": {
            "title": ["DOI Paper"],
            "author": [],
            "issued": {"date-parts": [[2020, 1, 1]]},
            "DOI": "10.1000/only",
        }
    }

    class FakeResp:
        def json(self) -> dict:
            return payload

    monkeypatch.setattr("tools.search_tools.rate_limited_get", lambda *a, **k: FakeResp())
    papers = tool_search_crossref("", filters={"doi": "10.1000/only", "max_results": 1})
    assert len(papers) == 1
    assert papers[0].title == "DOI Paper"


def test_search_semantic_scholar_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResp:
        def json(self) -> dict:
            raise ValueError("not json")

        text = "html"

    monkeypatch.setattr("tools.search_tools.rate_limited_get", lambda *a, **k: FakeResp())
    with pytest.raises(SearchToolError, match="non-JSON"):
        tool_search_semantic_scholar("q")


def test_search_arxiv_sorts_by_relevance_without_date_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    class FakeResp:
        text = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <title>Token compression survey</title>
            <summary>We study token compression.</summary>
            <id>http://arxiv.org/abs/2401.00001v1</id>
            <published>2024-01-01T00:00:00Z</published>
            <author><name>A. Author</name></author>
          </entry>
        </feed>"""

    def _fake_get(url: str, **kwargs: object) -> FakeResp:
        captured["url"] = url
        return FakeResp()

    monkeypatch.setattr("tools.search_tools.rate_limited_get", _fake_get)
    papers = tool_search_arxiv("token compression", filters={"max_results": 3})
    assert len(papers) == 1
    assert "sortBy=relevance" in captured["url"]
    assert "sortOrder=descending" in captured["url"]


def test_search_arxiv_sorts_by_submitted_date_when_date_filter_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    class FakeResp:
        text = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom"></feed>"""

    def _fake_get(url: str, **kwargs: object) -> FakeResp:
        captured["url"] = url
        return FakeResp()

    monkeypatch.setattr("tools.search_tools.rate_limited_get", _fake_get)
    tool_search_arxiv(
        "token compression",
        filters={
            "max_results": 3,
            "date_from": "2024-01-01",
            "date_to": "2024-12-31",
        },
    )
    assert "sortBy=submittedDate" in captured["url"]
