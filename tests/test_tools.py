"""
Unit tests for agent tools using mocks — no live external API calls.

Real-network integration tests are intentionally out of scope for this module.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from tools.analysis_tools import tool_deep_analyze_paper, tool_extract_citations
from tools.confirm import ConfirmationRequired
from tools.context import clear_tool_caches
from tools.file_cache import FileCache
from tools.paper_text import load_cached_paper_text
from tools.reading_list_tools import tool_add_paper_to_list
from tools.schemas import SearchFilters, validate_search_filters
from tools.search_tools import SearchToolError, tool_search_arxiv
from tools.storage_tools import tool_save_summary
from tools.text_heuristics import extract_github_urls, split_sections
from tools.tools_registry import TOOL_SPECS, list_tool_signatures


@pytest.fixture(autouse=True)
def _reset_caches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate cache and database paths per test."""
    clear_tool_caches()
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "t.sqlite3"))
    import config.config as cfg

    cfg.get_settings(reload=True)
    monkeypatch.setattr(
        "tools.context.get_cache_dir",
        lambda: tmp_path / "cache",
    )


def test_search_filters_validation() -> None:
    f = validate_search_filters({"max_results": 5, "author": "J. Smith"})
    assert isinstance(f, SearchFilters)
    assert f.max_results == 5


def test_split_sections_finds_headings() -> None:
    text = "Abstract\nWe study X.\n\nIntroduction\nRelated prior art.\n\nMethods\nOur model."
    sec = split_sections(text)
    assert "abstract" in sec or "preamble" in sec


def test_extract_github_urls() -> None:
    txt = "Code at https://github.com/foo/bar and mirror https://github.com/foo/bar/"
    assert "https://github.com/foo/bar" in extract_github_urls(txt)


def test_tool_deep_analyze_paper_heuristic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import config.config as cfg

    cfg.get_settings(reload=True)
    sample = (
        "Learning Widgets with Deep Flubber\n\nAbstract\nWe propose FlubberNets.\n"
        "Dataset: ImageNet. Code: https://github.com/org/repo\n"
    )
    pa = tool_deep_analyze_paper("unit-test-paper", full_text=sample)
    assert pa.paper_id == "unit-test-paper"
    assert pa.reproducibility_score >= 0.0


def test_tool_extract_citations_smoke() -> None:
    text = "References\n[1] Smith, J. Flubber Systems. In NeurIPS 2020."
    cites = tool_extract_citations("x", full_text=text)
    assert len(cites) >= 1


def test_cached_paper_text_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tools.paper_text.get_cache_dir", lambda: tmp_path)
    cache = FileCache(tmp_path, namespace="pdf_tool")
    cache.set_json(
        "paper:abc",
        {"paper_id": "abc", "full_text": "Hello world", "sections": {}},
    )
    t, _ = load_cached_paper_text("abc", None)
    assert "Hello" in t


def test_confirmation_gate_on_save_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from db.database import Database
    from db.init_db import initialize_schema
    from db.queries import insert_summary_row

    db_path = tmp_path / "d.sqlite3"
    db = Database(db_path)
    initialize_schema(db)
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO papers (title, authors, api_source) VALUES ('T','A','test');",
        )
        pid = int(conn.execute("SELECT id FROM papers LIMIT 1;").fetchone()["id"])
    insert_summary_row(db, paper_db_id=pid, summary_text="old", depth="shallow")

    monkeypatch.setenv("DATABASE_PATH", str(db_path))
    import config.config as cfg

    cfg.get_settings(reload=True)
    clear_tool_caches()

    with pytest.raises(ConfirmationRequired):
        tool_save_summary(str(pid), "new summary", "deep", user_confirmed=False)


def test_confirmation_add_paper_to_list() -> None:
    with pytest.raises(ConfirmationRequired):
        tool_add_paper_to_list("1", "1", "unread", user_confirmed=False)


def test_search_arxiv_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    atom = """<?xml version='1.0'?>
    <feed xmlns='http://www.w3.org/2005/Atom'>
      <entry>
        <id>http://arxiv.org/abs/1234.5678v1</id>
        <title>Demo &lt;br/&gt; Title</title>
        <summary>Abstract here</summary>
        <author><name>A. Test</name></author>
        <published>2024-01-01T00:00:00Z</published>
      </entry>
    </feed>"""

    class FakeResp:
        text = atom
        headers = {"content-type": "application/atom+xml"}

    monkeypatch.setattr(
        "tools.search_tools.rate_limited_get",
        lambda url, params=None, headers=None, host_key=None: FakeResp(),
    )
    papers = tool_search_arxiv("flubber", filters={"max_results": 3})
    assert len(papers) == 1
    assert papers[0].title
    assert papers[0].api_source == "arxiv"


def test_search_arxiv_builds_keyword_and_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """Natural-language questions become sanitized ``all:term AND ...`` queries."""
    captured: list[str] = []
    empty_feed = "<?xml version='1.0'?><feed xmlns='http://www.w3.org/2005/Atom'></feed>"

    class FakeResp:
        text = empty_feed

    def _capture(url: str, params=None, headers=None, host_key=None) -> FakeResp:
        captured.append(url)
        return FakeResp()

    monkeypatch.setattr("tools.search_tools.rate_limited_get", _capture)
    tool_search_arxiv(
        "What are the competing approaches to long-context retrieval in LLMs?"
    )

    from urllib.parse import parse_qs, urlparse

    first_query = parse_qs(urlparse(captured[0]).query)["search_query"][0]
    assert first_query == "all:long-context AND all:retrieval AND all:llms"
    # The AND query returned nothing, so the tool broadened to OR once.
    assert len(captured) == 2
    second_query = parse_qs(urlparse(captured[1]).query)["search_query"][0]
    assert second_query == "all:long-context OR all:retrieval OR all:llms"


def test_search_arxiv_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: object, **k: object) -> None:
        raise ConnectionError("offline")

    monkeypatch.setattr("tools.search_tools.rate_limited_get", boom)
    with pytest.raises(SearchToolError):
        tool_search_arxiv("q")


def test_tools_registry_covers_all_exports() -> None:
    assert len(TOOL_SPECS) == 21
    names = {spec.name for spec in TOOL_SPECS}
    assert "tool_lookup_forward_citations" in names
    assert "tool_export_list_to_bibtex" in names
    assert "tool_synthesize_literature_review" in names
    assert "tool_get_review_context" in names
    sigs = list_tool_signatures()
    assert "tool_search_arxiv" in sigs
    assert "->" in sigs["tool_deep_analyze_paper"]


def test_search_filters_rejects_invalid_max_results() -> None:
    with pytest.raises(Exception):
        validate_search_filters({"max_results": 0})
