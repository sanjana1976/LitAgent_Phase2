from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.citation_tools import (
    CitationToolError,
    resolve_semantic_scholar_paper_id,
    tool_lookup_forward_citations,
)


def test_resolve_s2_prefix() -> None:
    assert resolve_semantic_scholar_paper_id("s2:abc-uuid") == "abc-uuid"


def test_resolve_arxiv_prefix() -> None:
    assert resolve_semantic_scholar_paper_id("arxiv:1234.5678v2") == "ARXIV:1234.5678"


def test_resolve_crossref_and_doi() -> None:
    assert resolve_semantic_scholar_paper_id("crossref:10.1000/xyz") == "DOI:10.1000/xyz"
    assert resolve_semantic_scholar_paper_id("doi:10.1000/xyz") == "DOI:10.1000/xyz"
    assert resolve_semantic_scholar_paper_id("10.1000/xyz") == "DOI:10.1000/xyz"


def test_resolve_numeric_db_id_from_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from db.database import Database
    from db.init_db import initialize_schema

    db_path = tmp_path / "c.sqlite3"
    db = Database(db_path)
    initialize_schema(db)
    meta = json.dumps({"s2_paper_id": "s2paper99"})
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO papers (title, authors, api_source, metadata_json) VALUES (?,?,?,?);",
            ("T", "A", "semantic_scholar", meta),
        )
        pid = int(conn.execute("SELECT id FROM papers").fetchone()["id"])

    monkeypatch.setenv("DATABASE_PATH", str(db_path))
    import config.config as cfg

    cfg.get_settings(reload=True)
    from tools.context import clear_tool_caches

    clear_tool_caches()

    assert resolve_semantic_scholar_paper_id(str(pid)) == "s2paper99"


def test_resolve_numeric_db_id_missing_metadata_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from db.database import Database
    from db.init_db import initialize_schema

    db_path = tmp_path / "d.sqlite3"
    db = Database(db_path)
    initialize_schema(db)
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO papers (title, authors, api_source) VALUES ('T','A','local');",
        )
        pid = int(conn.execute("SELECT id FROM papers").fetchone()["id"])

    monkeypatch.setenv("DATABASE_PATH", str(db_path))
    import config.config as cfg

    cfg.get_settings(reload=True)
    from tools.context import clear_tool_caches

    clear_tool_caches()

    with pytest.raises(CitationToolError, match="no Semantic Scholar"):
        resolve_semantic_scholar_paper_id(str(pid))


def test_forward_citations_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "data": [
            {
                "citingPaper": {
                    "paperId": "abc123",
                    "title": "Building on Flubber",
                    "authors": [{"name": "A. Citer"}],
                    "year": 2024,
                    "abstract": "We extend prior work.",
                    "url": "https://example.com/p",
                    "citationCount": 3,
                }
            }
        ]
    }

    class FakeResp:
        def json(self) -> dict:
            return payload

    monkeypatch.setattr(
        "tools.citation_tools.rate_limited_get",
        lambda url, params=None, headers=None, host_key=None: FakeResp(),
    )
    papers = tool_lookup_forward_citations("s2:seed-paper-id", filters={"max_results": 5})
    assert len(papers) == 1
    assert papers[0].title == "Building on Flubber"
    assert papers[0].metadata.get("citation_direction") == "forward"


def test_forward_citations_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise ConnectionError("offline")

    monkeypatch.setattr("tools.citation_tools.rate_limited_get", boom)
    with pytest.raises(CitationToolError, match="citations request failed"):
        tool_lookup_forward_citations("s2:seed", filters={"max_results": 3})
