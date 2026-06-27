from __future__ import annotations

from pathlib import Path

import pytest

from tools.bibliography_tools import tool_generate_apa, tool_generate_bibtex
from tools.confirm import ConfirmationRequired
from tools.context import clear_tool_caches
from tools.reading_list_tools import (
    tool_add_paper_to_list,
    tool_create_reading_list,
    tool_get_list_contents,
    tool_list_all_lists,
    tool_remove_paper_from_list,
)
from tools.storage_tools import tool_export_list_to_bibtex


@pytest.fixture(autouse=True)
def _db_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clear_tool_caches()
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "lists.sqlite3"))
    import config.config as cfg

    cfg.get_settings(reload=True)
    from db.database import Database
    from db.init_db import initialize_schema

    initialize_schema(Database(tmp_path / "lists.sqlite3"))


def _seed_paper(tmp_path: Path) -> int:
    from db.database import Database

    db = Database(tmp_path / "lists.sqlite3")
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO papers (title, authors, doi, api_source, metadata_json) "
            "VALUES (?, ?, ?, ?, ?);",
            (
                "Attention Is All You Need",
                "Vaswani;Shazeer",
                "10.5555/test",
                "test",
                '{"year": 2017, "venue": "NeurIPS"}',
            ),
        )
        return int(conn.execute("SELECT id FROM papers").fetchone()["id"])


def test_reading_list_create_and_list(tmp_path: Path) -> None:
    created = tool_create_reading_list("Thesis reading", "Primary sources")
    assert created.name == "Thesis reading"
    all_lists = tool_list_all_lists()
    assert any(lst.list_id == created.list_id for lst in all_lists)


def test_add_and_get_list_contents(tmp_path: Path) -> None:
    pid = _seed_paper(tmp_path)
    lst = tool_create_reading_list("L1", "")
    tool_add_paper_to_list(str(pid), lst.list_id, "reading", user_confirmed=True)
    contents = tool_get_list_contents(lst.list_id)
    assert len(contents) == 1
    assert contents[0].reading_status == "reading"
    assert "Attention" in contents[0].title


def test_add_paper_invalid_status(tmp_path: Path) -> None:
    pid = _seed_paper(tmp_path)
    lst = tool_create_reading_list("L2", "")
    with pytest.raises(ValueError, match="reading_status"):
        tool_add_paper_to_list(str(pid), lst.list_id, "invalid_status", user_confirmed=True)


def test_remove_paper_requires_confirmation(tmp_path: Path) -> None:
    pid = _seed_paper(tmp_path)
    lst = tool_create_reading_list("L3", "")
    with pytest.raises(ConfirmationRequired):
        tool_remove_paper_from_list(str(pid), lst.list_id, user_confirmed=False)


def test_generate_bibtex_from_db_row(tmp_path: Path) -> None:
    pid = _seed_paper(tmp_path)
    bib = tool_generate_bibtex(str(pid))
    assert "@article{" in bib
    assert "Attention Is All You Need" in bib
    assert "10.5555/test" in bib


def test_generate_apa_from_db_row(tmp_path: Path) -> None:
    pid = _seed_paper(tmp_path)
    apa = tool_generate_apa(str(pid))
    assert "Attention Is All You Need" in apa


def test_export_list_requires_confirmation(tmp_path: Path) -> None:
    lst = tool_create_reading_list("Export me", "")
    with pytest.raises(ConfirmationRequired):
        tool_export_list_to_bibtex(lst.list_id, "out.bib", user_confirmed=False)


def test_export_list_writes_bib(tmp_path: Path) -> None:
    pid = _seed_paper(tmp_path)
    lst = tool_create_reading_list("Export ok", "")
    tool_add_paper_to_list(str(pid), lst.list_id, "unread", user_confirmed=True)
    ok = tool_export_list_to_bibtex(lst.list_id, "thesis.bib", user_confirmed=True)
    assert ok is True
    out = tmp_path / "thesis.bib"
    assert out.exists()
    assert "@article{" in out.read_text(encoding="utf-8")
