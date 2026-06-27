"""
Reading list CRUD backed by SQLite with explicit confirmation gates.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from db.database import DatabaseError
from db.queries import (
    add_paper_to_list,
    fetch_papers_for_list,
    insert_reading_list,
    list_reading_list_rows,
    remove_paper_from_list as delete_paper_list_mapping,
)
from tools.confirm import ConfirmationRequired
from tools.context import get_default_database
from tools.schemas import READING_STATUSES, Paper, ReadingList

logger = logging.getLogger(__name__)


class _ListId(BaseModel):
    """Validate numeric SQLite identifiers passed as strings."""

    value: str = Field(min_length=1)

    def as_int(self) -> int:
        return int(self.value)


def _row_to_reading_list(row: dict[str, Any]) -> ReadingList:
    raw_date = row.get("created_date")
    if isinstance(raw_date, str):
        try:
            created = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
        except ValueError:
            created = datetime.utcnow()
    else:
        created = datetime.utcnow()
    return ReadingList(
        list_id=str(row["id"]),
        name=str(row["name"]),
        description=str(row["description"]) if row.get("description") else None,
        created_at=created,
    )


def _db_row_to_tool_paper(row: dict[str, Any], *, reading_status: str | None) -> Paper:
    authors_raw = str(row.get("authors") or "")
    authors = [a.strip() for a in authors_raw.replace(" and ", ";").split(";") if a.strip()]
    if not authors:
        authors = [authors_raw] if authors_raw else []
    return Paper(
        paper_id=str(row["id"]),
        title=str(row["title"]),
        authors=authors,
        abstract=str(row["abstract"]) if row.get("abstract") else None,
        url=str(row["url"]) if row.get("url") else None,
        publication_date=None,
        doi=str(row["doi"]) if row.get("doi") else None,
        api_source=str(row.get("api_source") or "local"),
        metadata={},
        reading_status=reading_status,
    )


def tool_create_reading_list(
    name: str,
    description: str,
) -> ReadingList:
    """
    Persist a new reading list and return its metadata including string ``list_id``.
    """
    database = get_default_database()
    rid = insert_reading_list(database, name, description or None)
    rows = list_reading_list_rows(database)
    for r in rows:
        if int(r["id"]) == rid:
            return _row_to_reading_list(dict(r))
    raise RuntimeError("Inserted list row not found — database inconsistency")


def tool_add_paper_to_list(
    paper_id: str,
    list_id: str,
    reading_status: str,
    *,
    user_confirmed: bool = False,
) -> bool:
    """
    Attach a stored paper (numeric ``paper_id``) to a list with a reading status.

    Confirmation:
        Set ``user_confirmed=True`` after the user explicitly approves the mapping.
    """
    if reading_status not in READING_STATUSES:
        raise ValueError(f"reading_status must be one of {sorted(READING_STATUSES)}")

    if not user_confirmed:
        raise ConfirmationRequired(
            "Adding a paper to a reading list requires user confirmation "
            "(re-run with user_confirmed=True).",
            tool_name="tool_add_paper_to_list",
        )

    database = get_default_database()
    pid = _ListId(value=paper_id).as_int()
    lid = _ListId(value=list_id).as_int()
    try:
        add_paper_to_list(database, paper_db_id=pid, list_id=lid, reading_status=reading_status)
    except DatabaseError as exc:
        logger.exception("add_paper_to_list failed")
        raise
    return True


def tool_remove_paper_from_list(
    paper_id: str,
    list_id: str,
    *,
    user_confirmed: bool = False,
) -> bool:
    """Remove a paper from a list (destructive — needs confirmation)."""
    if not user_confirmed:
        raise ConfirmationRequired(
            "Removing a paper from a list requires user confirmation "
            "(re-run with user_confirmed=True).",
            tool_name="tool_remove_paper_from_list",
        )
    database = get_default_database()
    pid = _ListId(value=paper_id).as_int()
    lid = _ListId(value=list_id).as_int()
    return delete_paper_list_mapping(database, paper_db_id=pid, list_id=lid)


def tool_list_all_lists() -> list[ReadingList]:
    """Return metadata for every reading list."""
    database = get_default_database()
    rows = list_reading_list_rows(database)
    return [_row_to_reading_list(dict(r)) for r in rows]


def tool_get_list_contents(
    list_id: str,
) -> list[Paper]:
    """Return all papers in a list; each :class:`Paper` includes ``reading_status``."""
    database = get_default_database()
    lid = _ListId(value=list_id).as_int()
    rows = fetch_papers_for_list(database, lid)
    out: list[Paper] = []
    for row in rows:
        st = str(row.get("reading_status") or "unread")
        out.append(_db_row_to_tool_paper(row, reading_status=st))
    return out
