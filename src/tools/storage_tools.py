"""
Persist custom summaries and export BibTeX files (with confirmation gates).
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from db.queries import fetch_papers_for_list, insert_summary_row, list_summaries_for_paper
from tools.bibliography_tools import tool_generate_bibtex
from tools.confirm import ConfirmationRequired
from tools.context import get_default_database

logger = logging.getLogger(__name__)

SummaryDepth = Literal["shallow", "medium", "deep"]


class _SummarySave(BaseModel):
    paper_id: str = Field(min_length=1)
    summary_text: str = Field(min_length=1)
    depth: SummaryDepth


def tool_save_summary(
    paper_id: str,
    summary_text: str,
    depth: str,
    *,
    user_confirmed: bool = False,
) -> bool:
    """
    Save a user- or agent-authored summary scoped by depth tier.

    Overwriting:
        When summaries already exist for the paper, callers must pass ``user_confirmed=True``
        after warning the user.
    """
    payload = _SummarySave(paper_id=paper_id, summary_text=summary_text, depth=depth)  # type: ignore[arg-type]

    database = get_default_database()
    if not payload.paper_id.isdigit():
        raise ValueError("tool_save_summary expects numeric local paper_id")

    pid = int(payload.paper_id)
    existing = list_summaries_for_paper(database, pid)
    if existing and not user_confirmed:
        raise ConfirmationRequired(
            "A summary already exists for this paper — obtain confirmation to overwrite "
            "or append (re-run with user_confirmed=True).",
            tool_name="tool_save_summary",
        )

    insert_summary_row(
        database,
        paper_db_id=pid,
        summary_text=payload.summary_text,
        depth=str(payload.depth),
    )
    return True


def tool_export_list_to_bibtex(
    list_id: str,
    filename: str,
    *,
    user_confirmed: bool = False,
) -> bool:
    """
    Write a ``.bib`` file aggregating every paper currently in the list.

    Safety:
        Refuses to clobber an existing file unless ``user_confirmed=True``.
    """
    if not user_confirmed:
        raise ConfirmationRequired(
            "Exporting a BibTeX file can overwrite disk contents — confirm explicitly "
            "(re-run with user_confirmed=True).",
            tool_name="tool_export_list_to_bibtex",
        )

    database = get_default_database()
    if not list_id.strip().isdigit():
        raise ValueError("list_id must be numeric")

    lid = int(list_id)
    rows = fetch_papers_for_list(database, lid)
    if not rows:
        logger.warning("List %s is empty — writing minimal file header only", list_id)

    entries: list[str] = []
    for row in rows:
        pid = str(row["id"])
        try:
            entries.append(tool_generate_bibtex(pid))
        except Exception as exc:  # noqa: BLE001 — aggregate per-paper
            logger.warning("Skipping bib for paper %s: %s", pid, exc)

    body = "\n\n".join(entries) if entries else "% empty export\n"

    root = database.path.parent
    out_path = (root / filename).resolve()
    if out_path.exists():
        # Caller already confirmed overwrite intent via user_confirmed
        pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    return True
