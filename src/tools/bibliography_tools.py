"""
Generate BibTeX, APA 7th-style, and Chicago author-date strings for stored papers.
"""

from __future__ import annotations

import logging
import re
from datetime import date

from pydantic import BaseModel, Field

from db.queries import get_paper_by_id, row_to_metadata_dict
from tools.context import get_default_database
from tools.paper_text import load_cached_paper_text
from tools.text_heuristics import guess_title_from_text

logger = logging.getLogger(__name__)


class _BibMeta(BaseModel):
    """Internal normalized metadata for formatting."""

    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    publisher: str | None = None


def _authors_split(authors_str: str) -> list[str]:
    parts = re.split(r"\s*;\s*|\s+and\s+", authors_str, flags=re.I)
    return [p.strip() for p in parts if p.strip()]


def _authors_to_bibtex_key(names: list[str]) -> str:
    if not names:
        return "unknown"
    primary = names[0].split()[-1].lower()
    suffix = date.today().year
    return re.sub(r"[^a-z0-9]", "", primary) + str(suffix)


def _authors_apa(names: list[str]) -> str:
    if not names:
        return ""
    formatted: list[str] = []
    for i, n in enumerate(names):
        bits = n.split()
        if not bits:
            continue
        fam = bits[-1]
        initials = " ".join(f"{b[0]}." for b in bits[:-1])
        chunk = f"{fam}, {initials}" if initials else fam
        formatted.append(chunk)
    return ", ".join(formatted)


def _resolve_bib_meta(paper_id: str) -> _BibMeta:
    database = get_default_database()
    if paper_id.isdigit():
        row = get_paper_by_id(database, int(paper_id))
        if row:
            meta = row_to_metadata_dict(row)
            year = None
            for key in ("year", "publication_year"):
                if isinstance(meta.get(key), int):
                    year = int(meta[key])
                    break
            venue = meta.get("venue") or meta.get("container-title")
            return _BibMeta(
                title=str(row["title"]),
                authors=_authors_split(str(row["authors"])),
                year=year,
                venue=str(venue) if venue else None,
                doi=str(row["doi"]) if row.get("doi") else None,
                url=str(row["url"]) if row.get("url") else None,
                publisher=str(meta["publisher"]) if meta.get("publisher") else None,
            )
    # Fallback to cached PDF text
    try:
        text, _ = load_cached_paper_text(paper_id, None)
        title = guess_title_from_text(text, fallback=paper_id)
        return _BibMeta(title=title, authors=[], year=date.today().year)
    except Exception as exc:  # noqa: BLE001 — aggregate user-facing error
        logger.warning("Could not resolve metadata for %s: %s", paper_id, exc)
        raise ValueError(f"No bibliography metadata for paper_id={paper_id!r}") from exc


def tool_generate_bibtex(paper_id: str) -> str:
    """
    Emit a BibTeX ``@article`` entry with title, author, year, DOI, and URL when known.

    ``paper_id`` may be a local numeric id (preferred) or any id with cached PDF text.
    """
    m = _resolve_bib_meta(paper_id)
    cite_key = _authors_to_bibtex_key(m.authors) if m.authors else f"ref{paper_id}"
    lines = [
        f"@article{{{cite_key},",
        f"  title = {{{{{m.title}}}}},",
    ]
    if m.authors:
        author_field = " and ".join(m.authors)
        lines.append(f"  author = {{{author_field}}},")
    if m.year:
        lines.append(f"  year = {{{m.year}}},")
    if m.venue:
        lines.append(f"  journal = {{{m.venue}}},")
    if m.publisher:
        lines.append(f"  publisher = {{{m.publisher}}},")
    if m.doi:
        lines.append(f"  doi = {{{m.doi}}},")
    if m.url:
        lines.append(f"  url = {{{m.url}}},")
    lines.append("}")
    return "\n".join(lines)


def tool_generate_apa(paper_id: str) -> str:
    """Generate a concise APA-style reference string (7th edition inspired)."""
    m = _resolve_bib_meta(paper_id)
    authors = _authors_apa(m.authors) if m.authors else "(n.d.)"
    year = f"({m.year})." if m.year else "(n.d.)."
    venue = f" {m.venue}." if m.venue else ""
    doi = f" https://doi.org/{m.doi}" if m.doi else ""
    url = f" Retrieved from {m.url}" if m.url and not m.doi else ""
    return f"{authors} {year} {m.title}.{venue}{doi}{url}".strip()


def tool_generate_chicago(paper_id: str) -> str:
    """Author-date Chicago-style reference (notes/bibliography system, simplified)."""
    m = _resolve_bib_meta(paper_id)
    fams = []
    for n in m.authors:
        bits = n.split()
        if bits:
            fams.append(bits[-1])
    auth = ", ".join(fams) if fams else "Unknown"
    yr = str(m.year) if m.year else "n.d."
    ven = f" {m.venue}." if m.venue else ""
    doi = f" https://doi.org/{m.doi}." if m.doi else ""
    return f"{auth}. {yr}. \"{m.title}.\"{ven}{doi}".strip()
