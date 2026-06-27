"""
Pydantic models used by agent tools: search hits, analysis payloads, comparisons.

These types are distinct from persistence models under ``models/`` (e.g. ``PaperRecord``)
and are optimized for API returns and LLM-structured outputs.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchFilters(BaseModel):
    """Validated filter bag for scholarly search tools."""

    model_config = ConfigDict(extra="forbid")

    author: str | None = None
    category: str | None = Field(
        default=None,
        description="arXiv category (e.g. cs.LG) when the source supports it.",
    )
    date_from: date | None = None
    date_to: date | None = None
    max_results: int = Field(default=10, ge=1, le=100)
    venue: str | None = None
    doi: str | None = None


class Paper(BaseModel):
    """Normalized paper metadata returned by search tools."""

    model_config = ConfigDict(extra="allow")

    paper_id: str = Field(description="Stable external id (source-specific).")
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    url: str | None = None
    publication_date: date | None = None
    doi: str | None = None
    venue: str | None = None
    publisher: str | None = None
    citation_count: int | None = None
    influential_citations: int | None = None
    api_source: str = Field(description="Provenance label (arxiv, dblp, …).")
    metadata: dict[str, Any] = Field(default_factory=dict)
    reading_status: str | None = Field(
        default=None,
        description="When loaded from a reading list: unread|skimmed|reading|fully_read|to_cite.",
    )


class Citation(BaseModel):
    """A single parsed reference for cross-referencing."""

    cited_title: str
    cited_authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    raw: str | None = Field(default=None, description="Original reference line if available.")


class MethodologyBlock(BaseModel):
    """Structured methodology section used in deep analysis."""

    approach: str = ""
    datasets: list[str] = Field(default_factory=list)
    experimental_setup: str = ""
    baselines_compared: list[str] = Field(default_factory=list)


class KeyEquation(BaseModel):
    """Equation text plus a short gloss."""

    equation: str
    description: str = ""


class ResultsBlock(BaseModel):
    """Quantitative and qualitative results."""

    main_metrics: dict[str, float] = Field(default_factory=dict)
    improvements: dict[str, str] = Field(default_factory=dict)
    failure_cases: list[str] = Field(default_factory=list)


class CodeAvailability(BaseModel):
    """Reproducibility signals extracted from the paper text."""

    has_code: bool = False
    repo_links: list[str] = Field(default_factory=list)


class PaperAnalysis(BaseModel):
    """Structured deep analysis of a single paper."""

    paper_id: str
    title: str = ""
    key_contributions: list[str] = Field(default_factory=list)
    methodology: MethodologyBlock = Field(default_factory=MethodologyBlock)
    key_equations: list[KeyEquation] = Field(default_factory=list)
    results: ResultsBlock = Field(default_factory=ResultsBlock)
    limitations: list[str] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list)
    related_work_categories: dict[str, list[Citation]] = Field(default_factory=dict)
    reproducibility_score: float = Field(default=0.0, ge=0.0, le=1.0)
    code_availability: CodeAvailability = Field(default_factory=CodeAvailability)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("reproducibility_score", mode="before")
    @classmethod
    def _clamp_score(cls, v: object) -> float:
        if v is None:
            return 0.0
        x = float(v)
        return max(0.0, min(1.0, x))


class ComparisonMatrix(BaseModel):
    """Side-by-side comparison for 2–5 papers across fixed dimensions."""

    paper_ids: list[str]
    methodology: dict[str, str] = Field(
        default_factory=dict,
        description="paper_id -> narrative comparing approach.",
    )
    datasets_used: dict[str, str] = Field(default_factory=dict)
    results_metrics: dict[str, str] = Field(default_factory=dict)
    computational_cost: dict[str, str] = Field(default_factory=dict)
    novelty: dict[str, str] = Field(default_factory=dict)


READING_STATUSES = frozenset({"unread", "skimmed", "reading", "fully_read", "to_cite"})


class ReadingList(BaseModel):
    """Reading list metadata returned by list/create tools."""

    list_id: str
    name: str
    description: str | None
    created_at: datetime


# --- Input validators for tools that accept loose dicts from the agent ---


def validate_search_filters(filters: dict[str, Any] | None) -> SearchFilters:
    """Coerce an optional dict to ``SearchFilters`` (empty dict is legal)."""
    if filters is None:
        return SearchFilters()
    return SearchFilters.model_validate(filters)
