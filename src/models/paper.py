"""Paper-related Pydantic models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PaperCreate(BaseModel):
    """Validated payload for inserting a paper row."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    authors: str = Field(
        ...,
        description="Serialized author list or display string.",
    )
    abstract: str | None = None
    doi: str | None = None
    url: str | None = None
    api_source: str = Field(description="Which external API surfaced this paper.")
    metadata_json: str | None = Field(
        default=None,
        description="Opaque JSON blob for provider-specific metadata.",
    )


class PaperRecord(PaperCreate):
    """Paper as stored locally, including identifiers and timestamps."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    added_date: datetime

    def merged_metadata_preview(self, max_chars: int = 200) -> str:
        """Return a short snippet of metadata for logging (no parsing)."""
        if not self.metadata_json:
            return ""
        return (
            self.metadata_json[:max_chars] + "…"
            if len(self.metadata_json) > max_chars
            else self.metadata_json
        )
