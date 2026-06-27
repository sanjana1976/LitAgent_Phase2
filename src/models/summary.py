"""Summary / analysis artifacts tied to papers."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SummaryCreate(BaseModel):
    """Payload for persisting a generated summary or analysis block."""

    model_config = ConfigDict(extra="forbid")

    paper_id: int = Field(gt=0)
    summary_text: str = Field(min_length=1)
    analysis_depth: str = Field(
        default="standard",
        description="Label describing depth tier (e.g. skim, deep, critique).",
    )


class SummaryRecord(SummaryCreate):
    """Stored summary row."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    created_date: datetime
