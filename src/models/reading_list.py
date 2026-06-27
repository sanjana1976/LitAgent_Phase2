"""Reading list models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ReadingListCreate(BaseModel):
    """Payload for creating a named reading list."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=4096)


class ReadingListRecord(ReadingListCreate):
    """Stored reading list."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    created_date: datetime
