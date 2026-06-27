"""Shapes for conversational turns mirrored to SQLite."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ConversationTurnIn(BaseModel):
    """Inbound user/agent pair for archival."""

    model_config = ConfigDict(extra="forbid")

    user_message: str = Field(min_length=1)
    agent_response: str = Field(min_length=1)
    reading_list_context: str | None = Field(
        default=None,
        description="Optional opaque context key (list id/name) serialized as text.",
    )
    session_id: str | None = Field(
        default=None,
        description="Conversation session identifier for multi-turn resume support.",
    )


class ConversationTurnOut(ConversationTurnIn):
    """Turn as read back from persistence."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    timestamp: datetime
