"""Lightweight input validation prior to model calls."""

from __future__ import annotations

import logging

from guardrails.permissions import GuardrailError

logger = logging.getLogger(__name__)

_MAX_USER_MESSAGE_CHARS = 32_000


def validate_user_message(text: str) -> str:
    """
    Normalize and validate user-provided natural language.

    Args:
        text: Raw user text from CLI or API.

    Returns:
        Stripped message suitable for logging and model input.

    Raises:
        GuardrailError: on empty or oversized payloads.
    """
    cleaned = text.strip()
    if not cleaned:
        raise GuardrailError("User message cannot be empty.")
    if len(cleaned) > _MAX_USER_MESSAGE_CHARS:
        logger.warning("User message exceeded max length; rejecting before model call.")
        raise GuardrailError(
            f"User message exceeds maximum length of {_MAX_USER_MESSAGE_CHARS} characters."
        )
    return cleaned
