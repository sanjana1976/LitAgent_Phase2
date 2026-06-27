"""Shared pytest fixtures (deterministic OPENAI_API_KEY; no live API calls)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _default_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a deterministic key unless a test clears ``OPENAI_API_KEY``."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-for-production")
