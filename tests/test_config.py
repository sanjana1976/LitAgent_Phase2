from __future__ import annotations

from pathlib import Path

import pytest

from config.config import AppSettings, get_settings


def test_settings_loads_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("DATABASE_PATH", "/tmp/custom.sqlite3")
    settings = get_settings(reload=True)
    assert settings.openai_api_key == "k"
    assert settings.database_path == Path("/tmp/custom.sqlite3").resolve()
    assert "arxiv.org" in settings.arxiv_api_base_url


def test_settings_accepts_missing_api_key_for_db_only_flows(monkeypatch: pytest.MonkeyPatch) -> None:
    # Non-empty .env would win over a missing env var; empty string in the environment overrides .env.
    monkeypatch.setenv("OPENAI_API_KEY", "")
    settings = get_settings(reload=True)
    assert settings.openai_api_key is None


def test_guardrails_tool_lists_include_forward_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    settings = get_settings(reload=True)
    assert "tool_lookup_forward_citations" in settings.guardrails_autonomous_tools
    assert "tool_add_paper_to_list" in settings.guardrails_confirmation_tools
    assert "tool_search_arxiv" in settings.guardrails_autonomous_tools
