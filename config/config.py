"""
Central configuration for API endpoints, LLM defaults, persistence, and logging.

Loads secrets from environment variables; never commits API keys.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# -----------------------------------------------------------------------------
# Constants (paper search API base URLs — override via settings if extended)
# -----------------------------------------------------------------------------

ARXIV_API_BASE_URL: Final[str] = "https://export.arxiv.org/api/query"
"""arXiv Atom API endpoint for querying and atom feeds."""

DBLP_API_BASE_URL: Final[str] = "https://dblp.org/search/publ/api"
"""DBLP search API."""

SEMANTIC_SCHOLAR_API_BASE_URL: Final[str] = "https://api.semanticscholar.org/graph/v1"
"""Semantic Scholar Graph API."""

CROSSREF_API_BASE_URL: Final[str] = "https://api.crossref.org"
"""Crossref REST API."""

DEFAULT_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DB_FILENAME: Final[str] = "papers.sqlite3"


class AppSettings(BaseSettings):
    """
    Typed application settings sourced from environment and optional `.env`.

    Raises validation errors early if required values are malformed.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = Field(
        default=None,
        alias="OPENAI_API_KEY",
        description="Secret key for OpenAI API used by chat runtime and analysis helpers.",
    )
    openai_model: str = Field(
        default="gpt-4o",
        alias="OPENAI_MODEL",
        description="OpenAI model id for LangChain ChatOpenAI and analysis helpers.",
    )
    database_path: Path = Field(
        default_factory=lambda: Path.cwd() / "data" / DEFAULT_DB_FILENAME,
        alias="DATABASE_PATH",
        description="Filesystem path for the SQLite database file.",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    arxiv_api_base_url: str = Field(default=ARXIV_API_BASE_URL, alias="ARXIV_API_BASE_URL")
    dblp_api_base_url: str = Field(default=DBLP_API_BASE_URL, alias="DBLP_API_BASE_URL")
    semantic_scholar_api_base_url: str = Field(
        default=SEMANTIC_SCHOLAR_API_BASE_URL,
        alias="SEMANTIC_SCHOLAR_API_BASE_URL",
    )
    crossref_api_base_url: str = Field(
        default=CROSSREF_API_BASE_URL,
        alias="CROSSREF_API_BASE_URL",
    )
    project_root: Path = Field(
        default_factory=lambda: Path.cwd().resolve(),
        alias="PROJECT_ROOT",
        description="Absolute root directory; file writes must stay within this boundary.",
    )
    guardrails_autonomous_tools: tuple[str, ...] = Field(
        default=(
            "tool_search_arxiv",
            "tool_search_dblp",
            "tool_search_semantic_scholar",
            "tool_search_crossref",
            "tool_fetch_and_parse_pdf",
            "tool_deep_analyze_paper",
            "tool_extract_citations",
            "tool_lookup_forward_citations",
            "tool_compare_papers",
            "tool_generate_bibtex",
            "tool_generate_apa",
            "tool_generate_chicago",
            "tool_list_all_lists",
            "tool_get_list_contents",
            "tool_synthesize_literature_review",
        ),
        alias="GUARDRAILS_AUTONOMOUS_TOOLS",
    )
    guardrails_confirmation_tools: tuple[str, ...] = Field(
        default=(
            "tool_create_reading_list",
            "tool_add_paper_to_list",
            "tool_remove_paper_from_list",
            "tool_save_summary",
            "tool_export_list_to_bibtex",
        ),
        alias="GUARDRAILS_CONFIRMATION_TOOLS",
    )
    guardrails_blocked_tools: tuple[str, ...] = Field(
        default=(),
        alias="GUARDRAILS_BLOCKED_TOOLS",
    )

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def _empty_openai_api_key_none(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("openai_model")
    @classmethod
    def _nonempty_openai_model(cls, value: str) -> str:
        stripped = (value or "").strip()
        return stripped or "gpt-4o"

    @field_validator("database_path", mode="before")
    @classmethod
    def _coerce_db_path(cls, value: Path | str) -> Path:
        return Path(value).expanduser().resolve()

    @field_validator("project_root", mode="before")
    @classmethod
    def _coerce_project_root(cls, value: Path | str) -> Path:
        return Path(value).expanduser().resolve()


_settings_singleton: AppSettings | None = None


def get_settings(*, reload: bool = False) -> AppSettings:
    """
    Return process-wide cached settings unless ``reload`` is True.

    Use ``reload=True`` in tests that patch environment variables.
    """
    global _settings_singleton
    if reload or _settings_singleton is None:
        _settings_singleton = AppSettings()
    return _settings_singleton


def setup_logging(
    *,
    level: str | None = None,
    log_format: str = DEFAULT_LOG_FORMAT,
) -> None:
    """
    Configure root logging once (idempotent for repeated CLI invocations).

    Args:
        level: Log level name (defaults to ``AppSettings.log_level``).
        log_format: Format string passed to ``logging.Formatter``.
    """
    resolved = level or get_settings(reload=False).log_level.upper()
    numeric_level = getattr(logging, resolved, logging.INFO)
    root = logging.getLogger()

    # Avoid duplicate handlers when re-invoked from tests or REPL shells.
    if root.handlers:
        root.setLevel(numeric_level)
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(log_format))
    root.addHandler(handler)
    root.setLevel(numeric_level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
