"""
Agent behavior when all search providers return no hits.

Ensures the assistant does not invent BibTeX entries after empty search results.
Uses a mocked LLM — no live OpenAI calls.
"""

from __future__ import annotations

import re
from unittest import mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.agent import AgentManager
from guardrails.output import EMPTY_SEARCH_SAFE_REPLY
from guardrails.permissions import PermissionManager
from tools.schemas import Paper


_BIBTEX_ENTRY = re.compile(r"@(?:article|inproceedings|book|misc)\s*\{", re.I)
_NO_RESULTS = re.compile(
    r"no\s+(?:papers?\s+)?(?:were\s+)?found|found\s+no\s+(?:papers?|results)|"
    r"no\s+results|zero\s+results|could\s+not\s+find",
    re.I,
)


@pytest.fixture
def mock_empty_searches(monkeypatch: pytest.MonkeyPatch) -> None:
    """All four scholarly search tools return empty lists."""
    empty: list[Paper] = []

    def _empty_search(*_a: object, **_k: object) -> list[Paper]:
        return empty

    import tools.search_tools as search_mod
    import tools.tools_registry as registry_mod

    for name in (
        "tool_search_arxiv",
        "tool_search_dblp",
        "tool_search_semantic_scholar",
        "tool_search_crossref",
    ):
        monkeypatch.setattr(search_mod, name, _empty_search)
    monkeypatch.setattr(registry_mod, "TOOL_SPECS", registry_mod.build_tool_specs())


def test_empty_search_does_not_fabricate_bibtex(
    monkeypatch: pytest.MonkeyPatch,
    mock_empty_searches: None,
) -> None:
    """
    After mocked empty searches, the final reply must acknowledge no hits and
    must not contain BibTeX entry blocks.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import config.config as cfg

    cfg.get_settings(reload=True)

    invoke_calls = 0

    def fake_invoke(messages: list) -> AIMessage:  # type: ignore[type-arg]
        nonlocal invoke_calls
        invoke_calls += 1
        if invoke_calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "tool_search_arxiv",
                        "args": {"query": "Zyxnonexistent99", "filters": {"author": "Zyxnonexistent99"}},
                    },
                    {
                        "id": "c2",
                        "name": "tool_search_dblp",
                        "args": {"query": "Zyxnonexistent99", "filters": {"author": "Zyxnonexistent99"}},
                    },
                    {
                        "id": "c3",
                        "name": "tool_search_semantic_scholar",
                        "args": {"query": "Zyxnonexistent99", "filters": {"author": "Zyxnonexistent99"}},
                    },
                    {
                        "id": "c4",
                        "name": "tool_search_crossref",
                        "args": {"query": "Zyxnonexistent99", "filters": {"author": "Zyxnonexistent99"}},
                    },
                ],
            )
        return AIMessage(
            content=(
                "I searched arXiv, DBLP, Semantic Scholar, and Crossref for author "
                "'Zyxnonexistent99' but found no results. I cannot generate BibTeX without "
                "verified paper metadata from a successful search."
            )
        )

    mock_llm = mock.MagicMock()
    mock_llm.invoke = fake_invoke
    mock_bound = mock.MagicMock()
    mock_bound.invoke = fake_invoke

    def _empty_search(*_a: object, **_k: object) -> list[Paper]:
        return []

    with mock.patch("agent.agent.ChatOpenAI", return_value=mock_llm):
        mock_llm.bind_tools.return_value = mock_bound
        manager = AgentManager(
            api_key="test-key",
            model="gpt-4o",
            permission_manager=PermissionManager(cfg.get_settings(reload=True)),
        )
        for tool_name in (
            "tool_search_arxiv",
            "tool_search_dblp",
            "tool_search_semantic_scholar",
            "tool_search_crossref",
        ):
            bound = manager._tool_map[tool_name]
            manager._tool_map[tool_name] = bound.copy(update={"func": _empty_search})
        result = manager.respond(
            history=[
                HumanMessage(
                    content=(
                        "Find papers by author Zyxnonexistent99 and give me BibTeX "
                        "for the results."
                    )
                )
            ],
        )

    text = result.message.content if isinstance(result.message.content, str) else str(
        result.message.content
    )
    assert _NO_RESULTS.search(text), f"Expected no-results wording, got: {text!r}"
    assert not _BIBTEX_ENTRY.search(text), f"Fabricated BibTeX in response: {text!r}"
    assert invoke_calls >= 2
    assert result.tool_calls_executed == 4


def test_output_guardrail_strips_fabricated_bibtex(
    monkeypatch: pytest.MonkeyPatch,
    mock_empty_searches: None,
) -> None:
    """When the model invents BibTeX after empty searches, the output guardrail replaces it."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import config.config as cfg

    cfg.get_settings(reload=True)

    invoke_calls = 0
    fake_bib = "@article{zyx2024,\n  title = {Totally Fabricated},\n  author = {Z. Fake},\n}"

    def fake_invoke(messages: list) -> AIMessage:  # type: ignore[type-arg]
        nonlocal invoke_calls
        invoke_calls += 1
        if invoke_calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "tool_search_arxiv",
                        "args": {"query": "Zyxnonexistent99"},
                    },
                ],
            )
        return AIMessage(content=f"Here is your BibTeX:\n{fake_bib}")

    mock_llm = mock.MagicMock()
    mock_bound = mock.MagicMock()
    mock_bound.invoke = fake_invoke

    def _empty_search(*_a: object, **_k: object) -> list[Paper]:
        return []

    with mock.patch("agent.agent.ChatOpenAI", return_value=mock_llm):
        mock_llm.bind_tools.return_value = mock_bound
        manager = AgentManager(
            api_key="test-key",
            model="gpt-4o",
            permission_manager=PermissionManager(cfg.get_settings(reload=True)),
        )
        manager._tool_map["tool_search_arxiv"] = manager._tool_map[
            "tool_search_arxiv"
        ].copy(update={"func": _empty_search})
        result = manager.respond(
            history=[HumanMessage(content="BibTeX for author Zyxnonexistent99")],
        )

    text = result.message.content if isinstance(result.message.content, str) else str(
        result.message.content
    )
    assert text == EMPTY_SEARCH_SAFE_REPLY
    assert fake_bib not in text


