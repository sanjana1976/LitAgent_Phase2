"""
Agent behavior when all search providers return no hits.

Ensures the assistant does not invent BibTeX entries after empty search results.
Uses a scripted chat model and stubbed tools — no live OpenAI or network calls.
"""

from __future__ import annotations

import re

import pytest
from helpers_llm import ScriptedChatModel
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

_SEARCH_TOOLS = (
    "tool_search_arxiv",
    "tool_search_dblp",
    "tool_search_semantic_scholar",
    "tool_search_crossref",
)


def _empty_search(*_a: object, **_k: object) -> list[Paper]:
    return []


def _manager(monkeypatch: pytest.MonkeyPatch, responses: list[AIMessage]) -> AgentManager:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import config.config as cfg

    settings = cfg.get_settings(reload=True)
    return AgentManager(
        api_key="test-key",
        model="gpt-4o",
        permission_manager=PermissionManager(settings),
        model_instance=ScriptedChatModel(responses=responses),
        tool_overrides={name: _empty_search for name in _SEARCH_TOOLS},
    )


def test_empty_search_does_not_fabricate_bibtex(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    After stubbed empty searches, the final reply must acknowledge no hits and
    must not contain BibTeX entry blocks.
    """
    mgr = _manager(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"c{i}",
                        "name": name,
                        "args": {
                            "query": "Zyxnonexistent99",
                            "filters": {"author": "Zyxnonexistent99"},
                        },
                    }
                    for i, name in enumerate(_SEARCH_TOOLS, start=1)
                ],
            ),
            AIMessage(
                content=(
                    "I searched arXiv, DBLP, Semantic Scholar, and Crossref for author "
                    "'Zyxnonexistent99' but found no results. I cannot generate BibTeX "
                    "without verified paper metadata from a successful search."
                )
            ),
        ],
    )

    result = mgr.respond(
        history=[
            HumanMessage(
                content=(
                    "Find papers by author Zyxnonexistent99 and give me BibTeX "
                    "for the results."
                )
            )
        ],
    )

    text = (
        result.message.content
        if isinstance(result.message.content, str)
        else str(result.message.content)
    )
    assert _NO_RESULTS.search(text), f"Expected no-results wording, got: {text!r}"
    assert not _BIBTEX_ENTRY.search(text), f"Fabricated BibTeX in response: {text!r}"
    assert result.tool_calls_executed == 4


def test_output_guardrail_strips_fabricated_bibtex(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the model invents BibTeX after empty searches, the output guardrail replaces it."""
    fake_bib = "@article{zyx2024,\n  title = {Totally Fabricated},\n  author = {Z. Fake},\n}"
    mgr = _manager(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "tool_search_arxiv",
                        "args": {"query": "Zyxnonexistent99"},
                    },
                ],
            ),
            AIMessage(content=f"Here is your BibTeX:\n{fake_bib}"),
        ],
    )

    result = mgr.respond(
        history=[HumanMessage(content="BibTeX for author Zyxnonexistent99")],
    )

    text = (
        result.message.content
        if isinstance(result.message.content, str)
        else str(result.message.content)
    )
    assert text == EMPTY_SEARCH_SAFE_REPLY
    assert fake_bib not in text
