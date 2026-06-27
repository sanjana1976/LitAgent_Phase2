"""
AgentManager edge cases: confirmation denial, unknown tools, permission blocks.
"""

from __future__ import annotations

from unittest import mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.agent import AgentManager
from agent.errors import AgentError
from guardrails.permissions import PermissionManager
from tools.schemas import Paper


def _manager(monkeypatch: pytest.MonkeyPatch) -> AgentManager:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import config.config as cfg

    cfg.get_settings(reload=True)
    mock_llm = mock.MagicMock()
    mock_bound = mock.MagicMock()
    mock_llm.bind_tools.return_value = mock_bound

    with mock.patch("agent.agent.ChatOpenAI", return_value=mock_llm):
        mgr = AgentManager(
            api_key="test-key",
            model="gpt-4o",
            permission_manager=PermissionManager(cfg.get_settings(reload=True)),
        )
    mgr._llm = mock_bound
    return mgr


def test_agent_user_denies_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _manager(monkeypatch)
    calls = 0

    def fake_invoke(_messages: list) -> AIMessage:  # type: ignore[type-arg]
        nonlocal calls
        calls += 1
        if calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "w1",
                        "name": "tool_create_reading_list",
                        "args": {"name": "Test", "description": ""},
                    },
                ],
            )
        return AIMessage(content="User declined to create the list.")

    mgr._llm.invoke = fake_invoke  # type: ignore[method-assign]

    result = mgr.respond(
        history=[HumanMessage(content="Create a reading list called Test")],
        confirm_callback=lambda _p: False,
    )
    assert result.tool_calls_executed == 0
    assert "declined" in str(result.message.content).lower() or calls >= 2


def test_agent_unknown_tool_not_executed(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _manager(monkeypatch)
    calls = 0

    def fake_invoke(_messages: list) -> AIMessage:  # type: ignore[type-arg]
        nonlocal calls
        calls += 1
        if calls == 1:
            return AIMessage(
                content="",
                tool_calls=[{"id": "u1", "name": "tool_hallucinated_name", "args": {}}],
            )
        return AIMessage(content="That tool is not available.")

    mgr._llm.invoke = fake_invoke  # type: ignore[method-assign]
    result = mgr.respond(history=[HumanMessage(content="Do something")])
    assert result.tool_calls_executed == 0


def test_agent_blocked_tool_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GUARDRAILS_BLOCKED_TOOLS", '["tool_save_summary"]')
    import config.config as cfg

    cfg.get_settings(reload=True)
    mock_llm = mock.MagicMock()
    mock_bound = mock.MagicMock()
    mock_llm.bind_tools.return_value = mock_bound

    with mock.patch("agent.agent.ChatOpenAI", return_value=mock_llm):
        mgr = AgentManager(
            api_key="test-key",
            model="gpt-4o",
            permission_manager=PermissionManager(cfg.get_settings(reload=True)),
        )
    mgr._llm = mock_bound

    steps = 0

    def fake_invoke(_messages: list) -> AIMessage:  # type: ignore[type-arg]
        nonlocal steps
        steps += 1
        if steps == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "s1",
                        "name": "tool_save_summary",
                        "args": {
                            "paper_id": "1",
                            "summary_text": "x",
                            "depth": "shallow",
                        },
                    },
                ],
            )
        return AIMessage(content="Could not save.")

    mgr._llm.invoke = fake_invoke  # type: ignore[method-assign]
    result = mgr.respond(history=[HumanMessage(content="Save summary")])
    assert result.tool_calls_executed == 0


def test_agent_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import config.config as cfg

    cfg.get_settings(reload=True)
    with pytest.raises(AgentError, match="OPENAI_API_KEY"):
        AgentManager(api_key=None, model="gpt-4o")


def test_agent_max_steps_exceeded(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _manager(monkeypatch)
    mgr._max_steps = 2

    def fake_invoke(_messages: list) -> AIMessage:  # type: ignore[type-arg]
        return AIMessage(
            content="",
            tool_calls=[{"id": "t1", "name": "tool_search_arxiv", "args": {"query": "x"}}],
        )

    mgr._llm.invoke = fake_invoke  # type: ignore[method-assign]
    empty: list[Paper] = []
    mgr._tool_map["tool_search_arxiv"] = mgr._tool_map["tool_search_arxiv"].copy(
        update={"func": lambda *a, **k: empty}
    )
    with pytest.raises(AgentError, match="max tool-iteration"):
        mgr.respond(history=[HumanMessage(content="search forever")])
