"""
AgentManager edge cases: confirmation denial, unknown tools, permission blocks.

The agent runs on a LangGraph ``create_agent`` graph; tests inject a scripted
chat model (``model_instance``) and stub tool callables (``tool_overrides``)
instead of touching internals.
"""

from __future__ import annotations

import pytest
from helpers_llm import ScriptedChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agent.agent import AgentManager
from agent.errors import AgentError
from guardrails.permissions import PermissionManager


def _manager(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[AIMessage],
    *,
    tool_overrides: dict | None = None,
) -> AgentManager:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import config.config as cfg

    settings = cfg.get_settings(reload=True)
    return AgentManager(
        api_key="test-key",
        model="gpt-4o",
        permission_manager=PermissionManager(settings),
        model_instance=ScriptedChatModel(responses=responses),
        tool_overrides=tool_overrides,
    )


def test_agent_user_denies_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _manager(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "w1",
                        "name": "tool_create_reading_list",
                        "args": {"name": "Test", "description": ""},
                    },
                ],
            ),
            AIMessage(content="User declined to create the list."),
        ],
    )

    prompts: list[str] = []

    def deny(prompt: str) -> bool:
        prompts.append(prompt)
        return False

    result = mgr.respond(
        history=[HumanMessage(content="Create a reading list called Test")],
        confirm_callback=deny,
    )
    assert result.tool_calls_executed == 0
    assert prompts and "tool_create_reading_list" in prompts[0]
    assert "declined" in str(result.message.content).lower()


def test_agent_confirmation_approved_executes_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict] = []

    def fake_create(
        name: str, description: str = "", user_confirmed: bool = False
    ) -> str:
        created.append({"name": name, "description": description, "user_confirmed": user_confirmed})
        return "created list id=7"

    mgr = _manager(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "w2",
                        "name": "tool_create_reading_list",
                        "args": {"name": "Approved", "description": ""},
                    },
                ],
            ),
            AIMessage(content="Created the list."),
        ],
        tool_overrides={"tool_create_reading_list": fake_create},
    )

    result = mgr.respond(
        history=[HumanMessage(content="Create a reading list called Approved")],
        confirm_callback=lambda _p: True,
    )
    assert result.tool_calls_executed == 1
    assert created and created[0]["name"] == "Approved"
    # Approval injects the confirmed flag for the underlying tool.
    assert created[0].get("user_confirmed") is True


def test_agent_no_callback_means_denial(monkeypatch: pytest.MonkeyPatch) -> None:
    executed: list[str] = []

    mgr = _manager(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "w3",
                        "name": "tool_create_reading_list",
                        "args": {"name": "NoCallback", "description": ""},
                    },
                ],
            ),
            AIMessage(content="Cannot confirm."),
        ],
        tool_overrides={
            "tool_create_reading_list": lambda **kw: executed.append("ran") or "done"
        },
    )

    result = mgr.respond(history=[HumanMessage(content="Create a list")])
    assert result.tool_calls_executed == 0
    assert executed == []


def test_agent_unknown_tool_not_executed(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _manager(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[{"id": "u1", "name": "tool_hallucinated_name", "args": {}}],
            ),
            AIMessage(content="That tool is not available."),
        ],
    )
    result = mgr.respond(history=[HumanMessage(content="Do something")])
    assert result.tool_calls_executed == 0


def test_agent_blocked_tool_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GUARDRAILS_BLOCKED_TOOLS", '["tool_save_summary"]')
    executed: list[str] = []

    mgr = _manager(
        monkeypatch,
        [
            AIMessage(
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
            ),
            AIMessage(content="Could not save."),
        ],
        tool_overrides={
            "tool_save_summary": lambda **kw: executed.append("ran") or "saved"
        },
    )

    result = mgr.respond(history=[HumanMessage(content="Save summary")])
    assert result.tool_calls_executed == 0
    assert executed == []


def test_agent_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import config.config as cfg

    cfg.get_settings(reload=True)
    with pytest.raises(AgentError, match="OPENAI_API_KEY"):
        AgentManager(api_key=None, model="gpt-4o")


def test_agent_max_steps_exceeded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that calls tools forever hits the step budget and raises AgentError."""
    monkeypatch.setenv("GUARDRAILS_BLOCKED_TOOLS", '["tool_save_summary"]')
    # The scripted model repeats its last response forever: an endless tool
    # call against a blocked tool (so nothing ever actually executes).
    mgr = _manager(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "loop1",
                        "name": "tool_save_summary",
                        "args": {
                            "paper_id": "1",
                            "summary_text": "x",
                            "depth": "shallow",
                        },
                    },
                ],
            ),
        ],
    )
    mgr._max_steps = 2

    with pytest.raises(AgentError, match="max tool-iteration"):
        mgr.respond(history=[HumanMessage(content="loop forever")])
