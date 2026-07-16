"""Shared scripted chat model for agent tests (no network, no OpenAI)."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    """
    Chat model that replays a fixed list of ``AIMessage`` responses in order.

    The last response repeats if the agent asks for more turns than scripted,
    which keeps step-budget tests deterministic.
    """

    responses: list[AIMessage]
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        repeating = self.index >= len(self.responses)
        message = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        if repeating and message.tool_calls:
            # Real models never reuse tool_call ids; a repeated id makes the
            # agent router treat the call as already answered. Re-mint ids on
            # every repetition so looping-model tests behave like reality.
            from uuid import uuid4

            fresh_calls = [
                {**call, "id": uuid4().hex[:12]} for call in message.tool_calls
            ]
            message = AIMessage(content=message.content, tool_calls=fresh_calls)
        return ChatResult(generations=[ChatGeneration(message=message)])
