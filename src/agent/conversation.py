"""Conversation state manager with DB-backed session resume."""

from __future__ import annotations

import logging
import uuid
from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent.errors import AgentError
from db.database import Database, DatabaseError
from db.queries import (
    clear_conversation_session,
    fetch_conversation_by_session,
    fetch_recent_conversation_summaries,
    get_latest_synthesis_run_for_session,
    insert_conversation_turn,
    list_reading_list_rows,
)
from guardrails.permissions import GuardrailError
from guardrails.validators import validate_user_message
from models.conversation_turn import ConversationTurnIn

logger = logging.getLogger(__name__)


def _assistant_text(message: AIMessage) -> str:
    """Normalize assistant output to plain text suitable for archival."""
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


class ConversationManager:
    """
    Owns conversational state for a single logical session.

    Persisting turns is best-effort: DB failures are logged without crashing the UX.
    """

    def __init__(
        self,
        agent: object | None = None,
        database: Database | None = None,
        *,
        session_id: str | None = None,
        reading_list_context: str | None = None,
        seed_history: Sequence[BaseMessage] | None = None,
    ) -> None:
        self._agent = agent
        self._database = database
        self._session_id = session_id or str(uuid.uuid4())
        self._reading_list_context = reading_list_context
        self._history: list[BaseMessage] = list(seed_history or [])

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def messages(self) -> Sequence[BaseMessage]:
        """Read-only projection of buffered LangChain messages."""
        return tuple(self._history)

    def load_session(self, session_id: str) -> int:
        """
        Load persisted turns into in-memory history for a previous session.

        Returns number of turns loaded.
        """
        if self._database is None:
            return 0
        rows = fetch_conversation_by_session(self._database, session_id)
        self._history.clear()
        for user_message, assistant_message in rows:
            self._history.append(HumanMessage(content=user_message))
            self._history.append(AIMessage(content=assistant_message))
        self._session_id = session_id
        return len(rows)

    def clear_history(self, *, clear_persisted: bool = False) -> int:
        """
        Clear in-memory history and optionally delete persisted rows for this session.
        """
        self._history.clear()
        if clear_persisted and self._database is not None:
            return clear_conversation_session(self._database, self._session_id)
        return 0

    def context_snapshot(self) -> str:
        """
        Build lightweight context string for the agent prompt.
        """
        parts: list[str] = [f"session_id={self._session_id}"]
        if self._reading_list_context:
            parts.append(f"reading_list_context={self._reading_list_context}")

        if self._database is not None:
            try:
                lists = list_reading_list_rows(self._database)
                list_frag = ", ".join(f"{r['id']}:{r['name']}" for r in lists[:8]) if lists else "none"
                parts.append(f"reading_lists={list_frag}")
            except DatabaseError:
                logger.exception("Could not load reading list context")
            try:
                recent = fetch_recent_conversation_summaries(self._database, limit=5)
                parts.append(f"recent_turn_count={len(recent)}")
            except DatabaseError:
                logger.exception("Could not load recent conversation context")
            try:
                run = get_latest_synthesis_run_for_session(self._database, self._session_id)
                if run is not None:
                    topic = str(run.get("question") or "").strip()
                    if topic:
                        parts.append(
                            f"last_review_topic={topic!r} "
                            "(recall papers via tool_get_review_context)"
                        )
            except DatabaseError:
                logger.exception("Could not load last synthesis run context")
        return " | ".join(parts)

    def append_turn(
        self,
        user_message: str,
        *,
        persist: bool = True,
        reading_list_context: str | None = None,
    ) -> AIMessage:
        """
        Append a validated user utterance, query the model, and optionally persist.

        Returns:
            The assistant ``AIMessage`` appended to rolling history.

        Raises:
            GuardrailError: if input violates local policies.
            AgentError: when the upstream model fails.
        """
        cleaned = validate_user_message(user_message)
        if self._agent is None:
            raise AgentError("ConversationManager.append_turn requires an agent instance.")

        ctx = reading_list_context if reading_list_context is not None else self._reading_list_context

        human = HumanMessage(content=cleaned)
        self._history.append(human)

        try:
            assistant = self._agent.generate_reply(self._history)  # type: ignore[attr-defined]
        except AgentError:
            self._history.pop()  # drop unfulfilled user turn on failure
            raise

        self._history.append(assistant)

        response_text = _assistant_text(assistant)
        if persist and self._database is not None:
            turn = ConversationTurnIn(
                user_message=cleaned,
                agent_response=response_text,
                reading_list_context=ctx,
                session_id=self._session_id,
            )
            try:
                row_id = insert_conversation_turn(self._database, turn)
                logger.info("Recorded conversation_turn id=%s", row_id)
            except DatabaseError:
                logger.exception("Failed persisting conversation turn; continuing.")

        return assistant

    def append_completed_turn(
        self,
        *,
        user_message: str,
        assistant_message: str,
        persist: bool = True,
        reading_list_context: str | None = None,
    ) -> AIMessage:
        """
        Append an already-generated assistant response and optionally persist it.
        """
        cleaned = validate_user_message(user_message)
        self._history.append(HumanMessage(content=cleaned))
        ai = AIMessage(content=assistant_message)
        self._history.append(ai)
        if persist and self._database is not None:
            ctx = reading_list_context if reading_list_context is not None else self._reading_list_context
            turn = ConversationTurnIn(
                user_message=cleaned,
                agent_response=assistant_message,
                reading_list_context=ctx,
                session_id=self._session_id,
            )
            try:
                insert_conversation_turn(self._database, turn)
            except DatabaseError:
                logger.exception("Failed persisting conversation turn; continuing.")
        return ai

