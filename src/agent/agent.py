"""
LangGraph-backed tool-using agent manager for research workflows.

The chat agent is a ``langchain.agents.create_agent`` (LangGraph prebuilt)
graph over the registered research tools. Permission policy is enforced
*inside* each tool via a guard wrapper:

- blocked tools return their policy reason instead of executing;
- confirmation-tier tools pause the whole graph with ``interrupt()`` — the
  caller (CLI) answers via its confirm callback and the graph resumes with
  ``Command(resume=...)``, re-entering the tool with the user's decision;
- every decision is written to the permission audit table.

``AgentManager.respond`` keeps its original synchronous contract so the
conversation layer, CLI, and transcripts are unaffected by the orchestration
swap.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Sequence
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command, interrupt

from agent.errors import AgentError
from db.database import Database, DatabaseError
from db.queries import insert_permission_audit
from guardrails.output import TurnToolTracker, apply_output_guardrails
from guardrails.permissions import GuardrailError, PermissionManager
from tools.context import get_tool_session_id, set_tool_session_id
from tools.registry import get_registered_tools
from tools.tools_registry import list_tool_signatures

logger = logging.getLogger(__name__)

# Tool outputs with this prefix mark permission refusals (blocked or denied);
# they are shown to the model as tool results but never counted as executions.
_PERMISSION_PREFIX = "[permission]"
_TOOL_ERROR_PREFIX = "Tool error:"

DEFAULT_SYSTEM_PROMPT = """
You are Research Paper Analyzer, an expert assistant for graduate-level literature discovery, organization, and deep technical analysis. Your job is to help users build reading lists, inspect papers beyond the abstract, compare approaches rigorously, and generate high-quality citation outputs.

Core behavior:
- Prefer evidence-backed analysis from tool outputs over generic statements.
- For deep analysis, explicitly identify: problem framing, method assumptions, key equations or algorithmic components, datasets and splits, metrics, baselines, ablations, failure modes, reproducibility signals (code/data availability), and practical limitations.
- If the user asks for recommendations, state criteria and trade-offs clearly.
- If any request is ambiguous (scope, paper identity, metric preference, timeframe, depth), ask concise clarifying questions before acting.

Tool-use policy:
- Use search tools for discovery and metadata (`tool_search_*`).
- Use PDF and analysis tools (`tool_fetch_and_parse_pdf`, `tool_deep_analyze_paper`, `tool_extract_citations`) when the user asks for technical depth.
- Use `tool_lookup_forward_citations` when the user asks who cited a paper or what work built on it (forward citations).
- Use `tool_compare_papers` when a comparative judgment is requested.
- If all search tools return no papers, say so explicitly and do not invent BibTeX or other metadata.
- Use bibliography tools (`tool_generate_*`) for citation formatting requests.
- Use reading-list tools (`tool_list_all_lists`, `tool_get_list_contents`, `tool_create_reading_list`, `tool_add_paper_to_list`, `tool_remove_paper_from_list`) for organization tasks.
- Use storage tools (`tool_save_summary`, `tool_export_list_to_bibtex`) only when explicitly useful.
- Use `tool_synthesize_literature_review` when the user asks for a literature review, a synthesis, "what does the literature say about X", "competing approaches to X", or any request that requires reading multiple papers and producing an argument-driven section with citations. The tool returns JSON with `review_text`, `papers` (each with `paper_id`, `citation_key`, `url`), `hallucinated_citations`, and `contradictions`. After calling it, your reply to the user MUST be structured as follows so that follow-up questions have the paper identifiers available:
    1. Quote the `review_text` verbatim.
    2. Add a `### Papers used` section listing every paper from the `papers` field as a bullet in the exact format: ``- `<citation_key>` — paper_id=`<paper_id>` — url=<url>``. If a `url` is missing, write `url=n/a`.
    3. If `hallucinated_citations` is non-empty, add a `### Hallucinated citations (flagged)` section listing them.
    4. If `contradictions` is non-empty, add a `### Contradictions` section summarizing each one in plain English.
  This structured "Papers used" block is mandatory — it is the only way you and the user can refer back to specific papers in later turns. Do NOT omit it.
- Use `tool_get_review_context` when the user asks a follow-up about a previous literature review ("those papers", "the second paper", "which of them evaluate on X", "summarize the review again") and the paper identifiers are not already visible in this conversation — for example after resuming a session. It returns the last review's question, text, and full paper list for this session. Prefer answering follow-ups from that retained paper set; call `tool_fetch_and_parse_pdf` on a specific paper's URL when the user wants more depth than the review provides.

Guardrails (always enforce):
- Never delete papers or reading lists.
- Never fabricate paper metadata when APIs fail; instead say: "API unavailable, try another source."
- Never assist with paywall bypass; redirect users to institutional/library access.
- Never modify source PDF files.
- Never access or write files outside the project directory.
- Always require explicit user confirmation before any write/mutation operation.

Multi-turn memory:
- Use conversation context from prior turns and session state.
- Maintain continuity across follow-up questions, references like "that paper", and resumed sessions.
- If context is insufficient, ask a short disambiguation question.

Examples:
1) User: "Find recent graph RAG papers and compare top 3."
   Assistant: Uses search tools, asks for timeframe if missing, then compares methods/metrics and highlights reproducibility.
2) User: "Save this summary to paper 12."
   Assistant: Confirms before writing, then executes `tool_save_summary`.
3) User: "Delete list 4."
   Assistant: Refuses, explains data-loss guardrail, suggests removing items manually instead.
""".strip()


@dataclass(frozen=True)
class AgentTurnResult:
    """Container for a finalized assistant message and tool execution notes."""

    message: AIMessage
    tool_calls_executed: int


class ResearchPaperAgent:
    """
    Backwards-compatible wrapper around the new AgentManager.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.2,
        max_tokens: int = 4_096,
    ) -> None:
        self._manager = AgentManager(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate_reply(self, history: Sequence[BaseMessage]) -> AIMessage:
        """
        Produce the next assistant message given full prior ``history``.

        Raises:
            AgentError: on provider errors after guard-friendly logging.
        """
        result = self._manager.respond(history=list(history))
        return result.message


class AgentManager:
    """
    LangGraph agent + permission-guarded tools with interrupt confirmations.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        permission_manager: PermissionManager | None = None,
        database: Database | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.1,
        max_tokens: int = 4_096,
        model_instance: BaseChatModel | None = None,
        tool_overrides: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        """
        Args:
            model_instance: Optional pre-built chat model (tests inject a
                scripted model here). When omitted, a ``ChatOpenAI`` is built
                from ``api_key``/``model``.
            tool_overrides: Optional map of tool name -> replacement callable.
                The original tool's schema and description are kept; only the
                executed function is swapped (tests stub network tools here).
        """
        if not api_key and model_instance is None:
            raise AgentError("OPENAI_API_KEY is required for agent chat.")

        self._database = database
        self._permission_manager = permission_manager
        self._system_prompt = system_prompt
        self._max_steps = 8

        overrides = tool_overrides or {}
        guarded_tools: list[StructuredTool] = []
        for fn in get_registered_tools():
            base = StructuredTool.from_function(fn)
            inner = overrides.get(base.name, fn)
            guarded_tools.append(self._build_guarded_tool(base, inner))

        tool_signatures = list_tool_signatures()
        signatures_blob = "\n".join(f"- {name}: {sig}" for name, sig in tool_signatures.items())
        chat_model: BaseChatModel = (
            model_instance
            if model_instance is not None
            else ChatOpenAI(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        # The in-memory checkpointer exists to support interrupt()/resume for
        # tool confirmations; each respond() call runs on a fresh thread id.
        self._agent = create_agent(
            chat_model,
            guarded_tools,
            checkpointer=InMemorySaver(),
        )
        self._system_message = SystemMessage(
            content=(
                f"{self._system_prompt}\n\nAvailable tools and signatures:\n{signatures_blob}\n"
                "When a user asks for destructive actions, refuse and explain policy."
            )
        )

    def _build_guarded_tool(
        self, base: StructuredTool, inner: Callable[..., Any]
    ) -> StructuredTool:
        """
        Wrap one registered tool in the permission guard.

        The wrapper keeps the original schema (so the model sees the same
        tool signature) and enforces policy at execution time. Confirmation
        pauses the graph via ``interrupt()``; on resume the wrapper re-runs
        from the top and ``interrupt()`` returns the user's decision.
        """
        name = base.name
        schema = base.args_schema
        if isinstance(schema, dict):  # JSON-schema shape (langchain >= 1.x)
            schema_fields = schema.get("properties", {}) or {}
        else:  # pydantic-model shape
            schema_fields = getattr(schema, "model_fields", {}) or {}
        try:
            inner_params = inspect.signature(inner).parameters
        except (TypeError, ValueError):
            inner_params = {}
        accepts_confirm_flag = (
            "user_confirmed" in schema_fields or "user_confirmed" in inner_params
        )

        def guarded(**kwargs: Any) -> str:
            pm = self._permission_manager
            if pm is not None:
                decision = pm.check_permission(name, "execute")
                if not decision.allowed:
                    self._log_permission(
                        session_id=get_tool_session_id(),
                        tool_name=name,
                        action="execute",
                        allowed=False,
                        needs_confirmation=False,
                        reason=decision.reason,
                    )
                    return f"{_PERMISSION_PREFIX} {decision.reason or 'Blocked by policy.'}"

                if decision.needs_confirmation:
                    # Pauses the whole graph; respond() collects the user's
                    # answer and resumes. NOTE: nothing below this line runs
                    # on the first pass — logging must stay after the
                    # interrupt so it fires exactly once.
                    approved = bool(interrupt({"tool": name, "args": kwargs}))
                    self._log_permission(
                        session_id=get_tool_session_id(),
                        tool_name=name,
                        action="confirm",
                        allowed=approved,
                        needs_confirmation=True,
                        user_decision="yes" if approved else "no",
                        reason="user confirmation for write/mutation",
                    )
                    if not approved:
                        return f"{_PERMISSION_PREFIX} User denied execution of {name}."
                    if accepts_confirm_flag:
                        kwargs = {**kwargs, "user_confirmed": True}
                else:
                    self._log_permission(
                        session_id=get_tool_session_id(),
                        tool_name=name,
                        action="execute",
                        allowed=True,
                        needs_confirmation=False,
                        reason=decision.reason,
                    )

                if name == "tool_export_list_to_bibtex" and "filename" in kwargs:
                    try:
                        pm.validate_filesystem_target(str(kwargs["filename"]))
                    except GuardrailError as exc:
                        return f"{_PERMISSION_PREFIX} {exc}"

            try:
                output = inner(**kwargs)
            except GuardrailError as exc:
                return str(exc)
            except Exception as exc:  # noqa: BLE001 - errors go back to the model
                return f"{_TOOL_ERROR_PREFIX} {exc}"
            return output if isinstance(output, str) else str(output)

        return StructuredTool(
            name=name,
            description=base.description,
            args_schema=base.args_schema,
            func=guarded,
        )

    def _log_permission(
        self,
        *,
        session_id: str | None,
        tool_name: str,
        action: str,
        allowed: bool,
        needs_confirmation: bool,
        user_decision: str | None = None,
        reason: str | None = None,
    ) -> None:
        if self._database is None:
            return
        try:
            insert_permission_audit(
                self._database,
                session_id=session_id,
                tool_name=tool_name,
                action=action,
                allowed=allowed,
                needs_confirmation=needs_confirmation,
                user_decision=user_decision,
                reason=reason,
            )
        except DatabaseError:
            logger.exception("Failed to write permission audit log")

    def respond(
        self,
        *,
        history: Sequence[BaseMessage],
        context_note: str | None = None,
        session_id: str | None = None,
        confirm_callback: Callable[[str], bool] | None = None,
    ) -> AgentTurnResult:
        """
        Run the agent graph until a final assistant answer is produced.

        Confirmation-tier tool calls surface as graph interrupts: the run
        pauses, ``confirm_callback`` is asked (no callback means denial), and
        the graph resumes with the decision. Step budget is enforced by the
        graph itself — when exhausted it returns a final message instead of
        looping forever.
        """
        set_tool_session_id(session_id)
        messages: list[BaseMessage] = [self._system_message]
        if context_note:
            messages.append(SystemMessage(content=f"Session context: {context_note}"))
        messages.extend(history)

        run_config: dict[str, Any] = {
            "configurable": {"thread_id": uuid4().hex},
            "recursion_limit": 2 * self._max_steps + 1,
        }

        try:
            result = self._agent.invoke({"messages": messages}, run_config)
            while result.get("__interrupt__"):
                resume: dict[str, Any] = {}
                for intr in result["__interrupt__"]:
                    payload = intr.value if isinstance(intr.value, dict) else {}
                    tool_name = str(payload.get("tool", "unknown tool"))
                    tool_args = payload.get("args", {})
                    approved = False
                    if confirm_callback is not None:
                        approved = bool(
                            confirm_callback(
                                f"Confirm execution of {tool_name} with args={tool_args}? (yes/no)"
                            )
                        )
                    resume[intr.id] = approved
                result = self._agent.invoke(Command(resume=resume), run_config)
        except GraphRecursionError as exc:
            raise AgentError(
                "Agent exceeded max tool-iteration steps without final response."
            ) from exc
        except AgentError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent graph invocation failed")
            raise AgentError("Language model invocation failed.") from exc

        out_messages: list[BaseMessage] = list(result.get("messages", []))
        tool_tracker = TurnToolTracker()
        executed = 0
        for message in out_messages:
            if not isinstance(message, ToolMessage):
                continue
            text = message.content if isinstance(message.content, str) else str(message.content)
            tool_tracker.record(message.name or "", text)
            if (
                getattr(message, "status", None) != "error"
                and not text.startswith(_PERMISSION_PREFIX)
                and not text.startswith(_TOOL_ERROR_PREFIX)
            ):
                executed += 1

        final = out_messages[-1] if out_messages else None
        if not isinstance(final, AIMessage):
            raise AgentError("Unexpected model response shape.")

        content = final.content
        if isinstance(content, str):
            sanitized = apply_output_guardrails(content, tool_tracker)
            if sanitized != content:
                final = AIMessage(
                    content=sanitized,
                    additional_kwargs=getattr(final, "additional_kwargs", {}),
                    response_metadata=getattr(final, "response_metadata", {}),
                )
        return AgentTurnResult(message=final, tool_calls_executed=executed)
