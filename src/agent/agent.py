"""OpenAI-backed tool-using agent manager for research workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI

from agent.errors import AgentError
from db.database import Database, DatabaseError
from db.queries import insert_permission_audit
from guardrails.output import TurnToolTracker, apply_output_guardrails
from guardrails.permissions import GuardrailError, PermissionManager
from tools.context import set_tool_session_id
from tools.registry import get_registered_tools
from tools.tools_registry import list_tool_signatures

logger = logging.getLogger(__name__)

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
    OpenAI + tool orchestration loop with guardrails and confirmations.
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
    ) -> None:
        if not api_key:
            raise AgentError("OPENAI_API_KEY is required for agent chat.")

        self._database = database
        self._permission_manager = permission_manager
        self._system_prompt = system_prompt
        self._max_steps = 8
        self._tool_map: dict[str, BaseTool] = {}
        for fn in get_registered_tools():
            tool = StructuredTool.from_function(fn)
            self._tool_map[tool.name] = tool

        tool_signatures = list_tool_signatures()
        signatures_blob = "\n".join(f"- {name}: {sig}" for name, sig in tool_signatures.items())
        self._llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        ).bind_tools(list(self._tool_map.values()))
        self._system_message = SystemMessage(
            content=(
                f"{self._system_prompt}\n\nAvailable tools and signatures:\n{signatures_blob}\n"
                "When a user asks for destructive actions, refuse and explain policy."
            )
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
        Run iterative tool-use loop until a final assistant answer is produced.
        """
        set_tool_session_id(session_id)
        messages: list[BaseMessage] = [self._system_message]
        if context_note:
            messages.append(SystemMessage(content=f"Session context: {context_note}"))
        messages.extend(history)

        executed = 0
        tool_tracker = TurnToolTracker()
        for _step in range(self._max_steps):
            try:
                model_msg: AIMessage = self._llm.invoke(messages)  # type: ignore[assignment]
            except Exception as exc:  # noqa: BLE001
                logger.exception("OpenAI invocation failed")
                raise AgentError("Language model invocation failed.") from exc

            if not isinstance(model_msg, AIMessage):
                raise AgentError("Unexpected model response shape.")
            messages.append(model_msg)

            tool_calls = model_msg.tool_calls or []
            if not tool_calls:
                content = model_msg.content
                if isinstance(content, str):
                    sanitized = apply_output_guardrails(content, tool_tracker)
                    if sanitized != content:
                        model_msg = AIMessage(
                            content=sanitized,
                            additional_kwargs=getattr(model_msg, "additional_kwargs", {}),
                            response_metadata=getattr(model_msg, "response_metadata", {}),
                        )
                return AgentTurnResult(message=model_msg, tool_calls_executed=executed)

            for call in tool_calls:
                tool_name = call.get("name", "")
                args = call.get("args", {}) or {}
                call_id = call.get("id", tool_name)
                if tool_name not in self._tool_map:
                    messages.append(
                        ToolMessage(
                            tool_call_id=call_id,
                            content=f"Unknown tool: {tool_name}",
                        )
                    )
                    continue

                if self._permission_manager is not None:
                    decision = self._permission_manager.check_permission(tool_name, "execute")
                    self._log_permission(
                        session_id=session_id,
                        tool_name=tool_name,
                        action="execute",
                        allowed=decision.allowed,
                        needs_confirmation=decision.needs_confirmation,
                        reason=decision.reason,
                    )
                    if not decision.allowed:
                        messages.append(
                            ToolMessage(tool_call_id=call_id, content=decision.reason or "Blocked")
                        )
                        continue
                    if decision.needs_confirmation:
                        if confirm_callback is None:
                            messages.append(
                                ToolMessage(
                                    tool_call_id=call_id,
                                    content=f"User confirmation required for {tool_name}; not available.",
                                )
                            )
                            continue
                        approved = confirm_callback(
                            f"Confirm execution of {tool_name} with args={args}? (yes/no)"
                        )
                        self._log_permission(
                            session_id=session_id,
                            tool_name=tool_name,
                            action="confirm",
                            allowed=approved,
                            needs_confirmation=True,
                            user_decision="yes" if approved else "no",
                            reason="user confirmation for write/mutation",
                        )
                        if not approved:
                            messages.append(
                                ToolMessage(
                                    tool_call_id=call_id,
                                    content=f"User denied execution of {tool_name}.",
                                )
                            )
                            continue
                        args = {**args, "user_confirmed": True}

                try:
                    if (
                        self._permission_manager is not None
                        and tool_name == "tool_export_list_to_bibtex"
                        and isinstance(args, dict)
                        and "filename" in args
                    ):
                        self._permission_manager.validate_filesystem_target(str(args["filename"]))
                    output = self._tool_map[tool_name].invoke(args)
                    executed += 1
                except GuardrailError as exc:
                    output = str(exc)
                except Exception as exc:  # noqa: BLE001
                    output = f"Tool error: {exc}"
                output_text = output if isinstance(output, str) else str(output)
                tool_tracker.record(tool_name, output_text)
                messages.append(
                    ToolMessage(
                        tool_call_id=call_id,
                        content=output_text,
                    )
                )

        raise AgentError("Agent exceeded max tool-iteration steps without final response.")
