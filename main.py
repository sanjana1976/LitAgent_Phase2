"""
CLI entry point for the Research Paper Analyzer agent.

Commands: ``chat``, ``resume``, ``status``, ``init-db``.

Adds ``src/`` (packages) and the project root (for ``config/``) to ``sys.path`` so
``python main.py`` works without an editable install (``pip install -e .`` recommended).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent


def _ensure_import_paths() -> None:
    src = PROJECT_ROOT / "src"
    for path in (src, PROJECT_ROOT):
        serialized = str(path)
        if serialized not in sys.path:
            sys.path.insert(0, serialized)


_ensure_import_paths()

import json  # noqa: E402
import re  # noqa: E402
from datetime import datetime  # noqa: E402

from config.config import get_settings, setup_logging  # noqa: E402
from agent.agent import AgentManager  # noqa: E402
from agent.conversation import ConversationManager  # noqa: E402
from db.database import Database  # noqa: E402
from db.init_db import initialize_schema  # noqa: E402
from db.queries import (  # noqa: E402
    get_latest_synthesis_result_json_for_question,
    get_synthesis_run_result_json,
    list_recent_sessions,
    list_recent_synthesis_runs,
)
from guardrails.permissions import GuardrailError, PermissionManager  # noqa: E402
from agent.errors import AgentError  # noqa: E402
from cli.welcome import echo_chat_help, echo_chat_welcome  # noqa: E402
from synthesis.eval_harness import load_cases, score_synthesis, write_results  # noqa: E402
from synthesis.schemas import SynthesisResult  # noqa: E402
from synthesis.controller import ControllerConfig, run_agentic_synthesis  # noqa: E402

REVIEWS_DIR_NAME = "research reviews"


def _slugify_question(question: str, *, max_len: int = 40) -> str:
    """Render ``question`` as a filesystem-safe slug for default review filenames."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", question.lower()).strip("-")
    return (cleaned[:max_len] or "review").rstrip("-")


def _resolve_review_output_path(
    output: Path | None,
    *,
    question: str,
    project_root: Path,
) -> Path:
    """
    Always place the generated review under ``research reviews/`` unless an
    absolute path was given. When ``output`` is None, auto-generate a
    timestamped filename so every run captures a copy.
    """
    reviews_dir = project_root / REVIEWS_DIR_NAME
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return reviews_dir / f"review-{stamp}-{_slugify_question(question)}.md"
    if output.is_absolute():
        return output
    # Relative path: anchor it inside research reviews/, but only prepend the
    # folder if the user did not already include it.
    parts = output.parts
    if parts and parts[0].lower() == REVIEWS_DIR_NAME.lower():
        return project_root / output
    return reviews_dir / output


def _synthesis_result_from_json(raw: str) -> SynthesisResult:
    """Parse persisted synthesis result JSON into the pydantic result model."""
    return SynthesisResult.model_validate(json.loads(raw))


@click.group()
@click.option(
    "--log-level",
    type=str,
    default=None,
    help="Override LOG_LEVEL env (DEBUG, INFO, WARNING, ERROR).",
)
def cli(log_level: str | None) -> None:
    """Research Paper Analyzer — multi-turn CLI (search, PDF, lists, citations)."""
    setup_logging(level=log_level)


@cli.command("init-db")
@click.option(
    "--database-path",
    type=click.Path(path_type=Path),
    default=None,
    help="SQLite file location (defaults to settings.database_path).",
)
def init_db(database_path: Path | None) -> None:
    """Create tables and indexes if they do not already exist."""
    settings = get_settings()
    db_path = database_path or settings.database_path
    db = Database(db_path)
    initialize_schema(db)
    click.echo(f"Initialized schema at {db.path}")


@cli.command("chat")
@click.option("--reading-list-context", type=str, default=None, help="Optional context key for archiving.")
@click.option("--persist/--no-persist", default=True, help="Persist conversation turns to SQLite.")
@click.option("--session-id", type=str, default=None, help="Resume a specific conversation session id.")
@click.option(
    "--verbose",
    is_flag=True,
    help="Show INFO logs from tools and APIs (default: quiet terminal).",
)
def chat(
    reading_list_context: str | None,
    persist: bool,
    session_id: str | None,
    verbose: bool,
) -> None:
    """Start full REPL with OpenAI tool-calling, guardrails, and session resume."""
    ctx = click.get_current_context()
    parent_level = ctx.parent.params.get("log_level") if ctx.parent else None
    if not verbose and parent_level is None:
        setup_logging(level="WARNING")

    settings = get_settings()
    api_key = settings.openai_api_key
    model = settings.openai_model
    if not api_key:
        raise click.UsageError(
            "OPENAI_API_KEY is required for interactive chat. "
            "Set it in your environment or in a .env file in the project root (never commit secrets)."
        )

    db = Database(settings.database_path)
    initialize_schema(db)

    permission_manager = PermissionManager(settings)
    agent_manager = AgentManager(
        api_key=api_key,
        model=model.strip(),
        permission_manager=permission_manager,
        database=db if persist else None,
    )
    manager = ConversationManager(
        None,
        database=db if persist else None,
        session_id=session_id,
        reading_list_context=reading_list_context,
    )
    resumed_turns: int | None = None
    if session_id:
        resumed_turns = manager.load_session(session_id)

    echo_chat_welcome(
        session_id=manager.session_id,
        model=model.strip(),
        resumed_turns=resumed_turns,
    )

    while True:
        try:
            user_text = click.prompt("You", prompt_suffix="> ", show_default=False)
        except (EOFError, KeyboardInterrupt):
            click.echo("\nSession ended.")
            break

        normalized = user_text.strip().lower()
        if normalized in {"exit", "quit"}:
            click.echo("Goodbye.")
            break
        if normalized == "help":
            echo_chat_help()
            continue
        if normalized == "status":
            click.echo(f"session={manager.session_id} turns={len(manager.messages) // 2}")
            click.echo()
            continue
        if normalized == "clear history":
            deleted = manager.clear_history(clear_persisted=persist)
            click.echo(f"History cleared (deleted_rows={deleted}).")
            click.echo()
            continue
        if normalized == "list my lists":
            from tools.reading_list_tools import tool_list_all_lists

            lists = tool_list_all_lists()
            if not lists:
                click.echo("No reading lists yet.")
            else:
                for item in lists:
                    click.echo(f"- [{item.list_id}] {item.name}")
            click.echo()
            continue

        try:
            blocked = permission_manager.check_blocked_intent(user_text)
            if blocked is not None and not blocked.allowed:
                click.echo(blocked.reason or "This request is blocked by guardrails.")
                click.echo()
                continue

            from langchain_core.messages import HumanMessage

            history = list(manager.messages) + [HumanMessage(content=user_text.strip())]

            def _confirm(prompt: str) -> bool:
                ans = click.prompt(prompt, prompt_suffix=" ", default="no", show_default=True)
                return ans.strip().lower() in {"y", "yes", "confirm"}

            result = agent_manager.respond(
                history=history,
                context_note=manager.context_snapshot(),
                session_id=manager.session_id,
                confirm_callback=_confirm,
            )
            content = result.message.content
            response_text = content if isinstance(content, str) else str(content)
            reply = manager.append_completed_turn(
                user_message=user_text,
                assistant_message=response_text,
                persist=persist,
                reading_list_context=reading_list_context,
            )
        except GuardrailError as exc:
            click.echo(f"[guardrail] {exc}")
            click.echo()
            continue
        except AgentError as exc:
            click.echo(f"[agent] {exc}")
            click.echo()
            continue

        content = reply.content
        text = content if isinstance(content, str) else str(content)
        click.echo(text)
        click.echo()


@cli.command("status")
@click.option(
    "--database-path",
    type=click.Path(path_type=Path),
    default=None,
    help="SQLite file location (defaults to settings.database_path).",
)
@click.option("--recent-sessions", type=int, default=10, show_default=True)
def status(database_path: Path | None, recent_sessions: int) -> None:
    """Show database and session status for the assistant."""
    settings = get_settings()
    db_path = database_path or settings.database_path
    db = Database(db_path)
    initialize_schema(db)
    sessions = list_recent_sessions(db, limit=max(1, recent_sessions))

    click.echo(f"Database: {db.path}")
    click.echo(f"Recent sessions: {len(sessions)}")
    if not sessions:
        click.echo("No sessions found yet.")
        return
    for sid, turns, started_at, last_at in sessions:
        click.echo(f"- {sid} | turns={turns} | started={started_at} | last={last_at}")


@cli.command("resume")
@click.option("--session-id", type=str, default=None, help="Session id to resume directly.")
@click.option("--recent", type=int, default=10, show_default=True, help="How many sessions to show.")
@click.option("--reading-list-context", type=str, default=None)
@click.option("--persist/--no-persist", default=True)
def resume_cmd(
    session_id: str | None,
    recent: int,
    reading_list_context: str | None,
    persist: bool,
) -> None:
    """List recent sessions and optionally resume one in chat."""
    settings = get_settings()
    db = Database(settings.database_path)
    initialize_schema(db)

    chosen = session_id
    if not chosen:
        sessions = list_recent_sessions(db, limit=max(1, recent))
        if not sessions:
            click.echo("No previous sessions found.")
            return
        click.echo("Recent sessions:")
        for idx, (sid, turns, _started_at, last_at) in enumerate(sessions, start=1):
            click.echo(f"{idx}. {sid} (turns={turns}, last={last_at})")
        selected = click.prompt("Select session number", type=int)
        if selected < 1 or selected > len(sessions):
            raise click.UsageError("Invalid session number.")
        chosen = sessions[selected - 1][0]

    ctx = click.get_current_context()
    ctx.invoke(
        chat,
        reading_list_context=reading_list_context,
        persist=persist,
        session_id=chosen,
    )


@cli.command("synthesize")
@click.argument("question", nargs=-1, required=True)
@click.option("--word-budget", type=int, default=500, show_default=True)
@click.option(
    "--top-n",
    type=int,
    default=6,
    show_default=True,
    help="Maximum parsed-paper working set for the agentic controller.",
)
@click.option(
    "--source",
    "sources",
    multiple=True,
    type=click.Choice(["arxiv", "semantic_scholar", "dblp", "crossref"], case_sensitive=False),
    help="Repeat to add sources; default is arxiv only (semantic_scholar opt-in due to 429 rate limits).",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help=(
        "Optional path for the markdown copy. Relative paths are anchored under "
        "'research reviews/'. When omitted, a timestamped file is written there automatically."
    ),
)
@click.option(
    "--no-write",
    is_flag=True,
    help="Skip writing the markdown file (only print to stdout).",
)
@click.option("--session-id", type=str, default=None, help="Tag this run with a session id for resume.")
@click.option(
    "--verbose",
    is_flag=True,
    help="Show progress logs for each pipeline stage.",
)
def synthesize(
    question: tuple[str, ...],
    word_budget: int,
    top_n: int,
    sources: tuple[str, ...],
    output: Path | None,
    no_write: bool,
    session_id: str | None,
    verbose: bool,
) -> None:
    """Run the agentic LitSynth controller on one research question and print the review."""
    ctx = click.get_current_context()
    parent_level = ctx.parent.params.get("log_level") if ctx.parent else None
    if verbose and parent_level is None:
        setup_logging(level="INFO")
    elif not verbose and parent_level is None:
        setup_logging(level="WARNING")

    q = " ".join(question).strip()
    if not q:
        raise click.UsageError("question must be a non-empty string.")

    settings = get_settings()
    if not settings.openai_api_key:
        raise click.UsageError(
            "OPENAI_API_KEY is required for synthesis. Set it in your environment or .env file."
        )

    db = Database(settings.database_path)
    initialize_schema(db)

    paper_limit = max(2, min(12, top_n))
    cfg = ControllerConfig(
        word_budget=max(150, min(2000, word_budget)),
        min_relevant_papers=min(4, paper_limit),
        total_paper_limit=paper_limit,
        sources=tuple(s.lower() for s in sources) if sources else ControllerConfig().sources,
    )

    def _progress(label: str) -> None:
        click.echo(f"[synth] {label}", err=True)

    result = run_agentic_synthesis(
        q,
        config=cfg,
        database=db,
        session_id=session_id,
        progress=_progress if verbose else None,
    )

    click.echo(result.to_markdown())
    click.echo()
    click.echo(
        f"[synth] confidence={result.confidence_score:.2f} "
        f"papers_cited={len(result.citations_used)} "
        f"contradictions={result.contradictions_found} "
        f"hallucinated={len(result.hallucinated_citations)}"
    )
    if not no_write:
        target = _resolve_review_output_path(
            output,
            question=q,
            project_root=settings.project_root,
        )
        try:
            permission_manager = PermissionManager(settings)
            permission_manager.validate_filesystem_target(str(target))
        except GuardrailError as exc:
            raise click.UsageError(
                f"Refusing to write outside the project root: {exc}"
            ) from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result.to_markdown(), encoding="utf-8")
        click.echo(f"[synth] markdown written to {target}")


@cli.command("synth-history")
@click.option("--limit", type=int, default=10, show_default=True)
def synth_history(limit: int) -> None:
    """List the most recent persisted synthesis runs."""
    settings = get_settings()
    db = Database(settings.database_path)
    initialize_schema(db)
    rows = list_recent_synthesis_runs(db, limit=max(1, limit))
    if not rows:
        click.echo("No synthesis runs recorded yet.")
        return
    click.echo(f"Recent synthesis runs (latest {len(rows)}):")
    for row in rows:
        click.echo(
            f"- id={row['id']} | conf={row['confidence_score']:.2f} | "
            f"contradictions={row['contradictions_found']} | "
            f"hallucinated={row['hallucinated_count']} | "
            f"created={row['created_at']} | q={row['question'][:80]!r}"
        )


@cli.command("eval-synthesis")
@click.option(
    "--cases",
    "cases_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("eval") / "cases.json",
    show_default=True,
    help="Labeled eval cases JSON.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("eval") / "results.json",
    show_default=True,
    help="Where to write eval metrics JSON.",
)
@click.option(
    "--run-id",
    type=int,
    default=None,
    help="Score one specific persisted synthesis run against the first case.",
)
@click.option(
    "--live",
    is_flag=True,
    help="Generate missing case results with the live agentic synthesis controller.",
)
@click.option(
    "--word-budget",
    type=int,
    default=500,
    show_default=True,
    help="Word budget used only with --live.",
)
@click.option(
    "--top-n",
    type=int,
    default=6,
    show_default=True,
    help="Paper working-set size used only with --live.",
)
def eval_synthesis(
    cases_path: Path,
    output_path: Path,
    run_id: int | None,
    live: bool,
    word_budget: int,
    top_n: int,
) -> None:
    """
    Score LitSynth outputs against labeled eval cases.

    Default mode is offline: for each case, load the latest matching saved
    synthesis run from SQLite and score it. Use ``--live`` to generate missing
    runs. This keeps CI/test usage deterministic while still supporting real
    evaluation runs.
    """
    settings = get_settings()
    cases_file = cases_path if cases_path.is_absolute() else settings.project_root / cases_path
    output_file = output_path if output_path.is_absolute() else settings.project_root / output_path
    cases = load_cases(cases_file)
    if not cases:
        raise click.UsageError("eval cases file contains no cases.")

    db = Database(settings.database_path)
    initialize_schema(db)

    reports = []
    missing: list[str] = []

    if run_id is not None:
        raw = get_synthesis_run_result_json(db, run_id)
        if raw is None:
            raise click.UsageError(f"No synthesis run found with id={run_id}.")
        result = _synthesis_result_from_json(raw)
        reports.append(score_synthesis(result, cases[0]))
    else:
        paper_limit = max(2, min(12, top_n))
        cfg = ControllerConfig(
            word_budget=max(150, min(2000, word_budget)),
            min_relevant_papers=min(4, paper_limit),
            total_paper_limit=paper_limit,
            min_relevance_score=0.03,
        )
        for case in cases:
            raw = get_latest_synthesis_result_json_for_question(db, case.question)
            if raw is not None:
                result = _synthesis_result_from_json(raw)
            elif live:
                if not settings.openai_api_key:
                    raise click.UsageError(
                        "OPENAI_API_KEY is required for --live eval. "
                        "Run synthesize first or set the key."
                    )
                result = run_agentic_synthesis(
                    case.question,
                    config=cfg,
                    database=db,
                    session_id="eval",
                )
            else:
                missing.append(case.question)
                continue
            reports.append(score_synthesis(result, case))

    if missing:
        click.echo("Missing saved synthesis runs for:")
        for question in missing:
            click.echo(f"- {question}")
        raise click.UsageError(
            "Run `python main.py synthesize <question>` for each missing case, "
            "or rerun eval with --live."
        )

    written = write_results(reports, output_path=output_file)
    payload = json.loads(written.read_text(encoding="utf-8"))
    agg = payload["aggregate"]
    click.echo(f"[eval] wrote {written}")
    click.echo(
        "[eval] "
        f"cases={agg['case_count']:.0f} "
        f"faithfulness={agg['claim_faithfulness']:.2f} "
        f"hallucination_rate={agg['citation_hallucination_rate']:.2f} "
        f"contradiction_coverage={agg['contradiction_coverage']:.2f}"
    )


def main() -> None:
    """Allow ``python -m`` style execution when packaged later."""
    cli()


if __name__ == "__main__":
    main()
