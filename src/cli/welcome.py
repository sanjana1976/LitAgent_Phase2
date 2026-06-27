"""Terminal welcome and help text for the research assistant REPL."""

from __future__ import annotations

import click

_BOX_WIDTH = 64


def _box_line(inner: str) -> str:
    usable = _BOX_WIDTH - 4
    if len(inner) > usable:
        inner = inner[: usable - 1] + "…"
    return f"║ {inner:<{usable}} ║"


def echo_chat_welcome(
    *,
    session_id: str,
    model: str,
    resumed_turns: int | None = None,
) -> None:
    """Print a bordered welcome banner when the chat REPL starts."""
    top = "╔" + "═" * _BOX_WIDTH + "╗"
    mid = "╠" + "═" * _BOX_WIDTH + "╣"
    bot = "╚" + "═" * _BOX_WIDTH + "╝"

    click.echo()
    click.echo(top)
    click.echo(_box_line("📚  Research Assistant — ready for you"))
    click.echo(mid)
    click.echo(_box_line("🔎  Search arXiv · DBLP · Semantic Scholar · Crossref"))
    click.echo(_box_line("📄  PDF analysis · reading lists · BibTeX / APA / Chicago"))
    click.echo(_box_line("🛡️  Guardrails on writes, exports, and metadata"))
    click.echo(mid)
    click.echo(_box_line("Type a question, or: help · status · list my lists · exit"))
    click.echo(bot)
    click.echo()
    click.echo(f"  Session  {session_id}")
    click.echo(f"  Model    {model}")
    if resumed_turns is not None:
        click.echo(f"  Resumed  {resumed_turns} earlier turn(s)")
    click.echo()


def echo_chat_help() -> None:
    """Short in-session command reference."""
    click.echo()
    click.echo("  Commands")
    click.echo("  ─────────────────────────────────────────")
    click.echo("  help              Show this message")
    click.echo("  status            Session id and turn count")
    click.echo("  list my lists     Show reading lists")
    click.echo("  clear history     Wipe this session's chat history")
    click.echo("  exit / quit       End the session")
    click.echo()
    click.echo("  Ask naturally, e.g. search papers, compare methods, export BibTeX.")
    click.echo()
