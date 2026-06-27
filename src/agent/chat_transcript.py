"""
JSON transcript helpers for manual documentation (e.g. Cursor agent session logs).

Not used by the CLI REPL; create or update files under ``transcripts/`` yourself when needed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _utc_timestamp() -> str:
    """Return an ISO 8601 UTC timestamp with ``Z`` suffix."""
    now = datetime.now(timezone.utc)
    return now.isoformat().replace("+00:00", "Z")


@dataclass
class ChatTurnDict:
    """One user/assistant pair as stored inside the transcript JSON object."""

    timestamp: str
    user_message: str
    assistant_message: str


@dataclass
class ChatTranscriptDocument:
    """
    Top-level transcript document for hand-authored ``transcripts/*.json`` files.

    Fields use plain JSON-friendly types so the file stays human-readable outside Python.
    """

    name: str
    turns: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=_utc_timestamp)
    ended_at: str | None = None
    reading_list_context: str | None = None
    persist_sqlite: bool = True


def transcripts_dir(repo_root: Path) -> Path:
    """Resolved ``{repo_root}/transcripts`` path (directory may still need creation)."""
    return (repo_root / "transcripts").resolve()


def ensure_transcripts_directory(path: Path) -> None:
    """Create the transcripts folder when missing."""
    path.mkdir(parents=True, exist_ok=True)


def write_transcript_json(path: Path, document: ChatTranscriptDocument) -> None:
    """
    Serialize ``document`` to ``path`` with stable key ordering and trailing newline.

    Raises:
        OSError: if the path is not writable.
    """
    ensure_transcripts_directory(path.parent)
    payload = asdict(document)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    logger.debug("Wrote transcript %s (%s turns)", path, len(document.turns))


def append_turn(
    document: ChatTranscriptDocument,
    *,
    user_message: str,
    assistant_message: str,
) -> ChatTurnDict:
    """Record a completed exchange and return the turn for optional local use."""
    turn = ChatTurnDict(
        timestamp=_utc_timestamp(),
        user_message=user_message,
        assistant_message=assistant_message,
    )
    document.turns.append(asdict(turn))
    return turn


def mark_ended(document: ChatTranscriptDocument) -> None:
    """Stamp ``ended_at`` on a manually maintained transcript document."""
    document.ended_at = _utc_timestamp()
