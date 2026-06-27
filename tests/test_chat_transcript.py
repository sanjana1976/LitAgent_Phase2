from __future__ import annotations

import json
from pathlib import Path

from agent.chat_transcript import (
    ChatTranscriptDocument,
    append_turn,
    mark_ended,
    transcripts_dir,
    write_transcript_json,
)


def test_transcript_roundtrip(tmp_path: Path) -> None:
    repo = tmp_path / "proj"
    repo.mkdir()

    doc = ChatTranscriptDocument(name="chat1")
    append_turn(doc, user_message="hi", assistant_message="hello")

    dest = transcripts_dir(repo) / "chat1.json"
    write_transcript_json(dest, doc)

    assert dest.exists()
    loaded = json.loads(dest.read_text(encoding="utf-8"))
    assert loaded["name"] == "chat1"
    assert len(loaded["turns"]) == 1
    assert loaded["turns"][0]["user_message"] == "hi"
    assert loaded["turns"][0]["assistant_message"] == "hello"

    mark_ended(doc)
    write_transcript_json(dest, doc)
    final = json.loads(dest.read_text(encoding="utf-8"))
    assert final["ended_at"] is not None
