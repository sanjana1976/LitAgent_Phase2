"""Post-turn output checks (e.g. block fabricated BibTeX after empty searches)."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

SEARCH_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "tool_search_arxiv",
        "tool_search_dblp",
        "tool_search_semantic_scholar",
        "tool_search_crossref",
    }
)

BIBTEX_ENTRY_PATTERN = re.compile(
    r"@(?:article|inproceedings|book|misc|techreport|phdthesis|mastersthesis)\s*\{",
    re.IGNORECASE,
)

EMPTY_SEARCH_SAFE_REPLY = (
    "I searched the available sources but found no matching papers. "
    "I cannot provide BibTeX or other citation metadata without verified search results. "
    "Try different keywords, another author spelling, or a specific DOI or arXiv id."
)


@dataclass
class TurnToolTracker:
    """Records tool outputs for one ``AgentManager.respond`` turn."""

    _records: list[tuple[str, str]] = field(default_factory=list)

    def record(self, tool_name: str, output: object) -> None:
        text = output if isinstance(output, str) else str(output)
        self._records.append((tool_name, text))

    def searches_all_empty(self) -> bool:
        """True when every search tool invoked this turn returned an empty list."""
        search_runs = [out for name, out in self._records if name in SEARCH_TOOL_NAMES]
        if not search_runs:
            return False
        return all(_tool_output_is_empty_list(out) for out in search_runs)


def _tool_output_is_empty_list(output: str) -> bool:
    stripped = output.strip()
    if stripped in ("", "[]"):
        return True
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            value = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return False
        return isinstance(value, list) and len(value) == 0
    return False


def response_contains_bibtex_blocks(text: str) -> bool:
    """Detect BibTeX entry blocks in assistant-visible text."""
    return bool(BIBTEX_ENTRY_PATTERN.search(text))


def apply_output_guardrails(text: str, tracker: TurnToolTracker) -> str:
    """
    Sanitize assistant text before returning to the user.

    When all search tools returned empty results, strip fabricated BibTeX blocks
    and return a policy-safe explanation instead.
    """
    if not text.strip():
        return text
    if tracker.searches_all_empty() and response_contains_bibtex_blocks(text):
        return EMPTY_SEARCH_SAFE_REPLY
    return text
