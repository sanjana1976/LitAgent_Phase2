"""Regex and layout heuristics for sectioning PDF-extracted plain text."""

from __future__ import annotations

import re

_HEADING = re.compile(
    r"^(abstract|introduction|background|related work|method|methods|methodology|"
    r"experiments?|results?|discussion|conclusion|limitations|future work|references?)\s*$",
    re.I | re.M,
)

_EQ_START = re.compile(
    r"\\\[|\$\$|[\u2200-\u22FF]|\\begin\{(?:equation|align|gather)\}",
)


def split_sections(text: str) -> dict[str, str]:
    """
    Best-effort segmentation using common section titles.

    Unlabeled paragraphs accumulate under the active heading key.
    """
    if not text.strip():
        return {}

    lines = text.splitlines()
    sections: dict[str, list[str]] = {}
    current_key = "preamble"
    sections[current_key] = []

    for line in lines:
        stripped = line.strip()
        if len(stripped) < 80 and _HEADING.match(stripped):
            current_key = stripped.lower().replace(" ", "_")
            sections.setdefault(current_key, [])
            continue
        sections[current_key].append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


def extract_equation_like_lines(text: str, max_equations: int = 20) -> list[dict[str, str]]:
    """
    Pull lines that look like display equations or LaTeX equations.

    Returns:
        List of dicts with ``equation`` and ``description`` (empty string if unknown).
    """
    out: list[dict[str, str]] = []
    for block in text.splitlines():
        b = block.strip()
        if not b:
            continue
        if len(b) > 300:
            continue
        if _EQ_START.search(b) or (b.count("=") >= 2 and "\\" in b):
            out.append({"equation": b, "description": ""})
        if len(out) >= max_equations:
            break
    return out


def extract_github_urls(text: str) -> list[str]:
    """Return unique http(s) GitHub URLs mentioned in the text."""
    pat = re.compile(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?", re.I)
    seen: set[str] = set()
    out: list[str] = []
    for m in pat.finditer(text):
        u = m.group(0).rstrip(").,;")
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def extract_reference_block(text: str) -> str | None:
    """
    Heuristically isolate the bibliography/references region.
    """
    lower = text.lower()
    markers = ("references", "bibliography", "works cited")
    positions = [lower.find(m) for m in markers]
    positions = [p for p in positions if p != -1]
    if not positions:
        return None
    idx = min(positions)
    return text[idx:]


def guess_title_from_text(text: str, fallback: str = "") -> str:
    """Use the first substantial non-empty line as a provisional title."""
    for line in text.splitlines():
        s = line.strip()
        if 8 < len(s) < 400 and not s.lower().startswith("abstract"):
            return s
    return fallback
