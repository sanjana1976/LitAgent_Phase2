"""
Shared pydantic data shapes for the LitSynth (A4) pipeline.

Every stage of the synthesis pipeline imports from this module so that
inputs and outputs are typed end-to-end. Keep these models small, JSON-
serializable, and dependency-free beyond pydantic.

Conventions:
- ``paper_id`` always uses the A3 prefix convention (``arxiv:...``, ``s2:...``,
  ``crossref:...``, ``dblp:...``) so that downstream tools can route lookups
  back to the right source.
- LLM-produced fields (claims, contradictions, review text) are produced
  with strict JSON-object schemas and validated through these models before
  being trusted by later stages.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Stage 1: query decomposition
# ---------------------------------------------------------------------------


class ResearchQuestion(BaseModel):
    """Raw user research question plus LLM-generated sub-queries."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=3, description="Raw user question.")
    sub_queries: list[str] = Field(
        default_factory=list,
        description="3-5 LLM-generated angles used for retrieval.",
    )

    @field_validator("sub_queries")
    @classmethod
    def _strip_blank(cls, value: list[str]) -> list[str]:
        return [s.strip() for s in value if isinstance(s, str) and s.strip()]


# ---------------------------------------------------------------------------
# Stages 2 + 3 + 4: retrieved -> parsed -> ranked
# ---------------------------------------------------------------------------


# Grounding is two orthogonal axes, deliberately kept separate:
#   - TextTier (paper-level): what source text the system actually has.
#   - GroundingTier (claim-level): what kind of support a claim currently has,
#     which can be *upgraded* mid-run by the agent (e.g. abstract -> full_text
#     after a PDF hunt, or none -> corroborated after a support search).
# Collapsing these into one enum loses the loudest hallucination signal:
# "we had the full PDF (TextTier=full_text) yet the quote did not verify".
TextTier = Literal["full_text", "abstract", "none"]
GroundingTier = Literal["full_text", "abstract", "corroborated", "none"]


class ScoredPaper(BaseModel):
    """
    Paper after retrieval, parsing and relevance ranking.

    ``sections`` is best-effort: empty dict when the PDF could not be fetched
    (abstract-only fallback). ``full_text`` is included so downstream stages
    can ground claims without a second cache lookup.
    """

    model_config = ConfigDict(extra="allow")

    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    year: int | None = None
    venue: str | None = None
    url: str | None = None
    api_source: str = ""
    sections: dict[str, str] = Field(default_factory=dict)
    full_text: str | None = Field(
        default=None,
        description="Concatenated extracted text (truncated upstream if huge).",
    )
    has_pdf: bool = Field(
        default=False,
        description="True when full text came from a successful PDF parse.",
    )
    text_tier: TextTier = Field(
        default="none",
        description=(
            "Strongest source text available for this paper: 'full_text' (PDF "
            "body parsed), 'abstract' (abstract only), or 'none'. This is the "
            "source-of-truth for how strongly any claim from this paper can be "
            "grounded; it is a paper-level fact, not a per-claim one."
        ),
    )
    relevance_score: float = Field(default=0.0, ge=0.0)

    def short_citation_key(self) -> str:
        """
        Render a stable ``[FirstAuthor et al. YEAR]`` style key for prompts.

        Falls back to the paper id when no author / year is known.
        """
        author = self.authors[0].split()[-1] if self.authors else ""
        year = str(self.year) if self.year is not None else ""
        if author and year:
            tail = " et al." if len(self.authors) > 1 else ""
            return f"[{author}{tail} {year}]"
        if author:
            return f"[{author}]"
        if year:
            return f"[{year}]"
        return f"[{self.paper_id}]"


_YEAR_IN_KEY_RE = re.compile(r"(\d{4})(?=\D*\]?$)")


def _suffix_letter(index: int) -> str:
    """Render a bijective base-26 suffix: 0->'a', 25->'z', 26->'aa'."""
    n = index + 1
    letters = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters = chr(ord("a") + rem) + letters
    return letters


def assign_citation_keys(papers: Iterable[ScoredPaper]) -> dict[str, str]:
    """
    Map each ``paper_id`` to a citation key that is unique across ``papers``.

    Two distinct papers by the same first author in the same year collapse to
    an identical :meth:`ScoredPaper.short_citation_key` (e.g. two
    ``[Liu et al. 2025]`` papers). That ambiguity is what makes a perfectly
    legitimate citation unresolvable: the prompt offers one key for two papers
    and the validator cannot decide which one a citation meant, so it flags the
    citation as hallucinated.

    This function disambiguates colliding keys deterministically by appending a
    letter suffix to the year (``[Liu et al. 2025a]``, ``[Liu et al. 2025b]``)
    in input order. The validator's citation grammar already accepts a
    ``\\d{4}[a-z]?`` year, so the suffixed keys round-trip cleanly. Keys that do
    not collide are returned unchanged, and a key with no four-digit year (a
    rare fallback) is left as-is because it cannot be cited via the bracket
    grammar anyway.

    Both the prompt builder and the citation validator call this with the same
    ``papers`` list, so the keys they assign are guaranteed to agree.
    """
    base_by_id: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    for paper in papers:
        pid = paper.paper_id
        if pid in base_by_id:
            continue
        base = paper.short_citation_key()
        base_by_id[pid] = base
        groups.setdefault(base, []).append(pid)

    keys: dict[str, str] = {}
    for base, pids in groups.items():
        if len(pids) == 1:
            keys[pids[0]] = base
            continue
        for offset, pid in enumerate(pids):
            suffix = _suffix_letter(offset)
            disambiguated, count = _YEAR_IN_KEY_RE.subn(rf"\g<1>{suffix}", base, count=1)
            keys[pid] = disambiguated if count else base
    return keys


# ---------------------------------------------------------------------------
# Stage 5: claim extraction
# ---------------------------------------------------------------------------


class ClaimRecord(BaseModel):
    """A single grounded claim extracted from one paper."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(
        default_factory=lambda: uuid4().hex,
        description="Stable identifier so steps and gaps can reference this claim.",
    )
    paper_id: str
    claim: str = Field(..., min_length=3, max_length=600)
    evidence_quote: str = Field(
        ...,
        min_length=3,
        max_length=1200,
        description="Verbatim quote from the paper that supports the claim.",
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    grounded: bool = Field(
        default=False,
        description="Verdict: True iff the evidence quote appears in the source text.",
    )
    grounding_tier: GroundingTier = Field(
        default="none",
        description=(
            "What kind of support this claim currently has. 'full_text'/'abstract' "
            "mean the quote verified against the originating paper's own text at "
            "that tier; 'corroborated' means it was grounded via another paper "
            "(see supporting_paper_id); 'none' means unverified. Orthogonal to "
            "`grounded`: a claim can be grounded=False with text_tier=full_text, "
            "which is the strongest hallucination signal."
        ),
    )
    supporting_paper_id: str | None = Field(
        default=None,
        description=(
            "Paper whose text grounds this claim: the originating paper for "
            "self-grounding, or a different paper after corroboration. None when "
            "unverified."
        ),
    )


# ---------------------------------------------------------------------------
# Stage 6: contradiction detection
# ---------------------------------------------------------------------------


TensionType = Literal["contradiction", "scope", "methodology"]


class ContradictionPair(BaseModel):
    """A genuine tension between two papers' claims."""

    model_config = ConfigDict(extra="forbid")

    contradiction_id: str = Field(
        default_factory=lambda: uuid4().hex,
        description="Stable identifier so steps and UI can reference this contradiction.",
    )
    paper_a: str
    paper_b: str
    claim_a: str
    claim_b: str
    tension_type: TensionType = "contradiction"
    explanation: str = Field(default="", max_length=800)


# ---------------------------------------------------------------------------
# Stages 8 + 9: generated review + citation validation
# ---------------------------------------------------------------------------


class CitationCheck(BaseModel):
    """Per-citation result from the validator."""

    model_config = ConfigDict(extra="forbid")

    citation_key: str
    resolved_paper_id: str | None = None
    is_valid: bool = False


class SynthesisResult(BaseModel):
    """Final artifact returned by the pipeline."""

    model_config = ConfigDict(extra="allow")

    question: str
    review_text: str
    citations_used: list[str] = Field(default_factory=list)
    hallucinated_citations: list[str] = Field(default_factory=list)
    contradictions_found: int = 0
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)

    papers: list[ScoredPaper] = Field(default_factory=list)
    claims: list[ClaimRecord] = Field(default_factory=list)
    contradictions: list[ContradictionPair] = Field(default_factory=list)
    citation_checks: list[CitationCheck] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_markdown(self) -> str:
        """Render a self-contained markdown view (review + references)."""
        lines: list[str] = []
        lines.append(f"# Literature review: {self.question}")
        lines.append("")
        lines.append(self.review_text.strip())
        lines.append("")
        lines.append("## References")
        for paper in self.papers:
            if paper.paper_id not in self.citations_used:
                continue
            authors = ", ".join(paper.authors[:3]) or "Unknown"
            year = paper.year or "n.d."
            venue = f", {paper.venue}" if paper.venue else ""
            url = f" {paper.url}" if paper.url else ""
            lines.append(f"- {authors} ({year}). *{paper.title}*{venue}.{url}")
        if self.hallucinated_citations:
            lines.append("")
            lines.append(
                "> ⚠ Hallucinated citations detected and flagged: "
                + ", ".join(self.hallucinated_citations)
            )
        return "\n".join(lines).strip() + "\n"
