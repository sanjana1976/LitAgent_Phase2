"""
Stage 4 of the LitSynth pipeline: TF-IDF relevance ranking.

Given the parsed ``ScoredPaper`` corpus from stage 3 and the user's research
question, build a tiny in-memory TF-IDF index over ``(title + abstract +
first 4k chars of full_text)`` and score each paper by cosine similarity to
the question vector. The top ``N`` are passed downstream to claim extraction.

The implementation is deliberately dependency-free (no sklearn / numpy) so
the pipeline stays portable and unit tests stay deterministic. Stable sort
preserves the input order on ties so identical-score corpora remain stable
across runs.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from synthesis.schemas import ScoredPaper


_TOKEN_SPLIT_RE = re.compile(r"\W+", re.UNICODE)

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to",
        "for", "with", "by", "is", "are", "was", "were", "be", "been", "being",
        "as", "from", "that", "this", "these", "those", "it", "its", "we",
        "our", "their", "they", "them", "i", "you", "he", "she", "his", "her",
        "than", "then", "if", "else", "into", "such", "can", "could", "may",
        "might", "will", "would", "should", "do", "does", "did", "not", "no",
        "yes", "so", "also", "about", "between", "among", "via", "have", "has",
        "had", "most", "more", "less", "which", "what", "when", "where", "who",
        "how", "why", "there", "here",
    }
)

_FULL_TEXT_HEAD_CHARS = 4_000


def _tokenize(text: str) -> list[str]:
    """
    Lowercase ``text`` and split on non-word runs, dropping stopwords and 1-char tokens.

    Returns a list (not a set) so callers can compute term frequencies.
    """
    if not text:
        return []
    tokens = _TOKEN_SPLIT_RE.split(text.lower())
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """
    Cosine similarity between two sparse vectors keyed by term.

    Iterates the smaller vector for speed. Returns 0.0 if either vector is
    empty or has zero norm.
    """
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = 0.0
    for term, weight in a.items():
        other = b.get(term)
        if other:
            dot += weight * other
    if dot == 0.0:
        return 0.0
    norm_a = math.sqrt(sum(w * w for w in a.values()))
    norm_b = math.sqrt(sum(w * w for w in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _paper_text_for_ranking(paper: ScoredPaper, head_chars: int = _FULL_TEXT_HEAD_CHARS) -> str:
    """Compose the bag-of-words document for one paper: title + abstract + head of full text."""
    parts: list[str] = []
    if paper.title:
        parts.append(paper.title)
    if paper.abstract:
        parts.append(paper.abstract)
    if paper.full_text:
        parts.append(paper.full_text[:head_chars])
    return "\n".join(parts)


def _tfidf_vector(
    tokens: list[str],
    document_frequency: Counter[str],
    n_docs: int,
) -> dict[str, float]:
    """Build a single TF-IDF vector using smoothed IDF: ``log((N+1)/(df+1)) + 1``."""
    if not tokens:
        return {}
    tf = Counter(tokens)
    vec: dict[str, float] = {}
    for term, freq in tf.items():
        df = document_frequency.get(term, 0)
        idf = math.log((n_docs + 1) / (df + 1)) + 1.0
        vec[term] = float(freq) * idf
    return vec


def rank_papers(
    papers: list[ScoredPaper],
    *,
    question: str,
    top_n: int = 8,
    min_score: float = 0.0,
) -> list[ScoredPaper]:
    """
    Rank ``papers`` by TF-IDF cosine similarity to ``question`` and return the top N.

    Inputs are not mutated; each returned ``ScoredPaper`` is a fresh copy with
    its ``relevance_score`` set via ``model_copy``. Ties preserve input order
    because the sort is stable on a tuple of ``(-score, original_index)``.

    Args:
        papers: Parsed corpus from stage 3.
        question: The user's research question (already validated upstream).
        top_n: Maximum number of papers to keep.
        min_score: Minimum cosine relevance a paper must clear to be kept.
            The default keeps legacy behavior.

    Returns:
        Up to ``top_n`` ``ScoredPaper`` records ordered by descending relevance,
        excluding scores below ``min_score``. Returns ``[]`` when ``papers`` is
        empty. When the question contains no usable tokens, returns the first
        ``top_n`` input papers untouched (with ``relevance_score=0.0``).
    """
    if not papers:
        return []

    question_tokens = _tokenize(question)
    if not question_tokens:
        return [
            paper.model_copy(update={"relevance_score": 0.0})
            for paper in papers[:top_n]
        ]

    doc_tokens: list[list[str]] = [
        _tokenize(_paper_text_for_ranking(paper)) for paper in papers
    ]

    all_token_lists: list[list[str]] = [question_tokens, *doc_tokens]
    n_docs = len(all_token_lists)
    document_frequency: Counter[str] = Counter()
    for tokens in all_token_lists:
        for term in set(tokens):
            document_frequency[term] += 1

    question_vec = _tfidf_vector(question_tokens, document_frequency, n_docs)

    indexed_scores: list[tuple[int, float, ScoredPaper]] = []
    for index, (paper, tokens) in enumerate(zip(papers, doc_tokens)):
        paper_vec = _tfidf_vector(tokens, document_frequency, n_docs)
        score = _cosine(question_vec, paper_vec)
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0
        indexed_scores.append((index, score, paper))

    indexed_scores.sort(key=lambda triple: (-triple[1], triple[0]))

    floor = max(0.0, float(min_score))
    return [
        paper.model_copy(update={"relevance_score": score})
        for _, score, paper in indexed_scores[:top_n]
        if score >= floor
    ]
