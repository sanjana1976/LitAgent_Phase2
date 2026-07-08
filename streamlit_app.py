"""Streamlit UI for the agentic LitSynth controller.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
for path in (SRC, PROJECT_ROOT):
    serialized = str(path)
    if serialized not in sys.path:
        sys.path.insert(0, serialized)

import streamlit as st  # noqa: E402

from config.config import get_settings  # noqa: E402
from db.database import Database  # noqa: E402
from db.init_db import initialize_schema  # noqa: E402
from db.queries import get_synthesis_run_result_json, list_recent_synthesis_runs  # noqa: E402
from synthesis.controller import ControllerConfig, SynthesisController  # noqa: E402
from synthesis.persistence import load_synthesis_state_from_json, persist_synthesis_state  # noqa: E402
from synthesis.state import SynthesisState  # noqa: E402
from synthesis.trace_view import (  # noqa: E402
    claim_rows,
    contradiction_rows,
    gap_rows,
    paper_rows,
    state_metrics,
    summarize_effect,
    trace_rows,
)

# --------------------------------------------------------------------------- #
# Presentation vocab (text only, no icons)
# --------------------------------------------------------------------------- #

# Pastel badge backgrounds — always paired with black badge text in CSS.
_RESULT_COLORS: dict[str, str] = {
    "ok": "#c8f0d8",
    "insufficient": "#fff3c4",
    "failed": "#ffd6d6",
    "noop": "#e8e0f5",
    "pending": "#dbeafe",
}

_ACTION_LABELS: dict[str, str] = {
    "decompose": "Decompose",
    "search": "Search",
    "reformulate": "Reformulate",
    "fetch_pdf": "Fetch PDF",
    "extract_claims": "Extract Claims",
    "detect_contradictions": "Detect Contradictions",
    "hunt_support": "Hunt Support",
    "resolve_conflict": "Resolve Conflict",
    "synthesize": "Synthesize",
}

_CSS = """
<style>
:root {
    --ls-text: #000000;
    --ls-accent: #9b7ede;
    --ls-accent-soft: #f3edff;
    --ls-border: #d8c9f0;
    --ls-panel: #ffffff;
    --ls-panel-alt: #fff5f8;
    --ls-panel-warm: #fffbeb;
}
.stApp, .stApp p, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp h5, .stApp h6, .stMarkdown, .stCaption, .stSlider label,
.stMultiSelect label, .stTextArea label {
    color: #000000 !important;
}
.stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
    color: #000000 !important;
    background-color: #ffffff !important;
}
[data-testid="stSidebar"], [data-testid="stSidebar"] * {
    color: #000000 !important;
}
.block-container { padding-top: 2.4rem; max-width: 1400px; color: #000000; }
#MainMenu, footer { visibility: hidden; }

.ls-hero {
    background: linear-gradient(135deg, #f3edff 0%, #fff5f8 55%, #fffbeb 100%);
    border: 1px solid var(--ls-border);
    border-radius: 18px;
    padding: 30px 34px;
    margin-bottom: 22px;
    color: var(--ls-text);
}
.ls-hero h1 {
    margin: 0;
    font-size: 2.05rem;
    font-weight: 750;
    letter-spacing: -0.02em;
    color: #000000;
}
.ls-hero p {
    margin: 8px 0 0;
    color: #000000;
    font-size: 1.0rem;
    max-width: 720px;
}
.ls-pipeline {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 16px;
}
.ls-pill {
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #000000;
    background: #f3edff;
    border: 1px solid #c9b3ef;
    border-radius: 999px;
    padding: 5px 12px;
    font-weight: 600;
}
.ls-arrow { color: #000000; align-self: center; font-size: 0.8rem; }

.ls-metric-grid {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    gap: 12px;
    margin: 6px 0 18px;
}
.ls-card {
    background: var(--ls-panel);
    border: 1px solid var(--ls-border);
    border-radius: 14px;
    padding: 16px 16px 14px;
    color: #000000;
}
.ls-card:nth-child(3n+2) { background: var(--ls-panel-alt); }
.ls-card:nth-child(3n) { background: var(--ls-panel-warm); }
.ls-card .ls-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #000000;
    font-weight: 600;
}
.ls-card .ls-value {
    font-size: 1.7rem;
    font-weight: 720;
    margin-top: 4px;
    letter-spacing: -0.02em;
    color: #000000;
}

.ls-section-title {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: #000000;
    font-weight: 700;
    margin: 6px 0 14px;
}

.ls-review {
    background: #ffffff;
    border: 1px solid var(--ls-border);
    border-radius: 16px;
    padding: 26px 30px;
    line-height: 1.7;
    font-size: 1.02rem;
    color: #000000;
}

.ls-timeline { position: relative; margin-left: 6px; color: #000000; }
.ls-step {
    position: relative;
    padding: 0 0 18px 26px;
    border-left: 2px solid var(--ls-border);
    color: #000000;
}
.ls-step:last-child { border-left: 2px solid transparent; }
.ls-step::before {
    content: "";
    position: absolute;
    left: -7px;
    top: 4px;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--dot, var(--ls-accent));
    box-shadow: 0 0 0 4px rgba(155, 126, 222, 0.25);
}
.ls-step-head {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    color: #000000;
}
.ls-step-idx { color: #000000; font-variant-numeric: tabular-nums; font-size: 0.85rem; }
.ls-step-name { font-weight: 680; font-size: 1.0rem; letter-spacing: -0.01em; color: #000000; }
.ls-badge {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 700;
    border-radius: 6px;
    padding: 2px 8px;
    color: #000000;
    border: 1px solid #d8c9f0;
}
.ls-meta { color: #000000; font-size: 0.78rem; margin-left: auto; }
.ls-step-body { color: #000000; font-size: 0.9rem; margin-top: 5px; }
.ls-step-effect {
    color: #000000;
    font-size: 0.82rem;
    margin-top: 6px;
    font-family: "SFMono-Regular", ui-monospace, monospace;
}
.ls-link { color: #000000; font-size: 0.76rem; text-decoration: underline; }
</style>
"""


def _run_agentic_synthesis(
    question: str,
    *,
    min_relevant_papers: int,
    max_reformulations: int,
    word_budget: int,
    sources: tuple[str, ...],
    database: Database | None = None,
    session_id: str | None = None,
) -> SynthesisState:
    config = ControllerConfig(
        min_relevant_papers=min_relevant_papers,
        max_reformulations=max_reformulations,
        word_budget=word_budget,
        sources=sources,
    )
    state = SynthesisController(config=config).run(question)
    if database is not None:
        persist_synthesis_state(database, session_id, state)
    return state


def _render_hero() -> None:
    pipeline = [
        "Retrieval",
        "Claims",
        "Detect",
        "Gap Hunt",
        "Resolve",
        "Synthesize",
    ]
    pills = '<span class="ls-arrow">/</span>'.join(
        f'<span class="ls-pill">{stage}</span>' for stage in pipeline
    )
    st.markdown(
        f"""
        <div class="ls-hero">
            <h1>LitSynth</h1>
            <p>An agentic research-synthesis loop. Every retrieval, grounding check,
            and conflict resolution is a logged decision &mdash; read the final review
            beside the exact reasoning that produced it.</p>
            <div class="ls-pipeline">{pills}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metrics(state: SynthesisState) -> None:
    m = state_metrics(state)
    cards = [
        ("Papers", str(m["papers"])),
        ("Claims", str(m["claims"])),
        ("Grounded", f"{m['grounded_fraction']:.0%}"),
        ("Citations", f"{m['citation_validity']:.0%}"),
        ("Conflicts", str(m["contradictions"])),
        ("Open Gaps", str(m["open_gaps"])),
        ("Hallucinated", str(m["hallucinated_citations"])),
    ]
    cells = "".join(
        f'<div class="ls-card"><div class="ls-label">{label}</div>'
        f'<div class="ls-value">{value}</div></div>'
        for label, value in cards
    )
    st.markdown(f'<div class="ls-metric-grid">{cells}</div>', unsafe_allow_html=True)


def _render_timeline(state: SynthesisState) -> None:
    steps_html: list[str] = []
    for idx, step in enumerate(state.trace, start=1):
        color = _RESULT_COLORS.get(step.result, "#e8e0f5")
        name = _ACTION_LABELS.get(step.action, step.action)
        meta_bits: list[str] = []
        if step.duration_ms is not None:
            meta_bits.append(f"{step.duration_ms} ms")
        if step.llm_calls:
            meta_bits.append(f"{step.llm_calls} LLM")
        meta = " &middot; ".join(meta_bits)

        effect = summarize_effect(step)
        effect_html = (
            f'<div class="ls-step-effect">{html.escape(effect)}</div>' if effect else ""
        )
        note = step.result_note or step.rationale
        steps_html.append(
            f"""
            <div class="ls-step" style="--dot:{color}">
                <div class="ls-step-head">
                    <span class="ls-step-idx">{idx:02d}</span>
                    <span class="ls-step-name">{html.escape(name)}</span>
                    <span class="ls-badge" style="background:{color}">{step.result}</span>
                    <span class="ls-meta">{meta}</span>
                </div>
                <div class="ls-step-body">{html.escape(step.trigger)}</div>
                <div class="ls-step-body">{html.escape(note)}</div>
                {effect_html}
            </div>
            """
        )
    st.markdown(
        f'<div class="ls-timeline">{"".join(steps_html)}</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="LitSynth Agent Trace", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    _render_hero()

    settings = get_settings()
    if not settings.openai_api_key:
        st.warning(
            "OPENAI_API_KEY is not configured. The UI loads, but live synthesis needs an API key."
        )

    db = Database(settings.database_path)
    initialize_schema(db)

    with st.sidebar:
        st.markdown('<div class="ls-section-title">Run Settings</div>', unsafe_allow_html=True)
        min_relevant_papers = st.slider("Minimum papers before synthesis", 1, 8, 4)
        max_reformulations = st.slider("Max reformulations", 0, 4, 2)
        word_budget = st.slider("Word budget", 150, 1200, 500, step=50)
        selected_sources = st.multiselect(
            "Sources",
            ["arxiv", "semantic_scholar", "dblp", "crossref"],
            default=["arxiv"],
        )
        st.caption(
            "Semantic Scholar can rate-limit; arXiv is the safest demo default. "
            "Papers below relevance 0.03 are dropped after ranking."
        )

        st.divider()
        st.markdown('<div class="ls-section-title">Run History</div>', unsafe_allow_html=True)
        history_runs = list_recent_synthesis_runs(db, limit=12)
        if history_runs:
            labels = {
                row["id"]: f"#{row['id']} · conf={row['confidence_score']:.2f} · {row['question'][:48]}"
                for row in history_runs
            }
            selected_run_id = st.selectbox(
                "Load saved run",
                options=[None, *[row["id"] for row in history_runs]],
                format_func=lambda run_id: "— new run —" if run_id is None else labels[run_id],
            )
            if st.button("Load selected run", use_container_width=True) and selected_run_id is not None:
                raw = get_synthesis_run_result_json(db, int(selected_run_id))
                if raw:
                    loaded = load_synthesis_state_from_json(raw)
                    if loaded is not None:
                        st.session_state["synthesis_state"] = loaded
                        st.rerun()
                    else:
                        st.warning("Selected run has no decision trace (legacy row).")
        else:
            st.caption("No saved runs yet. Run a question to persist history.")

    question = st.text_area(
        "Research question",
        value="What are the competing approaches to long-context retrieval in LLMs?",
        height=90,
    )

    if st.button("Run Agentic Synthesis", type="primary", use_container_width=True):
        q = question.strip()
        if len(q) < 3:
            st.error("Enter a research question with at least 3 characters.")
            return
        with st.spinner("Retrieval -> Claims -> Detect -> Gap Hunt -> Resolve -> Synthesize"):
            try:
                st.session_state["synthesis_state"] = _run_agentic_synthesis(
                    q,
                    min_relevant_papers=min_relevant_papers,
                    max_reformulations=max_reformulations,
                    word_budget=word_budget,
                    sources=tuple(selected_sources) or ("arxiv",),
                    database=db,
                    session_id="streamlit",
                )
            except Exception as exc:  # noqa: BLE001 - surface UI failures clearly
                st.exception(exc)
                return

    state = st.session_state.get("synthesis_state")
    if state is None:
        st.info("Run a question to see the review and the decision trace that produced it.")
        return

    _render_metrics(state)
    result = state.to_result()

    review_col, trace_col = st.columns([1.05, 1.0], gap="large")
    with review_col:
        st.markdown('<div class="ls-section-title">Final Review</div>', unsafe_allow_html=True)
        review_text = result.review_text or "_No review text generated yet._"
        st.markdown(f'<div class="ls-review">{review_text}</div>', unsafe_allow_html=True)
        st.markdown("")
        st.download_button(
            "Download Markdown",
            data=result.to_markdown(),
            file_name="litsynth-review.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with trace_col:
        st.markdown('<div class="ls-section-title">Decision Trace</div>', unsafe_allow_html=True)
        _render_timeline(state)
        with st.expander("Raw trace table"):
            st.dataframe(trace_rows(state), use_container_width=True, hide_index=True)

    st.divider()
    papers_tab, claims_tab, contradictions_tab, gaps_tab = st.tabs(
        ["Papers", "Claims", "Contradictions", "Gaps"]
    )
    with papers_tab:
        st.dataframe(paper_rows(state), use_container_width=True, hide_index=True)
    with claims_tab:
        st.dataframe(claim_rows(state), use_container_width=True, hide_index=True)
    with contradictions_tab:
        st.dataframe(contradiction_rows(state), use_container_width=True, hide_index=True)
    with gaps_tab:
        st.dataframe(gap_rows(state), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
