## Pre-review by [your name], [your pid]
## Review of [review group name(s)] � LitSynth / Research Paper Analyzer (A3 + A4)

> **Note to submitter:** Replace bracketed fields with your name, PID, and the group you are reviewing. This draft is written for **this repository** (`A4_StudentClock_SJK`) so you can use it as a self-check before Tuesday or adapt it when reviewing a teammate.

---

### 1. Project summary/implementation

#### a. Summary

This is the **generation** track (A4), built on top of the prior **agentic programming** project (A3 � Research Paper Analyzer). Intended users are graduate-level researchers or students doing literature surveys from the terminal: they ask a research question and get an argument-driven literature-review section, or use the multi-turn REPL for search, PDF analysis, comparison, reading lists, and citations.

**LitSynth** (`src/synthesis/`) adds a 10-stage pipeline: decompose the question ? retrieve papers (arXiv by default) ? fetch/parse PDFs ? TF-IDF rank ? extract grounded claims ? detect contradictions ? build a synthesis prompt ? generate prose ? validate inline citations ? persist results. The headline artifact is **generated text with citations**, not a paper list � which matches the A4 �generation as the point� requirement.

#### b. Demo attempt

**Setup followed:** `README.md` quick start � venv, `pip install -r requirements.txt`, `pip install -e .`, `.env` with `OPENAI_API_KEY`, `python main.py init-db`.

**What worked:**
- `python -m pytest` ? **174 passed** (mocked; no live API required for CI).
- `python main.py synthesize "What are the competing approaches to long-context retrieval in LLMs?" --verbose` ? full pipeline ran: stage logs printed (`decomposing`, `retrieving`, `fetching`, `ranking`, `claims`, `contradictions`, `generating`, `validating`), markdown review printed, summary line e.g. `[synth] confidence=0.92 papers_cited=5 contradictions=1 hallucinated=0`, file auto-written under `research reviews/`.
- `python main.py synth-history` ? lists persisted rows from `synthesis_runs` (schema v4).
- `python main.py chat` ? REPL starts; literature-review request triggers `tool_synthesize_literature_review`; reply includes `### Papers used` with `paper_id` and `url` for follow-ups.

**Friction / errors observed:**
- **Semantic Scholar 429/504** when synthesis defaulted to arXiv + S2 � retriever logs warnings but continues on arXiv. **Fix in repo:** synthesis default is now **arXiv-only** (`PipelineConfig.sources = ("arxiv",)` in `src/synthesis/pipeline.py`). Chat synthesis still benefits from this default via `tool_synthesize_literature_review`.
- **Follow-up without �Papers used� block (older behavior):** second turn sometimes asked for URLs because tool JSON was not in persisted chat history. **Fix:** `src/agent/agent.py` system prompt now requires a mandatory `### Papers used` section after synthesis.

**If demo failed on a fresh machine:** most likely missing `.env` / `OPENAI_API_KEY`, or running from wrong directory (must be repo root with `main.py`). No `DEMO.md` exists � `README.md` + �Live demo prompts� table is the runbook.

**Intended flow (first deliverable user story), code trace:**

| Step | Module | Entry | In ? Out |
|------|--------|-------|----------|
| 1 | Query decomposer | `synthesis/decompose.py` ? `decompose_question()` | question ? `ResearchQuestion` + 3�5 sub-queries (LLM JSON; fallback templates) |
| 2 | Retriever | `synthesis/retrieve.py` ? `retrieve_papers()` | sub-queries ? deduped `list[Paper]` via A3 `tool_search_arxiv` |
| 3 | Fetch/parse | `synthesis/fetch_parse.py` ? `fetch_and_parse()` | papers ? `list[ScoredPaper]` via `tool_fetch_and_parse_pdf` or abstract fallback |
| 4 | Ranker | `synthesis/rank.py` ? `rank_papers()` | papers + question ? top N by TF-IDF cosine |
| 5 | Claims | `synthesis/claims.py` ? `extract_claims()` | per paper: LLM JSON `{claim, evidence_quote, confidence}`; substring grounding check |
| 6 | Contradictions | `synthesis/contradictions.py` ? `detect_contradictions()` | all claims ? `list[ContradictionPair]` (one LLM call) |
| 7 | Prompt | `synthesis/prompt.py` ? `build_synthesis_prompt()` | deterministic system+user prompt + `expected_citations` |
| 8 | Generate | `synthesis/generate.py` ? `generate_literature_review()` | `call_text()` ? 3�5 paragraph review |
| 9 | Validate | `synthesis/validate_cites.py` ? `validate_citations()` | regex `[Author et al. YEAR]` ? resolved vs hallucinated |
| Orchestrator | `synthesis/pipeline.py` ? `run_synthesis()` | wires 1�9; persists via `db/queries.insert_synthesis_run()` |

CLI entry: `main.py` ? `synthesize` command. Agent entry: `tools/synthesis_tools.py` ? `tool_synthesize_literature_review()`.

#### c. Proposal component check

Source: root `proposal.md` (no separate `proposal/` markup folder � inventory taken from proposal � Architecture).

**Implemented as written � verified in code:**

1. **Claim Extractor (stage 5)** � Proposal: per-paper `{paper_id, claim, evidence_quote, confidence}` with quote validation.  
   **Code:** `src/synthesis/claims.py` ? `extract_claims()`; `_is_grounded()` normalizes quote vs source text; ungrounded claims get `grounded=False` and confidence � 0.5.  
   **Tests:** `tests/test_synthesis_claims.py`.

2. **Citation Validator (stage 9)** � Proposal: flag hallucinated inline citations against source set.  
   **Code:** `src/synthesis/validate_cites.py` ? `validate_citations()` returns `CitationCheck`, `hallucinated_citations`, `confidence_score`.  
   **Tests:** `tests/test_synthesis_validate_cites.py`.

**Planned / after first deliverable � reasonable for final deadline:**

- **Multi-round refinement** (�go deeper on retrieval angle�) � not implemented; would need session-scoped synthesis state + partial re-run. Reasonable to defer if they ship a labeled eval suite first.
- **Hallucination eval suite (20+ labeled questions)** � `src/synthesis/eval_harness.py` exists with metrics (`claim_faithfulness`, `citation_hallucination_rate`, `contradiction_coverage`) but not yet a committed `eval/ground_truth.json` dataset. Reasonable next priority.

**Full component inventory (proposal stages 1�10):**

| # | Component | Location |
|---|-----------|----------|
| 1 | Query decomposer | `src/synthesis/decompose.py` |
| 2 | Paper retriever | `src/synthesis/retrieve.py` |
| 3 | Fetch + parse | `src/synthesis/fetch_parse.py` |
| 4 | Relevance ranker | `src/synthesis/rank.py` |
| 5 | Claim extractor | `src/synthesis/claims.py` |
| 6 | Contradiction detector | `src/synthesis/contradictions.py` |
| 7 | Synthesis prompt builder | `src/synthesis/prompt.py` |
| 8 | Literature review generator | `src/synthesis/generate.py` |
| 9 | Citation validator | `src/synthesis/validate_cites.py` |
| 10 | Eval harness | `src/synthesis/eval_harness.py` |
| � | Orchestrator | `src/synthesis/pipeline.py` |
| � | Agent tool | `src/tools/synthesis_tools.py` |
| � | CLI | `main.py` (`synthesize`, `synth-history`) |

#### d. One confusing thing

**Two ways to run synthesis with different persistence behavior.** `python main.py synthesize` always runs `run_synthesis()` directly and writes markdown to `research reviews/`. In `chat`, the agent calls `tool_synthesize_literature_review`, which returns compact JSON and relies on the model to format the reply (including the mandatory `### Papers used` block). The underlying pipeline is the same, but **chat history only stores the assistant�s formatted text**, not the raw tool payload � so the system-prompt contract for �Papers used� is critical. I had to read both `main.py` and `src/agent/agent.py` plus `src/tools/synthesis_tools.py` to see why follow-ups sometimes lost `paper_id`s before that prompt fix.

#### e. A conversation starter for Tuesday

**�Show me a run where two papers genuinely disagree, walk me through the contradiction detector�s inputs and outputs, and show what happens if the generator invents a citation that isn�t in the retrieved set � does the validator catch it and how does that affect `confidence_score`?�**

I want to see stage 6 (`contradictions`) and stage 9 (`validate_cites`) live, not just the final prose.

---

### 2. Suggestions

#### a. Scope feedback for the final deliverable

**Prioritize:**
1. **Labeled eval dataset + `eval/results.json` workflow** � the harness is built; adding 10�20 hand-labeled `EvalCase`s (question + expected contradiction keys) would satisfy �real eval� and give a demo graph for Tuesday.
2. **Relevance / retrieval quality** � some synthesis runs pulled papers only loosely related to �long-context retrieval� (broad arXiv hits). Tighter sub-query prompts or a minimum `relevance_score` threshold before claim extraction would improve trust more than more LLM tokens.
3. **Optional `DEMO.md`** � peers without your context will appreciate a 5-command script mirroring README�s �Live demo prompts� table.

**Gotchas:**
- **OpenAI cost/latency:** happy path ? 1 decompose + N claim calls + 1 contradiction + 1 generate per synthesis; chat + CLI double-running burns quota.
- **arXiv-only default** trades recall for reliability; CS papers outside arXiv need `--source dblp` or manual IDs.
- **Do not commit `.env`** � already gitignored; reviews in `research reviews/*.md` are gitignored too.

**Defer unless time:** multi-round refinement, BibTeX export of synthesis output, comparison-mode synthesis (RAG vs fine-tuning pro/con template).

#### b. One concrete suggestion

**Add a minimum relevance gate before claim extraction.** In `run_synthesis()`, after `rank_papers()`, drop papers with `relevance_score` below a configurable threshold (e.g. 0.15) or keep only the top cluster whose scores are within 50% of the best. Right now stage 5 will still spend an LLM call per marginal paper, and those claims can pollute contradiction detection and the final narrative. This is grounded in observed behavior where TF-IDF ranked tangentially related robotics/world-model papers into a �long-context retrieval� question � the ranker works mechanically but the pipeline has no �stop feeding the synthesizer weak papers� step.

#### c. Something you learned or thought was cool

**Hallucination resistance as a pipeline contract, not a prompt wish.** Stage 5 refuses to trust the LLM�s claims until `evidence_quote` appears in the source text; stage 7 whitelists citation keys; stage 9 regex-checks the generator�s brackets. The confidence score is literally `citation_validity � grounded_claim_fraction` � so you can point at two independent failure modes in the metrics line. That�s a clearer engineering story than �we told GPT to be careful,� and it�s testable in pytest without hitting OpenAI (`tests/test_synthesis_claims.py`, `tests/test_synthesis_validate_cites.py`).

---
