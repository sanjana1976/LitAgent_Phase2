# DESIGN

This document focuses on  design decisions for the Research Paper Analyzer Agent n why they were made.

## Architecture overview

The primary runtime is a **terminal REPL** (`main.py chat`) backed by OpenAI. A4 adds a **Streamlit decision-trace UI** (`streamlit run streamlit_app.py`) that drives the agentic synthesis controller and visualizes its reasoning. 
User (terminal)->
main.py chat
    ├── PermissionManager.check_blocked_intent(user text)   (before LLM)
    ├── ConversationManager (history + SQLite session rows)
    └── AgentManager.respond()
            ├── ChatOpenAI + bound tools (iterative loop, max 8 steps)
            ├── check_permission per tool call
            ├── TurnToolTracker + apply_output_guardrails on final reply
            └── optional permission_audit rows in SQLite


## Summary

- **Framework** :LangChain `ChatOpenAI` with tool binding and an iterative loop in `AgentManager` 
- **Outside world** : HTTP APIs (arXiv, DBLP, Semantic Scholar, Crossref), PDF fetch + cache, SQLite, BibTeX export under project root |
| **Multi-turn** | `ConversationManager` + SQLite sessions; `chat` / `resume` / `status` |
| **Guardrails: autonomous** | Layer 1 tool list (search, read, analyze, format citations) |
| **Guardrails: confirm** | Layer 2 writes and exports |
| **Guardrails: disallow** | Layer 3 blocked intents, optional blocked tools, filesystem boundary; **plus** output guardrail for fabricated BibTeX after empty searches |

Scope stays **narrowly research-assistant** (literature search, PDF-backed analysis, reading lists, citations, forward-citation discovery) rather than a general coding agent.

### Which agent class is used?(REVIEW CHANGE 1/3)

**`AgentManager`** is the  Primary runtime. Owns the LangChain loop, guardrails, confirmations, tool tracking, and output sanitization. `main.py` constructs this directly. is a legacy
**`ResearchPaperAgent`**is a legacy wrapper around `AgentManager` exposing only `generate_reply(history)` for older call sites. No permission manager, no output guardrail hook, no confirmation callback. 
 **`ConversationManager`**:History + persistence only.


### Data and external systems

- **SQLite** :papers, reading lists, summaries, conversation history, permission audit. Schema version **3** (`src/db/init_db.py`); lightweight `ALTER TABLE` migrations add `reading_status` and `session_id` on older files.
- **HTTP APIs:** arXiv Atom, DBLP XML, Semantic Scholar Graph API, Crossref REST (base URLs in `config/config.py`, overridable via env).
- **Local cache:** PDF text and parsed sections under `{database_parent}/cache/` (`tools/context.get_cache_dir`, `tools/file_cache.py`, `tools/paper_text.py`). Keys are SHA-256 hashed filenames to avoid path issues.
- **Transcripts (`transcripts/`):** **not** written by the CLI. JSON files there are **manual documentation** .

### CLI bootstrap and packaging

- **`main.py`** prepends `src/` and the repo root to `sys.path` so `python main.py` works without an editable install; **`pip install -e .`** is still recommended for IDE imports and pytest `pythonpath`.
- **Click** command group: `init-db`, `chat`, `resume`, `status`; global `--log-level` overrides `LOG_LEVEL`.
- **`chat --no-persist`:** skips SQLite writes for conversation turns and permission audit (AgentManager gets `database=None`); schema init still runs so reading-list tools work.
- **`chat --reading-list-context`:** optional string stored on each persisted turn for archival/tagging.
- **Confirmation UX:** write tools prompt `Confirm execution of …? (yes/no)` with default **`no`**; only `y`, `yes`, or `confirm` approve.

---

## Three Key Design Decisions

### 1) OpenAI-first agent orchestration with tool-calling + iterative control loop

**Decision:**  
Use a dedicated `AgentManager` around OpenAI (`ChatOpenAI` via LangChain) with a multi-step tool loop, instead of a one-shot chatbot response model.

Paper analysis requests are rarely single.ask discovery + comparison + citations in one flow. The agent needs to decide tools, inspect outputs, then continue reasoning before final response.

**How it works:**

- Tools are  in `src/tools/tools_registry.py.' Each turn: invoke model → if `tool_calls, run tools (with permissions) → append `ToolMessage` results → repeat until the model returns text with no tools or the step limit is hit.
- **Max 8 tool-iteration steps** per user message (`AgentManager._max_steps`); exceeding raises `AgentError`.
- **Unknown tool names** from the model produce a `ToolMessage` (`Unknown tool: …`) and the loop continues: no process crash.
- **Tool failures** (`GuardrailError`, provider errors, etc.) are stringified into `ToolMessage` content so the model can recover; the REPL does not exit.
- **`context_note`** (from `ConversationManager.context_snapshot()`) is injected as an extra `SystemMessage` after the main system prompt, not mixed into user history.
- The system prompt in `src/agent/agent.py` lists tool-use policy; **tool signatures** from `list_tool_signatures()` are appended at bind time so the model sees parameter shapes.

**Trade-offs:**

- Better multi-step reasoning for research workflows
- Flexible tool composition without hardcoding every user intent path
- Easier future extension for new tools
- More runtime complexity than direct single LLM call
- Tool loops need step limits and careful error handling

**Ownership split (me vs assistant):**

- I defined the high-level behavior and requirements (OpenAI backbone, deep analysis expectations, multi-turn continuity)
- Assistant implemented the code, slacked off about edge cases in the beginning till i forced it
---

### 2) Guardrails: input, tool permissions, output, and filesystem

**Decision:**  
Safety is split across **four mechanisms** (not only the three course tiers for tools):

1. **Input intent blocking**: phrase match on raw user text before the LLM runs.
2. **Tool permission tiers**: autonomous vs confirmation vs blocked tool names.
3. **Output guardrail**: post-process the assistant’s final message when searches were empty but BibTeX was invented.
4. **Filesystem boundary**: exports must stay under `PROJECT_ROOT`.


**Why this matters:**  
This assistant runs workflows touching local persistence and export paths. Explicit policy boundaries are needed for trust, repeatability, and safety review. Empty search results are a common case where models hallucinate citations; the output layer catches that even if the prompt is ignored.

#### ) Input intent (`check_blocked_intent`) and message validation

**Pre-LLM intent**: lightweight substring match in `PermissionManager` (not a full classifier)

#### 2b) Tool permission tiers (`check_permission`)

Values come from `AppSettings` in `config/config.py` (`guardrails_*_tools`) and `PermissionManager` in `src/guardrails/permissions.py`. Every tool name the model emits is checked before execution.

**Layer 1: autonomous (no confirmation):**

ctaegories and tools for layer 1:
| Search | `tool_search_arxiv`, `tool_search_dblp`, `tool_search_semantic_scholar`, `tool_search_crossref` |
| PDF & analysis | `tool_fetch_and_parse_pdf`, `tool_deep_analyze_paper`, `tool_extract_citations`, `tool_lookup_forward_citations`, `tool_compare_papers` |
| Bibliography formatting | `tool_generate_bibtex`, `tool_generate_apa`, `tool_generate_chicago` |
| Reading list read | `tool_list_all_lists`, `tool_get_list_contents` |

**Layer 2: confirmation required** (CLI prompts `yes` / `confirm`):

- `tool_create_reading_list`, `tool_add_paper_to_list`, `tool_remove_paper_from_list`
- `tool_save_summary`, `tool_export_list_to_bibtex`

**Layer 3: blocked or conservative defaults:**

- **`guardrails_blocked_tools`:** empty by default; extra tool names can be added via environment.
- **Unknown tool names:** confirmation-required (not silently auto-allowed).
- **Filesystem:** `validate_filesystem_target` for export paths (resolved path must be under `PROJECT_ROOT`).

**Dual confirmation path for writes:**

1. **Orchestration layer**: `check_permission` marks Layer 2 tools as `needs_confirmation`; `main.py` supplies `confirm_callback`; on approval, `AgentManager` injects **`user_confirmed=True`** into tool args before invoke.
2. **Tool layer**: mutating tools (`tool_save_summary`, `tool_export_list_to_bibtex`) raise **`ConfirmationRequired`** when overwriting existing summaries or clobbering an on-disk `.bib` file unless `user_confirmed=True` (defense if the model retries without going through the CLI prompt).

**Export path check:** `validate_filesystem_target` runs on `tool_export_list_to_bibtex`’s `filename` argument immediately before invoke (in addition to tool-internal checks).

**Permission audit:** rows in `permission_audit` are written only when `AgentManager` is constructed with a `Database` (i.e. `chat` with default `--persist`).

**System prompt:** `src/agent/agent.py` repeats complementary rules (no deletion, no paywall bypass, no writes outside the project, confirm mutations, refuse fabrication on API failure, do not invent BibTeX when all searches return nothing).

#### 2c) Output guardrail (`apply_output_guardrails`)

Implemented in `src/guardrails/output.py`, invoked at the end of `AgentManager.respond()`.
This is **defense in depth** beyond the system prompt. It does not block legitimate BibTeX from `tool_generate_bibtex` when searches returned papers; it targets BibTeX-shaped text in the model’s free-form reply after failed discovery.

#### Tool signatures and OpenAI JSON schema

Agent-bound tools must expose only JSON-serializable argument types. Reading-list, bibliography, and storage tools use `get_default_database()` internally instead of a `Database` parameter the model could not fill safely.

**Trade-offs:**

- Strong safety defaults and reviewability
- Clear user control for mutations
- Slightly slower interaction when many writes are requested
- Output guardrail may occasionally replace a borderline reply; tuning is possible via regex or tracker rules

**Ownership split (me vs assistant):**

- I drove the policy shape and blocked behaviors
- Assistant implemented guardrail classes, integration points, DB audit plumbing, output guardrail, and tests

---

### 3) Session-aware conversation persistence + resume-first CLI UX

**Decision:**  
Persist each turn with `session_id` in SQLite and expose conversational controls directly in CLI (`help`, `status`, `list my lists`, `clear history`, `resume`).

**Why this matters:**  
Research analysis is long-running and incremental. Users need continuity across restarts, not only in-memory context.

#### CLI presentation (`src/cli/welcome.py`)(REVIEW CHANGE 2/3

On `python main.py chat`, the REPL prints a **bordered welcome banner** (Unicode box, short feature lines, emojis) plus session id and model name. Resume shows turn count in the banner.

- **`help`**: formatted command reference via `echo_chat_help()`
- **`list my lists`**: bypasses the LLM; calls `tool_list_all_lists` directly
- **Quiet default**: unless `--verbose` or a parent `--log-level` is set, chat reconfigures logging to **WARNING** so tool/API `INFO` lines do not clutter the terminal during searches
- **`python main.py chat --verbose`**: keeps parent/root log level (does not force WARNING); use with `python main.py --log-level INFO chat` to see tool traffic
- **`setup_logging`** also sets **`httpx`** and **`openai`** loggers to WARNING to reduce SDK noise
- After each assistant reply, guardrail error, or special command, a blank line precedes the next `You>` prompt for readability

**Trade-offs:**

- Strong continuity for multi-turn technical discussions
- Easy operational visibility in terminal-only use
- More schema + command-surface maintenance
- Session data management grows over time without pruning strategy

**Ownership split (me vs assistant):**

- I prioritized resume-capable workflows and usability commands
- Assistant implemented session query helpers, command wiring, and welcome banner

---

## Tool catalog (19 tools)

Registry: `src/tools/tools_registry.py` · implementations under `src/tools/`.

| Tool | Purpose |
|------|---------|
| `tool_search_arxiv` | Keyword/author/category/date search via arXiv Atom API |
| `tool_search_dblp` | CS bibliography search (DBLP XML) |
| `tool_search_semantic_scholar` | Search with citation counts (S2 Graph API) |
| `tool_search_crossref` | Works search or DOI lookup (Crossref REST) |
| `tool_fetch_and_parse_pdf` | Download PDF, extract text, cache sections |
| `tool_deep_analyze_paper` | Structured analysis (OpenAI when key present; heuristics fallback) |
| `tool_extract_citations` | Parse reference block from cached PDF text (**backward** citations) |
| `tool_lookup_forward_citations` | Papers that **cite** a given work via S2 `/paper/{id}/citations` (**forward** citations) |
| `tool_compare_papers` | Side-by-side comparison of multiple cached/analyzed papers |
| `tool_generate_bibtex` | BibTeX string from DB row or cached text |
| `tool_generate_apa` | APA-style reference string |
| `tool_generate_chicago` | Chicago author-date string |
| `tool_create_reading_list` | New list (confirmation) |
| `tool_add_paper_to_list` | Add paper to list (confirmation) |
| `tool_remove_paper_from_list` | Remove mapping (confirmation; not full paper delete) |
| `tool_list_all_lists` | List reading list names/ids |
| `tool_get_list_contents` | Papers in a list with reading status |
| `tool_save_summary` | Persist analysis summary (confirmation) |
| `tool_export_list_to_bibtex` | Write `.bib` under project root (confirmation + path guard) |

### Forward vs backward citations (NEW FEATURE: REVIEW CHNAGE 3/3

- **`tool_extract_citations`**: from PDF full text: what *this paper cites*.
- **`tool_lookup_forward_citations`**: from Semantic Scholar: what *cites this paper* (“who built on this?”). Accepts `s2:…`, `arxiv:…`, DOI, `crossref:…`, or numeric SQLite `paper_id` when S2/DOI metadata exists (`src/tools/citation_tools.py`).


----


## A4 LitSynth: research synthesis pipeline

A4 (`proposal.md`) builds on A3. The point of A4 is to actually generate something: take one research question and write a structured literature-review section, with contradiction detection and citation checking so the model can't just make citations up. All of this lives under `src/synthesis/`.

There are two versions, and they share the same stage modules and `schemas`:

1. **Agentic controller** (`synthesis.controller.SynthesisController` / `run_agentic_synthesis`). A ReAct-style loop with explicit state and a logged decision trace. This is what the CLI, the chat tool, and the Streamlit UI all use now. See [§ A4.5 Agentic refactor](#a45-agentic-refactor-controller-state-and-decision-trace).
2. **Linear pipeline** (`synthesis.pipeline.run_synthesis`). The original 10-stage function. I kept it as a stable reference path and it still has tests, but it is not the main demo path anymore. Described right below.

The controller is my answer to the main review note. The first version looked like "agentic programming," but the hard part (deciding what to do next when you don't know what you will find) was buried inside a straight-line script. The refactor pulls that decision-making out into data you can actually look at.

### Pipeline (10 stages, one module each)

| # | Module | Responsibility |
|---|--------|----------------|
| 1 | `synthesis.decompose` | LLM splits the question into 3-5 angle-diverse sub-queries; deterministic fallback never raises |
| 2 | `synthesis.retrieve` | Fan out across A3 search tools (`tool_search_arxiv`, `tool_search_semantic_scholar`, …); dedupe by DOI → version-stripped arXiv id → normalized title; sort abstract-present, citation count desc, recency desc |
| 3 | `synthesis.fetch_parse` | Wrap `tool_fetch_and_parse_pdf` (auto-rewrites arXiv `/abs/` → `/pdf/`); abstract-only fallback when PDF fetch fails (`has_pdf=False`); truncates body to a paragraph boundary |
| 4 | `synthesis.rank` | Dependency-free TF-IDF + cosine ranks papers vs the question; returns top N as new `ScoredPaper` instances (immutable inputs) |
| 5 | `synthesis.claims` | One JSON-strict LLM call per paper for `{claim, evidence_quote, confidence}`; re-verifies each quote as a normalized substring of the source text; ungrounded claims get `grounded=False` and confidence halved |
| 6 | `synthesis.contradictions` | One LLM call across all claims; pydantic-validates each pair, drops unknown paper ids, drops self-contradictions, canonicalizes (A,B)/(B,A) duplicates; skips the LLM entirely when fewer than 2 papers have claims |
| 7 | `synthesis.prompt` | Deterministic builder: enforces citation format (`[Author et al. YEAR]`), word budget, abstract/claim/explanation length caps; returns the deduped `expected_citations` list the model is allowed to use |
| 8 | `synthesis.generate` | One LLM call (free-form text); returns a deterministic fallback string on any failure: never raises |
| 9 | `synthesis.validate_cites` | Regex extracts inline `[…]` citations (tolerates `[Doe 2024]`, `[Lee and Kim 2022]`, `[Doe, 2024]`, `[Smith et al. 2023a]`); resolves each against `ScoredPaper.short_citation_key()`; emits `CitationCheck`s and a `valid_unique / total_unique` score |
| 10 | `synthesis.eval_harness` | Loads hand-labeled `EvalCase`s from `eval/cases.json`, scores saved or live `SynthesisResult`s on **claim faithfulness**, **citation hallucination rate**, and **contradiction coverage**, then writes `eval/results.json` |

`synthesis.pipeline.run_synthesis(...)` is the original orchestrator. Every stage can be passed in through `PipelineHooks`, so the orchestrator tests can swap in fake stages without patching modules. It still works as a stable reference path, but the CLI, chat tool, and UI now use `synthesis.controller.run_agentic_synthesis(...)`.

### How it resists hallucination

There are three layers, and each one is tested on its own:

1. **Per-paper grounding (stage 5).** Every claim has to come with an exact `evidence_quote`. The extractor checks that the quote really shows up in the source text (it normalizes the text and also allows a 60-character prefix match). A claim that fails this check still survives, but its `confidence` is cut in half and it gets tagged `(UNGROUNDED)` in the prompt so the writer hedges it.
2. **Allowed citations only (stages 7 + 9).** The prompt lists the `expected_citations` and tells the model not to use anything else. The validator then pulls every `[...]` out of the generated text and checks it against that same list. Anything that does not match goes into `hallucinated_citations`, the score drops, and `SynthesisResult.to_markdown()` adds a warning note.
3. **Score penalty.** `confidence_score = citation_validity x grounded_claim_fraction`, clamped to [0, 1]. If either layer fails, the score goes down. The eval harness then averages this across a labeled dataset (`synthesis.eval_harness.aggregate`). The CLI command is `python main.py eval-synthesis`.

### Integration with A3

- New SQLite table `synthesis_runs` (schema bumped to **v4**): `id, session_id, question, review_text, result_json, confidence_score, contradictions_found, hallucinated_count, created_at`. Helpers in `db.queries` (`insert_synthesis_run`, `list_recent_synthesis_runs`).
- New agent-callable tool `tool_synthesize_literature_review` in `src/tools/synthesis_tools.py`, registered in `tools.tools_registry.TOOL_SPECS` (count now **20**) and listed in `GUARDRAILS_AUTONOMOUS_TOOLS`. The agent prompt in `src/agent/agent.py` was extended with a single bullet telling the model when to call it.
- Three CLI commands in `main.py`: `synthesize "..."` now calls `run_agentic_synthesis` (with `--word-budget`, `--top-n`, `--source` repeatable, `--output`, `--session-id`, `--verbose`), `synth-history` inspects persisted runs, and `eval-synthesis` scores those runs against `eval/cases.json` (or uses `--live` to generate missing cases first).
- A new pydantic-only module `synthesis.schemas` for cross-stage types (`ResearchQuestion`, `ScoredPaper`, `ClaimRecord`, `ContradictionPair`, `CitationCheck`, `SynthesisResult`) so each stage can be unit-tested in complete isolation.
- A tiny OpenAI wrapper `synthesis.llm.call_json` / `call_text` (uses the existing `OPENAI_API_KEY` + `OPENAI_MODEL`: no new config). Always asks for `response_format={"type":"json_object"}` for structured stages; raises `SynthesisLLMError` on parse failure so the calling stage can fail soft.

### Trade-offs

- **More moving parts** than a single "summarize these abstracts" prompt. The upside is that each stage is small, type-checked, and easy to swap out.
- **Several LLM calls** on the normal path (1 decompose, N claim extractions, 1 contradictions, 1 generate). That is the real cost. Saving runs in `synthesis_runs` makes resume and re-export free, and the per-paper claim calls fail soft, so a flaky provider only loses one paper's claims instead of the whole run.
- **TF-IDF instead of embeddings** for ranking. It is deterministic, has no extra dependencies, and costs no API calls. I can swap in an embedding ranker later without changing the public surface.

### Ownership split (me vs assistant)

- I defined the proposal, the staged architecture, and the hallucination-resistance properties (verbatim grounding + whitelisted citations + confidence penalty).
- The assistant implemented the ten stage modules, the orchestrator hooks, the agent-callable tool, the two new CLI commands, the SQLite migration, and the mocked tests.

---

## A4.5 Agentic refactor: controller, state, and decision trace

The linear pipeline gives an answer but it can't explain itself. It runs the same stages in the same order no matter what it finds. The agentic layer swaps that straight line for a loop that reads its own state, picks the next action, and writes down every decision as data you can read back. Three new modules do this, all under `src/synthesis/`.

### First decision: keep "text source" and "claim verdict" separate

Before writing any controller code, I locked one schema decision because everything else depends on it. The source of a paper's text and whether a claim checks out are two different things, so they get two different fields. If I had jammed them into one enum (like `grounding_source` being one of `full_text`, `abstract`, `unverified`), I would lose the most important signal: "we had the full PDF and the quote still did not verify."

- `ScoredPaper.text_tier: Literal["full_text", "abstract", "none"]`. A fact about the paper: the best text we actually have for it.
- `ClaimRecord.grounded: bool`. The verdict: did the exact quote check out against the source text.
- `ClaimRecord.grounding_tier: Literal["full_text", "abstract", "corroborated", "none"]`. How the claim is supported right now. This can go up during a run (for example `none` to `corroborated` after a successful gap hunt).
- `ClaimRecord.supporting_paper_id`. Which paper backs the claim (its own paper normally, or a different one after corroboration).

I added stable ids for the same reason: it is easier to point at data by id than by list position. So `ClaimRecord.claim_id` and `ContradictionPair.contradiction_id` (both default to `uuid4().hex`).

### `synthesis.trace`: the decision trace

A `DecisionStep` is the main building block. The same object is the runtime trace, the demo timeline, and the evidence the eval harness reads. It is not a flat log line. It has two phases (observe, decide, act, result):

- **Two phases.** `DecisionStep.start(...)` creates the step when the decision is made, with `result="pending"`. Later, `.complete(result=..., effect=..., ...)` fills in what happened. If something crashes in the middle, the step stays visible as `pending` instead of disappearing. `complete()` refuses `"pending"` as a guard.
- **Typed params.** `StepParams` is a union keyed by `kind`, one model per action type (`DecomposeParams`, `SearchParams`, `ReformulateParams`, `ExtractClaimsParams`, `DetectContradictionsParams`, `GapHuntParams`, `ResolveConflictParams`, `SynthesizeParams`). This lets the eval harness group by action type without parsing text, and a `model_validator` checks that `params.kind == action`.
- **Cause links.** `parent_step_id` points a step back at the step it was reacting to. A gap hunt points back at the `extract_claims` step that produced the ungrounded claim, and a conflict resolution carries the `contradiction_id` it worked on. So the trace explains itself, it is not just a list.
- **Effect.** `StepEffect` records what changed (`added_paper_ids`, `claim_ref`/`claim_refs`, `contradiction_ids`, `tier_before`/`tier_after`, `resolved_conflict`, `gap_ref`), so the UI can say "upgraded claim C from none to corroborated using X".
- **Numbers.** `llm_calls` and `duration_ms` per step feed the baseline comparison.

### `synthesis.state`: the working memory

`SynthesisState` is everything the agent knows at one moment: `question`, `sub_queries`, `papers`, `claims`, `contradictions`, `gaps`, `review_text`, `citation_checks`, and the full `trace`. I followed two rules, both carried over from the schema work:

- **Don't store what you can compute.** `current_step`, `status`, `reformulation_count`, `open_gaps`, `grounded_fraction()`, `grounded_by_tier()`, and `citation_validity()` are all derived from the raw lists, not stored next to them. That way they can't drift out of sync.
- **Don't mix separate signals.** Citation validity and grounded fraction are reported on their own. The old single `confidence_score` is only built inside `to_result()`, and only so the `synthesis_runs` table keeps working.

`Gap` is one model with a `kind` field (`ungrounded_claim` or `coverage`) and a simple lifecycle (`open`, `hunting`, then `resolved` or `flagged_unverified`). A failed hunt is just the existing fields combined (`grounded=False`, `grounding_tier="none"`, and a `Gap` with `status="flagged_unverified"`), not a new redundant flag. I kept `to_result()` on purpose: `SynthesisState` is the mutable thing the agent edits while it works, and `SynthesisResult` is the frozen output you hand off. Different uses, different lifetimes.

### `synthesis.controller`: the decision loop

`next_action(state, config)` is the policy function. It reads the state and returns the next pending `DecisionStep` (decompose, then search, then reformulate until a cap, then synthesize). `SynthesisController` runs the steps and owns the feature loops. Each loop is a small method with its own tests:

| Method | What it does | Trace evidence |
|--------|----------|--------------------|
| `run_retrieval_loop` | **Adaptive retrieval.** After every `search`, it ranks the papers it has so far against the question (TF-IDF cosine from `synthesis.rank`) and keeps the top `total_paper_limit`, so on-topic papers win instead of just the newest ones. If a search still comes back with fewer than `min_relevant_papers`, the agent rewrites the query and tries again up to `max_reformulations`, then stops. | `reformulate` steps with `original/new_query`; `parent_step_id` chains search to reformulate to search |
| `run_claims_loop` | One `extract_claims` step per paper, with grounding stamped from the paper's `text_tier`. If an extraction fails it is logged as `failed` and the loop keeps going. | `StepEffect.claim_refs` per paper |
| `run_detect_contradictions_loop` | One pass over all claims. Each `ContradictionPair` gets a stable id and is added to the state. | `StepEffect.contradiction_ids`; parent is the last `extract_claims` step |
| `run_gap_detection_loop` | **Gap hunting.** Every ungrounded claim opens a `Gap` and starts a focused `hunt_support` search. The keywords go out as one combined query (not one search per word), and the results are ranked before grounding, so a claim no longer fans out into off-topic single-word searches. A hit upgrades the claim to `corroborated`; a miss marks it `flagged_unverified`. | `tier_before`/`tier_after`; parent is the origin `extract_claims` step |
| `run_conflict_resolution_loop` | **Conflict resolution.** For each contradiction, look for a third paper using one combined, ranked query. If that paper has a grounded claim, it becomes the resolution, grounded at its own `text_tier` like any other claim, so no new hallucination is introduced. | `StepEffect.resolved_conflict` = `contradiction_id` |
| `run_synthesize_loop` | Build the prompt, generate the review, validate the citations, and write `review_text`, `citation_checks`, and `hallucinated_citations` back to state. It reuses the linear pipeline's stage modules through hooks. | final `synthesize` step, `llm_calls=1` |

`run()` calls the loops in order: retrieval, claims, detect contradictions, gap detection, conflict resolution, synthesize (detection has to come before resolution; gap detection is independent). Every stage can be passed in through `ControllerHooks` for tests, and the real stage functions are imported lazily inside the methods to dodge the `tools` to `pipeline` import cycle. An integration test checks that the cause links stay unbroken before `synthesize` reads the final state.

### `synthesis.trace_view` + `streamlit_app.py`: decision trace UI

`trace_view` is a plain formatting layer with no Streamlit imports (`state_metrics`, `trace_rows`, `summarize_effect`, `claim_rows`, `paper_rows`, `contradiction_rows`, `gap_rows`), so the display logic can be tested without running a UI. `streamlit_app.py` runs the controller and shows two panels: the final review on the left, and the decision trace as a colored timeline on the right (color-coded result badges, the cause and effect of each step, and timing). Below that are tabs for papers, claims, contradictions, and gaps. This is the demo: a reviewer reads the right panel and sees exactly why the left panel came out the way it did. The theme is in `.streamlit/config.toml`, and the app also runs headless under Streamlit's `AppTest` harness.

### Retrieval relevance: fixing it at three layers

An early run gave a bad result: a long-context-retrieval question returned an astrophysics paper. When I dug in, relevance was being lost in three different places, so the fix is in three places too:

1. **The provider** (`tools.search_tools`). arXiv was searched with `sortBy=submittedDate` every time, so a narrow query just returned the newest papers that loosely matched. It now sorts by `relevance`, and only falls back to `submittedDate` when the caller sets a date range (where newest-within-range is actually what you want).
2. **The controller** (`_execute_search`). After every search, the agent re-ranks the papers it has by TF-IDF cosine against the question (`synthesis.rank`) and keeps the top `total_paper_limit`, so relevant papers win instead of just recent ones. The gap and conflict hunts also stopped sending one search per keyword (`all:video`, `all:data`). Now `_single_query_rq` sends one combined query and `_rank_candidates` ranks the results before grounding.
3. **The synthesis step** (`_synthesis_inputs`). Right before writing, the collected papers (which can include loosely related ones the hunts pulled in) get ranked one more time and filtered by a `min_relevance_score` floor (the CLI sets it to `0.03`). Claims and contradictions are then cut down to the papers that survive. So a weak result turns into an honest "not enough evidence" instead of an off-topic review.

For layer 3, `rank_papers` got a `min_score` floor. It defaults to `0.0` (keep everything, the old behavior) and is ignored when the question has no usable words.

### Making citation keys unique

Two different papers by the same first author in the same year produce the same `short_citation_key` (two `[Liu et al. 2025]` papers). This used to hurt twice. `_synthesis_inputs` would drop one of the papers, which lost real evidence, and the validator would mark any key that pointed to more than one paper as hallucinated, so a real citation got called fake. `synthesis.schemas.assign_citation_keys(papers)` fixes this by looking at the whole list at once: when keys collide, it adds a letter to the year (`[Liu et al. 2025a]`, `[Liu et al. 2025b]`) in order. The prompt builder and the validator both call it on the same paper list, so they always agree on the keys. The validator already accepts a `\d{4}[a-z]?` year, so the suffixed keys work without any other change. Now both papers are kept and both can be cited, and the prompt was also tightened to say: only use the keys in the list, and if you can't, leave the statement out.

### Trade-offs

- **More types and modules** than the linear pipeline. The trade is that each one (`trace`, `state`, `controller`, `trace_view`) is small, type-checked, and tested on its own, and the decision trace is reused three ways (runtime, demo, eval).
- **The controller and the linear pipeline both exist.** The CLI, chat tool, and UI all use `run_agentic_synthesis` now, so the demo is consistently agentic, and `run_synthesis` stays as a reference path. They share the same stage modules, so there is no second copy of the logic, just two ways to run it.
- **Reformulation is deterministic** by default (`default_reformulate`), which keeps the loop testable and saves an LLM call. The hook can be swapped for an LLM reformulator later without changing the trace.

### Ownership split (me vs assistant)

- I ran the design review and made the hard calls: splitting text source from claim verdict, the two-phase typed `DecisionStep`, stable ids, the `SynthesisState` shape, how conflict resolution stays grounded, and the call to wire the loops together with an integration test before adding synthesize.
- The assistant wrote the `trace`, `state`, `controller`, and `trace_view` modules, each feature loop, the Streamlit UI, and the mocked tests.


