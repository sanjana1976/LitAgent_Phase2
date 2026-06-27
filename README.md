# Research Paper Analyzer Agent + LitSynth
Multi-turn terminal research assistant for literature discovery, PDF-backed analysis, reading lists, and citation formatting (A3). A4 extends it with LitSynth, a synthesis system that turns one research question into a structured, citation-grounded literature review with contradiction detection and citation validation.

A4 proposal: proposal/proposal.md
Marked-up proposal: proposal/proposal-markedup.md
Architecture and design rationale: Design.md

## Features
- Search: arXiv, DBLP, Semantic Scholar, Crossref.
- PDF: fetch, parse, and cache papers under data/cache/.
- Analysis: deep analysis with OpenAI when configured, plus heuristic fallback.
- Citations: PDF reference extraction and forward citations through Semantic Scholar.
- Compare: compare 2 to 5 cached papers.
- Bibliography: BibTeX, APA, and Chicago.
- Reading lists: create lists, add or remove papers, and track reading status.
- Sessions: resume chat through SQLite with chat, resume, and status.
- Synthesis (A4): agentic LitSynth controller that runs retrieval, claims, contradiction detection, gap hunting, conflict resolution, then synthesis.
- Decision trace UI (A4): Streamlit app showing the final review beside every traced agent decision, causal parent link, and state effect.
- Hallucination resistance (A4): verbatim-quote grounding per claim, full-text / abstract / corroborated tiers, whitelisted citations, and validator checks for fake citations.
- Auto-saved reviews (A4): every synthesize run writes a timestamped markdown file to research reviews/.
- Guardrails: input blocks, tool tiers, output BibTeX check, and PROJECT_ROOT exports.

## Requirements
- Python 3.11+
- OPENAI_API_KEY for chat and OpenAI-backed analysis
- Network access for scholarly APIs when using search and PDF tools
- Streamlit for the decision-trace UI (included in requirements.txt)

## Quick Start
Create and activate a virtual environment:

python -m venv .venv
.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
pip install -e .

Copy the environment file and set OPENAI_API_KEY:

copy .env.example .env

Initialize the database:

python main.py init-db

Run chat:

python main.py chat

Run A4 synthesis:

python main.py synthesize "What are the competing approaches to long-context retrieval in LLMs?" --verbose

Run the decision-trace UI:

streamlit run streamlit_app.py

The chat starts with a welcome banner and then You> prompts. In-session commands include help, status, list my lists, clear history, and exit.

## Docker

Prerequisites: Docker Desktop, and a `.env` file with `OPENAI_API_KEY` (same as Quick Start).

Build and run the Streamlit decision-trace UI:

docker compose up --build

Open http://localhost:8501. On first start the entrypoint creates `data/` and `research reviews/` (if missing), runs `init-db` when SQLite is absent, then starts Streamlit.

One-shot synthesis (uses the same image as `ui`):

docker compose run --rm ui python main.py synthesize "What are the competing approaches to long-context retrieval in LLMs?" --verbose

Interactive terminal chat:

docker compose --profile cli run --rm cli

Run the mocked test suite inside the image (no `.env` or live APIs required):

docker compose --profile test run --rm test

Stop the UI: `docker compose down`.

Persistence: `./data` (SQLite + PDF cache) and `./research reviews` (auto-saved markdown) are bind-mounted into the container. They survive restarts.

Deploy notes:

- The output folder is literally named `research reviews` (with a space), matching `REVIEWS_DIR_NAME` in `main.py`. Compose uses long-form bind mounts with quoted paths; do not rename the folder without updating the app and compose file together.
- `docker/entrypoint.sh` must use Unix (LF) line endings. `.gitattributes` enforces this; CRLF breaks the shebang in Linux containers.
- The image runs as root so bind mounts work reliably on Windows and macOS hosts. For a production deploy with no host volumes, switch to a non-root `USER` in the Dockerfile.

## A4 In One Minute
- One-shot literature review: python main.py synthesize "Compare RAG and fine-tuning for LLM domain adaptation" --verbose
- Inspect persisted runs: python main.py synth-history
- Use the same synthesis tool inside chat: python main.py chat, then ask for a literature review on competing approaches to long-context retrieval in LLMs.
- Follow up in chat: Which of those papers actually evaluate on needle-in-a-haystack benchmarks?

## Two Synthesis Entry Points
- Agentic controller: synthesis.controller.SynthesisController is the main A4 path. It reads explicit SynthesisState, decides the next action, logs each DecisionStep, and backs the Streamlit UI, CLI synthesize command, and chat tool.
- Linear pipeline: synthesis.pipeline.run_synthesis is the original deterministic 10-stage function. It is kept as a compatibility and reference path, but it is no longer the main demo path.

See Design.md for the full architecture and the reason behind the agentic refactor.

## CLI
- python main.py chat: REPL with quiet logs.
- python main.py chat --verbose: do not force quiet logging.
- python main.py chat --session-id ID: resume a session.
- python main.py chat --no-persist: skip SQLite turns and permission audit.
- python main.py chat --reading-list-context TAG: attach an optional reading-list tag to persisted turns.
- python main.py resume: pick a recent session.
- python main.py status: show DB path and recent sessions.
- python main.py init-db: create the schema.
- python main.py synthesize "...": run the agentic LitSynth controller, print a markdown review, and auto-save it to research reviews/.
- python main.py synthesize "..." --verbose: stream coarse controller progress while the trace records each internal decision.
- python main.py synthesize "..." --output review.md: override the output filename. Relative paths are anchored under research reviews/.
- python main.py synthesize "..." --no-write: print to stdout only.
- python main.py synthesize "..." --source semantic_scholar: opt into Semantic Scholar. Default is arXiv-only to avoid rate-limit delays.
- python main.py synthesize "..." --top-n 8 --word-budget 700: tune ranked-paper count and review length.
- python main.py synth-history --limit 20: list recent persisted synthesis runs.
- python main.py eval-synthesis: score saved synthesis runs against eval/cases.json and write eval/results.json.
- python main.py eval-synthesis --run-id ID: score one saved run.
- python main.py eval-synthesis --live: generate missing case results with the live controller, then score. Requires OPENAI_API_KEY.
- streamlit run streamlit_app.py: run the agentic controller and inspect the review beside its decision trace.

Global log override: python main.py --log-level DEBUG chat

## Environment
- OPENAI_API_KEY: required for chat and OpenAI-backed analysis.
- OPENAI_MODEL: optional, defaults to gpt-4o.
- DATABASE_PATH: optional, defaults to ./data/papers.sqlite3.
- LOG_LEVEL: optional, defaults to INFO.
- PROJECT_ROOT: optional, defaults to the current working directory.

Optional guardrail and API overrides are documented in .env.example and Design.md.

## Layout
- Dockerfile, docker-compose.yml, docker/entrypoint.sh: containerized Streamlit UI, optional CLI profile, and offline test profile.
- main.py: Click CLI for chat, resume, status, init-db, synthesize, synth-history, and eval-synthesis.
- config/: settings, guardrail tool lists, and API bases.
- src/agent/: AgentManager, ConversationManager, and transcript helpers.
- src/tools/: tools plus registry, including tool_synthesize_literature_review.
- src/synthesis/: LitSynth pipeline plus the agentic controller, state, trace, trace view, and stage modules.
- streamlit_app.py: two-panel review and decision-trace demo.
- src/guardrails/: permissions, validators, and output guardrails.
- src/cli/: welcome banner and help.
- src/db/: SQLite schema and queries. Version 4 adds synthesis_runs.
- src/models/: persistence models.
- tests/: pytest suite with mocked LLM/API calls. Current suite is 242 tests: 78 A3 and 164 A4.
- research reviews/: auto-generated markdown reviews from synthesize.
- eval/: labeled cases.json and generated results.json for LitSynth eval.
- Design.md: source of truth for design decisions.
- proposal/: original and marked-up A4 proposal.
- review/: reviewer quick start and session log.
- transcripts/: optional manual JSON logs.

## Testing
Run the test suite:

python -m pytest

There are 242 mocked tests. They cover the A3 code, every LitSynth stage, the agentic controller and its loops, the decision-trace/state models, eval CLI, and UI formatting helpers. CI does not call live APIs. conftest.py sets a dummy OPENAI_API_KEY.

In Docker: `docker compose --profile test run --rm test` (tests are baked into the image; no extra volume mount).

## LitSynth Eval
The eval path has three pieces:

- eval/cases.json: labeled research questions with expected_contradiction_keys.
- python main.py eval-synthesis: scores saved synthesis runs from SQLite and writes eval/results.json.
- python main.py eval-synthesis --live: generates missing case results first, then scores them. Requires OPENAI_API_KEY.

The committed eval/results.json is the baseline: claim faithfulness 0.88, citation hallucination rate 0.20, contradiction coverage 0.00 over 1 scored run. Coverage is 0.00 because the starter cases have no expected paper-pair keys yet. After reviewing a saved run, fill expected_contradiction_keys in eval/cases.json to make that metric meaningful.

The SQLite database in data/ is gitignored, so a fresh clone has no saved runs. To regenerate the numbers from scratch, run:

python main.py eval-synthesis --live

On a machine that already has saved runs, python main.py eval-synthesis scores them offline with no API calls.

## Live Demo Prompts
- Headline A4 story: python main.py synthesize "What are the competing approaches to long-context retrieval in LLMs?" --verbose
- Agentic decision trace UI: streamlit run streamlit_app.py, then run a question and read the final review beside the live decision timeline.
- Contradiction detection: python main.py synthesize "What are the tradeoffs between RAG and fine-tuning for LLM domain adaptation?" --verbose
- Multi-turn follow-up: python main.py chat, ask for a literature review, then ask which papers evaluate on needle-in-a-haystack benchmarks.
- Compare two papers: python main.py chat, then ask it to compare https://arxiv.org/abs/2310.06825 and https://arxiv.org/abs/2401.04088.
- Guardrail refusal: python main.py chat, then ask delete reading list 1.
- Eval from a fresh clone: python main.py eval-synthesis --live, or open committed eval/results.json.

## Course Review
- Design.md: architecture, guardrails, tools, and oracles.
- proposal/proposal.md: original A4 proposal.
- proposal/proposal-markedup.md: final marked-up proposal.
- REVIEW-PLAN.md: review-day feedback and changes made in response.
- PEER_REVIEW.md: filled peer-review template.
- review/README.md: review session quick start.
- review/session-log.md: demo script and traces.

## License
Use and modify for your course assignment as permitted by your instructor.
