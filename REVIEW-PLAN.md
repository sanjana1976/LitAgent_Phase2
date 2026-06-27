# REVIEW-PLAN

Feedback I got on review day (given verbally, I do not have a written reviewer doc) and what I changed in response. I will rewrite these in my own voice before submitting.

## Feedback I received

- The ten-stage pipeline was redundant. It ran every stage in the same fixed order no matter what the question needed.
- Latency was slow.
- There was no UI. A CLI-only demo felt underwhelming.

## What I changed in response

### 1. Redundant ten-stage pipeline to agentic controller

Replaced the fixed linear pipeline with an agentic controller (src/synthesis/controller.py) that reads its own state and picks the next action instead of always running all ten stages. It only does what is needed (reformulates retrieval when results are thin, hunts gaps only for ungrounded claims, resolves conflicts only when contradictions exist), and logs every decision as a DecisionStep (src/synthesis/trace.py).

### 2. Slow latency

Cut a typical run from about 2 to 3 minutes down to around 1 minute (rough timing from my own demo runs, not a controlled benchmark). The wins were defaulting to arXiv-only retrieval to skip slow Semantic Scholar rate-limit retries, trimming low-relevance papers before claim extraction, reusing cached PDF parses, and failing soft on individual calls.

### 3. No UI to Streamlit decision-trace UI

Added streamlit_app.py: a two-panel view with the final review on the left and the agent's live decision timeline on the right, plus tabs for papers, claims, contradictions, and gaps. This makes the on-the-spot behavior visible.

### Related feature: paper-vs-paper comparison

The agent compares two papers' claims to find genuine disagreements (contradiction detection), then hunts a third paper to put the disagreement in context (conflict resolution).
