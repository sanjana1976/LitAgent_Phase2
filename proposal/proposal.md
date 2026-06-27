# A4 Proposal: Research Synthesis Agent

## Title
LitSynth: AI Research Synthesis Agent

## One Sentence Description
An agentic terminal assistant that reads real papers and writes a structured, argument-driven literature review section with contradiction detection and hallucination-resistant citations. Generation as the point.

## Past Project Reference
Built on top of A3, Research Paper Analyzer Agent. The base already has arXiv/Semantic Scholar/DBLP search, PDF fetch and parse, citation formatting, reading lists, and SQLite sessions. This project extends that foundation by adding the synthesis layer, the hard generative piece that doesn't exist yet.

GitHub: (your existing A3 repo link)

## Planned Technologies
- Language: Python 3.11+
- LLM: OpenAI (gpt-4o by default, configurable via OPENAI_MODEL) via the official OpenAI Python SDK and langchain-openai.ChatOpenAI
- Agent framework: LangChain (already in A3)
- PDF parsing: existing pipeline from A3 (tool_fetch_and_parse_pdf + pypdf)
- Storage: SQLite (already in A3) plus new synthesis cache tables
- Eval: custom pytest-based eval harness with hand-labeled ground truth
- Interface: terminal REPL and a new synthesize CLI subcommand (extending the existing Click CLI)

## First Deliverable

Single user story: The user types a research question like "What are the competing approaches to long-context retrieval in LLMs?" and the agent returns a 3 to 5 paragraph literature review section that:

1. Finds 5 to 8 relevant papers using existing search tools
2. Reads their abstracts and key sections (PDF when available, abstract-only fallback)
3. Groups them into themes (not just a flat list)
4. Identifies at least one genuine contradiction or tension between papers
5. Produces a structured argument with inline citations like [Smith et al. 2023]

This forces every part of the system to be exercised: retrieval, PDF parsing, prompting, output formatting, and eval.

## Architecture (First Deliverable)

1. Query Decomposer: input is the raw research question. Output is 3 to 5 search sub-queries covering different angles. Why: one query never captures the full space.

2. Paper Retriever: input is the list of sub-queries. Output is a ranked, deduped list of paper metadata. Hits arXiv and Semantic Scholar APIs (already built in A3). Dedupes by DOI / arXiv id / normalized title.

3. Paper Fetcher and Parser: input is paper ids. Output is a structured paper object {title, abstract, sections[], references[]}. Downloads and caches PDFs under data/cache/. Falls back to abstract-only if PDF unavailable.

4. Relevance Ranker: input is parsed papers plus the original research question. Output is the top N papers scored by TF-IDF cosine relevance (deterministic, no external embedding API). Keeps synthesis focused.

5. Claim Extractor: input is the top N paper objects. Output is a per-paper list of {paper_id, claim, evidence_quote, confidence}. One LLM call per paper with strict JSON schema output. Hallucination risk is highest here, so every evidence_quote is validated against the source text.

6. Contradiction Detector: input is all extracted claims across papers. Output is a list of tension pairs {claim_a, claim_b, paper_a, paper_b, tension_type}. One LLM call with all claims in context.

7. Synthesis Prompt Builder: input is claims, contradictions, original question, paper metadata. Output is a structured prompt with citation format, word budget, and argument structure enforced.

8. Literature Review Generator: input is the synthesis prompt. Output is a 3 to 5 paragraph literature review with inline citations. One LLM call with a high token budget. The main generative artifact.

9. Citation Validator: input is the generated review plus source paper metadata. Output is the validated review with flagged hallucinated citations and a confidence score. Every inline citation is checked against the source set.

10. Eval Harness: input is the research question plus generated review plus labeled ground truth. Output is claim faithfulness, hallucination rate, contradiction coverage. Writes results to eval/results.json.

## Data Shapes (canonical, see src/synthesis/schemas.py)

class ResearchQuestion:        # the raw question + decomposed sub-queries
    question: str
    sub_queries: list[str]

class ScoredPaper:             # retrieved + parsed + ranked
    paper_id: str
    title: str
    authors: list[str]
    abstract: str | None
    year: int | None
    venue: str | None
    url: str | None
    sections: dict[str, str]   # may be empty when abstract-only
    relevance_score: float

class ClaimRecord:             # extracted from a single paper
    paper_id: str
    claim: str
    evidence_quote: str        # verbatim from paper text
    confidence: float          # 0..1
    grounded: bool             # set by validator: quote appears in source text

class ContradictionPair:
    paper_a: str
    paper_b: str
    claim_a: str
    claim_b: str
    tension_type: str          # contradiction | scope | methodology
    explanation: str

class SynthesisResult:
    question: str
    review_text: str
    citations_used: list[str]       # paper ids actually cited
    hallucinated_citations: list[str]
    contradictions_found: int
    confidence_score: float         # weighted by grounded claims + citation accuracy
    papers: list[ScoredPaper]
    claims: list[ClaimRecord]
    contradictions: list[ContradictionPair]

## After-First-Deliverable Goals

- Hallucination eval suite: hand-labeled dataset of 20+ research questions with known correct claims; measure what percent of generated claims are faithful to source papers.
- Contradiction detection eval: curate 10 known disagreements in ML literature (for example scaling laws debates) and measure whether the agent finds them.
- Multi-round refinement: user can say "go deeper on the retrieval angle" and the agent fetches more targeted papers and regenerates that section.
- Gap detection: agent identifies what the literature does not answer and flags it explicitly in the review.
- Export to markdown / BibTeX: one command dumps the review plus full bibliography in ready-to-paste format.
- Session persistence: resume a synthesis session, add more papers, regenerate; stored in SQLite.
- Confidence scoring: every claim in the output has a confidence score based on how many papers agree; low-confidence claims are flagged inline.
- Comparison mode: given two competing methods (for example RAG vs fine-tuning), generate a structured pro/con synthesis specifically.

## What Makes This Hard

The easy version is: fetch abstracts, concatenate, ask GPT to summarize. The hard version, what this is, requires:

1. Faithful claim extraction: the agent has to ground every claim in a verbatim quote before synthesizing.
2. Citation validation: every inline citation in the output gets checked against the source set; hallucinated citations are flagged, not silently included.
3. Contradiction detection as a first-class feature: not just summarizing each paper, but finding where they genuinely disagree.
4. A real eval: the eval harness measures claim faithfulness, citation accuracy, and contradiction coverage with labeled ground truth.

This is the gap between tools like Elicit and what researchers actually need. Elicit finds papers. It doesn't write the argument.
