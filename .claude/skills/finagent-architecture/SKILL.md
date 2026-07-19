---
name: finagent-architecture
description: Implementation/architecture reference for the FinAgent-RAG thesis system — document processing pipeline, PDF/table parsing, chunking, the 7-agent adaptive RAG architecture, query complexity routing, per-agent prompting design, and the full 5-family evaluation metric formulas. Load this before writing or reviewing any pipeline, agent, chunker, prompt, or metric code in this workspace, so implementation stays faithful to Kasi_Research_Proposal.pdf rather than improvised.
---

# FinAgent-RAG: Implementation & Architecture Skill

Grounded in a full read of `Kasi_Research_Proposal.pdf` (34 pages, sections 7.1–7.12). This is the
ground-truth architecture for **all coding/implementation work** in this workspace. If any other
document, memory, or prior assumption conflicts with this file or the PDF, the PDF wins.

Use this skill when: building the document-processing pipeline, implementing any of the 7 agents,
writing chunking/embedding code, designing prompts for an agent, implementing an evaluation metric,
or reviewing code for architectural fidelity to the proposal.

For which of the 14 experiments to build and how to record results, use the **finagent-experiments** skill instead — this skill is the shared architecture underneath all 14.

## 0. Production-grade implementation standard (mandatory)

This is a research-grade thesis codebase, not a prototype. Every module, agent, and experiment built
in this workspace must meet this bar — no exceptions for "I'll flesh it out later":

- **No skeleton/scaffold functions.** Every function must have a complete, working implementation.
  A function with only a signature, a docstring, and `pass`/`raise NotImplementedError`/`# TODO` is
  not acceptable output — either implement it fully or don't create it yet.
- **Fully parameterized.** No hardcoded model names, chunk sizes, top-k values, temperatures, file
  paths, or thresholds inside function bodies — they come from `config.py` constants or explicit
  function parameters with sensible defaults matching §1 below. This is what makes the ablation
  experiments (EXP-12/13/14) a config toggle instead of a forked copy of EXP-11.
- **Full docstrings.** Every public function/class gets a docstring covering: what it does, why (the
  design rationale from the proposal, not just a restatement of the code), parameters, return value,
  and any Groq API cost implications (does it call the LLM? how many times?).
- **Complete prompt designs.** Every agent's LLM prompt(s) must be fully written out — explicit
  system/user prompt text addressing exactly what that agent is responsible for (see §4), not a
  placeholder like `f"Answer this: {question}"`. Prompts should specify: the agent's role, the exact
  task, the expected output format (plain text vs. structured/JSON), and constraints (e.g. "only use
  information present in the provided evidence — do not use outside knowledge").
- **Groq API call efficiency is a first-class design constraint.** Every agent/experiment implementation
  must be reviewed against this before being considered done:
  - Minimize the number of LLM calls per question. Combine steps into a single structured-output call
    wherever the proposal doesn't require them to be separate reasoning stages (e.g. query
    understanding + complexity classification can be one call with a structured JSON response, not
    two).
  - Skip agents entirely when the adaptive route doesn't need them (§5) — a Simple-route question
    must not invoke Query Refinement or multi-query expansion; that's not an optimization, it's the
    architecture.
  - Always request `temperature=0.0` (§1) and set an explicit `max_tokens` sized to the expected
    output — don't leave it unbounded.
  - Retry only on transient failures (rate limit / timeout / 5xx) with exponential backoff; never
    silently retry on a bad/malformed response more than a small bounded number of times.
  - Track and return token usage (input + output) from every LLM call so `Token Usage`/`Cost per
    Answer` (§7E) can be computed without a separate accounting pass.
  - Cache/reuse embeddings and retrieval results within a single question's pipeline run — don't
    re-embed or re-query ChromaDB for information already fetched earlier in the same request.
  - Batch embedding generation during ingestion (chunk embedding), not per-chunk API calls, wherever
    the embedding backend supports it.
- **Reliability over cleverness.** The 70B model via Groq is fast but not infallible — agents that
  expect structured output (JSON) must validate/parse defensively (e.g. `json.loads` in a try/except
  with a single bounded repair-retry), not assume the model always returns valid JSON on the first try.

## 1. Fixed controlled setup (must NEVER vary across experiments)

This is the independent-variable discipline of the whole thesis: only the retrieval/reasoning
pipeline design changes between experiments. Everything below must be held identical (confirmed by
both the proposal Table 7.12 and `Coding_Sheet.xlsx` "Common Metrics"):

| Component | Fixed value |
|---|---|
| Dataset | FinanceBench (same PDF collection across all experiments) |
| LLM | Groq Llama 3.3 70B Versatile |
| Vector Database | ChromaDB |
| Embedding Model | Same embedding model for all RAG experiments (e.g. `all-MiniLM-L6-v2`) |
| Chunk Size | 500 tokens |
| Chunk Overlap | 100 tokens |
| Retrieval Top-k | Top 5 chunks |
| Temperature | 0.0 |
| Output Storage | question, generated answer, retrieved chunks, reference answer, latency, token usage, metric scores |

Never hardcode a different chunk size, top-k, or temperature "just for this experiment" — that
breaks the controlled-comparison design that the entire benchmarking ladder depends on.

## 2. Financial document processing & knowledge base pipeline

Source: proposal §7.5, Table 7.4, Table 7.5. FinanceBench PDFs (10-K, 10-Q, 8-K, earnings reports,
regulatory disclosures) mix structured tables and unstructured narrative — the pipeline must
preserve structure, not just dump raw text.

Pipeline order:

1. **PDF Text Extraction** — extract text preserving headings, tables, section boundaries. (Use
   `pypdf`'s `PdfReader(...).pages[i].extract_text()` per this workspace's environment notes —
   poppler/`pdftoppm` is not installed here.)
2. **Table Extraction** — extract financial tables/numerical statements where possible (balance
   sheet, income statement, cash flow). Numerical questions depend on this succeeding.
3. **Cleaning** — remove repeated headers, page numbers, navigation elements, formatting noise.
4. **Section Detection** — identify sections: balance sheet, income statement, risk factors, notes,
   management discussion & analysis, etc.
5. **Chunking** — split into retrieval-ready, section-aware semantic chunks (500 tokens / 100
   overlap, per §1 above).
6. **Metadata Tagging** — attach company name, filing year, report type, section name, page number,
   chunk ID to every chunk.
7. **Storage** — embed chunks and store chunks + metadata in ChromaDB for semantic indexing.

Sample processed chunk record (Table 7.5 — match this schema):

```
Chunk ID:     AAPL_2022_AR_CH_015
Company:      Apple
Year:         2022
Report Type:  Annual Report
Section:      Consolidated Statements of Operations
Page Number:  34
Text:         "Net sales increased to $394.3 billion in 2022, primarily driven by strong demand
               for iPhone, Mac, and Services across multiple geographic segments."
```

## 3. FinanceBench dataset fields

Source: proposal §7.4, Table 7.3. Every QA record has:

| Field | Description |
|---|---|
| `question` | The financial question — factual retrieval, numerical reasoning, comparison, or interpretation |
| `answer` | Ground-truth answer (textual, numerical, or explanatory) |
| `evidence` | Supporting evidence text from the source document — used for grounding/faithfulness eval |
| `page_number` | Page where evidence appears — enables fine-grained retrieval evaluation |
| `company` | Company associated with the filing |
| `document_type` | 10-K, 10-Q, 8-K, or earnings report |
| `document_name` | Filing identifier/filename |
| `document_year` | Filing year |
| `question_type` | Complexity/reasoning category |
| `gics_sector` | Industry sector classification |
| `justification` | Explanation/reasoning info for some QA pairs |

Source: https://github.com/patronus-ai/financebench/tree/main/pdfs

## 4. The 7-agent adaptive architecture

Source: proposal §7.6, Table 7.6. Not a fixed pipeline — routing depth adapts to query complexity
(§5 below). Full pipeline order when all stages activate:

```
User Question
  → Query Understanding Agent
  → (if ambiguous/broad) Query Refinement Agent
  → Retrieval Agent
  → Evidence Filtering Agent
  → Reasoning Agent
  → Answer Generation Agent
  → Verification Agent
  → Final Answer
```

| Agent | Role |
|---|---|
| Query Understanding | Identifies intent, company, financial terminology, reporting year, and **complexity level** |
| Query Refinement | Rewrites broad/ambiguous financial questions into retrieval-friendly form. Only invoked when Query Understanding flags the question as needing it. |
| Retrieval | Semantic + metadata search over the ChromaDB knowledge base (report type, filing year, section, page) |
| Evidence Filtering | Removes weak, noisy, or redundant retrieved chunks before reasoning |
| Reasoning | Numerical interpretation, cross-section analysis, comparison, aggregation |
| Answer Generation | Produces the final grounded natural-language response from filtered evidence + reasoning output |
| Verification | Validates the generated answer is actually supported by retrieved evidence; reduces unsupported claims and numerical inconsistencies |

Note EXP-12/13/14 (ablations, see finagent-experiments skill) each remove exactly one of Query
Refinement / Evidence Filtering / Verification from this pipeline — so each agent should be a
swappable/toggleable stage, not hardcoded into a monolithic function.

## 5. Query complexity detection & adaptive routing

Source: proposal §7.7, Tables 7.7–7.9. Classification is performed by the Query Understanding Agent.

| Level | Meaning | Example |
|---|---|---|
| Simple | Direct factual/numerical lookup, single chunk sufficient | "What was Microsoft's revenue in 2022?" |
| Moderate | Comparison across periods / multiple related values | "How did Apple's revenue change from 2021 to 2022?" |
| Complex | Multi-step, multi-section, explanation-oriented | "What factors contributed to the change in operating income?" |

Reasoning signals used for classification (Table 7.8):

| Signal | Meaning |
|---|---|
| Financial Entity | Company name, reporting year, financial metric |
| Question Type | lookup / comparison / explanation / trend / reasoning |
| Number of Required Values | one value vs. multiple |
| Need for Calculation | does it require arithmetic/comparison? |
| Need for Multiple Evidence Chunks | does info span different sections? |

Routing rules (Table 7.9):

| Condition | Route |
|---|---|
| One company + one year + one metric | Simple |
| Multiple years or comparison terms | Moderate |
| Contains "why", "explain", "compare", "trend", "reason" | Complex |
| Requires values from different sections/statements | Complex |
| Needs arithmetic calculation or financial interpretation | Moderate or Complex |
| Requires evidence from multiple pages/disclosures | Complex |

Simple queries: short path — direct top-k retrieval + generation (skip refinement, minimize
reasoning depth). Complex queries: full path — refinement, multi-query expansion, evidence
filtering, deeper reasoning, verification.

## 6. Retrieval, evidence selection, and reasoning workflow

Source: proposal §7.8. The 8-step operational sequence:

1. User financial question is received.
2. Query Understanding Agent identifies intent and query complexity.
3. Query Refinement Agent improves the question if refinement is required.
4. Retrieval Agent fetches relevant financial document chunks (direct top-k for simple; query
   rewriting/expansion/multi-query for complex).
5. Evidence Filtering Agent selects the strongest supporting evidence, considering metadata
   (company, year, page, section) for contextual consistency and traceability.
6. Reasoning Agent combines evidence and performs numerical interpretation, comparison,
   aggregation, explanation-oriented reasoning.
7. Answer Generation Agent produces the final grounded response.
8. Verification Agent validates the answer is supported by retrieved evidence.

Sample evidence selection structure (Table 7.10) — retrieval agent output should look like:

```
Evidence ID | Company   | Year | Section                              | Retrieved Text                    | Relevance Score
EV_001      | Microsoft | 2022 | Consolidated Statements of Income    | "Revenue increased primarily..."  | 0.94
```

## 7. Evaluation metrics (5 families, with formulas)

Source: proposal §7.10, Tables 7.13–7.17. Correctness alone is insufficient — grounding, retrieval
quality, reasoning validity, and efficiency must all be measured. Implement metric functions once,
shared across all 14 experiments (per finagent-experiments skill), not reimplemented per experiment.

**A. Answer Quality**
| Metric | Formula |
|---|---|
| Answer Relevance | cosine(Q_embedding, A_embedding) = (Q·A)/(\|Q\|\|A\|) |
| Exact Match (EM) | 1 if generated == reference else 0 |
| F1-Score | 2·Precision·Recall / (Precision+Recall), token-level overlap |
| Semantic Similarity | cosine(E_generated, E_reference) |

**B. Evidence & Retrieval**
| Metric | Formula |
|---|---|
| Context Recall | Relevant Chunks Retrieved / Total Relevant Chunks |
| Context Precision | Relevant Retrieved Chunks / Total Retrieved Chunks |
| Hit@K | Queries With Correct Evidence In Top-K / Total Queries |
| MRR | (1/N)·Σ(1/rank_i) over first correct evidence chunk rank |

**C. Grounding & Trust**
| Metric | Formula |
|---|---|
| Faithfulness | Supported Statements / Total Statements In Generated Answer |
| Hallucination Rate | Unsupported Claims / Total Claims |
| Evidence Coverage | Evidence Points Used / Total Relevant Evidence Points |
| Citation Correctness | Correct Supporting Citations / Total Citations |

**D. Financial Reasoning**
| Metric | Formula |
|---|---|
| Numerical Accuracy | Correct Numerical Values / Total Numerical Values |
| Calculation Accuracy | Correct Calculations / Total Calculations |
| Multi-step Reasoning Score | Correct Reasoning Steps / Total Reasoning Steps |
| Explanation Completeness | Covered Reasoning Points / Total Expected Reasoning Points |

**E. Efficiency**
| Metric | Formula |
|---|---|
| Latency | Response End Time − Response Start Time |
| Token Usage | Input Tokens + Output Tokens |
| Retrieval Time | Retrieval End Time − Retrieval Start Time |
| Cost per Answer | Total API/Compute Cost / Total Generated Answers |

Note: `Coding_Sheet.xlsx` "Common Metrics" narrows this to an 11-metric core reporting set (answer
relevance, exact match, F1, semantic similarity, context precision, context recall, faithfulness,
hallucination rate, numerical accuracy, citation correctness, latency) — use the full 5-family set
above for deep/diagnostic evaluation, and the 11-metric core for the main comparison tables (see
finagent-experiments skill).

## 8. End-to-end implementation workflow (15 steps)

Source: proposal §7.11. This is the build order for the whole system, not just one experiment:

1. Collect FinanceBench documents (PDFs + QA/evidence files).
2. Extract text and tables from financial PDFs.
3. Clean and structure extracted content.
4. Create financial document chunks (section-aware).
5. Add metadata to chunks (company, year, report type, section, page, chunk ID).
6. Generate embeddings and store in ChromaDB.
7. Prepare benchmark questions, classified into simple/moderate/complex.
8. Build baseline systems: Direct LLM, Naïve RAG, Query-Rewritten RAG, Multi-Query RAG.
9. Build the proposed adaptive multi-agent RAG system (all 7 agents).
10. Run all benchmark questions through every system.
11. Store generated answers, retrieved chunks, evidence IDs, model name, system type, latency.
12. Evaluate results using the metrics in §7.
13. Compare baseline vs. proposed system.
14. Perform error analysis (wrong numbers, missing evidence, weak reasoning, hallucinated claims).
15. Summarize the best-performing approach (accuracy, grounding, reasoning, efficiency balance).

Steps 10–15 map directly onto the finagent-experiments skill's result-recording workflow.

## 9. Tech stack & environment notes

- Python 3.12, LangChain 0.3+, ChromaDB 0.5+, Pandas 2.2+, NumPy 2.2+, Scikit-learn 1.6+, FAISS 1.10+.
- Hardware target: i7/Ryzen 7, 16GB+ RAM, RTX 3060+ GPU, 500GB+ SSD (informs batch sizes / local vs.
  cloud embedding choices — Colab is the proposal's stated fallback for constrained compute).
- This machine has no poppler/`pdftoppm` — use `pypdf` for PDF text extraction, not
  `pdf2image`/`pdftoppm`-based tooling.
- Printing non-ASCII characters (e.g. "→", "Naïve") directly to this machine's git-bash terminal can
  crash on cp1252 encoding — write extraction/analysis output to a file and Read it, or strip
  non-ASCII before printing, rather than printing directly in Bash.
