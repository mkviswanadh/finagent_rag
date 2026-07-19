# FinAgent-RAG — Implementation Workspace

Master's thesis (Machine Learning & AI, LJMU) implementation for **Kasi Viswanadh Maddala** (Student ID: 1224024, Supervisor: Shubham Gupta).

This workspace is for **coding/development/implementation only**. The literature review and thesis-document work lives in a separate documentation workspace — this one is scoped to building the actual FinAgent-RAG system, running the 14 experiments, and capturing results.

## Thesis Title

**FinAgent-RAG: Adaptive Multi-Agent Retrieval-Augmented Generation for Financial Document Intelligence**

## Problem & Motivation

Financial documents (10-K, 10-Q, 8-K, earnings reports) mix structured tables and unstructured narrative, with evidence fragmented across sections, footnotes, and cross-references. Fixed-pipeline RAG struggles with ambiguous, multi-step, or multi-section financial queries. This project builds and benchmarks an adaptive multi-agent RAG framework against progressively more capable baselines, all under one controlled setup, to determine where adaptive/agentic reasoning actually earns its cost.

## Research Questions

1. How effectively does the adaptive multi-agent framework retrieve relevant evidence and generate grounded answers from FinanceBench under controlled conditions?
2. Which approach — Direct LLM, Naïve RAG, Metadata-Aware RAG, Query-Rewritten RAG, Multi-Query RAG, or Adaptive Multi-Agent RAG — produces the most accurate/faithful/context-supported answers under identical dataset, embedding model, LLM, and metrics?
3. How does query complexity affect retrieval quality, numerical reasoning, and evidence grounding?
4. What impact do adaptive routing, query refinement, and multi-agent reasoning have on hallucination reduction and evidence-grounded generation?

## Dataset — FinanceBench

Source: https://github.com/patronus-ai/financebench/tree/main/pdfs
Real financial filings (10-K, 10-Q, 8-K, earnings reports, regulatory disclosures) with QA pairs + evidence annotations, evaluable at document/page/chunk level.

Fields: `question`, `answer`, `evidence`, `page_number`, `company`, `document_type`, `document_name`, `document_year`, `question_type`, `gics_sector`, `justification`.

## Document Processing Pipeline

PDF text extraction (preserving headings/tables/sections) → table extraction → cleaning (headers, page numbers, noise) → section detection (balance sheet, income statement, risk factors, notes) → chunking (section-aware) → metadata tagging (company, year, report type, page, section) → embeddings → ChromaDB storage.

Sample chunk record: `Chunk ID` (e.g. `AAPL_2022_AR_CH_015`), `Company`, `Year`, `Report Type`, `Section`, `Page Number`, `Text`.

## Agentic Architecture (7 agents)

| Agent | Role |
|---|---|
| Query Understanding | Detects intent, company, reporting period, terminology, complexity level |
| Query Refinement | Rewrites ambiguous/broad questions into retrieval-friendly form |
| Retrieval | Semantic + metadata search over the knowledge base |
| Evidence Filtering | Removes redundant/weak/noisy chunks |
| Reasoning | Numerical interpretation, cross-section reasoning |
| Answer Generation | Produces the final grounded natural-language response |
| Verification | Validates the answer is supported by retrieved evidence |

**Pipeline order:** Query received → Query Understanding → (if needed) Query Refinement → Retrieval → Evidence Filtering → Reasoning → Answer Generation → Verification.

## Adaptive Routing (Query Complexity)

| Level | Meaning | Example |
|---|---|---|
| Simple | Direct factual/numerical lookup, single chunk sufficient | "What was Microsoft's revenue in 2022?" |
| Moderate | Comparison across periods / multiple related values | "How did Apple's revenue change from 2021 to 2022?" |
| Complex | Multi-step, multi-section, explanation-oriented | "What factors contributed to the change in operating income?" |

Reasoning signals used for classification: financial entity (company/year/metric), question type (lookup/comparison/explanation/trend/reasoning), number of required values, need for calculation, need for multiple evidence chunks.

Routing rules: 1 company+1 year+1 metric → simple; multiple years/comparison terms → moderate; "why/explain/compare/trend/reason" or cross-statement values → complex; arithmetic/interpretation → moderate or complex; multi-page/disclosure evidence → complex.

Simple queries take a short path (direct top-k retrieval + generation); complex queries trigger query refinement, multi-query expansion, evidence filtering, and deeper reasoning.

## Benchmarking Ladder — 14-Experiment Design (CONFIRMED)

Confirmed via `Coding_Sheet.xlsx` (sheet "Overall Performance Results of") — this is the authoritative, resolved experiment design:

| Exp. | Name | Category |
|---|---|---|
| EXP-01 | Direct LLM with Zero-Shot Prompting | Direct LLM |
| EXP-02 | Direct LLM with Role-Based Financial Analyst Prompting | Direct LLM |
| EXP-03 | Direct LLM with Few-Shot Prompting | Direct LLM |
| EXP-04 | Direct LLM with Stepwise Financial Reasoning Prompting | Direct LLM |
| EXP-05 | Direct LLM with Self-Verification Prompting | Direct LLM |
| EXP-06 | Direct LLM with Structured Output Prompting | Direct LLM |
| EXP-07 | Naïve RAG using ChromaDB | Basic RAG |
| EXP-08 | Metadata-Aware Naïve RAG using ChromaDB | Improved RAG |
| EXP-09 | Query-Rewritten RAG using ChromaDB | Query Reformulation RAG |
| EXP-10 | Multi-Query RAG using ChromaDB | Multi-Query RAG |
| EXP-11 | Adaptive Multi-Agent RAG using ChromaDB | Proposed System |
| EXP-12 | Adaptive Multi-Agent RAG without Query Refinement | Ablation Study |
| EXP-13 | Adaptive Multi-Agent RAG without Evidence Filtering | Ablation Study |
| EXP-14 | Adaptive Multi-Agent RAG without Verification Agent | Ablation Study |

**Confirmed fixed setup** (from `Coding_Sheet.xlsx`, sheet "Common Metrics") — held identical across all 14 experiments so any performance difference is attributable to architecture alone:

| Component | Fixed Setup |
|---|---|
| Dataset | FinanceBench (same PDF collection across all experiments) |
| LLM | Groq Llama 3.3 70B Versatile |
| Vector Database | ChromaDB |
| Embedding Model | Same embedding model for all RAG experiments (example given: all-MiniLM-L6-v2) |
| Chunk Size | 500 tokens |
| Chunk Overlap | 100 tokens |
| Retrieval Top-k | Top 5 chunks |
| Temperature | 0.0 |
| Output Storage | Question, generated answer, retrieved chunks, reference answer, latency, token usage, metric scores |

## Evaluation Metrics

**Main metrics** (per `Coding_Sheet.xlsx` "Common Metrics"): Answer relevance, exact match, F1-score, semantic similarity, context precision, context recall, faithfulness, hallucination rate, numerical accuracy, citation correctness, latency.

**Full 5-family breakdown** (per the approved Research Proposal):
- **A. Answer Quality:** Answer Relevance (cosine of Q·A embeddings), Exact Match, F1-Score, Semantic Similarity.
- **B. Evidence & Retrieval:** Context Recall, Context Precision, Hit@K, Mean Reciprocal Rank (MRR).
- **C. Grounding & Trust:** Faithfulness, Hallucination Rate, Evidence Coverage, Citation Correctness.
- **D. Financial Reasoning:** Numerical Accuracy, Calculation Accuracy, Multi-step Reasoning Score, Explanation Completeness.
- **E. Efficiency:** Latency, Token Usage, Retrieval Time, Cost per Answer.

## Tech Stack / Requirements

- **Software:** Python 3.12, VS Code 1.99+, Jupyter Notebook 7.4+, LangChain 0.3+, ChromaDB 0.5+, Pandas 2.2+, NumPy 2.2+, Scikit-learn 1.6+, FAISS 1.10+.
- **Hardware:** i7/Ryzen 7, 16GB+ RAM, NVIDIA RTX 3060+ GPU, 500GB+ SSD, stable internet, Windows 11 or Ubuntu.

## Key Resource Files

- `Kasi_Research_Proposal.pdf` — the approved 34-page proposal. Contains the full architecture description, PDF parsing/chunking/prompting design rationale, and detailed metric definitions. Fully read; distilled into the `finagent-architecture` skill (see below) — load that skill rather than re-reading the PDF for implementation work.
- `Coding_Sheet.xlsx` — the authoritative experiment tracker, all 7 sheets fully read: "Common Metrics", "Overall Performance Results of", "Final Guidance Sheet", "Short summary", "Retrieval and Evidence Groundin", "Final Comparative Ranking and A", "Query Complexity-Wise Final Res". The four result sheets (all but "Common Metrics" and "Final Guidance Sheet") are empty templates to be filled in as experiments run. Distilled into the `finagent-experiments` skill (see below).
- `Coding_Sheet_RESULTS.xlsx` — working copy of `Coding_Sheet.xlsx` for recording actual experiment results as EXP-01–14 are run. `Coding_Sheet.xlsx` itself must never be edited; it stays the untouched reference/original.

Read PDFs via `pypdf` (`PdfReader(...).pages[i].extract_text()`) — `pdftoppm`/poppler is not installed on this machine. Read the xlsx via `openpyxl.load_workbook(path, data_only=True)`. Note: printing cell values containing non-ASCII characters (e.g. "→", "Naïve") directly to this machine's git-bash terminal can crash on cp1252 encoding — write to a file instead of printing, or strip non-ASCII first.

## Skills

Two workspace skills are built and available (`.claude/skills/`):

1. **`finagent-architecture`** — implementation/architecture reference: document processing pipeline, PDF/table parsing, chunking, the 7-agent adaptive RAG architecture, query complexity routing, per-agent prompting design, and the full 5-family evaluation metric formulas. Load before writing or reviewing any pipeline, agent, chunker, prompt, or metric code.
2. **`finagent-experiments`** — experiment-execution reference: what each of the 14 experiments (EXP-01–14) implements, its exact pipeline flow, purpose, metrics, and how/where to record results into `Coding_Sheet_RESULTS.xlsx`. Load before running, scripting, or recording results for any experiment.

## Provenance

This workspace was seeded on 2026-07-19 from a separate documentation workspace (`C:/Experiments/documentation`) where the thesis's Chapter 2 Literature Review and thesis-document skeleton were built. That workspace remains documentation-only going forward; this one is implementation-only. If anything here seems to disagree with the documentation workspace, treat `Kasi_Research_Proposal.pdf` and `Coding_Sheet.xlsx` (both copied fresh into this workspace) as the ground truth, not memory of past conversations.
