---
name: finagent-experiments
description: Experiment-execution reference for the FinAgent-RAG thesis's 14-experiment benchmarking ladder (EXP-01 to EXP-14) — what each experiment implements, its exact pipeline flow, purpose, and metrics, plus how and where to record results into the Coding_Sheet_RESULTS.xlsx workbook without ever touching the original Coding_Sheet.xlsx. Load this before running, scripting, or recording results for any of the 14 experiments.
---

# FinAgent-RAG: Experiment-Execution Skill

Grounded in a full read of every tab of `Coding_Sheet.xlsx` (7 sheets: Common Metrics, Overall
Performance Results of, Final Guidance Sheet, Short summary, Retrieval and Evidence Grounding,
Final Comparative Ranking and Analysis, Query Complexity-Wise Final Results). This is the
ground-truth experiment tracker — treat it as authoritative over any restated summary elsewhere.

Use this skill when: implementing/running any of EXP-01 through EXP-14, deciding what an experiment
should do, or writing results back into the results workbook. For the shared architecture underneath
these experiments (agents, pipeline, chunking, metric formulas), use the **finagent-architecture**
skill — this skill is "which experiment, what to build, where results go," not "how the system works
internally."

## 0a. Production-grade implementation standard (mandatory)

See `finagent-architecture` skill §0 for the full standard — it applies identically to every
experiment runner built here. In particular for experiments specifically:

- Each `EXP-XX` runner is a thin, fully-implemented composition of the shared agents/pipeline
  (see finagent-architecture §4), parameterized by which stages are enabled — **not** a
  copy-pasted variant. EXP-12/13/14 must literally reuse EXP-11's code path with one stage's
  `enabled=False`, proving the ablation is real rather than a rewritten approximation.
- Every experiment runner must record, per question: the exact prompts sent, the raw LLM response(s),
  parsed answer, retrieved chunk IDs (if any), per-agent latency, and token usage — this raw trace is
  what step 14 (error analysis, finagent-architecture §8) and the metric computation in §3 below both
  depend on. Don't discard intermediate agent outputs after producing the final answer.
- Minimize Groq calls per question according to each experiment's actual pipeline string in §1 below
  — e.g. EXP-01–06 make exactly one LLM call per question (the pipeline strings show no branching);
  EXP-10 makes 1 call to generate query variants + 1 final generation call (not 3 separate generation
  calls); EXP-11's call count varies by adaptive route (§5 of finagent-architecture) — a Simple-route
  question should cost meaningfully less than a Complex-route one, and the implementation must reflect
  that rather than always running the full agent chain.

## 0. Critical rule: results go in the COPY, never the original

- `Coding_Sheet.xlsx` = reference / original. **Never write to it.**
- `Coding_Sheet_RESULTS.xlsx` = the working copy created for recording actual experiment results. All
  results, scores, and rankings get written here.
- If `Coding_Sheet_RESULTS.xlsx` doesn't exist yet, create it as an exact copy of `Coding_Sheet.xlsx`
  before writing anything.
- Sheet names in this workbook are truncated to Excel's 31-character limit — use these **exact**
  strings (including the trailing space on one of them) when opening sheets with openpyxl, or you'll
  silently create a new blank sheet instead of writing to the right one:
  - `'Common Metrics'`
  - `'Overall Performance Results of '` ← note trailing space
  - `'Final Guidance Sheet'`
  - `'Short summary'`
  - `'Retrieval and Evidence Groundin'` ← truncated from "Grounding"
  - `'Final Comparative Ranking and A'` ← truncated from "Analysis"
  - `'Query Complexity-Wise Final Res'` ← truncated from "Results"
- Read with `openpyxl.load_workbook(path, data_only=True)`. When writing, load without `data_only`
  (or reload fresh) so you don't strip formulas/formatting unintentionally — use
  `openpyxl.load_workbook(path)` for the write pass, edit cell `.value`s, then `.save(path)`.
- Experiment names contain non-ASCII characters (e.g. "Naïve"). Copy them from the existing cell
  values rather than retyping, and don't print raw cell values to this machine's git-bash terminal
  (cp1252 encoding can crash on them) — write to a file and Read it instead.

## 1. The 14-experiment ladder (Final Guidance Sheet — full detail)

Fixed setup for every experiment (see finagent-architecture skill §1): FinanceBench dataset, Groq
Llama 3.3 70B Versatile, ChromaDB, same embedding model, 500-token chunks/100 overlap, top-5
retrieval, temperature 0.0. Only the pipeline design below varies.

### Direct LLM group (EXP-01–06) — no retrieval, no ChromaDB

**EXP-01 — Direct LLM with Zero-Shot Prompting**
- Implements: question passed directly to Groq Llama 3.3, no examples, no document context.
- Pipeline: `User Question → Groq Llama 3.3 → Answer`
- Purpose: simplest baseline — how does the LLM answer without retrieval?
- Metrics: answer relevance, exact match, F1-score, semantic similarity, hallucination rate, numerical accuracy, latency
- Expected finding: fluent but likely wrong on exact financial numbers (no FinanceBench evidence).

**EXP-02 — Direct LLM with Role-Based Financial Analyst Prompting**
- Implements: LLM instructed to behave as a financial analyst before answering.
- Pipeline: `User Question → Financial Analyst Role Prompt → Groq Llama 3.3 → Answer`
- Purpose: does role-based prompting improve financial tone/explanation/relevance?
- Metrics: answer relevance, semantic similarity, explanation completeness, hallucination rate, latency
- Expected finding: tests whether a domain role improves quality even without retrieval.

**EXP-03 — Direct LLM with Few-Shot Prompting**
- Implements: prompt includes 2–3 sample financial Q&A pairs before the real question.
- Pipeline: `Example Q&A Pairs → User Question → Groq Llama 3.3 → Answer`
- Purpose: do examples improve format/numerical explanation/consistency?
- Metrics: answer relevance, F1-score, semantic similarity, numerical accuracy, hallucination rate
- Expected finding: may improve format/style, not factual correctness.

**EXP-04 — Direct LLM with Stepwise Financial Reasoning Prompting**
- Implements: LLM identifies company, year, metric, and calculation requirement before final answer.
- Pipeline: `User Question → Stepwise Reasoning Prompt → Groq Llama 3.3 → Final Answer`
- Purpose: does reasoning-focused prompting improve comparison/calculation/explanation answers?
- Metrics: numerical accuracy, calculation accuracy, explanation completeness, semantic similarity, hallucination rate
- Expected finding: may still hallucinate — no retrieved document grounding it.

**EXP-05 — Direct LLM with Self-Verification Prompting**
- Implements: LLM answers, then internally checks completeness/relevance/numerical consistency.
- Pipeline: `User Question → Generate Answer → Self-Check → Final Answer`
- Purpose: does self-verification reduce overconfident/unsupported answers?
- Metrics: faithfulness proxy, hallucination rate, numerical accuracy, answer relevance, latency
- Expected finding: verification is only internal (no external evidence) — limited effect expected.

**EXP-06 — Direct LLM with Structured Output Prompting**
- Implements: fixed output format — answer, company, year, metric, reasoning, confidence, limitation.
- Pipeline: `User Question → Structured Output Prompt → Groq Llama 3.3 → Structured Answer`
- Purpose: does a fixed format improve consistency and ease of evaluation?
- Metrics: answer relevance, explanation completeness, semantic similarity, hallucination rate, latency
- Expected finding: improves presentation, doesn't solve evidence grounding.

### Basic RAG group (EXP-07–08) — retrieval + ChromaDB begin here

**EXP-07 — Naïve RAG using ChromaDB**
- Implements: FinanceBench PDFs chunked/embedded/stored in ChromaDB; question retrieves top-5 chunks, then generates.
- Pipeline: `Question → ChromaDB Top-k Retrieval → Retrieved Chunks → Groq Llama 3.3 → Answer`
- Purpose: basic benefit of retrieval grounding vs. Direct LLM group.
- Metrics: context precision, context recall, Hit@K, faithfulness, numerical accuracy, hallucination rate, latency
- Expected finding: improves factual grounding, but retrieval may miss evidence for complex questions.

**EXP-08 — Metadata-Aware Naïve RAG using ChromaDB**
- Implements: same as EXP-07, but retrieval/filtering uses metadata (company, year, document type, section, page).
- Pipeline: `Question → Metadata Extraction → ChromaDB Retrieval with Metadata Filter → Top-k Chunks → Groq Llama 3.3 → Answer`
- Purpose: does metadata improve retrieval precision and reduce wrong-company/wrong-year evidence?
- Metrics: context precision, citation correctness, numerical accuracy, faithfulness, hallucination rate
- Expected finding: fewer "right metric, wrong company/year" errors.

### Advanced RAG group (EXP-09–10)

**EXP-09 — Query-Rewritten RAG using ChromaDB**
- Implements: original question rewritten into a clearer retrieval-friendly query before search.
- Pipeline: `Original Question → Query Rewrite → ChromaDB Retrieval → Top-k Chunks → Groq Llama 3.3 → Answer`
- Purpose: does rewriting improve retrieval for unclear/complex questions?
- Metrics: context recall, context precision, Hit@K, answer relevance, faithfulness, latency
- Expected finding: helps when user wording doesn't match report wording.

**EXP-10 — Multi-Query RAG using ChromaDB**
- Implements: LLM generates 3 alternative query versions; each retrieves from ChromaDB; results merged and deduplicated before generation.
- Pipeline: `Original Question → Generate 3 Query Variants → ChromaDB Retrieval → Merge Chunks → Groq Llama 3.3 → Answer`
- Purpose: does multi-query improve evidence coverage for complex questions?
- Metrics: context recall, Hit@K, MRR, faithfulness, numerical accuracy, latency, token usage
- Expected finding: may improve recall but increases latency and can add noisy chunks.

### Proposed system + ablations (EXP-11–14)

**EXP-11 — Adaptive Multi-Agent RAG using ChromaDB (Proposed Full System)**
- Implements: full 7-agent system — query understanding, complexity detection, adaptive routing,
  query refinement, retrieval, evidence filtering, reasoning, answer generation, verification.
- Pipeline: `Question → Query Understanding Agent → Complexity Detection → Adaptive Routing → Query Refinement / Multi-Query if Needed → ChromaDB Retrieval → Evidence Filtering → Reasoning Agent → Answer Generation → Verification Agent → Final Answer`
- Purpose: evaluate the complete proposed framework against all baselines.
- Metrics: answer relevance, context recall, faithfulness, hallucination rate, numerical accuracy, citation correctness, reasoning consistency, latency
- Expected finding: this is the main result — should outperform baselines on moderate/complex questions specifically.

**EXP-12 — Adaptive Multi-Agent RAG without Query Refinement (Ablation)**
- Implements: proposed system minus the Query Refinement Agent — original question goes straight to retrieval after complexity detection.
- Pipeline: `Question → Query Understanding → Adaptive Routing → ChromaDB Retrieval → Evidence Filtering → Reasoning → Verification → Answer`
- Purpose: isolate the contribution of query refinement.
- Metrics: context recall, context precision, answer relevance, faithfulness, numerical accuracy
- Expected finding: if scores drop vs. EXP-11, refinement matters.

**EXP-13 — Adaptive Multi-Agent RAG without Evidence Filtering (Ablation)**
- Implements: proposed system minus the Evidence Filtering Agent — all retrieved top-k chunks go straight to reasoning.
- Pipeline: `Question → Query Understanding → Adaptive Routing → Query Refinement → ChromaDB Retrieval → Reasoning → Verification → Answer`
- Purpose: isolate the contribution of evidence filtering.
- Metrics: faithfulness, hallucination rate, citation correctness, context precision, numerical accuracy
- Expected finding: if hallucination rises vs. EXP-11, filtering matters.

**EXP-14 — Adaptive Multi-Agent RAG without Verification Agent (Ablation)**
- Implements: proposed system minus the Verification Agent — generated answer returned directly.
- Pipeline: `Question → Query Understanding → Adaptive Routing → Query Refinement → ChromaDB Retrieval → Evidence Filtering → Reasoning → Answer Generation → Final Answer`
- Purpose: isolate the contribution of the verification step.
- Metrics: hallucination rate, faithfulness, citation correctness, numerical accuracy, answer relevance
- Expected finding: if unsupported claims rise vs. EXP-11, verification matters.

Implementation implication: build EXP-11's agents as independently toggleable stages (flags, not
copy-pasted variants) so EXP-12/13/14 are the same codebase with one stage disabled — not three
separate forked implementations.

## 2. Experiment groups (Short summary sheet)

| Group | Experiments | Purpose |
|---|---|---|
| Direct LLM Prompting Experiments | EXP-01–06 | Can prompt engineering alone solve financial QA without retrieval? |
| Basic RAG Experiments | EXP-07–08 | Does ChromaDB retrieval improve grounding; does metadata improve precision? |
| Advanced RAG Experiments | EXP-09–10 | Do query rewriting and multi-query retrieval improve evidence retrieval? |
| Proposed and Ablation Experiments | EXP-11–14 | Does the full adaptive multi-agent system work, and which components actually drive the improvement? |

Run roughly in this group order — later groups' baselines (e.g. EXP-11's comparison claims) depend
on earlier groups' results already being recorded.

## 3. Where results get recorded — 4 result sheets, 4 different purposes

All writes target `Coding_Sheet_RESULTS.xlsx` (never the original). Each experiment run should feed
all four of these, not just one:

### 3a. `'Overall Performance Results of '` — the main per-experiment scorecard

One row per experiment (EXP-01…EXP-14), columns:
`Exp. No. | Experiment Name | Category | Answer Relevance | Exact Match | F1-Score | Semantic Similarity | Numerical Accuracy | Faithfulness | Hallucination Rate | Avg. Latency | Avg. Token Usage | Overall Rank`

Fill this after running all questions for an experiment and averaging metrics across them.
`Overall Rank` is filled last, once all 14 experiments are done, by ranking on whatever composite the
thesis uses (see §3c).

### 3b. `'Retrieval and Evidence Groundin'` — retrieval-specific detail, RAG experiments only

Only applies to EXP-07 through EXP-14 (the 8 experiments that actually do retrieval — EXP-01–06 have
no retrieval stage, leave them out of this sheet). Columns:
`Exp. No. | Experiment Name | Retrieval Type | Top-k Used | Context Precision | Context Recall | Hit@K | MRR | Evidence Coverage | Citation Correctness | Wrong Company/Year Retrieval Cases | Missing Evidence Cases | Retrieval Quality Rank`

`Retrieval Type` is pre-filled per experiment (e.g. "Direct top-k retrieval" for EXP-07,
"Metadata-filtered retrieval" for EXP-08, "Adaptive routed retrieval" for EXP-11) — don't overwrite
that column, just fill the metric columns. `Wrong Company/Year Retrieval Cases` and `Missing Evidence
Cases` are counts you tally during error analysis (proposal step 14), not computed metrics.

### 3c. `'Final Comparative Ranking and A'` — cross-experiment synthesis, fill LAST

One row per experiment, all 14. Columns:
`Exp. No. | Experiment Name | Group | Answer Quality Score | Retrieval Quality Score | Grounding Score | Financial Reasoning Score | Efficiency Score | Overall Weighted Score | Improvement Over Naïve RAG | Improvement Over Direct LLM | Final Rank | Main Finding`

This sheet aggregates the four metric families (Answer Quality / Retrieval / Grounding / Reasoning /
Efficiency — see finagent-architecture skill §7) into composite scores per experiment, then computes
`Improvement Over Naïve RAG` and `Improvement Over Direct LLM` as relative deltas against EXP-07 and
EXP-01 respectively. Only fill this after all 14 experiments' raw results (§3a, §3b, §3d) are
recorded — it's a derived/summary sheet, not a raw-results sheet. `Main Finding` is a short free-text
note per experiment (e.g. "Metadata filtering cut wrong-company retrieval by X%").

### 3d. `'Query Complexity-Wise Final Res'` — per-experiment breakdown by Simple/Moderate/Complex

42 rows: each of the 14 experiments × 3 complexity levels (Simple, Moderate, Complex), pre-populated
in that order. Columns:
`Exp. No. | Experiment Name | Query Complexity | No. of Questions | Answer Relevance | Context Recall | Numerical Accuracy | Faithfulness | Hallucination Rate | Avg. Latency | Key Result Observation`

This directly answers Research Question 3 (how does query complexity affect performance) — when
running an experiment, tag each FinanceBench question with its complexity level (per
finagent-architecture skill §5's routing rules) *before* running, so results can be split three ways
afterward instead of re-classified retroactively. `Key Result Observation` is a short free-text note
(e.g. "Faithfulness drops sharply on Complex queries without evidence filtering").

## 4. Recording workflow per experiment run

1. Confirm `Coding_Sheet_RESULTS.xlsx` exists (create from `Coding_Sheet.xlsx` if not — see §0).
2. Run the experiment's pipeline (per §1's exact pipeline string) across all FinanceBench benchmark
   questions, with each question pre-tagged by complexity level.
3. Compute per-question metrics, then aggregate: overall averages (→ 3a), retrieval-specific
   averages if applicable (→ 3b), and per-complexity-level averages (→ 3d).
4. Write the three sheets' rows for this experiment via openpyxl (load without `data_only`, set cell
   `.value`, save).
5. Only after EXP-01 through EXP-14 are all recorded, compute and fill §3c (comparative
   ranking/synthesis) in one final pass.
6. Never edit `Coding_Sheet.xlsx` (the original) at any point in this workflow.
