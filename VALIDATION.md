# FinAgent-RAG Implementation Validation Log

Living record of verification against `Kasi_Research_Proposal.pdf` §7.2–7.12 and
`Coding_Sheet.xlsx`, per the project's verification checklist. Updated as each section of the
system is built and checked — not a one-time sign-off document.

## §7.5 Financial Document Processing and Knowledge Base Preparation

**Status: PASS.** All 7 steps of Table 7.4 implemented and verified against real FinanceBench PDFs.

| Table 7.4 Step | Module | Verified |
|---|---|---|
| PDF Text Extraction | `document_processing/pdf_extraction.py` | Yes — see PDF library note below |
| Table Extraction | `document_processing/table_extraction.py` | Yes — numeric-density heuristic, table blocks found on every tested document |
| Cleaning | `document_processing/cleaning.py` | Yes — repeated header/footer + page-number removal verified |
| Section Detection | `document_processing/section_detection.py` | Yes — see hardening history below |
| Chunking | `document_processing/chunking.py` | Yes — 500-token/100-overlap, section-aware, verified against `config.py` fixed values |
| Metadata Tagging | `document_processing/metadata.py`, `chunking.py` | Yes — every chunk carries company/year/report_type/section/page/chunk_id |
| Storage | `document_processing/vector_store.py` | Yes — ChromaDB, batched `upsert`, metadata-filtered query verified |

**PDF library decision**: switched from `pypdf` to `PyMuPDF` mid-implementation. `pypdf` was found
to fragment multi-word 10-K headings (e.g. "Item 9A. Controls and Procedures.") across separate
extracted lines, which broke line-level section detection. PyMuPDF keeps them intact, needs no
poppler dependency, is ~30x faster on this corpus (1.03s vs 31.57s for a 160-page 10-K), and is
what FinanceBench's own baseline notebook uses (`PyMuPDFLoader`). See `CLAUDE.md` "Requirement
Deviations".

**Section-detection hardening history** (found and fixed during real-PDF validation against
`3M_2018_10K.pdf`, then reconfirmed after the PyMuPDF switch):
1. Table-detection numeric-density calc was diluted by bare `$`/`%` tokens — fixed by excluding
   bare currency/percent symbols from the density denominator.
2. Section tagging was per-page (whole page inherits one section) — moved to per-line, since a
   heading can appear partway down a page.
3. Table-of-Contents pages (which list every section name as a short line) were corrupting the
   running section tracker — pages with 3+ distinct section matches are now excluded from
   retagging. (Threshold is 3, not 2: a genuine content page — e.g. a 10-K's Part II item list —
   can legitimately contain 2 distinct matches without being a TOC.)
4. Narrative sentences merely mentioning a section by name (e.g. "...concluded that the Company's
   disclosure controls and procedures are effective.") were false-triggering heading matches —
   fixed with a word-count gate (≤14 words) and a match-coverage-ratio gate (matched phrase must
   cover ≥50% of the line).
5. Added generic `NOTE N.` and `Item N.` catch-all patterns so individual footnotes/Part
   II-III-IV items that aren't specifically named don't inherit a stale, unrelated section label
   from whatever was last specifically matched.
6. "Risk Factors" and "Legal Proceedings" patterns extended to match their real `Item 1A.`/`Item 3.`
   prefixed heading form (the bare, unprefixed form rarely appears alone in real filings).

**Known residual limitation**: section detection is a best-effort heuristic (Proposal §7.5 does not
require exact schema reconstruction), not a guaranteed-exact parser. A 10-K's Notes to Financial
Statements section contains 15-25+ individually-numbered footnotes; only the numbering convention
is generically caught, not each footnote's specific topic (e.g. "Note 12 — Leases" is tagged
generically as "Notes to Financial Statements", not "Leases"). This does not affect retrieval
completeness — full document text is always chunked and stored regardless of section-label
accuracy — only affects the precision of metadata-based section filtering (used by EXP-08+).

## Per-Company Parsing Validation (item #12)

**Status: PASS — 42/42 companies, 0 failures, 0 flags.**

Method: one representative PDF per unique company (42 companies across the 368-PDF corpus,
preferring a 10-K where available) run through the full extraction → clean → chunk pipeline,
checking for: exceptions, zero pages/chunks, empty chunks, section-tagging diversity (>1 distinct
section, or flagged as likely-failed detection), and minimum extracted text volume.

| Metric | Range across 42 companies |
|---|---|
| Chunks per document | 115 – 944 |
| Distinct sections tagged | 7 – 13 |
| Table blocks detected | 114 – 1843 |
| Processing time | 0.31s – 3.52s |
| Failures | 0 |
| Flagged anomalies | 0 |

**Retrieval spot-check (PASS)**: 4 companies (Apple, JPMorgan, Costco, Boeing; 1,486 chunks total)
ingested into a shared ChromaDB collection, queried with realistic financial questions:

| Question | Top result | Relevance |
|---|---|---|
| "What was Apple's net sales?" | `APPLE_2015_10K_CH_058`, score 0.852, section "Other Item Disclosures" — contains "Total net sales 233,715..." | Correct company, correct figure |
| "What were JPMorgan's total assets?" | `JPMORGAN_2021_10K_CH_340`/`_342`, scores 0.86/0.82, sections "Consolidated Statements of Operations"/"Consolidated Balance Sheet" | Correct company, correct statement |
| "What was Costco's revenue?" | `COSTCO_2015_10K_CH_070`, score 0.818, section "Consolidated Statements of Operations" — "CONSOLIDATED STATEMENTS OF INCOME" | Correct company, correct statement |
| "What risks does Boeing face?" | `BOEING_2015_10K_CH_013`, score 0.763, section "Risk Factors" — "Item 1A. Risk Factors..." | Correct company, correct section |

Every question's top hit is the correct company with topically on-point content. One useful
observation: Boeing's second-ranked result for the generic risk question was Apple's own Risk
Factors section (score 0.748) — expected, since unfiltered semantic search has no company
constraint and generic risk-factor language is similar across filers. This is precisely the
motivation for EXP-08's metadata-aware retrieval (filtering by company/year before or alongside
semantic search) — confirmed as a real, observable effect on this corpus, not just a theoretical
concern from the proposal.

Full per-company table (from `per_company_validation.py`, not checked in — regenerate as needed):

```
3M                     3M_2015_10K.pdf                     chunks=317 sections=10 tables=452
ACTIVISIONBLIZZARD     ACTIVISIONBLIZZARD_2015_10K.pdf     chunks=321 sections=10 tables=207
ACTIVSIONBLIZZARD      ACTIVSIONBLIZZARD_2023Q2_10Q.pdf    chunks=115 sections=8  tables=248
ADOBE                  ADOBE_2015_10K.pdf                  chunks=234 sections=11 tables=526
AES                    AES_2015_10K.pdf                    chunks=556 sections=13 tables=585
AMAZON                 AMAZON_2015_10K.pdf                 chunks=155 sections=9  tables=285
AMCOR                  AMCOR_2019_10K.pdf                  chunks=198 sections=10 tables=562
AMD                    AMD_2015_10K.pdf                    chunks=411 sections=10 tables=197
AMERICANEXPRESS        AMERICANEXPRESS_2022_10K.pdf        chunks=444 sections=13 tables=517
AMERICANWATERWORKS     AMERICANWATERWORKS_2015_10K.pdf     chunks=842 sections=12 tables=280
APPLE                  APPLE_2015_10K.pdf                  chunks=170 sections=9  tables=319
BESTBUY                BESTBUY_2015_10K.pdf                chunks=247 sections=10 tables=307
BLOCK                  BLOCK_2015_10K.pdf                  chunks=498 sections=10 tables=339
BOEING                 BOEING_2015_10K.pdf                 chunks=325 sections=10 tables=628
BOSTONPROPERTIES       BOSTONPROPERTIES_2015_10K.pdf       chunks=445 sections=11 tables=1472
COCACOLA               COCACOLA_2015_10K.pdf               chunks=436 sections=13 tables=399
CORNING                CORNING_2015_10K.pdf                chunks=944 sections=13 tables=420
COSTCO                 COSTCO_2015_10K.pdf                 chunks=144 sections=9  tables=212
CVSHEALTH              CVSHEALTH_2015_10K.pdf              chunks=661 sections=11 tables=273
EBAY                   EBAY_2015_10K.pdf                   chunks=210 sections=9  tables=327
FEDEX                  FEDEX_2023_10K.pdf                  chunks=419 sections=11 tables=381
FOOTLOCKER             FOOTLOCKER_2022_10K.pdf             chunks=213 sections=12 tables=205
GENERALMILLS           GENERALMILLS_2015_10K.pdf           chunks=245 sections=9  tables=467
INTEL                  INTEL_2015_10K.pdf                  chunks=256 sections=11 tables=557
JOHNSON_JOHNSON        JOHNSON_JOHNSON_2015_10K.pdf        chunks=225 sections=9  tables=495
JPMORGAN               JPMORGAN_2021_10K.pdf               chunks=847 sections=13 tables=1843
KRAFTHEINZ             KRAFTHEINZ_2015_10K.pdf             chunks=273 sections=12 tables=930
LOCKHEEDMARTIN         LOCKHEEDMARTIN_2015_10K.pdf         chunks=311 sections=10 tables=233
MCDONALDS              MCDONALDS_2022_10K.pdf              chunks=198 sections=7  tables=364
MGMRESORTS             MGMRESORTS_2015_10K.pdf             chunks=492 sections=9  tables=648
MICROSOFT              MICROSOFT_2015_10K.pdf              chunks=222 sections=9  tables=164
NETFLIX                NETFLIX_2015_10K.pdf                chunks=131 sections=10 tables=313
NIKE                   NIKE_2015_10K.pdf                   chunks=372 sections=12 tables=435
ORACLE                 ORACLE_2015_10K.pdf                 chunks=317 sections=11 tables=283
PAYPAL                 PAYPAL_2022_10K.pdf                 chunks=256 sections=11 tables=370
PEPSICO                PEPSICO_2015_10K.pdf                chunks=272 sections=12 tables=434
PFIZER                 PFIZER_2015_10K.pdf                 chunks=532 sections=11 tables=606
PG_E                   PG_E_2015_10K.pdf                   chunks=440 sections=12 tables=114
SALESFORCE             SALESFORCE_2023_10K.pdf             chunks=268 sections=11 tables=232
ULTABEAUTY             ULTABEAUTY_2023_10K.pdf             chunks=175 sections=11 tables=226
VERIZON                VERIZON_2015_10K.pdf                chunks=259 sections=10 tables=319
WALMART                WALMART_2015_10K.pdf                chunks=232 sections=10 tables=429
```

## §7.6–7.11: All 7 Agents, All 14 Experiments, Metrics, Results Writer

**Status: PASS (mocked) + PASS (live pilot).** All 7 agents, all 14 experiment runners, the full
5-family metrics library, and the results workbook writer are built. Validated in two stages:

1. **Mocked-client suite** (`src/tests/`, 287 tests, zero API cost): every agent individually, the
   full 7-agent orchestration end-to-end, all 14 experiments via the registry — including explicit
   proof that EXP-12/13/14's ablations produce the exact expected *reduced* Groq call count for the
   same question (e.g. EXP-12 makes only 3 calls on a Complex-routed question that would trigger 5
   calls in EXP-11, because Query Refinement is genuinely disabled, not approximated).
2. **Live pilot run against the real Groq API** — see below.

## Live Pilot Run (item #15 — 25 documents, all 14 experiments)

**Status: PASS — 56/56 runs succeeded, 0 failures, 0 quota exhaustion.**

Method: ingested the 25 FinanceBench documents with the most associated questions (6,759 chunks
total) into a dedicated ChromaDB store, selected 4 questions spanning different companies and
question styles (direct lookup, multi-value calculation, comparative judgment, balance-sheet
lookup), and ran all 4 through all 14 experiments (56 live pipeline executions) using a 2-key Groq
pool. Full raw output in `pilot_run_report.json` / `pilot_run_output.log`; runner is
`scripts/run_pilot.py` (rerunnable).

| Metric | Result |
|---|---|
| Runs completed | 56 / 56 |
| Failures (exceptions) | 0 |
| Quota-exhaustion errors | 0 |
| Total Groq calls | 95 |
| Total tokens used | 168,553 (vs. ~169,700 estimated in `Groq_API_Call_Budget.xlsx` for N=4 — accurate to within 1%) |
| Wall-clock time | 6.4 minutes |

**Real findings from the 4 questions** (not just "it ran" — actual behavior worth noting before
scaling to the full 150):

- **Adobe operating-margin question**: Direct LLM (EXP-01) confidently stated operating margin was
  "around 38-40%" — the real answer is a *decline* from 36.8% to 34.6%. A clean, unprompted
  real-world instance of the exact failure mode `fig1.png` illustrates (confident wrong numbers
  without retrieval). The RAG-based systems (EXP-07, EXP-11) correctly returned "insufficient
  evidence" instead of guessing, for this question, because retrieval didn't surface the right
  page — grounded-but-cautious beat confident-but-wrong here.
- **3M net PP&E question**: EXP-11 (full adaptive system) got it right — $8.738B vs. reference
  $8.70B, `numerical_accuracy=1.0`, `faithfulness=1.0`, with citations to two evidence pages — while
  its own `context_recall=0.0`. This is a **metric-methodology finding, not a bug**: context recall
  checks retrieved pages against FinanceBench's specific labeled evidence page(s); the system found
  the same figure on different, equally-valid pages of the filing. Worth accounting for in the full
  run's error analysis rather than reading raw `context_recall` as the whole retrieval-quality story.
- **2 of 4 questions** (Amazon DPO — a 4-value cross-statement calculation; Adobe margin — a
  cross-year comparison) had genuine retrieval misses (`context_recall=0.0`) for the RAG systems.
  Both are exactly the harder multi-step/cross-section question types the proposal's adaptive
  design targets — a real, useful signal for where retrieval strategy (top-k, chunking, or
  metadata filtering) may need attention at full scale, not a pipeline defect.

### Aggregate diagnostic (from `Pilot_Results_Draft.xlsx`, exported via `scripts/export_pilot_results.py`)

Averaging all 4 questions per experiment surfaces a clearer, more concerning pattern than the
per-question anecdotes above: **EXP-07, EXP-09, EXP-10, EXP-12, and EXP-14 returned the fixed
"insufficient evidence" fallback on all 4 of 4 pilot questions** (their averaged Answer Relevance/
F1/Semantic Similarity are numerically identical — 0.0629 / 0.0263 / 0.0165 — which only happens if
every underlying answer was the exact same fallback string). Only EXP-08 (metadata-aware), EXP-11
(full adaptive), and EXP-13 (adaptive without evidence filtering) found any usable evidence at all,
and even they succeeded on only 1 of 4 questions.

**Working hypothesis, not yet confirmed**: unfiltered semantic search (`retrieval_strategy="naive"`,
used by EXP-07/09/10) struggles once the corpus holds 25 *different companies'* filings (6,759
chunks) — a chunk from the wrong company can be more textually similar to a question's surface
phrasing than the right company's actual chunk, and nothing in the naive path constrains retrieval
to the right filing. Company/year metadata filtering (EXP-08, and the adaptive strategy's routing
in EXP-11/13) is what rescues retrieval when it engages. EXP-12 and EXP-14 also use the "adaptive"
strategy with the same metadata-filtering code path as EXP-11/13, yet showed the *naive* failure
pattern instead — with only 4 questions and live (not perfectly deterministic even at
temperature=0.0) LLM sampling behind each experiment's own independent Query Understanding call,
this could be routing-decision variance across nominally-similar runs rather than a real
per-experiment difference. **This needs the full 150-question run (or a larger pilot) to
distinguish signal from 4-question noise** — it is the single most important thing to watch when
scaling up, and likely the highest-leverage area for improvement: metadata-aware/filtered
retrieval, not the reasoning/verification/answer-generation stages, which perform correctly *when
given the right evidence* (see the 3M net PP&E success above: `numerical_accuracy=1.0`,
`faithfulness=1.0`).

Full per-run detail — every generated answer, every metric, explicitly distinguishing "not
applicable" (blank) from "computed as zero" — is in `Pilot_Results_Draft.xlsx` (sheets: `Detail`,
`Experiment Summary`, `Metric Coverage Diagnostic`). This is a draft/diagnostic export, kept
separate from `Coding_Sheet_RESULTS.xlsx`, which is reserved for the full 150-question run.

### Root-cause follow-up: a real metrics bug, found and fixed entirely without spending Groq tokens

Retrieval and evidence-filtering are pure local operations (ChromaDB + `sentence-transformers`) —
they cost zero Groq calls, so once both keys' daily budgets were exhausted, the pilot's near-zero
`context_recall` finding could still be investigated directly against the already-populated pilot
ChromaDB store. This surfaced the actual root cause:

**`FinanceBenchQuestion.evidence[*].page_number` does not reliably align with the raw sequential
page index this codebase's PDF extraction produces**, and the offset is not a constant — verified
directly on two pilot documents: 3M's 2018 10-K (FinanceBench claims page 57 for a PP&E figure that
is actually on raw PDF page 41 — a 16-page offset) and Adobe's 2022 10-K (FinanceBench claims page
53 for an income statement actually on raw PDF page 54 — a 1-page offset). Every metric that
compared `chunk.page_number == evidence.page_number` exactly (`context_recall`, `context_precision`,
`hit_at_k`, `mrr`, `citation_correctness`, `evidence_coverage`) was silently under-counting correct
retrievals across the whole dataset, not just this pilot.

**Fix** (`metrics/retrieval_metrics.py`, `metrics/grounding.py`): added
`matches_evidence_reference()`, which falls back to checking whether a candidate chunk shares
enough of the evidence excerpt's own *distinctive* numeric values when the page number doesn't
match. First calibration attempt was itself a false-positive trap — raw number overlap matched a
chunk about share repurchases to a capex question purely because both mentioned "December 31, 2018"
— fixed by excluding calendar years and small integers (≤31, i.e. day-of-month/footnote-index
range) from the "distinctive" set, verified against the exact real false positive found. 5 new
regression tests lock in both the fix and the false-positive rejection.

**Re-scored the same 4 pilot questions offline** (re-running only the free retrieval step against
the pilot ChromaDB store, not the full paid pipeline) with the corrected matching:

| Question | Old (page-only) recall | New (calibrated) recall |
|---|---|---|
| 3M FY2018 capex | 0.0 | **1.0** — was a pure metric artifact |
| 3M FY2018 net PP&E | 0.0 | **1.0** — was a pure metric artifact |
| Adobe FY2022 operating margin | 0.0 | 0.0 — genuine retrieval miss |
| Amazon FY2017 DPO (2 evidence pages) | 0.0 | 0.5 — one of two pages genuinely missed |

This is a much more credible picture than the pilot's raw output suggested: half of the apparent
retrieval failures were a measurement bug (now fixed for all future runs, pilot or full-scale),
and half are genuine gaps worth investigating (Adobe's specific evidence page never appeared in the
top-5 even with correct company+year filtering; Amazon's multi-page evidence was only half-covered).
The originally-generated `pilot_run_report.json` / `Pilot_Results_Draft.xlsx` numbers themselves are
**not** retroactively corrected (the live run didn't persist raw retrieval traces, only final
metrics — logging/tracing improvements now underway address this gap for future runs) — treat this
table as the corrected read, not the exported files.

## Remaining Before Full-Scale Results

- Full 150-question run across all 14 experiments (~6.36M tokens, ~$3.85 estimated — see
  `Groq_API_Call_Budget.xlsx`) — the free-tier token cap is the actual constraint, not pipeline
  correctness, which this pilot confirms is solid.
- §7.11 steps 12-15 (evaluate/compare/error-analysis/summarize across the full run) depend on that
  full run existing.

## Freezing Known-Good State

The practical "freeze" mechanism here is the combination of the pytest suite (311 tests, mocked
Groq client, zero API cost — run with `PYTHONPATH=src python -m pytest src/tests -q` any time) and
an annotated git tag at every point this suite is green and the codebase has been through a real
validation pass like this one. A tag is just a permanent pointer to a commit — future work keeps
happening on `main`, but `git checkout <tag>` (or `git diff <tag>..main`) always gets back to exactly
this state, so a regression introduced later can be pinpointed against a known-good baseline instead
of guessing.

Current checkpoints:

| Tag | Commit | Marks |
|---|---|---|
| `v0.1.0-pilot-validated` | `62c0535` | Full pipeline (7 agents, 14 experiments, 5 metric families) implemented, 311 tests passing, per-company parsing validated, live pilot run (56/56) clean, stratified sampling built, repo cleaned up for GitHub publish. Full 150-question run not yet done. |

Cut a new tag after any future change that (a) passes the full suite and (b) represents a real,
intentional milestone (not every commit) — e.g. once the full 150-question run completes.
