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

## Not Yet Validated

- §7.6 Agentic RAG Architecture — 1 of 7 agents built (Query Understanding). Remaining: Query
  Refinement, Retrieval, Evidence Filtering, Reasoning, Answer Generation, Verification.
- §7.7 Query Complexity Detection and Adaptive Routing — routing logic implemented in Query
  Understanding Agent with a hybrid LLM + deterministic-rule design; not yet tested against live
  Groq calls (no `GROQ_API_KEY` configured yet) or a labeled question set.
- §7.8 Retrieval, Evidence Selection, and Reasoning Workflow — Retrieval Agent not yet built
  (though the underlying `ChromaVectorStore.query` is validated above); Evidence Filtering,
  Reasoning, Answer Generation, Verification agents not yet built.
- §7.9 Benchmarking Framework and Baseline Comparison — none of the 14 experiment runners built yet.
- §7.10 Evaluation Metrics — metrics library not yet built.
- §7.11 End-to-End Implementation Workflow — steps 1-6 (dataset, extraction, cleaning, chunking,
  metadata, embedding/storage) validated above; steps 7-15 (benchmark question classification,
  baseline systems, proposed system, running/recording/evaluating/comparing/error-analysis) pending.
