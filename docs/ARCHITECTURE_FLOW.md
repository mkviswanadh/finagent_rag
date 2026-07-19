# FinAgent-RAG: End-to-End Technical Flow

This document shows every stage the system goes through, from a raw PDF on disk to a final
verified answer. It reflects what is actually implemented in `src/finagent/`, not just the
proposal's description — see `.claude/skills/finagent-architecture/SKILL.md` for the design
rationale behind each stage, and `docs/FUNCTION_GUIDE.md` for a plain-language walkthrough of
every function shown here.

## 1. Document Processing Pipeline (offline, one-time per PDF)

Runs once per PDF to populate the shared ChromaDB knowledge base. Every RAG experiment (EXP-07
through EXP-14) reads from this same store; it is never rebuilt per-experiment.

```mermaid
flowchart TD
    A["PDF file\n(data/pdfs/*.pdf)"] --> B["extract_pdf_pages()\nPyMuPDF text extraction,\none PageContent per page"]
    B --> C["clean_text()\nremoves repeated headers/footers\nand standalone page-number lines"]
    C --> D["chunk_document()\nsection-aware chunking:\n500 tokens / 100 overlap,\nforces a new chunk at every\ndetected section change"]
    D -->|"internally calls"| D1["match_section_heading() / detect_section()\nregex heading patterns +\nTOC-page exclusion +\nnarrative false-positive filtering"]
    D -->|"internally calls"| D2["extract_tables_from_page()\nnumeric-density heuristic keeps\ntable rows contiguous"]
    D --> E["list[Chunk]\neach with company/year/report_type/\nsection/page_number/chunk_id"]
    E --> F["ChromaVectorStore.add_chunks()\nembeds via all-MiniLM-L6-v2,\nbatched upsert"]
    F --> G[("ChromaDB collection\nfinagent_chunks")]
```

## 2. Query-Time Pipeline: the Adaptive Multi-Agent System (EXP-11..14)

This is the proposed system (EXP-11) and its three ablations (EXP-12/13/14, each disabling exactly
one stage). Every box after "Query Understanding" is conditional on the routing decision — a
Simple question skips straight to Reasoning.

```mermaid
flowchart TD
    Q["User question"] --> QU["QueryUnderstandingAgent.analyze()\n1 Groq call: extracts company/year/metric,\nclassifies Simple / Moderate / Complex\n(LLM judgment + deterministic keyword rules\nthat can only escalate, never downgrade)"]

    QU --> ROUTE{"Complexity tier?"}

    ROUTE -->|Simple| RET1["RetrievalAgent.retrieve()\noriginal question,\nmetadata filter from extracted entities"]

    ROUTE -->|"Moderate\n(needs_refinement?)"| MODCHECK{needs_refinement}
    MODCHECK -->|No| RET1
    MODCHECK -->|Yes| REFINE["QueryRefinementAgent.refine()\n1 Groq call: rewrites into a\nretrieval-friendly single query"]
    REFINE --> RET1

    ROUTE -->|Complex| COMPCHECK{needs_refinement}
    COMPCHECK -->|Yes| REFINE2["QueryRefinementAgent.refine()\n1 Groq call"]
    COMPCHECK -->|No| MULTI
    REFINE2 --> MULTI["QueryRefinementAgent.expand_multi_query()\n1 Groq call: 3 alternative\nphrasings of the question"]
    MULTI --> RETMULTI["RetrievalAgent.retrieve_multi()\nqueries each variant,\nmerges + dedups by chunk_id"]

    RET1 --> FILT["EvidenceFilteringAgent.filter()\n0 Groq calls: relevance threshold +\nnear-duplicate dedup (SequenceMatcher) +\ncap at top_k=5\n[SKIPPED in EXP-13]"]
    RETMULTI --> FILT

    FILT --> REASON["ReasoningAgent.reason()\n1 Groq call: numerical interpretation +\ncross-section reasoning + drafts a\ngrounded answer, all in one structured call"]

    REASON --> INSUFF{insufficient_evidence?}
    INSUFF -->|Yes| FALLBACK["Fixed 'insufficient evidence' message\n(no fabricated answer)"]
    INSUFF -->|No| ANSGEN["AnswerGenerationAgent.generate()\n0 Groq calls: finalizes the\nReasoning Agent's draft answer,\nappends evidence citations"]

    ANSGEN --> VERCHECK{"Verification enabled?\n[SKIPPED in EXP-14]"}
    VERCHECK -->|Yes| VERIFY["VerificationAgent.verify()\n1 Groq call: checks every claim\nin the answer against the evidence,\nfails closed on a parse error"]
    VERCHECK -->|No| FINAL
    VERIFY --> FINAL["Final answer returned\n+ full PipelineTrace recorded\n(prompts, responses, tokens, latency)"]
    FALLBACK --> FINAL

    style FILT fill:#fff3cd
    style VERCHECK fill:#fff3cd
```

**Groq call count by route** (validated by the mocked-client test suite, `src/tests/experiments/test_registry.py`):

| Route | Calls | Which agents |
|---|---|---|
| Simple | 3 | Query Understanding, Reasoning, Verification |
| Moderate, no refinement needed | 3 | same as Simple |
| Moderate, refinement needed | 4 | + Query Refinement (refine) |
| Complex, no refinement needed | 4 | Query Understanding, Multi-Query, Reasoning, Verification |
| Complex, refinement needed | 5 | + Query Refinement (refine) before Multi-Query |

EXP-12 (no Query Refinement) forces every route to behave like the Simple row (3 calls, always).
EXP-13 (no Evidence Filtering) has identical call counts to EXP-11 — the ablation shows up in
*which* chunks reach Reasoning, not in call count, since filtering is free. EXP-14 (no
Verification) is always exactly 1 call fewer than the equivalent EXP-11 row.

## 3. The Simpler RAG Pipelines (EXP-07..10)

Each is a fixed (non-adaptive) point on the same spectrum — no complexity routing at all.

```mermaid
flowchart LR
    subgraph EXP07["EXP-07: Naïve RAG"]
        direction TB
        A7["Question"] --> B7["Retrieve (unfiltered top-k)"] --> C7["Reasoning + Answer"]
    end
    subgraph EXP08["EXP-08: Metadata-Aware RAG"]
        direction TB
        A8["Question"] --> B8["Query Understanding\n(entity extraction only)"] --> C8["Retrieve (metadata-filtered)"] --> D8["Reasoning + Answer"]
    end
    subgraph EXP09["EXP-09: Query-Rewritten RAG"]
        direction TB
        A9["Question"] --> B9["Query Refinement\n(always rewrites)"] --> C9["Retrieve (unfiltered top-k)"] --> D9["Reasoning + Answer"]
    end
    subgraph EXP10["EXP-10: Multi-Query RAG"]
        direction TB
        A10["Question"] --> B10["Query Refinement\n(always expands to 3 variants)"] --> C10["Retrieve + Merge"] --> D10["Reasoning + Answer"]
    end
```

None of EXP-07..10 use Evidence Filtering or Verification — evidence goes straight from retrieval
to Reasoning, and the Reasoning Agent's draft answer is returned as-is.

## 4. The Direct LLM Baselines (EXP-01..06)

No retrieval, no ChromaDB, no agents at all — one Groq call per question, differing only in the
system prompt (`experiments/direct_llm_prompts.py`).

```mermaid
flowchart LR
    A["User question"] --> B["System prompt variant\n(zero-shot / role-based / few-shot /\nstepwise / self-verification / structured)"]
    B --> C["1 Groq call\n(llama-3.3-70b-versatile, temp=0.0)"]
    C --> D["Answer returned directly\n(no evidence, no citations)"]
```

## 5. Evaluation — What Happens After an Answer Is Produced

Every experiment, regardless of which pipeline above produced the answer, funnels into the same
metric computation and results-recording path:

```mermaid
flowchart TD
    A["QuestionResult\n(PipelineTrace + generated_answer)"] --> B["compute_all_metrics()\nzero additional Groq calls:\nembeddings + string comparison +\ncross-referencing FinanceBench's own\nground-truth evidence pages"]
    B --> C1["Answer Quality\n(relevance, EM, F1, semantic similarity)"]
    B --> C2["Evidence & Retrieval\n(context recall/precision, Hit@K, MRR)\n[RAG experiments only]"]
    B --> C3["Grounding & Trust\n(faithfulness, hallucination rate,\nevidence coverage, citation correctness)"]
    B --> C4["Financial Reasoning\n(numerical/calculation accuracy,\nmulti-step score, explanation completeness)"]
    B --> C5["Efficiency\n(latency, tokens, cost)"]
    C1 & C2 & C3 & C4 & C5 --> D["ResultsWorkbookWriter\nwrites averaged per-experiment metrics\ninto Coding_Sheet_RESULTS.xlsx\n(never touches the original Coding_Sheet.xlsx)"]
```
