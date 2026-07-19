# FinAgent-RAG: Function-by-Function Guide (Plain Language)

Every public function in `src/finagent/`, explained simply: what goes in, what happens, what comes
out. Organized in the same order the system actually runs — start at Section 1 and read straight
through to follow one question's entire journey from a PDF on disk to a verified answer.

Pair this with `docs/ARCHITECTURE_FLOW.md` for the visual diagrams of the same flow.

---

## 1. Turning a PDF into searchable pieces (`document_processing/`)

This whole section runs once per document, ahead of time, to build the knowledge base every RAG
experiment searches later.

### `extract_pdf_pages(pdf_path)` — *pdf_extraction.py*
- **Input:** the file path to one PDF (a 10-K, 10-Q, 8-K, etc.).
- **What happens:** opens the PDF with PyMuPDF and pulls the text out of every page, one page at a time.
- **Output:** a list of `PageContent` objects — one per page, each holding the page number and its raw text.

### `clean_text(pages)` — *cleaning.py*
- **Input:** the list of pages from `extract_pdf_pages`.
- **What happens:** looks for short lines that repeat on many pages (like a running header
  "Apple Inc. | 2022 Form 10-K" or a page number by itself) and strips those out, since they add
  noise without adding information. Only checked in documents with 4+ pages.
- **Output:** the same list of pages, cleaned up.

### `match_section_heading(line)` — *section_detection.py*
- **Input:** one line of text.
- **What happens:** checks whether that single line looks like a real section title ("Risk
  Factors", "Item 9A. Controls and Procedures.", "NOTE 6. ...") rather than a sentence that just
  happens to mention one of those phrases in passing. Uses both a list of known heading patterns
  and two sanity checks (the line can't be too long, and the matched phrase has to make up most of
  the line, not just a small piece of it).
- **Output:** the section name it matched, or nothing if the line isn't a heading.

### `detect_section(page_text, previous_section)` — *section_detection.py*
- **Input:** the text of one page, plus whatever section the previous page ended in.
- **What happens:** scans the page for heading lines and remembers the last one found. If none are
  found, it assumes the page is still part of whatever section came before.
- **Output:** the section name this page belongs to.

### `extract_tables_from_page(page_text)` — *table_extraction.py*
- **Input:** the text of one page.
- **What happens:** looks for runs of consecutive lines where most of the words are actually
  numbers (a tell-tale sign of a financial table, like a row of a balance sheet). Bare currency
  symbols ($ signs sitting alone) don't count against a line's "number density."
- **Output:** a list of the table-like blocks found on that page, so the chunker can try to keep
  them together instead of splitting a table row in half.

### `parse_filing_metadata(filename)` — *metadata.py*
- **Input:** a PDF's filename, e.g. `"3M_2018_10K.pdf"`.
- **What happens:** reads the company name, year, and filing type straight out of the filename
  pattern FinanceBench uses.
- **Output:** a `FilingMetadata` object with those three fields filled in (or `"Unknown"` if the
  filename doesn't match the expected pattern).

### `normalize_report_type(raw_doc_type)` — *metadata.py*
- **Input:** a filing-type string in any format ("10k", "10K", "Earnings", "10k_annualreport"...).
- **What happens:** looks it up in a table of known variants.
- **Output:** the same filing type written consistently ("10-K", "Earnings Report", "Annual Report").

### `chunk_document(pages, company, year, report_type, source_document, ...)` — *chunking.py*
- **Input:** the cleaned pages of a document, plus the company/year/filing-type metadata to attach.
- **What happens:** walks through the document line by line, grouping lines into chunks of about
  500 tokens each (with 100 tokens repeated at the start of the next chunk, so nothing gets lost at
  a boundary). Every time the section changes, it starts a brand-new chunk rather than mixing two
  sections together. It also skips re-tagging pages that look like a Table of Contents (multiple
  different section names crammed onto one page), so a TOC page near the front of the document
  can't accidentally mislabel everything that follows it.
- **Output:** a list of `Chunk` objects, each with a unique ID (like `AAPL_2022_AR_CH_015`), its
  text, and the company/year/section/page it came from.

### `ChromaVectorStore.add_chunks(chunks)` — *vector_store.py*
- **Input:** a list of chunks.
- **What happens:** converts each chunk's text into a numeric "embedding" (using the
  `all-MiniLM-L6-v2` model) and stores it in the shared ChromaDB database, along with its metadata.
  Re-adding the same chunk ID just overwrites the old copy instead of duplicating it.
- **Output:** nothing returned — the chunks are now searchable.

### `ChromaVectorStore.query(query_text, top_k, metadata_filter)` — *vector_store.py*
- **Input:** a search query (a question, or part of one), how many results to return, and an
  optional filter (e.g. "only chunks from Apple's 2022 filing").
- **What happens:** embeds the query the same way the chunks were embedded, then asks ChromaDB for
  the most similar stored chunks, narrowed down by the filter if one was given.
- **Output:** a ranked list of `EvidenceItem`s (the matching chunk plus a relevance score from 0 to 1).

### `DocumentProcessingPipeline.process_pdf(pdf_path)` — *pipeline.py*
- **Input:** a PDF file path.
- **What happens:** runs everything above in order — extract, clean, chunk — for one document.
- **Output:** the finished list of chunks for that document.

### `DocumentProcessingPipeline.ingest_directory(pdf_dir)` — *pipeline.py*
- **Input:** a folder full of PDFs.
- **What happens:** runs `process_pdf` on every PDF in the folder and stores the results in ChromaDB.
- **Output:** the total number of chunks created across every document.

---

## 2. Loading the questions to test (`data/`)

### `load_document_info(path)` — *financebench_loader.py*
- **Input:** the path to FinanceBench's filing-metadata file (361 filings).
- **What happens:** reads that file and builds a lookup table from filing name to its company,
  year, sector, and filing type.
- **Output:** a dictionary you can look up any filing's details in.

### `load_financebench_questions(qa_path, document_info)` — *financebench_loader.py*
- **Input:** the path to FinanceBench's 150-question file, plus the lookup table from
  `load_document_info`.
- **What happens:** reads every question, and for each one looks up its filing's company/year/type
  from the lookup table, and organizes its supporting evidence (which can span more than one page)
  into a clean list.
- **Output:** 150 `FinanceBenchQuestion` objects, each fully self-contained (question, correct
  answer, evidence pages, company, year, sector, etc.).

---

## 3. Talking to the Groq LLM (`llm/`)

### `GroqClient.complete(agent_name, system_prompt, user_prompt, max_tokens, ...)` — *groq_client.py*
- **Input:** which part of the system is asking (for tracking), the instructions for the model, the
  actual question/content, and a cap on how long the answer can be.
- **What happens:** sends one request to the Groq API. If it hits a temporary problem (rate limit,
  timeout), it waits a bit and tries again, up to 4 times. If multiple API keys are configured, a
  rate-limit error switches to a different key immediately instead of waiting.
- **Output:** an `LLMCallRecord` — the model's raw reply, plus how many tokens it used and how long
  it took.

### `GroqClient.complete_json(...)` — *groq_client.py*
- **Input:** the same as `complete`, but the model is expected to reply with a structured JSON object.
- **What happens:** does everything `complete` does, then tries to parse the reply as JSON. If the
  model's reply isn't valid JSON, it shows the model its own broken output and asks it to fix it —
  up to 2 extra tries — before giving up.
- **Output:** the same `LLMCallRecord`, with the parsed JSON attached (or `None` if it never parsed).

### `GroqKeyPool.select_key(exclude)` — *key_pool.py*
- **Input:** a set of API keys to skip (e.g. ones that just failed).
- **What happens:** looks at how much of each key's daily budget is left and picks the one with the
  most room remaining.
- **Output:** the best available API key, or nothing if every key is used up for the day.

### `GroqKeyPool.record_usage(api_key, tokens)` / `mark_exhausted(api_key)` — *key_pool.py*
- **Input:** which key was used and how many tokens it cost (or, for `mark_exhausted`, just the key).
- **What happens:** updates that key's running total for today and saves it to disk, so the count
  survives even if the program restarts.
- **Output:** nothing returned — it's just bookkeeping.

---

## 4. The seven agents (`agents/`)

Each agent takes a `GroqClient` (except Retrieval and Evidence Filtering, which never call the LLM
at all) and does one specific job in the pipeline.

### `QueryUnderstandingAgent.analyze(question)`
- **Input:** the raw question text.
- **What happens:** one call to the LLM asks it to identify the company, year, and financial metric
  involved, decide what *kind* of question it is (a lookup, a comparison, an explanation...), and
  judge how complex it is. That judgment is then double-checked against a fixed set of rules (e.g.
  a question containing "why" or "explain" is always at least Complex) — the rules can only make
  the complexity *higher* than the LLM said, never lower, so an under-confident model can't
  accidentally route a hard question down the easy path.
- **Output:** a `QueryAnalysis` — company, year, metric, complexity level, and whether the question
  needs to be rewritten before searching.

### `QueryRefinementAgent.refine(question, analysis)`
- **Input:** the original question and the analysis from the step above.
- **What happens:** one call to the LLM asks it to rewrite a vague or oddly-phrased question into
  something that reads more like the language an actual financial filing would use.
- **Output:** the rewritten question (or the original, unchanged, if the model's reply was empty).

### `QueryRefinementAgent.expand_multi_query(question, n)`
- **Input:** a question, and how many alternative versions to generate (3, by default).
- **What happens:** one call to the LLM asks it to phrase the same question three different ways,
  to widen the net when searching.
- **Output:** a list of `n` alternative phrasings.

### `RetrievalAgent.retrieve(query, top_k, metadata_filter)`
- **Input:** a query string, how many results to fetch, and an optional company/year filter.
- **What happens:** searches the ChromaDB knowledge base. No LLM call is involved at all — this is
  a pure database search.
- **Output:** the top-k most relevant chunks, each with a relevance score.

### `RetrievalAgent.retrieve_multi(queries, top_k_per_query, metadata_filter)`
- **Input:** several query variants instead of just one.
- **What happens:** searches once per variant, then combines all the results together, removing any
  chunk that showed up more than once (keeping its best score).
- **Output:** one merged, deduplicated, ranked list of chunks.

### `RetrievalAgent.build_metadata_filter(analysis)`
- **Input:** a `QueryAnalysis`.
- **What happens:** turns the extracted company/year into a database filter, if either was found
  confidently. If neither was found, it returns no filter, so the search stays unrestricted instead
  of accidentally filtering out everything.
- **Output:** a filter dictionary, or nothing.

### `EvidenceFilteringAgent.filter(evidence, min_relevance, max_items, ...)`
- **Input:** a list of retrieved chunks.
- **What happens:** no LLM call — this is plain arithmetic and text comparison. Drops any chunk
  scoring below a relevance threshold, removes near-duplicate chunks (ones whose text is almost
  identical to one already kept), and caps the final list at 5 chunks.
- **Output:** the strongest, most distinct chunks left over.

### `ReasoningAgent.reason(question, evidence, analysis)`
- **Input:** the question and the filtered evidence chunks.
- **What happens:** one call to the LLM asks it to work through the evidence step by step — pull
  out the exact numbers needed, do any arithmetic the question requires, connect information across
  chunks if necessary — and then draft an answer, citing exactly which evidence chunks it used. If
  the evidence genuinely isn't enough to answer confidently, the model is told to say so rather than
  guess.
- **Output:** a `ReasoningOutput` — the reasoning steps, the values it pulled out, a draft answer,
  and a list of which evidence IDs it actually relied on.

### `AnswerGenerationAgent.generate(reasoning_output, evidence, include_citations)`
- **Input:** the output of the Reasoning Agent.
- **What happens:** no LLM call — the Reasoning Agent already wrote a grounded draft answer, so this
  step just finalizes it, attaching a short "Sources: ..." line naming the company/filing/page for
  each piece of evidence cited. If the evidence was flagged as insufficient, this returns a fixed
  "not enough information" message instead of a fabricated answer.
- **Output:** the final answer text.

### `VerificationAgent.verify(question, answer, evidence)`
- **Input:** the question, the finished answer, and the evidence it's supposed to be grounded in.
- **What happens:** one call to the LLM checks every factual claim in the answer against the
  evidence and flags anything that isn't actually supported. If the model's reply can't be parsed,
  this deliberately treats the answer as *not* verified (rather than assuming it's fine) — a broken
  check must never silently pass something through.
- **Output:** a `VerificationResult` — whether it passed, which claims (if any) weren't supported,
  and a confidence score.

---

## 5. Scoring the answer (`metrics/`)

All of these run with **zero** extra LLM calls — they use local embeddings, plain text comparison,
or cross-checking against FinanceBench's own labeled evidence pages.

### `answer_relevance(question, generated_answer)`
- **What it measures:** whether the answer is actually about the question that was asked.
- **How:** compares their embeddings — how "close" they are in meaning.

### `exact_match(generated_answer, reference_answer)`
- **What it measures:** whether the answer matches the official answer exactly, after ignoring
  formatting differences like `$`, commas, and capitalization.

### `f1_score(generated_answer, reference_answer)`
- **What it measures:** how much word-for-word overlap there is between the two answers, balancing
  "did it say the right things" against "did it avoid padding with extra stuff."

### `semantic_similarity(generated_answer, reference_answer)`
- **What it measures:** like `answer_relevance`, but comparing the generated answer directly to the
  official answer instead of to the question.

### `context_recall(retrieved, question)` / `context_precision(retrieved, question)`
- **What they measure:** recall = "did we find all the pages FinanceBench says we needed?"
  precision = "of what we found, how much of it was actually relevant?"
- **How:** checks each retrieved chunk's page number against FinanceBench's labeled evidence pages
  for that question.

### `hit_at_k(retrieved, question, k)` / `mean_reciprocal_rank(retrieved, question)`
- **What they measure:** hit@k = "was at least one correct page somewhere in our top-k results?"
  MRR = "how close to the top was the first correct result?" (1st place scores 1.0, 2nd place
  scores 0.5, and so on.)

### `faithfulness(verification_result, generated_answer)` / `hallucination_rate(...)`
- **What they measure:** faithfulness = what fraction of the answer's sentences the Verification
  Agent found to be actually supported. Hallucination rate is just the flip side of that number.

### `evidence_coverage(citations, evidence, question)` / `citation_correctness(...)`
- **What they measure:** coverage = "did the answer end up using all the evidence pages it needed
  to?" Correctness = "of the sources the answer cited, how many were actually the right ones?"

### `numerical_accuracy(generated_answer, reference_answer)`
- **What it measures:** whether the numbers in the answer match the numbers in the official answer
  (allowing a small 1% rounding tolerance).

### `calculation_accuracy(generated_answer, reference_answer, needs_calculation)`
- **What it measures:** the same numeric check as above, but only counted for questions that
  actually required doing math — it returns "not applicable" for questions that didn't.

### `multi_step_reasoning_score(reasoning_output, question)`
- **What it measures:** whether the Reasoning Agent actually pulled together as many distinct pieces
  of evidence as the question genuinely needed, rather than answering off of just one chunk.

### `explanation_completeness(generated_answer, reference_answer)`
- **What it measures:** how much of the official answer's content shows up somewhere in the
  generated answer — did it leave anything important out?

### `latency_seconds(trace)` / `token_usage(trace)` / `cost_per_answer(trace)`
- **What they measure:** how long the whole question took end to end, how many tokens it used
  across every LLM call involved, and roughly what that cost in dollars.

### `compute_all_metrics(trace, question)` — *aggregate.py*
- **Input:** everything recorded about how one question was answered, plus the question itself.
- **What happens:** calls every relevant metric function above in one place. Metrics that don't
  apply (like retrieval metrics for a Direct LLM experiment that never searched anything) are left
  out of the result entirely, rather than reported as a misleading zero.
- **Output:** one dictionary of metric name → score, ready to be averaged and written to the results
  spreadsheet.

---

## 6. Running an experiment (`experiments/`)

### `DirectLLMExperiment.run_question(question)` — *direct_llm_runner.py*
- **Input:** one FinanceBench question.
- **What happens:** sends the question straight to the LLM with this experiment's specific prompt
  style (zero-shot, role-based, few-shot, stepwise, self-verifying, or structured-output) — one
  call, no retrieval at all.
- **Output:** a `QuestionResult` with the generated answer and its computed metrics.

### `AdaptiveRAGPipeline.run_question(question)` — *adaptive_pipeline.py*
- **Input:** one FinanceBench question.
- **What happens:** runs the full agent pipeline described in Section 4 above, shaped by this
  experiment's specific configuration (which of Query Refinement / Evidence Filtering /
  Verification are switched on, and whether retrieval is naive, metadata-filtered, always-rewritten,
  always-multi-query, or fully adaptive). This single class implements all 8 of EXP-07 through
  EXP-14 — the three ablation experiments (EXP-12/13/14) are the exact same code as the full system
  (EXP-11) with one switch turned off, not separate rewritten versions.
- **Output:** a `QuestionResult`, same as above.

### `get_experiment(experiment_id, llm_client, vector_store)` — *registry.py*
- **Input:** which experiment to run (e.g. `"EXP-11"`), the shared LLM client, and (for RAG
  experiments) the shared vector store.
- **What happens:** looks up that experiment's configuration and builds the right runner object.
- **Output:** a ready-to-use experiment runner.

### `BaseExperiment.run_batch(questions)` — *base.py*
- **Input:** a list of questions.
- **What happens:** runs `run_question` on each one in turn, logging progress as it goes.
- **Output:** a list of `QuestionResult`s, one per question, in the same order.

---

## 7. Recording the results (`results/`)

### `ReportWriter.write_overall_performance(exp_id, question_results)`
- **Input:** an experiment ID and every question result from that experiment's run.
- **What happens:** averages the core metrics (relevance, exact match, F1, faithfulness,
  hallucination rate, latency, token usage...) across all the questions, then writes those averages
  into that experiment's row in the "Overall Performance" sheet of `Coding_Sheet_RESULTS.xlsx`.
- **Output:** nothing returned — the spreadsheet is updated in place. `Coding_Sheet.xlsx` (the
  original) is never touched.

### `ReportWriter.write_retrieval_grounding(exp_id, question_results)`
- **What happens:** same idea, but for retrieval-specific metrics (context precision/recall,
  Hit@K, MRR...) — only meaningful for the 8 RAG experiments, since Direct LLM experiments never
  retrieve anything.

### `ReportWriter.write_query_complexity_breakdown(exp_id, question_results)`
- **What happens:** splits the results by whether each question was Simple, Moderate, or Complex,
  and writes one averaged row per complexity tier — this is what lets the thesis directly answer
  "does complexity affect performance?"

### `ReportWriter.write_comparative_ranking_row(exp_id, ...)`
- **What happens:** writes the final cross-experiment comparison row (composite scores, rank,
  headline finding) — filled in only after all 14 experiments have already been run and recorded,
  since it's a summary of everything else, not a fresh measurement.
