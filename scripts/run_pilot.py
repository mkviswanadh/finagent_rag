"""25-document pilot run: validates the full pipeline against LIVE Groq calls before scaling to
the full 150-question run.

Scope, deliberately budget-conscious (Groq free tier, see CLAUDE.md "Requirement Deviations" and
Groq_API_Call_Budget.xlsx):
- Ingests the 25 FinanceBench documents with the most associated questions (realistic multi-document
  retrieval corpus — not a single trivially-easy document).
- Runs a small, deliberately-diverse set of questions from among those 25 documents through ALL 14
  experiments, to catch integration bugs the mocked test suite can't (real JSON reliability, real
  ChromaDB retrieval quality, real latency) — breadth (every experiment) over depth (many questions),
  since the mocked suite already covers pipeline logic exhaustively.
- Records full traces, computes metrics, and writes a summary — into a SEPARATE pilot workbook, never
  the real Coding_Sheet_RESULTS.xlsx.

Run with: PYTHONPATH=src python scripts/run_pilot.py
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finagent.config import PROJECT_ROOT, Settings
from finagent.data import load_document_info, load_financebench_questions
from finagent.document_processing.metadata import FilingMetadata
from finagent.document_processing.pipeline import DocumentProcessingPipeline
from finagent.document_processing.vector_store import ChromaVectorStore
from finagent.experiments.registry import list_experiment_ids, get_experiment
from finagent.llm.groq_client import GroqCallError, GroqClient

NUM_DOCUMENTS = 25
PILOT_QUESTION_IDS_HINT = None  # set to a list of financebench_id strings to pin an exact set

PILOT_CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_store_pilot"
PILOT_REPORT_PATH = PROJECT_ROOT / "pilot_run_report.json"


def select_documents_and_questions() -> tuple[list[str], list]:
    questions = load_financebench_questions()
    doc_counts = Counter(q.document_name for q in questions)
    top_docs = [d for d, _ in doc_counts.most_common(NUM_DOCUMENTS)]
    top_docs_set = set(top_docs)

    candidates = [q for q in questions if q.document_name in top_docs_set]

    # Pick a small, deliberately diverse set: different companies, and a mix of question phrasing
    # that should route to different complexity tiers (plain lookup vs. comparison vs. "why"/"explain").
    picked = []
    seen_companies = set()
    complex_keywords = ("why", "explain", "factors", "drove", "driving")
    moderate_keywords = ("compare", "change", "increase", "decrease", "grow")

    def bucket(q) -> str:
        text = q.question.lower()
        if any(k in text for k in complex_keywords):
            return "complex"
        if any(k in text for k in moderate_keywords):
            return "moderate"
        return "simple"

    buckets: dict[str, list] = {"simple": [], "moderate": [], "complex": []}
    for q in candidates:
        buckets[bucket(q)].append(q)

    # One from each bucket if available, plus one more from whichever bucket has the most options,
    # for a target of 4 questions spanning different companies and (hoped-for) complexity tiers.
    for bucket_name in ("simple", "moderate", "complex"):
        for q in buckets[bucket_name]:
            if q.company not in seen_companies:
                picked.append(q)
                seen_companies.add(q.company)
                break
    remaining_pool = [q for q in candidates if q not in picked]
    if remaining_pool:
        picked.append(remaining_pool[0])

    return top_docs, picked[:4]


def ingest_pilot_documents(doc_names: list[str]) -> ChromaVectorStore:
    doc_info = load_document_info()
    store = ChromaVectorStore(persist_dir=PILOT_CHROMA_DIR, collection_name="pilot_25docs")
    pipeline = DocumentProcessingPipeline(vector_store=store)

    if store.count() > 0:
        print(f"Pilot vector store already populated ({store.count()} chunks) — skipping re-ingestion.")
        return store

    total_chunks = 0
    for doc_name in doc_names:
        pdf_path = PROJECT_ROOT / "data" / "pdfs" / f"{doc_name}.pdf"
        meta = doc_info.get(doc_name)
        override = (
            None
            if meta is None
            else FilingMetadata(
                company=meta.company, year=meta.year, report_type=meta.report_type, document_name=doc_name
            )
        )
        chunks = pipeline.process_pdf(pdf_path, metadata_override=override)
        store.add_chunks(chunks)
        total_chunks += len(chunks)
        print(f"  ingested {doc_name}: {len(chunks)} chunks")

    print(f"Total: {total_chunks} chunks from {len(doc_names)} documents.")
    return store


def run_pilot() -> None:
    print(f"=== FinAgent-RAG Pilot Run ===\n")

    doc_names, pilot_questions = select_documents_and_questions()
    print(f"Selected {len(doc_names)} documents for the knowledge base.")
    print(f"Selected {len(pilot_questions)} pilot questions:")
    for q in pilot_questions:
        print(f"  [{q.question_id}] ({q.company}) {q.question}")
    print()

    print("Ingesting documents into pilot ChromaDB store...")
    store = ingest_pilot_documents(doc_names)
    print()

    settings = Settings()
    llm_client = GroqClient(settings)
    if llm_client.key_pool:
        print(f"Using key pool with {llm_client.key_pool.size} keys.")
        for status in llm_client.key_pool.status():
            print(f"  {status}")
    print()

    results = []
    start_time = time.time()

    for exp_id in list_experiment_ids():
        is_direct_llm = exp_id in [f"EXP-{i:02d}" for i in range(1, 7)]
        vector_store = None if is_direct_llm else store
        experiment = get_experiment(exp_id, llm_client=llm_client, vector_store=vector_store)

        for question in pilot_questions:
            row = {"exp_id": exp_id, "question_id": question.question_id, "status": "OK"}
            t0 = time.time()
            try:
                qr = experiment.run_question(question)
                row["elapsed_seconds"] = round(time.time() - t0, 2)
                row["generated_answer"] = qr.trace.generated_answer
                row["reference_answer"] = question.reference_answer
                row["complexity_used"] = qr.trace.complexity_used.value
                row["num_llm_calls"] = len(qr.trace.llm_calls)
                row["total_tokens"] = qr.trace.total_tokens
                row["metrics"] = {k: round(v, 4) for k, v in qr.metrics.items()}
            except GroqCallError as exc:
                row["status"] = "GROQ_ERROR"
                row["error"] = str(exc)
            except Exception as exc:
                row["status"] = "FAILED"
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc()

            status_marker = {"OK": "OK  ", "GROQ_ERROR": "QUOTA", "FAILED": "FAIL"}[row["status"]]
            print(
                f"[{status_marker}] {exp_id} / {question.question_id} "
                f"({row.get('elapsed_seconds', '?')}s, {row.get('num_llm_calls', '?')} calls, "
                f"{row.get('total_tokens', '?')} tokens)"
            )
            if row["status"] != "OK":
                print(f"       -> {row.get('error')}")
            results.append(row)

    total_elapsed = time.time() - start_time
    print(f"\n=== Pilot run complete in {round(total_elapsed / 60, 1)} minutes ===")

    ok = [r for r in results if r["status"] == "OK"]
    failed = [r for r in results if r["status"] == "FAILED"]
    quota = [r for r in results if r["status"] == "GROQ_ERROR"]
    total_tokens = sum(r.get("total_tokens", 0) for r in ok)
    total_calls = sum(r.get("num_llm_calls", 0) for r in ok)

    print(f"Total runs: {len(results)}  OK: {len(ok)}  FAILED: {len(failed)}  QUOTA-EXHAUSTED: {len(quota)}")
    print(f"Total Groq calls made: {total_calls}")
    print(f"Total tokens used: {total_tokens:,}")

    if llm_client.key_pool:
        print("\nFinal key pool status:")
        for status in llm_client.key_pool.status():
            print(f"  {status}")

    if failed:
        print("\nFAILURES (real bugs, need investigation):")
        for r in failed:
            print(f"  {r['exp_id']} / {r['question_id']}: {r['error']}")

    PILOT_REPORT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull report written to {PILOT_REPORT_PATH}")


if __name__ == "__main__":
    run_pilot()
