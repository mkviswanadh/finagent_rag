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
- Records full traces (including retrieved evidence, query variants, and per-stage timing — not just
  final metrics, so results can be re-scored offline later without spending more Groq tokens) and
  writes a summary — into a SEPARATE pilot workbook, never the real Coding_Sheet_RESULTS.xlsx.

Every stage of every question (Query Understanding, Retrieval, Evidence Filtering, Reasoning,
Answer Generation, Verification) logs its own start/result via the `finagent` logger — this script
just configures where those logs go (console + file) and adds the per-experiment/overall summary.

Run with: PYTHONPATH=src python scripts/run_pilot.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finagent.config import PROJECT_ROOT, Settings
from finagent.data import load_document_info, load_financebench_questions
from finagent.document_processing.metadata import FilingMetadata
from finagent.document_processing.pipeline import DocumentProcessingPipeline
from finagent.document_processing.vector_store import ChromaVectorStore
from finagent.experiments.registry import get_experiment, list_experiment_ids
from finagent.llm.groq_client import GroqCallError, GroqClient
from finagent.logging_config import configure_logging

logger = logging.getLogger(__name__)

NUM_DOCUMENTS = 25
PILOT_CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_store_pilot"
PILOT_REPORT_PATH = PROJECT_ROOT / "pilot_run_report.json"
PILOT_LOG_PATH = PROJECT_ROOT / "pilot_run.log"


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
        logger.info("Pilot vector store already populated (%d chunks) — skipping re-ingestion.", store.count())
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
        logger.info("Ingested %s: %d chunks", doc_name, len(chunks))

    logger.info("Ingestion complete: %d chunks from %d documents.", total_chunks, len(doc_names))
    return store


def _trace_to_dict(qr) -> dict:
    """Flatten a QuestionResult's trace into a JSON-serializable dict with full raw detail —
    not just final metrics — so results can be re-scored offline later without new Groq calls."""
    trace = qr.trace
    return {
        "generated_answer": trace.generated_answer,
        "complexity_used": trace.complexity_used.value,
        "num_llm_calls": len(trace.llm_calls),
        "total_input_tokens": trace.total_input_tokens,
        "total_output_tokens": trace.total_output_tokens,
        "total_tokens": trace.total_tokens,
        "elapsed_seconds": round(trace.total_latency_seconds, 3),
        "stage_timings": {k: round(v, 3) for k, v in trace.stage_timings.items()},
        "query_variants": trace.query_variants,
        "refined_query": trace.refined_query,
        "query_analysis": (
            None if trace.query_analysis is None else {
                "company": trace.query_analysis.company,
                "year": trace.query_analysis.year,
                "metric": trace.query_analysis.metric,
                "question_type": trace.query_analysis.question_type,
                "needs_calculation": trace.query_analysis.needs_calculation,
                "needs_refinement": trace.query_analysis.needs_refinement,
                "routing_rationale": trace.query_analysis.routing_rationale,
            }
        ),
        "retrieved_chunk_ids": trace.retrieved_chunk_ids,
        "filtered_chunk_ids": trace.filtered_chunk_ids,
        "verification_passed": None if trace.verification_result is None else trace.verification_result.passed,
        "metrics": {k: round(v, 4) for k, v in qr.metrics.items()},
    }


def run_pilot() -> None:
    configure_logging(log_file=PILOT_LOG_PATH)
    logger.info("=== FinAgent-RAG Pilot Run ===")

    doc_names, pilot_questions = select_documents_and_questions()
    logger.info("Selected %d documents for the knowledge base.", len(doc_names))
    logger.info("Selected %d pilot questions:", len(pilot_questions))
    for q in pilot_questions:
        logger.info("  [%s] (%s) %s", q.question_id, q.company, q.question)

    logger.info("Ingesting documents into pilot ChromaDB store...")
    store = ingest_pilot_documents(doc_names)

    settings = Settings()
    llm_client = GroqClient(settings)
    if llm_client.key_pool:
        logger.info("Using key pool with %d keys.", llm_client.key_pool.size)
        for status in llm_client.key_pool.status():
            logger.info("  %s", status)

    results = []
    pipeline_start = time.perf_counter()
    per_experiment_seconds: dict[str, float] = {}

    for exp_id in list_experiment_ids():
        is_direct_llm = exp_id in [f"EXP-{i:02d}" for i in range(1, 7)]
        vector_store = None if is_direct_llm else store
        experiment = get_experiment(exp_id, llm_client=llm_client, vector_store=vector_store)

        exp_start = time.perf_counter()
        for question in pilot_questions:
            row = {"exp_id": exp_id, "question_id": question.question_id, "status": "OK"}
            try:
                qr = experiment.run_question(question)
                row.update(_trace_to_dict(qr))
                row["reference_answer"] = question.reference_answer
            except GroqCallError as exc:
                row["status"] = "GROQ_ERROR"
                row["error"] = str(exc)
                logger.warning("[%s] %s: quota/rate-limit error: %s", exp_id, question.question_id, exc)
            except Exception as exc:
                row["status"] = "FAILED"
                row["error"] = f"{type(exc).__name__}: {exc}"
                logger.exception("[%s] %s: unexpected failure", exp_id, question.question_id)
            results.append(row)

        per_experiment_seconds[exp_id] = round(time.perf_counter() - exp_start, 2)
        logger.info("[%s] experiment total: %.1fs across %d questions", exp_id, per_experiment_seconds[exp_id], len(pilot_questions))

    total_elapsed = time.perf_counter() - pipeline_start
    logger.info("=== Pilot run complete in %.1f minutes ===", total_elapsed / 60)

    ok = [r for r in results if r["status"] == "OK"]
    failed = [r for r in results if r["status"] == "FAILED"]
    quota = [r for r in results if r["status"] == "GROQ_ERROR"]
    total_tokens = sum(r.get("total_tokens", 0) for r in ok)
    total_calls = sum(r.get("num_llm_calls", 0) for r in ok)

    logger.info(
        "Total runs: %d  OK: %d  FAILED: %d  QUOTA-EXHAUSTED: %d",
        len(results), len(ok), len(failed), len(quota),
    )
    logger.info("Total Groq calls made: %d", total_calls)
    logger.info("Total tokens used: %s", f"{total_tokens:,}")
    logger.info("Per-experiment wall time (seconds): %s", per_experiment_seconds)

    if llm_client.key_pool:
        logger.info("Final key pool status:")
        for status in llm_client.key_pool.status():
            logger.info("  %s", status)

    if failed:
        logger.warning("FAILURES (real bugs, need investigation):")
        for r in failed:
            logger.warning("  %s / %s: %s", r["exp_id"], r["question_id"], r["error"])

    report = {
        "pilot_questions": [q.question_id for q in pilot_questions],
        "documents": doc_names,
        "total_elapsed_seconds": round(total_elapsed, 1),
        "per_experiment_seconds": per_experiment_seconds,
        "total_tokens": total_tokens,
        "total_calls": total_calls,
        "runs": results,
    }
    PILOT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Full report written to %s", PILOT_REPORT_PATH)
    logger.info("Full log written to %s", PILOT_LOG_PATH)


if __name__ == "__main__":
    run_pilot()
