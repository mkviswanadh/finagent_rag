"""Zero-Groq-cost diagnostic: validates document parsing, retrieval, and evidence filtering
across the FULL 150-question FinanceBench set before spending any API budget on Reasoning/
Verification.

Motivation: the pilot draft (`Pilot_Results_Draft.xlsx`) showed several experiments' Evidence &
Retrieval columns (context recall/precision, Hit@K, MRR) at or near zero on most questions.
Retrieval and Evidence Filtering are pure local operations (ChromaDB + sentence-transformers) —
they cost **zero** Groq calls regardless of how many questions are run — so that finding can be
fully investigated, for every one of the 150 questions, without spending a single token. This
script does exactly that, for two strategies that don't require a live Query Understanding call:

- "naive": unfiltered semantic search on the raw question text — what EXP-07/09/10 reduce to
  before their own (Groq-costing) query rewriting/expansion.
- "oracle_metadata": semantic search filtered to the question's own ground-truth company/year
  (`FinanceBenchQuestion.company` / `.document_year`, straight from the dataset) — an upper bound
  on what EXP-08's metadata filtering could achieve *if* Query Understanding's entity extraction
  were perfect, without paying for the Groq call that would normally produce those entities.

If oracle_metadata still scores near-zero on a question, that question's problem is retrieval/
chunking quality, not entity extraction or routing — worth fixing before the full run. If
oracle_metadata recovers a question naive missed, that question specifically needs Query
Understanding to extract the right company/year live.

Also reports, per referenced document, whether it produced any chunks at all — the parsing
pre-flight check ("are documents getting parsed successfully") — before retrieval is even
attempted.

Run with: PYTHONPATH=src python scripts/diagnose_retrieval.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finagent.agents.evidence_filtering import EvidenceFilteringAgent
from finagent.agents.retrieval import RetrievalAgent
from finagent.config import PROJECT_ROOT
from finagent.data import load_document_info, load_financebench_questions
from finagent.data.schemas import QueryAnalysis, QueryComplexity
from finagent.document_processing.metadata import FilingMetadata
from finagent.document_processing.pipeline import DocumentProcessingPipeline
from finagent.document_processing.vector_store import ChromaVectorStore
from finagent.logging_config import configure_logging
from finagent.metrics.retrieval_metrics import context_precision, context_recall, hit_at_k, mean_reciprocal_rank
from finagent.results.archive import archive_file

logger = logging.getLogger(__name__)

DIAGNOSTIC_CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_store_diagnostic"
DIAGNOSTIC_REPORT_PATH = PROJECT_ROOT / "retrieval_diagnostic_report.json"
DIAGNOSTIC_LOG_PATH = PROJECT_ROOT / "retrieval_diagnostic.log"


def ingest_referenced_documents(doc_names: list[str]) -> tuple[ChromaVectorStore, dict[str, int]]:
    """Ingest every document referenced by at least one question; returns per-document chunk
    counts so a zero-chunk document (a parsing failure) is visible before retrieval is attempted."""
    doc_info = load_document_info()
    store = ChromaVectorStore(persist_dir=DIAGNOSTIC_CHROMA_DIR, collection_name="diagnostic_full")
    pipeline = DocumentProcessingPipeline(vector_store=store)

    chunk_counts: dict[str, int] = {}
    already_populated = store.count() > 0
    if already_populated:
        logger.info("Diagnostic vector store already populated (%d chunks) — skipping re-ingestion.", store.count())

    for doc_name in doc_names:
        pdf_path = PROJECT_ROOT / "data" / "pdfs" / f"{doc_name}.pdf"
        if not pdf_path.exists():
            logger.error("PARSE FAILURE: %s — PDF file not found at %s", doc_name, pdf_path)
            chunk_counts[doc_name] = 0
            continue
        if already_populated:
            chunk_counts[doc_name] = -1  # unknown — not re-counted when reusing an existing store
            continue
        meta = doc_info.get(doc_name)
        override = (
            None
            if meta is None
            else FilingMetadata(
                company=meta.company, year=meta.year, report_type=meta.report_type, document_name=doc_name
            )
        )
        try:
            chunks = pipeline.process_pdf(pdf_path, metadata_override=override)
            store.add_chunks(chunks)
            chunk_counts[doc_name] = len(chunks)
            if len(chunks) == 0:
                logger.error("PARSE FAILURE: %s produced 0 chunks", doc_name)
            else:
                logger.info("Ingested %s: %d chunks", doc_name, len(chunks))
        except Exception:
            logger.exception("PARSE FAILURE: %s raised an exception during processing", doc_name)
            chunk_counts[doc_name] = 0

    return store, chunk_counts


def _oracle_metadata_filter(question) -> dict | None:
    """Ground-truth-derived metadata filter — zero Groq calls, upper bound on EXP-08."""
    analysis = QueryAnalysis(
        complexity=QueryComplexity.SIMPLE,
        company=question.company,
        year=question.document_year,
        metric=None,
        question_type="lookup",
        needs_calculation=False,
        needs_multiple_evidence_chunks=len(question.evidence) > 1,
        needs_refinement=False,
        routing_rationale="oracle diagnostic — ground truth, not a live routing decision",
    )
    return RetrievalAgent.build_metadata_filter(analysis)


def run_diagnostic() -> None:
    configure_logging(log_file=DIAGNOSTIC_LOG_PATH)
    logger.info("=== FinAgent-RAG Retrieval Diagnostic (zero Groq calls) ===")

    questions = load_financebench_questions()
    doc_names = sorted({q.document_name for q in questions})
    logger.info("Loaded %d questions referencing %d unique documents.", len(questions), len(doc_names))

    store, chunk_counts = ingest_referenced_documents(doc_names)
    failed_docs = [d for d, c in chunk_counts.items() if c == 0]
    if failed_docs:
        logger.warning("PARSING FAILURES (%d/%d documents produced 0 chunks): %s", len(failed_docs), len(doc_names), failed_docs)
    else:
        logger.info("Parsing pre-flight: all %d referenced documents produced at least 1 chunk.", len(doc_names))

    retrieval_agent = RetrievalAgent(store)
    filtering_agent = EvidenceFilteringAgent()

    rows = []
    start = time.perf_counter()
    for question in questions:
        row = {"question_id": question.question_id, "company": question.company, "document_name": question.document_name}
        for strategy, metadata_filter in (
            ("naive", None),
            ("oracle_metadata", _oracle_metadata_filter(question)),
        ):
            retrieved = retrieval_agent.retrieve(question.question, metadata_filter=metadata_filter)
            filtered = filtering_agent.filter(retrieved)
            row[f"{strategy}_context_recall"] = context_recall(retrieved, question)
            row[f"{strategy}_context_precision"] = context_precision(retrieved, question)
            row[f"{strategy}_hit_at_k"] = hit_at_k(retrieved, question, 5)
            row[f"{strategy}_mrr"] = mean_reciprocal_rank(retrieved, question)
            row[f"{strategy}_context_recall_post_filter"] = context_recall(filtered, question)
        rows.append(row)
    elapsed = time.perf_counter() - start
    logger.info("Scored %d questions x 2 strategies in %.1fs (zero Groq calls).", len(rows), elapsed)

    def _agg(key: str) -> float:
        return round(mean(r[key] for r in rows), 4)

    summary = {
        "questions_scored": len(rows),
        "documents_referenced": len(doc_names),
        "documents_failed_parsing": failed_docs,
        "naive": {
            "avg_context_recall": _agg("naive_context_recall"),
            "avg_context_precision": _agg("naive_context_precision"),
            "avg_hit_at_5": _agg("naive_hit_at_k"),
            "avg_mrr": _agg("naive_mrr"),
            "zero_recall_questions": sum(1 for r in rows if r["naive_context_recall"] == 0),
        },
        "oracle_metadata": {
            "avg_context_recall": _agg("oracle_metadata_context_recall"),
            "avg_context_precision": _agg("oracle_metadata_context_precision"),
            "avg_hit_at_5": _agg("oracle_metadata_hit_at_k"),
            "avg_mrr": _agg("oracle_metadata_mrr"),
            "zero_recall_questions": sum(1 for r in rows if r["oracle_metadata_context_recall"] == 0),
        },
    }
    still_zero_with_oracle = [
        r["question_id"] for r in rows
        if r["oracle_metadata_context_recall"] == 0 and r["naive_context_recall"] == 0
    ]
    summary["questions_failing_even_with_perfect_metadata"] = still_zero_with_oracle

    logger.info("=== Summary ===")
    logger.info("Naive retrieval:           avg context_recall=%.3f, zero-recall on %d/%d questions",
                summary["naive"]["avg_context_recall"], summary["naive"]["zero_recall_questions"], len(rows))
    logger.info("Oracle-metadata retrieval: avg context_recall=%.3f, zero-recall on %d/%d questions",
                summary["oracle_metadata"]["avg_context_recall"], summary["oracle_metadata"]["zero_recall_questions"], len(rows))
    logger.info("%d questions fail retrieval even with perfect ground-truth metadata (a chunking/embedding "
                "problem, not an entity-extraction problem): %s", len(still_zero_with_oracle), still_zero_with_oracle)

    report = {"summary": summary, "chunk_counts": chunk_counts, "rows": rows}
    archived = archive_file(DIAGNOSTIC_REPORT_PATH, category="diagnostic")
    if archived:
        logger.info("Previous diagnostic report archived to %s", archived)
    DIAGNOSTIC_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Full report written to %s", DIAGNOSTIC_REPORT_PATH)


if __name__ == "__main__":
    run_diagnostic()
