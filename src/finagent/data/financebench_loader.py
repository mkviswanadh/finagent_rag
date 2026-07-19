"""Loader for the real FinanceBench dataset files (Proposal §7.4, Table 7.3).

`financebench_open_source.jsonl` (150 questions) and `financebench_document_information.jsonl`
(361 filings) are the actual upstream files from https://github.com/patronus-ai/financebench —
their real field names differ from the proposal's condensed Table 7.3 summary, so this module is
the single place that maps raw dataset JSON onto this codebase's `FinanceBenchQuestion` /
`EvidenceReference` schema. Every other module (agents, experiments, metrics) should consume
`FinanceBenchQuestion` objects from here, never re-parse the raw JSONL itself.

Real schema, as verified against the actual files (not just the proposal's summary):
- QA record keys: financebench_id, question, answer, company, doc_name, question_type,
  question_reasoning, evidence (list of {doc_name, evidence_page_num, evidence_text}),
  justification, dataset_subset_label, domain_question_num.
- Document-info record keys: doc_name, company, doc_type ("10k", "10q", "8k", "Earnings",
  "10k_annualreport"), doc_period (int year), gics_sector, doc_link.
- 23% of the 150 open-source questions have 2-3 evidence excerpts (not always 1), which is why
  `FinanceBenchQuestion.evidence` is a list (see `schemas.EvidenceReference`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from finagent.config import DOCUMENT_INFO_PATH, QA_DATASET_PATH
from finagent.data.schemas import EvidenceReference, FinanceBenchQuestion
from finagent.document_processing.metadata import normalize_report_type


@dataclass(frozen=True)
class FinanceBenchDocumentInfo:
    """One row of `financebench_document_information.jsonl`, fully resolved.

    Distinct from `document_processing.metadata.FilingMetadata` (which is the generic,
    filename-derivable subset used by the PDF ingestion pipeline for arbitrary documents) because
    this carries `gics_sector` and `doc_link`, which only the FinanceBench dataset itself supplies.
    """

    doc_name: str
    company: str
    year: int
    report_type: str
    gics_sector: str
    doc_link: str


def load_document_info(
    path: str | Path = DOCUMENT_INFO_PATH,
) -> dict[str, FinanceBenchDocumentInfo]:
    """Load `financebench_document_information.jsonl` into a doc_name -> info table.

    Args:
        path: Path to the document-information JSONL file.

    Returns:
        Mapping from `doc_name` (e.g. "3M_2018_10K") to its resolved `FinanceBenchDocumentInfo`,
        with `report_type` normalized through the same table `document_processing.metadata` uses
        for filename parsing (e.g. raw "10k" -> "10-K"), so downstream code never has to branch on
        which source resolved a filing's report type.
    """
    path = Path(path)
    table: dict[str, FinanceBenchDocumentInfo] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            report_type = normalize_report_type(str(record["doc_type"]))
            table[record["doc_name"]] = FinanceBenchDocumentInfo(
                doc_name=record["doc_name"],
                company=record["company"],
                year=int(record["doc_period"]),
                report_type=report_type,
                gics_sector=record.get("gics_sector", ""),
                doc_link=record.get("doc_link", ""),
            )
    return table


def load_financebench_questions(
    qa_path: str | Path = QA_DATASET_PATH,
    document_info: dict[str, FinanceBenchDocumentInfo] | None = None,
) -> list[FinanceBenchQuestion]:
    """Load and join the FinanceBench open-source QA split into `FinanceBenchQuestion` objects.

    Args:
        qa_path: Path to `financebench_open_source.jsonl`.
        document_info: Pre-loaded doc_name -> `FinanceBenchDocumentInfo` table (from
            `load_document_info`). If `None`, it is loaded automatically from
            `config.DOCUMENT_INFO_PATH`.

    Returns:
        All 150 questions from the open-source split, each with its `evidence` field resolved to
        one or more `EvidenceReference`s and its `document_type`/`document_year`/`gics_sector`
        resolved via `document_info` (falling back to "Unknown"/`None`/`""` for the rare case where
        a question's `doc_name` has no matching document-info entry).

    Raises:
        FileNotFoundError: if `qa_path` does not exist.
    """
    qa_path = Path(qa_path)
    if not qa_path.exists():
        raise FileNotFoundError(
            f"FinanceBench QA dataset not found at {qa_path}. Expected the upstream "
            "financebench_open_source.jsonl file (see https://github.com/patronus-ai/financebench)."
        )
    if document_info is None:
        document_info = load_document_info()

    questions: list[FinanceBenchQuestion] = []
    with qa_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            evidence = [
                EvidenceReference(
                    doc_name=e.get("doc_name", record["doc_name"]),
                    page_number=e.get("evidence_page_num"),
                    text=e.get("evidence_text", ""),
                )
                for e in record.get("evidence", [])
            ]

            doc_name = record["doc_name"]
            meta = document_info.get(doc_name)

            questions.append(
                FinanceBenchQuestion(
                    question_id=record["financebench_id"],
                    question=record["question"],
                    reference_answer=record["answer"],
                    evidence=evidence,
                    company=record.get("company") or (meta.company if meta else "Unknown"),
                    document_type=meta.report_type if meta else "Unknown",
                    document_name=doc_name,
                    document_year=meta.year if meta else None,
                    gics_sector=meta.gics_sector if meta else "",
                    justification=record.get("justification") or "",
                    dataset_question_type=record.get("question_type", ""),
                    question_reasoning=record.get("question_reasoning") or "",
                )
            )
    return questions
