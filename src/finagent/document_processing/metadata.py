"""Filing metadata resolution (Proposal Table 7.4, Step 5: "Metadata Tagging").

FinanceBench PDF filenames follow a `{COMPANY}_{YEAR}_{DOCTYPE}[_extra].pdf` convention (e.g.
`3M_2018_10K.pdf`, `AMAZON_2022_8K_dated-2022-06-06.pdf`). This module parses that convention as
the default source of truth, with an explicit override path (`load_filing_metadata_table`) for
cases where the FinanceBench-supplied question metadata (`company`, `document_type`,
`document_year`, `document_name` — Proposal Table 7.3) disagrees with or is more reliable than the
filename, since the QA dataset's own fields should always win when both are available (a chunk's
metadata must match what FinanceBench's evidence annotations expect, or context-recall evaluation
silently breaks).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

_FILENAME_PATTERN = re.compile(
    r"^(?P<company>[A-Za-z0-9&]+)_(?P<year>\d{4})(Q\d)?_(?P<doctype>10K|10Q|8K|EARNINGS|ER|AR|ANNUALREPORT)",
    re.IGNORECASE,
)

# Covers both the FinanceBench PDF filename convention (upper-case: "10K", "ANNUALREPORT") and the
# `doc_type` values found in financebench_document_information.jsonl (lower-case/underscored:
# "10k", "10k_annualreport", "Earnings" — see financebench_loader.py for the latter's exact
# provenance). Normalizing both through the same table keeps `Chunk.report_type` consistent
# regardless of which source (filename vs. dataset metadata) resolved it.
_DOCTYPE_NORMALIZATION = {
    "10K": "10-K",
    "10Q": "10-Q",
    "8K": "8-K",
    "EARNINGS": "Earnings Report",
    "ER": "Earnings Report",
    "AR": "Annual Report",
    "ANNUALREPORT": "Annual Report",
    "10K_ANNUALREPORT": "Annual Report",
}


@dataclass(frozen=True)
class FilingMetadata:
    """Resolved filing-level metadata for one source PDF."""

    company: str
    year: int
    report_type: str
    document_name: str


def parse_filing_metadata(filename: str) -> FilingMetadata:
    """Parse company/year/report-type from a FinanceBench-convention PDF filename.

    Args:
        filename: The PDF's filename (with or without extension/directory), e.g.
            "3M_2018_10K.pdf".

    Returns:
        A `FilingMetadata` with best-effort parsed fields. `report_type` falls back to "Unknown"
        and `year` falls back to `0` if the filename doesn't match the expected convention —
        callers ingesting real FinanceBench data should prefer `load_filing_metadata_table` (or
        the QA dataset's own `document_type`/`document_year` fields) over relying on this fallback.

    Raises:
        ValueError: if `filename` is empty.
    """
    if not filename:
        raise ValueError("filename must be non-empty")

    stem = Path(filename).stem
    match = _FILENAME_PATTERN.match(stem)
    if not match:
        return FilingMetadata(company="Unknown", year=0, report_type="Unknown", document_name=stem)

    company = match.group("company")
    year = int(match.group("year"))
    report_type = normalize_report_type(match.group("doctype"))
    return FilingMetadata(company=company, year=year, report_type=report_type, document_name=stem)


def normalize_report_type(raw_doc_type: str) -> str:
    """Normalize a raw document-type token to its canonical display form.

    Accepts both the FinanceBench PDF filename convention ("10K", "ANNUALREPORT") and the
    `doc_type` values found in `financebench_document_information.jsonl` ("10k", "Earnings",
    "10k_annualreport") — see `finagent.data.financebench_loader`, the other caller of this
    function, for where the latter comes from. Unrecognized tokens are returned unchanged so
    callers can decide how to handle an unknown filing type rather than having it silently
    swallowed.

    Args:
        raw_doc_type: A document-type token from a filename or dataset record, in any casing.

    Returns:
        The canonical form (e.g. "10-K", "Annual Report", "Earnings Report") if recognized,
        otherwise `raw_doc_type` unchanged.
    """
    key = raw_doc_type.strip().upper().replace(" ", "_")
    return _DOCTYPE_NORMALIZATION.get(key, raw_doc_type)


def load_filing_metadata_table(path: str | Path) -> dict[str, FilingMetadata]:
    """Load an explicit filing-metadata manifest (CSV or JSON) keyed by document name.

    Use this when the FinanceBench dataset's own metadata (company/document_type/document_year
    fields on each QA record, Proposal Table 7.3) should override filename parsing — e.g. when
    building the manifest directly from the `financebench_qa.jsonl` dataset during ingestion.

    Expected CSV columns / JSON object keys per entry: `document_name`, `company`, `year`,
    `report_type`.

    Args:
        path: Path to a `.csv` or `.json` manifest file.

    Returns:
        A mapping from `document_name` (filename stem) to `FilingMetadata`.

    Raises:
        ValueError: if `path`'s extension is neither `.csv` nor `.json`.
        FileNotFoundError: if `path` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata manifest not found: {path}")

    entries: list[dict]
    if path.suffix.lower() == ".json":
        entries = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as fh:
            entries = list(csv.DictReader(fh))
    else:
        raise ValueError(f"Unsupported metadata manifest format: {path.suffix}")

    table: dict[str, FilingMetadata] = {}
    for entry in entries:
        document_name = entry["document_name"]
        table[document_name] = FilingMetadata(
            company=entry["company"],
            year=int(entry["year"]),
            report_type=entry["report_type"],
            document_name=document_name,
        )
    return table
