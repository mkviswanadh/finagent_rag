"""Per-company document parsing validation (VALIDATION.md item #12 — "THIS IS VERY VERY CRITICAL").

One representative PDF per unique company in `data/pdfs/` (preferring a 10-K where available) is
run through the full extraction -> clean -> chunk pipeline. This is the automated, repeatable
version of the ad-hoc validation originally run during implementation (see VALIDATION.md for the
one-time results table) — every company must parse without exceptions, produce a healthy chunk
count, and show real section-tagging diversity (not a single generic bucket, which would indicate
section detection failed entirely for that filer's layout).
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

from finagent.document_processing.chunking import chunk_document
from finagent.document_processing.cleaning import clean_text
from finagent.document_processing.metadata import parse_filing_metadata
from finagent.document_processing.pdf_extraction import extract_pdf_pages
from finagent.document_processing.table_extraction import extract_tables_from_page
from tests.conftest import PDF_DIR

MIN_SECTION_DIVERSITY = 2  # a real multi-page filing should never collapse to one section bucket
MIN_TEXT_CHARS = 500  # a healthy filing has far more extracted text than this; below suggests a parse failure


def _select_one_pdf_per_company() -> list[tuple[str, Path]]:
    if not PDF_DIR.exists():
        return []
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    companies: dict[str, list[Path]] = {}
    for p in pdfs:
        m = re.match(r"^([A-Za-z0-9&]+)_", p.stem)
        company = m.group(1).upper() if m else p.stem.upper()
        companies.setdefault(company, []).append(p)

    selected: list[tuple[str, Path]] = []
    for company, files in sorted(companies.items()):
        tenk = [f for f in files if "10K" in f.stem.upper()]
        selected.append((company, tenk[0] if tenk else files[0]))
    return selected


_COMPANY_CASES = _select_one_pdf_per_company()
_CASE_IDS = [company for company, _ in _COMPANY_CASES]


@pytest.mark.skipif(not _COMPANY_CASES, reason="data/pdfs/ (FinanceBench PDF corpus) not present in this environment")
@pytest.mark.parametrize("company,pdf_path", _COMPANY_CASES, ids=_CASE_IDS)
def test_company_document_parses_successfully(company: str, pdf_path: Path):
    pages = extract_pdf_pages(pdf_path)
    assert len(pages) > 0, f"{company}: zero pages extracted"

    total_text_chars = sum(len(p.text) for p in pages)
    assert total_text_chars >= MIN_TEXT_CHARS, f"{company}: only {total_text_chars} chars extracted"

    cleaned = clean_text(pages)
    meta = parse_filing_metadata(pdf_path.name)
    chunks = chunk_document(
        cleaned, company=meta.company, year=meta.year, report_type=meta.report_type,
        source_document=pdf_path.name,
    )
    assert len(chunks) > 0, f"{company}: zero chunks produced"
    assert all(c.text.strip() for c in chunks), f"{company}: at least one empty chunk"

    section_counts = Counter(c.section for c in chunks)
    if len(chunks) > 5:  # section diversity is only meaningful once there's enough content to diversify
        assert len(section_counts) >= MIN_SECTION_DIVERSITY, (
            f"{company}: only {len(section_counts)} distinct section(s) across {len(chunks)} chunks "
            "— section detection may have failed for this filer's layout"
        )

    table_block_count = sum(len(extract_tables_from_page(p.text)) for p in cleaned)
    # A 10-K/10-Q with financial statements should have at least some detected tabular content.
    assert table_block_count > 0, f"{company}: zero table blocks detected"
