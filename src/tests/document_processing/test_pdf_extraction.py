"""Tests for pdf_extraction.py — PDF text extraction via PyMuPDF."""

from __future__ import annotations

import pytest

from finagent.document_processing.pdf_extraction import extract_pdf_pages


def test_extract_pdf_pages_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.pdf"
    with pytest.raises(FileNotFoundError):
        extract_pdf_pages(missing)


def test_extract_pdf_pages_real_document():
    """Extraction against a real corpus PDF: correct page count, non-empty text, 1-indexed pages."""
    pytest.importorskip("fitz")
    from tests.conftest import PDF_DIR

    pdf_path = PDF_DIR / "3M_2018_10K.pdf"
    if not pdf_path.exists():
        pytest.skip("data/pdfs/3M_2018_10K.pdf not present in this environment")

    pages = extract_pdf_pages(pdf_path)

    assert len(pages) == 160  # verified page count for this specific filing
    assert pages[0].page_number == 1
    assert pages[-1].page_number == 160
    # Every page should have extracted some text — a real 10-K has no blank pages.
    assert all(p.text.strip() for p in pages)
