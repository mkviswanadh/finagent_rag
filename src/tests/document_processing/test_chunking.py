"""Tests for chunking.py — section-aware, token-bounded chunking and TOC-page exclusion."""

from __future__ import annotations

from finagent.document_processing.chunking import _is_toc_like_page, chunk_document
from finagent.document_processing.pdf_extraction import PageContent


def test_chunk_document_produces_expected_chunk_id_format():
    pages = [PageContent(page_number=1, text="Some narrative text about the business.")]
    chunks = chunk_document(
        pages, company="Apple", year=2022, report_type="Annual Report", source_document="AAPL_2022_AR.pdf"
    )
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "APPLE_2022_AR_CH_001"


def test_chunk_document_forces_boundary_at_section_change():
    pages = [
        PageContent(page_number=1, text="Risk Factors\nSome risk narrative text here."),
        PageContent(page_number=2, text="CONSOLIDATED BALANCE SHEETS\nCash and cash equivalents $ 100 $ 90"),
    ]
    chunks = chunk_document(
        pages, company="X", year=2022, report_type="10-K", source_document="X_2022_10K.pdf"
    )
    sections = {c.section for c in chunks}
    assert "Risk Factors" in sections
    assert "Consolidated Balance Sheet" in sections
    # No single chunk should mix both sections' text.
    for chunk in chunks:
        if chunk.section == "Risk Factors":
            assert "Cash and cash equivalents" not in chunk.text


def test_chunk_document_respects_size_and_overlap():
    long_text = "\n".join(f"Sentence number {i} about company operations and results." for i in range(200))
    pages = [PageContent(page_number=1, text=long_text)]
    chunks = chunk_document(
        pages, company="X", year=2022, report_type="10-K", source_document="X.pdf",
        chunk_size_tokens=100, chunk_overlap_tokens=20,
    )
    assert len(chunks) > 1
    # Every chunk_id is unique and sequential.
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_document_empty_pages_returns_no_chunks():
    assert chunk_document([], company="X", year=2022, report_type="10-K", source_document="X.pdf") == []


def test_is_toc_like_page_detects_multiple_distinct_headings():
    toc_page_text = "\n".join([
        "Risk Factors",
        "Legal Proceedings",
        "Controls and Procedures",
    ])
    assert _is_toc_like_page(toc_page_text) is True


def test_is_toc_like_page_allows_two_legitimate_headings_together():
    """Regression test: a real content page (e.g. 10-K Part II item list) with exactly 2 distinct
    headings must NOT be treated as a Table of Contents page."""
    content_page_text = "\n".join([
        "Item 9. Changes in and Disagreements With Accountants on Accounting and Financial Disclosure.",
        "None.",
        "Item 9A. Controls and Procedures.",
        "The Company carried out an evaluation of its disclosure controls and procedures.",
    ])
    assert _is_toc_like_page(content_page_text) is False


def test_is_toc_like_page_false_for_plain_narrative():
    assert _is_toc_like_page("Just some plain narrative text with no headings at all.") is False
