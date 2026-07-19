"""Tests for cleaning.py — repeated header/footer and page-number removal."""

from __future__ import annotations

from finagent.document_processing.cleaning import clean_text
from finagent.document_processing.pdf_extraction import PageContent

HEADER = "Apple Inc. | 2022 Form 10-K"


def _build_pages(n: int = 7) -> list[PageContent]:
    pages = []
    for i in range(1, n + 1):
        body_lines = [f"Narrative sentence {j} about operations for page {i}." for j in range(6)]
        text = "\n".join([HEADER, str(i), ""] + body_lines)
        pages.append(PageContent(page_number=i, text=text))
    return pages


def test_clean_text_removes_repeated_header():
    cleaned = clean_text(_build_pages())
    assert all(HEADER not in page.text for page in cleaned)


def test_clean_text_removes_standalone_page_numbers():
    cleaned = clean_text(_build_pages())
    for page in cleaned:
        assert not any(line.strip() == str(page.page_number) for line in page.text.splitlines())


def test_clean_text_preserves_narrative_content():
    cleaned = clean_text(_build_pages())
    assert "Narrative sentence 0 about operations for page 1." in cleaned[0].text


def test_clean_text_preserves_page_numbers_field():
    pages = _build_pages()
    cleaned = clean_text(pages)
    assert [p.page_number for p in cleaned] == [p.page_number for p in pages]


def test_clean_text_empty_input():
    assert clean_text([]) == []


def test_clean_text_below_repeat_threshold_page_count_keeps_header():
    """With fewer than the minimum page count, no header/footer detection is attempted at all."""
    pages = _build_pages(n=2)
    cleaned = clean_text(pages)
    # Below _HEADER_FOOTER_MIN_PAGE_COUNT (4), the header is not considered "repeated" and survives.
    assert any(HEADER in page.text for page in cleaned)
