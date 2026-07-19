"""Document cleaning (Proposal Table 7.4, Step 3: "Cleaning").

Removes repeated headers/footers, standalone page-number lines, and formatting noise that would
otherwise dilute retrieval quality (Proposal §7.5: "cleaning to remove repeated headers, page
numbers, navigation elements, and other irrelevant content that may reduce retrieval quality").

Repeated headers/footers (e.g. "Apple Inc. | 2022 Form 10-K", "See accompanying notes...") are
detected by cross-page frequency rather than a fixed denylist, since exact boilerplate text varies
by filer and filing type — a frequency heuristic generalizes across the whole FinanceBench PDF
collection without per-company special-casing.
"""

from __future__ import annotations

import re

from finagent.document_processing.pdf_extraction import PageContent

_PAGE_NUMBER_LINE = re.compile(r"^\s*(page\s+)?[-–—]?\s*\d{1,4}\s*[-–—]?\s*$", re.IGNORECASE)
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_MULTI_SPACES = re.compile(r"[ \t]{2,}")

# A candidate header/footer line must be short (real content rarely repeats verbatim across many
# pages at this length) and must recur on at least this fraction of the document's pages.
_HEADER_FOOTER_MAX_LEN = 90
_HEADER_FOOTER_MIN_FREQUENCY = 0.35
_HEADER_FOOTER_MIN_PAGE_COUNT = 4


def clean_text(pages: list[PageContent]) -> list[PageContent]:
    """Remove repeated headers/footers, page-number lines, and noise from an extracted document.

    Args:
        pages: Pages as returned by `extract_pdf_pages`, in document order.

    Returns:
        A new list of `PageContent` with boilerplate removed and whitespace normalized. Page
        numbers (the `page_number` field) are preserved unchanged — only `text` is modified.
    """
    if not pages:
        return []

    boilerplate_lines = _detect_repeated_lines(pages) if len(pages) >= _HEADER_FOOTER_MIN_PAGE_COUNT else set()

    cleaned: list[PageContent] = []
    for page in pages:
        lines = page.text.splitlines()
        kept_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                kept_lines.append("")
                continue
            if stripped in boilerplate_lines:
                continue
            if _PAGE_NUMBER_LINE.match(stripped):
                continue
            kept_lines.append(_MULTI_SPACES.sub(" ", stripped))

        text = "\n".join(kept_lines)
        text = _MULTI_BLANK_LINES.sub("\n\n", text).strip()
        cleaned.append(PageContent(page_number=page.page_number, text=text))

    return cleaned


def _detect_repeated_lines(pages: list[PageContent]) -> set[str]:
    """Identify short lines that recur across a large fraction of pages (headers/footers)."""
    line_page_counts: dict[str, int] = {}
    for page in pages:
        seen_on_this_page: set[str] = set()
        for line in page.text.splitlines():
            stripped = line.strip()
            if not stripped or len(stripped) > _HEADER_FOOTER_MAX_LEN:
                continue
            if stripped in seen_on_this_page:
                continue
            seen_on_this_page.add(stripped)
            line_page_counts[stripped] = line_page_counts.get(stripped, 0) + 1

    threshold_count = max(
        _HEADER_FOOTER_MIN_PAGE_COUNT, int(len(pages) * _HEADER_FOOTER_MIN_FREQUENCY)
    )
    return {line for line, count in line_page_counts.items() if count >= threshold_count}
