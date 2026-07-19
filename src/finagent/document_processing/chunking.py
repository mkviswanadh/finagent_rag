"""Section-aware chunking (Proposal Table 7.4, Step 4: "Chunking" + Step 5: "Metadata Tagging").

Produces `Chunk` objects sized to `config.CHUNK_SIZE_TOKENS` (500) with `config.CHUNK_OVERLAP_TOKENS`
(100) overlap — the fixed values held constant across all RAG experiments (finagent-architecture
skill §1). Chunking is section-aware: a chunk boundary is always forced at a detected section
change (Proposal §7.5: "documents are divided into smaller retrieval-ready chunks" after "important
report sections... are identified"), so a single chunk never silently mixes, e.g., balance-sheet
rows with risk-factor narrative. Token overlap is only carried across a size-triggered split within
the same section, never across a section boundary, since carrying topically unrelated context
across a section change would not improve retrieval — it would dilute it.
"""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from finagent.config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS
from finagent.data.schemas import Chunk
from finagent.document_processing.pdf_extraction import PageContent
from finagent.document_processing.section_detection import UNCLASSIFIED_SECTION, match_section_heading

# cl100k_base is used purely as a consistent, fast token-counting proxy for sizing chunks — it is
# not the exact Llama 3.3 tokenizer, but chunk-size consistency (not exact token parity with the
# generation model) is what the fixed 500/100 setup is protecting.
_ENCODING = tiktoken.get_encoding("cl100k_base")

_REPORT_TYPE_ABBREVIATIONS = {
    "10-K": "10K",
    "10-Q": "10Q",
    "8-K": "8K",
    "ANNUAL REPORT": "AR",
    "EARNINGS REPORT": "ER",
    "EARNINGS RELEASE": "ER",
    "REGULATORY DISCLOSURE": "RD",
}


@dataclass(frozen=True)
class _Unit:
    page_number: int
    section: str
    text: str

    @property
    def token_count(self) -> int:
        return len(_ENCODING.encode(self.text))


# A page where 2+ *distinct* canonical sections all match is almost certainly a Table of Contents
# or index page (financial-report TOCs list every section name as a short standalone line), not
# real section content — a real content page essentially never contains two different section
# headings. Such pages are excluded from section (re)tagging entirely, or the TOC's first mention
# of e.g. "Controls and Procedures" would prematurely claim that label for everything up to the
# next matched heading, potentially the bulk of the document.
_TOC_PAGE_DISTINCT_SECTION_THRESHOLD = 2


def _is_toc_like_page(page_text: str) -> bool:
    matched_sections = {
        match_section_heading(line) for line in page_text.splitlines() if match_section_heading(line)
    }
    return len(matched_sections) >= _TOC_PAGE_DISTINCT_SECTION_THRESHOLD


def _units_from_pages(pages: list[PageContent]) -> list[_Unit]:
    """Flatten pages into line-level units, retagging the running section on every heading line.

    Section is tracked per-line rather than once per page: a heading can appear partway down a
    page, with everything below it already belonging to the new section. Tagging the whole page
    with a single `detect_section` call (as an earlier version of this function did) would
    misattribute that tail to whichever section was active at the top of the page. Pages detected
    as Table-of-Contents-like (`_is_toc_like_page`) are skipped for retagging purposes so their
    listing of section names doesn't corrupt the running section for real content pages that follow.
    """
    units: list[_Unit] = []
    current_section = UNCLASSIFIED_SECTION
    for page in pages:
        page_is_toc_like = _is_toc_like_page(page.text)
        for line in page.text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if not page_is_toc_like:
                heading_match = match_section_heading(stripped)
                if heading_match:
                    current_section = heading_match
            units.append(_Unit(page_number=page.page_number, section=current_section, text=stripped))
    return units


def _report_type_code(report_type: str) -> str:
    return _REPORT_TYPE_ABBREVIATIONS.get(report_type.strip().upper(), "DOC")


def _company_code(company: str) -> str:
    return "".join(ch for ch in company.upper() if ch.isalnum()) or "UNK"


def chunk_document(
    pages: list[PageContent],
    *,
    company: str,
    year: int,
    report_type: str,
    source_document: str,
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Split a cleaned, section-detected document into metadata-tagged, retrieval-ready chunks.

    Args:
        pages: Cleaned pages (post `clean_text`), in document order.
        company: Company name for metadata tagging and chunk ID generation.
        year: Filing year for metadata tagging and chunk ID generation.
        report_type: Filing type, e.g. "10-K", "10-Q", "8-K", "Annual Report" — used both as
            stored metadata and to build the chunk ID's report-type code (Proposal Table 7.5:
            "AR" in "AAPL_2022_AR_CH_015").
        source_document: Filename of the originating PDF, stored on every chunk for traceability.
        chunk_size_tokens: Target maximum chunk size. Defaults to the fixed experimental value;
            override only for deliberate, documented experimentation outside the controlled runs.
        chunk_overlap_tokens: Token overlap carried across a size-triggered split within the same
            section. Defaults to the fixed experimental value.

    Returns:
        Chunks in document order, each with a unique, deterministic `chunk_id` of the form
        `{COMPANY}_{YEAR}_{REPORTTYPE}_CH_{seq:03d}`.
    """
    units = _units_from_pages(pages)
    company_code = _company_code(company)
    report_code = _report_type_code(report_type)

    chunks: list[Chunk] = []
    current_units: list[_Unit] = []
    current_tokens = 0
    seq = 1

    def _flush(carry_overlap: bool) -> None:
        nonlocal current_units, current_tokens, seq
        if not current_units:
            return
        text = "\n".join(u.text for u in current_units)
        page_number = current_units[0].page_number
        section = current_units[0].section
        chunk_id = f"{company_code}_{year}_{report_code}_CH_{seq:03d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                company=company,
                year=year,
                report_type=report_type,
                section=section,
                page_number=page_number,
                text=text,
                source_document=source_document,
            )
        )
        seq += 1

        if carry_overlap:
            overlap_units: list[_Unit] = []
            overlap_tokens = 0
            for unit in reversed(current_units):
                overlap_tokens += unit.token_count
                overlap_units.insert(0, unit)
                if overlap_tokens >= chunk_overlap_tokens:
                    break
            current_units = overlap_units
            current_tokens = overlap_tokens
        else:
            current_units = []
            current_tokens = 0

    active_section = units[0].section if units else None
    for unit in units:
        if current_units and unit.section != active_section:
            _flush(carry_overlap=False)
            active_section = unit.section

        if current_units and current_tokens + unit.token_count > chunk_size_tokens:
            _flush(carry_overlap=True)

        current_units.append(unit)
        current_tokens += unit.token_count
        active_section = unit.section

    _flush(carry_overlap=False)
    return chunks
