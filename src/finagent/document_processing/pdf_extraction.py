"""PDF text extraction (Proposal Table 7.4, Step 1: "PDF Text Extraction").

Uses `pypdf` exclusively — this development machine has no poppler/`pdftoppm` installed, so any
extraction path depending on `pdf2image` or `pdftoppm` will fail here (see finagent-architecture
skill §9). `pypdf`'s layout-preserving text extraction is sufficient to keep headings and rough
table row structure intact for the downstream table-heuristic and section-detection stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class PageContent:
    """Raw extracted text for a single PDF page, 1-indexed to match human-readable page numbers."""

    page_number: int
    text: str


def extract_pdf_pages(pdf_path: str | Path) -> list[PageContent]:
    """Extract text from every page of a financial PDF, preserving page boundaries.

    Page boundaries are preserved (rather than concatenating the whole document into one string)
    because `page_number` is a required metadata field on every `Chunk` (Proposal Table 7.5) and
    is used for fine-grained retrieval evaluation (Proposal §7.4.1).

    Args:
        pdf_path: Path to the source financial PDF (10-K, 10-Q, 8-K, earnings report, etc.).

    Returns:
        One `PageContent` per page, in document order, 1-indexed.

    Raises:
        FileNotFoundError: if `pdf_path` does not exist.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages: list[PageContent] = []
    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(PageContent(page_number=index + 1, text=text))
    return pages
