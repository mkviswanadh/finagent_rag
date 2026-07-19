"""PDF text extraction (Proposal Table 7.4, Step 1: "PDF Text Extraction").

Uses `PyMuPDF` (the `fitz` module) — the same library FinanceBench's own authors used for their
published baselines (`notebooks_evaluation_playground_upstream.ipynb`, via LangChain's
`PyMuPDFLoader`). It requires no external poppler/`pdftoppm` binary (this development machine has
neither installed — see finagent-architecture skill §9), and its layout reconstruction keeps
multi-word headings intact on a single line (e.g. "Item 9A. Controls and Procedures.") where
`pypdf`'s extraction was observed to fragment the same heading across three separate lines
("Item 9A." / "Controls" / "and Procedures.") on real 10-K filings in this dataset — which broke
line-level heading detection in `section_detection.py`. This was verified directly against
`data/pdfs/3M_2018_10K.pdf` page 128 during implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


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

    pages: list[PageContent] = []
    with fitz.open(str(pdf_path)) as doc:
        for index, page in enumerate(doc):
            text = page.get_text() or ""
            pages.append(PageContent(page_number=index + 1, text=text))
    return pages
