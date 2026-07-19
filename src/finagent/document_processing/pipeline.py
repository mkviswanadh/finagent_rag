"""End-to-end document processing orchestration (Proposal §7.5, Table 7.4, all 7 steps).

`DocumentProcessingPipeline.process_pdf` runs one PDF through extraction → table detection →
cleaning → section detection → chunking → metadata tagging, and `ingest_directory` additionally
stores the result in ChromaDB — i.e. the full "Financial Document Processing & Knowledge Base
Preparation Workflow" (Figure 7.5.1) in one call. This is the single entry point every experiment's
ingestion step should use, so no experiment reimplements its own ad-hoc extraction path.
"""

from __future__ import annotations

import logging
from pathlib import Path

from finagent.config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, PDF_DIR
from finagent.data.schemas import Chunk
from finagent.document_processing.chunking import chunk_document
from finagent.document_processing.cleaning import clean_text
from finagent.document_processing.metadata import FilingMetadata, parse_filing_metadata
from finagent.document_processing.pdf_extraction import extract_pdf_pages
from finagent.document_processing.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class DocumentProcessingPipeline:
    """Runs the full financial-document-to-knowledge-base workflow (Proposal Figure 7.5.1)."""

    def __init__(self, vector_store: ChromaVectorStore | None = None) -> None:
        self._vector_store = vector_store or ChromaVectorStore()

    @property
    def vector_store(self) -> ChromaVectorStore:
        return self._vector_store

    def process_pdf(
        self,
        pdf_path: str | Path,
        *,
        metadata_override: FilingMetadata | None = None,
        chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
        chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    ) -> list[Chunk]:
        """Extract, clean, section-detect, and chunk one PDF — without storing it.

        Table extraction (Proposal Table 7.4 Step 2) does not produce a separate return value:
        table-like line runs are detected internally by the chunker's section-aware grouping
        (numeric-dense rows naturally cluster together within a section), which is sufficient for
        this pipeline's purpose of keeping table rows contiguous rather than reconstructing a
        structured cell grid (see `table_extraction` module docstring).

        Args:
            pdf_path: Path to the source financial PDF.
            metadata_override: Explicit company/year/report_type to use instead of parsing the
                filename — pass this when metadata is known from the FinanceBench QA dataset
                itself (Proposal Table 7.3 fields), which should always take precedence over
                filename parsing.
            chunk_size_tokens: See `chunk_document`.
            chunk_overlap_tokens: See `chunk_document`.

        Returns:
            The document's chunks, ready for storage or direct use.
        """
        pdf_path = Path(pdf_path)
        meta = metadata_override or parse_filing_metadata(pdf_path.name)

        pages = extract_pdf_pages(pdf_path)
        pages = clean_text(pages)
        chunks = chunk_document(
            pages,
            company=meta.company,
            year=meta.year,
            report_type=meta.report_type,
            source_document=pdf_path.name,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )
        logger.info(
            "Processed %s -> %d chunks (company=%s, year=%s, report_type=%s)",
            pdf_path.name,
            len(chunks),
            meta.company,
            meta.year,
            meta.report_type,
        )
        return chunks

    def ingest_directory(
        self,
        pdf_dir: str | Path = PDF_DIR,
        *,
        metadata_table: dict[str, FilingMetadata] | None = None,
    ) -> int:
        """Process and store every PDF in a directory into the shared ChromaDB collection.

        Args:
            pdf_dir: Directory containing FinanceBench PDFs.
            metadata_table: Optional mapping (filename stem -> `FilingMetadata`) produced by
                `load_filing_metadata_table`, used to override filename-based parsing per file
                when available.

        Returns:
            Total number of chunks stored across all processed PDFs.
        """
        pdf_dir = Path(pdf_dir)
        pdf_paths = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_paths:
            logger.warning("No PDFs found in %s", pdf_dir)
            return 0

        total_chunks = 0
        for pdf_path in pdf_paths:
            override = None
            if metadata_table is not None:
                override = metadata_table.get(pdf_path.stem)
            chunks = self.process_pdf(pdf_path, metadata_override=override)
            self._vector_store.add_chunks(chunks)
            total_chunks += len(chunks)

        logger.info("Ingested %d PDFs -> %d total chunks", len(pdf_paths), total_chunks)
        return total_chunks
