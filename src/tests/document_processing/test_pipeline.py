"""Tests for pipeline.py — the end-to-end DocumentProcessingPipeline orchestration."""

from __future__ import annotations

import pytest

from finagent.document_processing.metadata import FilingMetadata
from finagent.document_processing.pipeline import DocumentProcessingPipeline
from tests.conftest import PDF_DIR, requires_real_pdfs


@requires_real_pdfs
def test_process_pdf_real_document_produces_chunks(temp_vector_store):
    pipeline = DocumentProcessingPipeline(vector_store=temp_vector_store)
    chunks = pipeline.process_pdf(PDF_DIR / "3M_2018_10K.pdf")

    assert len(chunks) > 100  # a 160-page 10-K should produce well over 100 chunks
    assert all(c.company == "3M" for c in chunks)
    assert all(c.year == 2018 for c in chunks)
    assert all(c.text.strip() for c in chunks)
    # Chunk IDs must be unique.
    assert len({c.chunk_id for c in chunks}) == len(chunks)


@requires_real_pdfs
def test_process_pdf_metadata_override_takes_precedence(temp_vector_store):
    pipeline = DocumentProcessingPipeline(vector_store=temp_vector_store)
    override = FilingMetadata(company="Minnesota Mining", year=1999, report_type="Special Filing", document_name="x")
    chunks = pipeline.process_pdf(PDF_DIR / "3M_2018_10K.pdf", metadata_override=override)

    assert all(c.company == "Minnesota Mining" for c in chunks)
    assert all(c.year == 1999 for c in chunks)
    assert all(c.report_type == "Special Filing" for c in chunks)


@requires_real_pdfs
def test_ingest_directory_stores_chunks_in_vector_store(temp_vector_store, tmp_path):
    """Ingest a single-PDF subdirectory to keep the test fast, rather than all 368 PDFs."""
    import shutil

    single_pdf_dir = tmp_path / "single_pdf"
    single_pdf_dir.mkdir()
    shutil.copy(PDF_DIR / "APPLE_2015_10K.pdf", single_pdf_dir / "APPLE_2015_10K.pdf")

    pipeline = DocumentProcessingPipeline(vector_store=temp_vector_store)
    total = pipeline.ingest_directory(single_pdf_dir)

    assert total > 0
    assert temp_vector_store.count() == total


def test_ingest_directory_empty_dir_returns_zero(temp_vector_store, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    pipeline = DocumentProcessingPipeline(vector_store=temp_vector_store)
    assert pipeline.ingest_directory(empty_dir) == 0
