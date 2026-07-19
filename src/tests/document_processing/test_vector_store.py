"""Tests for vector_store.py — ChromaDB storage, semantic retrieval, and metadata filtering."""

from __future__ import annotations

from finagent.data.schemas import Chunk


def test_add_and_count(temp_vector_store, sample_chunk):
    assert temp_vector_store.count() == 0
    temp_vector_store.add_chunks([sample_chunk])
    assert temp_vector_store.count() == 1


def test_add_chunks_empty_list_is_noop(temp_vector_store):
    temp_vector_store.add_chunks([])
    assert temp_vector_store.count() == 0


def test_upsert_is_idempotent(temp_vector_store, sample_chunk):
    """Re-adding a chunk with the same ID overwrites rather than duplicating."""
    temp_vector_store.add_chunks([sample_chunk])
    temp_vector_store.add_chunks([sample_chunk])
    assert temp_vector_store.count() == 1


def test_query_empty_store_returns_no_results(temp_vector_store):
    assert temp_vector_store.query("any query", top_k=5) == []


def test_query_returns_relevant_chunk_first(populated_vector_store):
    results = populated_vector_store.query("What was Microsoft's revenue?", top_k=3)
    assert len(results) >= 1
    assert results[0].chunk.chunk_id == "MSFT_2022_10K_CH_001"
    assert 0.0 <= results[0].relevance_score <= 1.0


def test_query_with_matching_metadata_filter_returns_result(populated_vector_store):
    results = populated_vector_store.query(
        "revenue", top_k=3, metadata_filter={"company": "Microsoft"}
    )
    assert len(results) == 1


def test_query_with_non_matching_metadata_filter_returns_nothing(populated_vector_store):
    results = populated_vector_store.query(
        "revenue", top_k=3, metadata_filter={"company": "Apple"}
    )
    assert results == []


def test_query_respects_top_k(temp_vector_store):
    chunks = [
        Chunk(
            chunk_id=f"C{i}", company="X", year=2022, report_type="10-K", section="Notes",
            page_number=i, text=f"Some financial narrative text number {i}.", source_document="X.pdf",
        )
        for i in range(10)
    ]
    temp_vector_store.add_chunks(chunks)
    results = temp_vector_store.query("financial narrative", top_k=3)
    assert len(results) == 3
