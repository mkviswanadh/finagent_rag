"""Tests for RetrievalAgent — zero-LLM-call vector search, multi-query merge, metadata filter builder."""

from __future__ import annotations

from finagent.agents.retrieval import RetrievalAgent
from finagent.data.schemas import Chunk, QueryAnalysis, QueryComplexity


def _make_analysis(company=None, year=None) -> QueryAnalysis:
    return QueryAnalysis(
        complexity=QueryComplexity.SIMPLE, company=company, year=year, metric=None,
        question_type="lookup", needs_calculation=False, needs_multiple_evidence_chunks=False,
        needs_refinement=False, routing_rationale="",
    )


class TestRetrieve:
    def test_returns_results_from_vector_store(self, populated_vector_store):
        agent = RetrievalAgent(populated_vector_store)
        results = agent.retrieve("Microsoft revenue", top_k=3)
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "MSFT_2022_10K_CH_001"

    def test_makes_no_llm_calls(self, populated_vector_store):
        """RetrievalAgent has no LLM client at all — this is a structural guarantee, not a mock check."""
        agent = RetrievalAgent(populated_vector_store)
        assert not hasattr(agent, "_llm")


class TestRetrieveMulti:
    def test_merges_and_dedups_across_query_variants(self, temp_vector_store):
        chunks = [
            Chunk(chunk_id="C1", company="X", year=2022, report_type="10-K", section="S",
                  page_number=1, text="Revenue was strong this year.", source_document="X.pdf"),
            Chunk(chunk_id="C2", company="X", year=2022, report_type="10-K", section="S",
                  page_number=2, text="Expenses increased due to inflation.", source_document="X.pdf"),
        ]
        temp_vector_store.add_chunks(chunks)
        agent = RetrievalAgent(temp_vector_store)

        results = agent.retrieve_multi(["revenue", "expenses", "revenue again"], top_k_per_query=2)

        chunk_ids = [r.chunk.chunk_id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))  # no duplicates despite overlapping variants

    def test_sorted_by_descending_relevance(self, temp_vector_store):
        chunks = [
            Chunk(chunk_id=f"C{i}", company="X", year=2022, report_type="10-K", section="S",
                  page_number=i, text=f"Some financial content number {i}.", source_document="X.pdf")
            for i in range(5)
        ]
        temp_vector_store.add_chunks(chunks)
        agent = RetrievalAgent(temp_vector_store)
        results = agent.retrieve_multi(["financial content"], top_k_per_query=5)
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)


class TestBuildMetadataFilter:
    def test_returns_none_with_no_entities(self):
        assert RetrievalAgent.build_metadata_filter(_make_analysis()) is None

    def test_single_condition_company_only(self):
        result = RetrievalAgent.build_metadata_filter(_make_analysis(company="Apple"))
        assert result == {"company": "Apple"}

    def test_single_condition_year_only(self):
        result = RetrievalAgent.build_metadata_filter(_make_analysis(year=2022))
        assert result == {"year": 2022}

    def test_and_condition_company_and_year(self):
        result = RetrievalAgent.build_metadata_filter(_make_analysis(company="Apple", year=2022))
        assert result == {"$and": [{"company": "Apple"}, {"year": 2022}]}
