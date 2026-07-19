"""Tests for QueryRefinementAgent — single rewrite and multi-query expansion."""

from __future__ import annotations

from finagent.agents.query_refinement import QueryRefinementAgent
from finagent.data.schemas import QueryAnalysis, QueryComplexity

_ANALYSIS = QueryAnalysis(
    complexity=QueryComplexity.COMPLEX, company="Apple", year=2022, metric="operating income",
    question_type="explanation", needs_calculation=True, needs_multiple_evidence_chunks=True,
    needs_refinement=True, routing_rationale="test",
)


class TestRefine:
    def test_returns_rewritten_query(self, mock_groq_client):
        mock_groq_client.set_response(
            "query_refinement", "What was Apple's operating income for fiscal year 2022?"
        )
        agent = QueryRefinementAgent(mock_groq_client)
        rewritten, record = agent.refine("apple money made 2022", _ANALYSIS)

        assert rewritten == "What was Apple's operating income for fiscal year 2022?"
        assert record.agent_name == "query_refinement"

    def test_falls_back_to_original_on_empty_response(self, mock_groq_client):
        mock_groq_client.set_response("query_refinement", "   ")
        agent = QueryRefinementAgent(mock_groq_client)
        rewritten, _ = agent.refine("original question", _ANALYSIS)
        assert rewritten == "original question"

    def test_makes_exactly_one_call(self, mock_groq_client):
        mock_groq_client.set_response("query_refinement", "rewritten")
        agent = QueryRefinementAgent(mock_groq_client)
        agent.refine("q", _ANALYSIS)
        assert mock_groq_client.call_log == ["query_refinement"]


class TestExpandMultiQuery:
    def test_returns_n_variants(self, mock_groq_client):
        mock_groq_client.set_response("query_refinement_multi", {
            "variants": ["variant one", "variant two", "variant three"]
        })
        agent = QueryRefinementAgent(mock_groq_client)
        variants, record = agent.expand_multi_query("original question", n=3)

        assert variants == ["variant one", "variant two", "variant three"]
        assert record.agent_name == "query_refinement_multi"

    def test_pads_short_variant_list_with_original_question(self, mock_groq_client):
        mock_groq_client.set_response("query_refinement_multi", {"variants": ["only one"]})
        agent = QueryRefinementAgent(mock_groq_client)
        variants, _ = agent.expand_multi_query("original question", n=3)

        assert len(variants) == 3
        assert variants[0] == "only one"
        assert variants[1] == "original question"

    def test_falls_back_to_n_copies_on_malformed_response(self, mock_groq_client):
        mock_groq_client.set_response("query_refinement_multi", {"not_variants_key": []})
        agent = QueryRefinementAgent(mock_groq_client)
        variants, _ = agent.expand_multi_query("original question", n=3)
        assert variants == ["original question"] * 3

    def test_truncates_excess_variants(self, mock_groq_client):
        mock_groq_client.set_response("query_refinement_multi", {
            "variants": ["v1", "v2", "v3", "v4", "v5"]
        })
        agent = QueryRefinementAgent(mock_groq_client)
        variants, _ = agent.expand_multi_query("q", n=3)
        assert len(variants) == 3

    def test_makes_exactly_one_call(self, mock_groq_client):
        mock_groq_client.set_response("query_refinement_multi", {"variants": ["a", "b", "c"]})
        agent = QueryRefinementAgent(mock_groq_client)
        agent.expand_multi_query("q", n=3)
        assert mock_groq_client.call_log == ["query_refinement_multi"]
