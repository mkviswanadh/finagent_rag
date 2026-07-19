"""Tests for QueryUnderstandingAgent — entity extraction, complexity classification, routing rules."""

from __future__ import annotations

from finagent.agents.query_understanding import QueryUnderstandingAgent, _apply_deterministic_routing_rules
from finagent.data.schemas import QueryComplexity


class TestAnalyze:
    def test_extracts_entities_and_complexity_from_llm_response(self, mock_groq_client):
        mock_groq_client.set_response("query_understanding", {
            "company": "Microsoft", "year": 2022, "metric": "revenue",
            "question_type": "lookup", "needs_calculation": False,
            "needs_multiple_evidence_chunks": False, "complexity": "Simple",
            "needs_refinement": False, "routing_rationale": "single lookup",
        })
        agent = QueryUnderstandingAgent(mock_groq_client)
        analysis, record = agent.analyze("What was Microsoft's revenue in 2022?")

        assert analysis.company == "Microsoft"
        assert analysis.year == 2022
        assert analysis.metric == "revenue"
        assert analysis.complexity == QueryComplexity.SIMPLE
        assert analysis.needs_refinement is False
        assert record.agent_name == "query_understanding"
        assert mock_groq_client.call_log == ["query_understanding"]

    def test_handles_null_entity_fields(self, mock_groq_client):
        mock_groq_client.set_response("query_understanding", {
            "company": None, "year": None, "metric": None,
            "question_type": "reasoning", "needs_calculation": False,
            "needs_multiple_evidence_chunks": True, "complexity": "Complex",
            "needs_refinement": True, "routing_rationale": "vague question",
        })
        agent = QueryUnderstandingAgent(mock_groq_client)
        analysis, _ = agent.analyze("Why did things change?")

        assert analysis.company is None
        assert analysis.year is None

    def test_invalid_question_type_falls_back_to_lookup(self, mock_groq_client):
        mock_groq_client.set_response("query_understanding", {
            "company": "X", "year": 2022, "metric": "m",
            "question_type": "not_a_real_type", "needs_calculation": False,
            "needs_multiple_evidence_chunks": False, "complexity": "Simple",
            "needs_refinement": False, "routing_rationale": "",
        })
        agent = QueryUnderstandingAgent(mock_groq_client)
        analysis, _ = agent.analyze("some question")
        assert analysis.question_type == "lookup"

    def test_makes_exactly_one_llm_call(self, mock_groq_client):
        mock_groq_client.set_response("query_understanding", {
            "company": "X", "year": 2022, "metric": "m", "question_type": "lookup",
            "needs_calculation": False, "needs_multiple_evidence_chunks": False,
            "complexity": "Simple", "needs_refinement": False, "routing_rationale": "",
        })
        agent = QueryUnderstandingAgent(mock_groq_client)
        agent.analyze("any question")
        assert len(mock_groq_client.call_log) == 1


class TestDeterministicRoutingRules:
    """Table 7.9 routing rules must be enforced even if the LLM under-calls complexity."""

    def test_escalates_to_complex_on_why_keyword(self):
        result = _apply_deterministic_routing_rules(
            question="Why did revenue increase?", llm_complexity=QueryComplexity.SIMPLE,
            needs_calculation=False, needs_multiple_evidence_chunks=False,
        )
        assert result == QueryComplexity.COMPLEX

    def test_escalates_to_complex_on_explain_keyword(self):
        result = _apply_deterministic_routing_rules(
            question="Explain the change in operating income.", llm_complexity=QueryComplexity.MODERATE,
            needs_calculation=False, needs_multiple_evidence_chunks=False,
        )
        assert result == QueryComplexity.COMPLEX

    def test_escalates_to_moderate_on_multiple_years(self):
        # Deliberately avoids "compare"/"why"/"explain" etc. so only the multi-year rule fires,
        # not the separate Complex-trigger-keyword rule (see test_escalates_to_complex_* above).
        result = _apply_deterministic_routing_rules(
            question="What was revenue in 2021 and 2022?", llm_complexity=QueryComplexity.SIMPLE,
            needs_calculation=False, needs_multiple_evidence_chunks=False,
        )
        assert result == QueryComplexity.MODERATE

    def test_escalates_on_needs_multiple_evidence_chunks(self):
        result = _apply_deterministic_routing_rules(
            question="What was the revenue?", llm_complexity=QueryComplexity.SIMPLE,
            needs_calculation=False, needs_multiple_evidence_chunks=True,
        )
        assert result == QueryComplexity.COMPLEX

    def test_escalates_on_needs_calculation(self):
        result = _apply_deterministic_routing_rules(
            question="What was the revenue?", llm_complexity=QueryComplexity.SIMPLE,
            needs_calculation=True, needs_multiple_evidence_chunks=False,
        )
        assert result == QueryComplexity.MODERATE

    def test_never_downgrades_llm_complex_judgment(self):
        """The LLM's own Complex judgment must never be downgraded, even with no trigger keywords."""
        result = _apply_deterministic_routing_rules(
            question="What was the revenue in 2022?", llm_complexity=QueryComplexity.COMPLEX,
            needs_calculation=False, needs_multiple_evidence_chunks=False,
        )
        assert result == QueryComplexity.COMPLEX

    def test_simple_question_stays_simple(self):
        result = _apply_deterministic_routing_rules(
            question="What was Microsoft's revenue in 2022?", llm_complexity=QueryComplexity.SIMPLE,
            needs_calculation=False, needs_multiple_evidence_chunks=False,
        )
        assert result == QueryComplexity.SIMPLE
