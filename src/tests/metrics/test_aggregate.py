"""Tests for aggregate.py — compute_all_metrics, the single entry point every experiment uses."""

from __future__ import annotations

from finagent.data.schemas import (
    LLMCallRecord,
    PipelineTrace,
    QueryAnalysis,
    QueryComplexity,
    ReasoningOutput,
    VerificationResult,
)
from finagent.metrics.aggregate import compute_all_metrics


def _full_trace(sample_question, sample_evidence_item):
    trace = PipelineTrace(
        experiment_id="EXP-11", question=sample_question, complexity_used=QueryComplexity.SIMPLE
    )
    trace.retrieved_evidence = [sample_evidence_item]
    trace.filtered_evidence = [sample_evidence_item]
    trace.reasoning_output = ReasoningOutput(
        reasoning_steps=["s"], extracted_values={}, draft_answer="x", citations=["EV_001"],
        insufficient_evidence=False,
    )
    trace.verification_result = VerificationResult(passed=True, unsupported_claims=[], confidence=0.9, notes="")
    trace.generated_answer = "Microsoft's revenue in fiscal year 2022 was $198.3 billion."
    trace.query_analysis = QueryAnalysis(
        complexity=QueryComplexity.SIMPLE, company="Microsoft", year=2022, metric="revenue",
        question_type="lookup", needs_calculation=False, needs_multiple_evidence_chunks=False,
        needs_refinement=False, routing_rationale="",
    )
    trace.llm_calls.append(LLMCallRecord(
        agent_name="reasoning", model="m", system_prompt="s", user_prompt="u", raw_response="r",
        input_tokens=200, output_tokens=100, latency_seconds=0.5, temperature=0.0,
    ))
    trace.mark_finished()
    return trace


class TestComputeAllMetrics:
    def test_full_rag_trace_includes_all_5_metric_families(self, sample_question, sample_evidence_item):
        trace = _full_trace(sample_question, sample_evidence_item)
        metrics = compute_all_metrics(trace, sample_question)

        # A. Answer Quality
        for key in ("answer_relevance", "exact_match", "f1_score", "semantic_similarity"):
            assert key in metrics
        # B. Evidence & Retrieval
        for key in ("context_recall", "context_precision", "hit_at_k", "mrr"):
            assert key in metrics
        # C. Grounding & Trust
        for key in ("faithfulness", "hallucination_rate", "evidence_coverage", "citation_correctness"):
            assert key in metrics
        # D. Financial Reasoning
        for key in ("numerical_accuracy", "explanation_completeness", "multi_step_reasoning_score"):
            assert key in metrics
        # E. Efficiency
        for key in ("latency_seconds", "token_usage", "cost_per_answer_usd"):
            assert key in metrics

    def test_direct_llm_trace_omits_retrieval_and_grounding_metrics(self, sample_question):
        """A Direct LLM experiment (no retrieval at all) must not report retrieval/grounding
        metrics as a misleading 0.0 — they should be absent from the dict entirely."""
        trace = PipelineTrace(
            experiment_id="EXP-01", question=sample_question, complexity_used=QueryComplexity.SIMPLE
        )
        trace.generated_answer = "Microsoft's revenue was $198.3 billion."
        trace.llm_calls.append(LLMCallRecord(
            agent_name="EXP-01", model="m", system_prompt="s", user_prompt="u", raw_response="r",
            input_tokens=100, output_tokens=50, latency_seconds=0.2, temperature=0.0,
        ))
        trace.mark_finished()

        metrics = compute_all_metrics(trace, sample_question)

        assert "answer_relevance" in metrics
        assert "latency_seconds" in metrics
        for key in ("context_recall", "context_precision", "hit_at_k", "mrr", "faithfulness",
                    "evidence_coverage", "citation_correctness"):
            assert key not in metrics

    def test_calculation_accuracy_only_present_when_needed(self, sample_question, sample_evidence_item):
        trace = _full_trace(sample_question, sample_evidence_item)
        trace.query_analysis.needs_calculation = False
        metrics = compute_all_metrics(trace, sample_question)
        assert "calculation_accuracy" not in metrics

        trace.query_analysis.needs_calculation = True
        metrics = compute_all_metrics(trace, sample_question)
        assert "calculation_accuracy" in metrics

    def test_all_metric_values_are_floats_in_valid_range(self, sample_question, sample_evidence_item):
        trace = _full_trace(sample_question, sample_evidence_item)
        metrics = compute_all_metrics(trace, sample_question)
        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} is not a float: {value!r}"
            if key not in ("latency_seconds", "token_usage", "cost_per_answer_usd"):
                assert 0.0 <= value <= 1.0, f"{key}={value} out of [0,1] range"
