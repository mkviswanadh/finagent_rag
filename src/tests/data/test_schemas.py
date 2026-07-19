"""Tests for schemas.py — PipelineTrace's derived properties (the ones with real logic)."""

from __future__ import annotations

import time

from finagent.data.schemas import EvidenceReference, LLMCallRecord, PipelineTrace, QueryComplexity


def _call(input_tokens: int, output_tokens: int) -> LLMCallRecord:
    return LLMCallRecord(
        agent_name="test", model="m", system_prompt="s", user_prompt="u", raw_response="r",
        input_tokens=input_tokens, output_tokens=output_tokens, latency_seconds=0.1, temperature=0.0,
    )


class TestPipelineTrace:
    def test_total_tokens_sums_all_calls(self, sample_question):
        trace = PipelineTrace(experiment_id="EXP-11", question=sample_question, complexity_used=QueryComplexity.SIMPLE)
        trace.llm_calls = [_call(100, 50), _call(200, 30)]
        assert trace.total_input_tokens == 300
        assert trace.total_output_tokens == 80
        assert trace.total_tokens == 380

    def test_latency_measured_after_mark_finished(self, sample_question):
        trace = PipelineTrace(experiment_id="EXP-11", question=sample_question, complexity_used=QueryComplexity.SIMPLE)
        time.sleep(0.01)
        trace.mark_finished()
        assert trace.total_latency_seconds >= 0.01

    def test_latency_still_computable_before_mark_finished(self, sample_question):
        trace = PipelineTrace(experiment_id="EXP-11", question=sample_question, complexity_used=QueryComplexity.SIMPLE)
        assert trace.total_latency_seconds >= 0.0

    def test_retrieved_and_filtered_chunk_ids(self, sample_question, sample_evidence_item):
        trace = PipelineTrace(experiment_id="EXP-11", question=sample_question, complexity_used=QueryComplexity.SIMPLE)
        trace.retrieved_evidence = [sample_evidence_item]
        trace.filtered_evidence = [sample_evidence_item]
        assert trace.retrieved_chunk_ids == ["MSFT_2022_10K_CH_001"]
        assert trace.filtered_chunk_ids == ["MSFT_2022_10K_CH_001"]


class TestFinanceBenchQuestion:
    def test_evidence_page_numbers_deduplicated_and_sorted(self, sample_question):
        sample_question.evidence = [
            EvidenceReference(doc_name="X", page_number=30, text=""),
            EvidenceReference(doc_name="X", page_number=10, text=""),
            EvidenceReference(doc_name="X", page_number=30, text=""),  # duplicate page
        ]
        assert sample_question.evidence_page_numbers == [10, 30]

    def test_evidence_text_combined_joins_all_excerpts(self, sample_question):
        sample_question.evidence = [
            EvidenceReference(doc_name="X", page_number=1, text="first excerpt"),
            EvidenceReference(doc_name="X", page_number=2, text="second excerpt"),
        ]
        combined = sample_question.evidence_text_combined
        assert "first excerpt" in combined
        assert "second excerpt" in combined

    def test_evidence_page_numbers_excludes_none(self, sample_question):
        sample_question.evidence = [
            EvidenceReference(doc_name="X", page_number=None, text=""),
            EvidenceReference(doc_name="X", page_number=5, text=""),
        ]
        assert sample_question.evidence_page_numbers == [5]
