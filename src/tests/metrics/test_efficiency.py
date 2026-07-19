"""Tests for efficiency.py — Latency, Token Usage, Retrieval Time, Cost per Answer."""

from __future__ import annotations

import time

from finagent.data.schemas import LLMCallRecord, PipelineTrace, QueryComplexity
from finagent.metrics.efficiency import cost_per_answer, latency_seconds, retrieval_time_seconds, token_usage


def _call(input_tokens: int, output_tokens: int) -> LLMCallRecord:
    return LLMCallRecord(
        agent_name="test", model="m", system_prompt="s", user_prompt="u", raw_response="r",
        input_tokens=input_tokens, output_tokens=output_tokens, latency_seconds=0.1, temperature=0.0,
    )


def test_token_usage_sums_all_calls(sample_question):
    trace = PipelineTrace(experiment_id="EXP-11", question=sample_question, complexity_used=QueryComplexity.SIMPLE)
    trace.llm_calls = [_call(100, 50), _call(200, 80)]
    assert token_usage(trace) == 430


def test_token_usage_zero_with_no_calls(sample_question):
    trace = PipelineTrace(experiment_id="EXP-01", question=sample_question, complexity_used=QueryComplexity.SIMPLE)
    assert token_usage(trace) == 0


def test_latency_seconds_measures_elapsed_time(sample_question):
    trace = PipelineTrace(experiment_id="EXP-01", question=sample_question, complexity_used=QueryComplexity.SIMPLE)
    time.sleep(0.01)
    trace.mark_finished()
    assert latency_seconds(trace) >= 0.01


def test_retrieval_time_seconds():
    start = time.perf_counter()
    time.sleep(0.01)
    end = time.perf_counter()
    assert retrieval_time_seconds(None, start, end) >= 0.01


def test_cost_per_answer_positive_for_nonzero_tokens(sample_question):
    trace = PipelineTrace(experiment_id="EXP-11", question=sample_question, complexity_used=QueryComplexity.SIMPLE)
    trace.llm_calls = [_call(1000, 500)]
    assert cost_per_answer(trace) > 0.0


def test_cost_per_answer_zero_with_no_calls(sample_question):
    trace = PipelineTrace(experiment_id="EXP-01", question=sample_question, complexity_used=QueryComplexity.SIMPLE)
    assert cost_per_answer(trace) == 0.0
