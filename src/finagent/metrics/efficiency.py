"""Efficiency metrics (Proposal Table 7.17).

All four are derived directly from a `PipelineTrace`'s recorded `LLMCallRecord`s — no additional
computation or API calls needed, since latency and token usage are captured at the source (every
`GroqClient.complete`/`complete_json` call already returns them — see `llm/groq_client.py`).
"""

from __future__ import annotations

from finagent.config import GROQ_PRICE_PER_MILLION_INPUT_TOKENS_USD, GROQ_PRICE_PER_MILLION_OUTPUT_TOKENS_USD
from finagent.data.schemas import PipelineTrace


def latency_seconds(trace: PipelineTrace) -> float:
    """Latency: total wall-clock time for one question's pipeline run.

    Formula (Proposal Table 7.17): `Response End Time - Response Start Time`.

    Args:
        trace: The completed pipeline trace (`PipelineTrace.mark_finished` must have been called).

    Returns:
        Elapsed seconds.
    """
    return trace.total_latency_seconds


def token_usage(trace: PipelineTrace) -> int:
    """Token Usage: total input + output tokens across every Groq call in this question's run.

    Formula (Proposal Table 7.17): `Input Tokens + Output Tokens`.

    Args:
        trace: The pipeline trace.

    Returns:
        Total tokens across all `LLMCallRecord`s in the trace.
    """
    return trace.total_tokens


def retrieval_time_seconds(trace: PipelineTrace, retrieval_started_at: float, retrieval_finished_at: float) -> float:
    """Retrieval Time: duration of the retrieval stage specifically.

    Formula (Proposal Table 7.17): `Retrieval End Time - Retrieval Start Time`. Retrieval timing is
    not tracked on `PipelineTrace` itself (retrieval makes no Groq call, so it isn't an
    `LLMCallRecord`) — callers (experiment runners) must time the retrieval call themselves and
    pass the boundaries in.

    Args:
        trace: The pipeline trace (unused directly, kept for a consistent metric-function signature
            across this module — accepted for API symmetry with the other three metrics here).
        retrieval_started_at: `time.perf_counter()` value at retrieval start.
        retrieval_finished_at: `time.perf_counter()` value at retrieval end.

    Returns:
        Elapsed seconds.
    """
    del trace  # unused — see docstring
    return retrieval_finished_at - retrieval_started_at


def cost_per_answer(trace: PipelineTrace) -> float:
    """Cost per Answer: estimated Groq API cost for this question's run.

    Formula (Proposal Table 7.17): `Total API/Compute Cost / Total Generated Answers` — this
    function returns the numerator for one question; averaging it across an experiment's questions
    produces the per-experiment "Cost per Answer" (the denominator is 1 for a single question).

    Args:
        trace: The pipeline trace.

    Returns:
        Estimated cost in USD, using `config.GROQ_PRICE_PER_MILLION_*_TOKENS_USD`.
    """
    input_cost = trace.total_input_tokens / 1_000_000 * GROQ_PRICE_PER_MILLION_INPUT_TOKENS_USD
    output_cost = trace.total_output_tokens / 1_000_000 * GROQ_PRICE_PER_MILLION_OUTPUT_TOKENS_USD
    return input_cost + output_cost
