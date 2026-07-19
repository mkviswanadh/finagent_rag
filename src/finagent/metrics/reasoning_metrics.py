"""Financial Reasoning metrics (Proposal Table 7.16).

Numerical Accuracy and Calculation Accuracy are computed via regex-extracted numeric comparison
(zero Groq calls) — a genuine approximation of the proposal's "Correct Numerical Values / Total
Numerical Values" formula, not exact claim-level grading, since that would require an LLM judge on
every question. Multi-step Reasoning Score and Explanation Completeness are structural/lexical
proxies over the Reasoning Agent's own output, documented as such rather than claimed to be exact.
This mirrors how `document_processing`'s heuristics are framed throughout this codebase: best-effort
and clearly labeled, not silently over-claiming precision.
"""

from __future__ import annotations

import math

from finagent.data.schemas import FinanceBenchQuestion, ReasoningOutput
from finagent.metrics.text_normalization import extract_numbers, normalize_answer

_RELATIVE_TOLERANCE = 0.01  # 1% — tolerates minor rounding differences in stated financial figures


def _numbers_match(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_RELATIVE_TOLERANCE, abs_tol=1e-6)


def numerical_accuracy(generated_answer: str, reference_answer: str) -> float:
    """Numerical Accuracy: fraction of the reference answer's numeric values reproduced correctly.

    Formula (Proposal Table 7.16): `Correct Numerical Values / Total Numerical Values`. A reference
    number counts as "correct" if some number in the generated answer matches it within 1% relative
    tolerance (financial figures are often reported with minor rounding: "$394.3B" vs "$394.32B").

    Args:
        generated_answer: The system's generated answer.
        reference_answer: The FinanceBench ground-truth answer.

    Returns:
        Accuracy in `[0, 1]`. `1.0` if the reference answer contains no numbers (nothing numeric to
        get wrong) — callers computing an experiment-wide average should be aware this means
        non-numeric questions trivially score 1.0 on this metric, not that a system did well on
        numerical extraction for them.
    """
    reference_numbers = extract_numbers(reference_answer)
    if not reference_numbers:
        return 1.0
    generated_numbers = extract_numbers(generated_answer)
    if not generated_numbers:
        return 0.0

    correct = sum(
        1 for ref_num in reference_numbers if any(_numbers_match(ref_num, gen_num) for gen_num in generated_numbers)
    )
    return correct / len(reference_numbers)


def calculation_accuracy(
    generated_answer: str, reference_answer: str, *, needs_calculation: bool
) -> float | None:
    """Calculation Accuracy: numerical accuracy specifically for calculation-requiring questions.

    Formula (Proposal Table 7.16): `Correct Calculations / Total Calculations`. Uses the same
    numeric-matching mechanism as `numerical_accuracy` — the distinction this metric adds is scope
    (only calculation-flagged questions), not a different comparison method.

    Args:
        generated_answer: The system's generated answer.
        reference_answer: The FinanceBench ground-truth answer.
        needs_calculation: Whether the Query Understanding Agent flagged this question as requiring
            arithmetic (`QueryAnalysis.needs_calculation`).

    Returns:
        Accuracy in `[0, 1]`, or `None` if `needs_calculation` is `False` — this metric is only
        meaningful for calculation questions; callers aggregating across an experiment should
        average only the non-`None` values, not treat `None` as `0`.
    """
    if not needs_calculation:
        return None
    return numerical_accuracy(generated_answer, reference_answer)


def multi_step_reasoning_score(reasoning_output: ReasoningOutput, question: FinanceBenchQuestion) -> float:
    """Multi-step Reasoning Score: how much of the question's required evidence was actually combined.

    Formula (Proposal Table 7.16): `Correct Reasoning Steps / Total Reasoning Steps`. Approximated
    here as the fraction of the question's ground-truth evidence points (`len(question.evidence)`,
    the number of annotated excerpts — a proxy for how many distinct pieces of information the
    question genuinely requires) that the Reasoning Agent actually cited, capped at 1.0. This
    measures reasoning breadth (did it draw on enough distinct evidence) rather than step-by-step
    logical correctness, which would require an LLM judge to assess.

    Args:
        reasoning_output: Output of `ReasoningAgent.reason`.
        question: The FinanceBench question, providing the ground-truth evidence-point count.

    Returns:
        Score in `[0, 1]`.
    """
    expected_points = max(1, len(question.evidence))
    actual_points = len(set(reasoning_output.citations))
    return min(1.0, actual_points / expected_points)


def explanation_completeness(generated_answer: str, reference_answer: str) -> float:
    """Explanation Completeness: how much of the reference answer's content is covered.

    Formula (Proposal Table 7.16): `Covered Reasoning Points / Total Expected Reasoning Points`.
    Approximated as token-level recall of the reference answer's normalized tokens found in the
    generated answer (i.e. F1's recall component in isolation, without penalizing extra explanatory
    detail the way precision/F1 would) — completeness cares about not omitting content, not about
    verbosity.

    Args:
        generated_answer: The system's generated answer.
        reference_answer: The FinanceBench ground-truth answer.

    Returns:
        Recall in `[0, 1]`. `0.0` if the reference answer normalizes to no tokens.
    """
    ref_tokens = normalize_answer(reference_answer).split()
    if not ref_tokens:
        return 0.0
    gen_tokens = set(normalize_answer(generated_answer).split())

    ref_counts: dict[str, int] = {}
    for tok in ref_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1

    covered = sum(count for tok, count in ref_counts.items() if tok in gen_tokens)
    return covered / len(ref_tokens)
