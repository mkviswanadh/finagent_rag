"""Single entry point computing every applicable metric for one question's pipeline trace.

Every experiment runner (finagent-experiments skill §4) should call `compute_all_metrics` exactly
once per question rather than calling individual metric functions ad hoc — this is what guarantees
the `QuestionResult.metrics` dict has consistent keys across all 14 experiments, which the results
writer (`finagent.results.workbook_writer`) depends on to populate `Coding_Sheet_RESULTS.xlsx`'s
columns generically.

Not every metric applies to every experiment: Direct LLM experiments (EXP-01..06) have no
retrieval, so retrieval/grounding metrics that require evidence are simply omitted from the
returned dict rather than reported as a misleading `0.0` — callers must use `.get(key)` and treat
a missing key as "not applicable", not "zero".
"""

from __future__ import annotations

from finagent.config import RETRIEVAL_TOP_K
from finagent.data.schemas import FinanceBenchQuestion, PipelineTrace
from finagent.metrics.answer_quality import answer_relevance, exact_match, f1_score, semantic_similarity
from finagent.metrics.efficiency import cost_per_answer, latency_seconds, token_usage
from finagent.metrics.grounding import citation_correctness, evidence_coverage, faithfulness, hallucination_rate
from finagent.metrics.reasoning_metrics import (
    calculation_accuracy,
    explanation_completeness,
    multi_step_reasoning_score,
    numerical_accuracy,
)
from finagent.metrics.retrieval_metrics import context_precision, context_recall, hit_at_k, mean_reciprocal_rank


def compute_all_metrics(
    trace: PipelineTrace,
    question: FinanceBenchQuestion,
    *,
    top_k: int = RETRIEVAL_TOP_K,
) -> dict[str, float]:
    """Compute every metric applicable to a completed pipeline trace.

    Args:
        trace: The completed `PipelineTrace` for one question (must have `mark_finished` called,
            and `generated_answer` populated).
        question: The `FinanceBenchQuestion` this trace answers, providing the reference answer and
            ground-truth evidence.
        top_k: `k` for `hit_at_k`. Defaults to the fixed experimental retrieval depth.

    Returns:
        A dict of metric name -> score. Always includes Answer Quality (A) and Efficiency (E)
        metrics, since those require only the generated answer and the trace's own timing/token
        data. Evidence & Retrieval (B) and Grounding & Trust (C) metrics are included only if
        `trace.retrieved_evidence` is non-empty. Financial Reasoning (D) metrics are included only
        if `trace.reasoning_output` is present. `calculation_accuracy` is included only when the
        question's complexity analysis flagged `needs_calculation` — see
        `reasoning_metrics.calculation_accuracy`.
    """
    metrics: dict[str, float] = {}
    generated_answer = trace.generated_answer
    reference_answer = question.reference_answer

    # A. Answer Quality — always computable.
    metrics["answer_relevance"] = answer_relevance(question.question, generated_answer)
    metrics["exact_match"] = exact_match(generated_answer, reference_answer)
    metrics["f1_score"] = f1_score(generated_answer, reference_answer)
    metrics["semantic_similarity"] = semantic_similarity(generated_answer, reference_answer)

    # E. Efficiency — always computable from the trace itself.
    metrics["latency_seconds"] = latency_seconds(trace)
    metrics["token_usage"] = float(token_usage(trace))
    metrics["cost_per_answer_usd"] = cost_per_answer(trace)

    # B. Evidence & Retrieval — requires retrieval to have happened.
    if trace.retrieved_evidence:
        metrics["context_recall"] = context_recall(trace.retrieved_evidence, question)
        metrics["context_precision"] = context_precision(trace.retrieved_evidence, question)
        metrics["hit_at_k"] = hit_at_k(trace.retrieved_evidence, question, top_k)
        metrics["mrr"] = mean_reciprocal_rank(trace.retrieved_evidence, question)

    # C. Grounding & Trust — faithfulness/hallucination need a Verification result; evidence
    # coverage/citation correctness need a Reasoning result with citations.
    if trace.verification_result is not None:
        metrics["faithfulness"] = faithfulness(trace.verification_result, generated_answer)
        metrics["hallucination_rate"] = hallucination_rate(trace.verification_result, generated_answer)
    if trace.reasoning_output is not None and trace.retrieved_evidence:
        metrics["evidence_coverage"] = evidence_coverage(
            trace.reasoning_output.citations, trace.filtered_evidence or trace.retrieved_evidence, question
        )
        metrics["citation_correctness"] = citation_correctness(
            trace.reasoning_output.citations, trace.filtered_evidence or trace.retrieved_evidence, question
        )

    # D. Financial Reasoning.
    metrics["numerical_accuracy"] = numerical_accuracy(generated_answer, reference_answer)
    metrics["explanation_completeness"] = explanation_completeness(generated_answer, reference_answer)
    if trace.reasoning_output is not None:
        metrics["multi_step_reasoning_score"] = multi_step_reasoning_score(trace.reasoning_output, question)
    needs_calculation = trace.query_analysis is not None and trace.query_analysis.needs_calculation
    calc_accuracy = calculation_accuracy(generated_answer, reference_answer, needs_calculation=needs_calculation)
    if calc_accuracy is not None:
        metrics["calculation_accuracy"] = calc_accuracy

    return metrics
