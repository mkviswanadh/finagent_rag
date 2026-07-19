"""Deterministic, zero-Groq-cost cross-experiment analysis (finagent-experiments skill §3c).

`ReportWriter.write_comparative_ranking_row` was built anticipating exactly this: "the scores here
are derived composites... the caller (a final synthesis pass, not a per-experiment run) is
responsible for computing them from the other three sheets before calling this method." This module
is that synthesis pass — run it once, after all 14 experiments have produced their `QuestionResult`
lists, to get per-experiment composite scores, a final ranking, ablation-value findings (does Query
Refinement/Evidence Filtering/Verification actually earn its cost? — Research Question 4), and
plain-language issue flags (e.g. "most answers are the identical fallback string — a retrieval
problem, not a reasoning one").

Everything here is pure arithmetic over metrics `compute_all_metrics` already produced — no LLM
call, no new Groq cost, safe to re-run as many times as useful while interpreting results.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from finagent.data.schemas import QueryComplexity, QuestionResult
from finagent.experiments.registry import EXPERIMENT_NAMES

# Metrics whose family average should use (1 - value) instead of value, because for these lower is
# better — "grounding_score" must reward LOW hallucination, not average in the raw rate.
_INVERTED_METRICS = {"hallucination_rate"}

_ANSWER_QUALITY_METRICS = ["answer_relevance", "exact_match", "f1_score", "semantic_similarity"]
_RETRIEVAL_METRICS = ["context_recall", "context_precision", "hit_at_k", "mrr"]
_GROUNDING_METRICS = ["faithfulness", "hallucination_rate", "evidence_coverage", "citation_correctness"]
_REASONING_METRICS = ["numerical_accuracy", "calculation_accuracy", "multi_step_reasoning_score", "explanation_completeness"]
_EFFICIENCY_METRICS = ["latency_seconds", "token_usage", "cost_per_answer_usd"]  # lower is better, normalized separately

# A question fails "retrievably" if its recall is this low or its answer collapses to the fixed
# insufficient-evidence fallback — thresholds chosen loosely (not tuned against a labeled set),
# meant to flag patterns worth a human look, not to make a final judgment automatically.
_HIGH_FALLBACK_RATE_THRESHOLD = 0.4
_LOW_RECALL_THRESHOLD = 0.3
_HIGH_PRECISION_LOW_RECALL_GAP = 0.4
_HIGH_HALLUCINATION_THRESHOLD = 0.3
_COMPLEX_VS_SIMPLE_DEGRADATION_THRESHOLD = 0.25

_BASELINE_DIRECT_LLM = "EXP-01"
_BASELINE_NAIVE_RAG = "EXP-07"
_ABLATION_PAIRS = {
    "EXP-12": ("EXP-11", "Query Refinement"),
    "EXP-13": ("EXP-11", "Evidence Filtering"),
    "EXP-14": ("EXP-11", "Verification"),
}


@dataclass
class FamilyScores:
    """Composite family scores, each 0-1 (or `None` if the family doesn't apply, e.g. Retrieval
    for a Direct LLM experiment) — higher is always better here, including `grounding` (which
    already folds in `1 - hallucination_rate`, not the raw rate)."""

    answer_quality: float | None = None
    retrieval_quality: float | None = None
    grounding: float | None = None
    financial_reasoning: float | None = None
    efficiency: float | None = None


@dataclass
class ExperimentAnalysis:
    exp_id: str
    experiment_name: str
    n_questions: int
    n_ok: int
    n_failed: int
    fallback_rate: float
    family_scores: FamilyScores
    overall_weighted_score: float | None = None
    improvement_over_naive_rag: float | None = None
    improvement_over_direct_llm: float | None = None
    final_rank: int | None = None
    main_finding: str = ""
    issues: list[str] = field(default_factory=list)


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _family_average(metrics_list: list[dict[str, float]], metric_keys: list[str]) -> float | None:
    """Mean, across questions and metric keys, of whichever `metric_keys` are actually present —
    inverting the ones in `_INVERTED_METRICS` first so every value contributing to the average is
    on a consistent "higher is better" scale."""
    values: list[float] = []
    for metrics in metrics_list:
        for key in metric_keys:
            if key not in metrics:
                continue
            value = metrics[key]
            values.append(1.0 - value if key in _INVERTED_METRICS else value)
    return _mean(values)


def _fallback_rate(results: list[QuestionResult]) -> float:
    """Fraction of successfully-answered questions sharing the single most common answer text —
    high values indicate the pipeline collapsed to a fixed fallback (e.g. "insufficient evidence")
    rather than genuinely answering most questions, which raw Answer Quality metrics alone don't
    surface (a fallback string can still score non-zero relevance/similarity against some
    references)."""
    answers = [r.trace.generated_answer for r in results if r.trace.generated_answer]
    if not answers:
        return 0.0
    most_common_count = Counter(answers).most_common(1)[0][1]
    return most_common_count / len(answers)


def _detect_issues(exp_id: str, results: list[QuestionResult], scores: FamilyScores, fallback_rate: float) -> list[str]:
    issues: list[str] = []

    if fallback_rate >= _HIGH_FALLBACK_RATE_THRESHOLD:
        issues.append(
            f"{fallback_rate:.0%} of answered questions share the identical answer text — likely a "
            "fixed fallback (e.g. 'insufficient evidence'), meaning this is predominantly a "
            "retrieval/evidence problem, not a reasoning or generation one."
        )

    if scores.retrieval_quality is not None:
        recalls = [r.metrics.get("context_recall") for r in results if "context_recall" in r.metrics]
        precisions = [r.metrics.get("context_precision") for r in results if "context_precision" in r.metrics]
        avg_recall, avg_precision = _mean(recalls), _mean(precisions)
        if avg_recall is not None and avg_recall < _LOW_RECALL_THRESHOLD:
            issues.append(
                f"Low context recall ({avg_recall:.2f}) — retrieval is missing the annotated evidence "
                "on most questions; check chunking granularity, top_k, or metadata filtering before "
                "trusting downstream reasoning/grounding scores."
            )
        if (
            avg_recall is not None and avg_precision is not None
            and avg_precision - avg_recall > _HIGH_PRECISION_LOW_RECALL_GAP
        ):
            issues.append(
                f"Precision ({avg_precision:.2f}) far exceeds recall ({avg_recall:.2f}) — retrieval finds "
                "few chunks but they're usually right; increasing top_k may recover more evidence "
                "without a large precision cost."
            )

    if scores.grounding is not None:
        hallucination_rates = [r.metrics.get("hallucination_rate") for r in results if "hallucination_rate" in r.metrics]
        avg_hallucination = _mean(hallucination_rates)
        if avg_hallucination is not None and avg_hallucination > _HIGH_HALLUCINATION_THRESHOLD:
            issues.append(
                f"High hallucination rate ({avg_hallucination:.2f}) — verification and/or evidence "
                "filtering may not be catching unsupported claims; review VerificationAgent output for "
                "this experiment's questions."
            )

    complexity_scores: dict[QueryComplexity, list[float]] = {}
    for r in results:
        if r.trace.complexity_used is None or "answer_relevance" not in r.metrics:
            continue
        complexity_scores.setdefault(r.trace.complexity_used, []).append(r.metrics["answer_relevance"])
    simple_avg = _mean(complexity_scores.get(QueryComplexity.SIMPLE, []))
    complex_avg = _mean(complexity_scores.get(QueryComplexity.COMPLEX, []))
    if simple_avg is not None and complex_avg is not None and simple_avg - complex_avg > _COMPLEX_VS_SIMPLE_DEGRADATION_THRESHOLD:
        issues.append(
            f"Answer relevance drops sharply from Simple ({simple_avg:.2f}) to Complex ({complex_avg:.2f}) "
            "questions — adaptive routing/refinement isn't fully compensating for question difficulty."
        )

    return issues


def _main_finding(analysis: ExperimentAnalysis) -> str:
    s = analysis.family_scores
    named = {
        "Answer Quality": s.answer_quality,
        "Retrieval Quality": s.retrieval_quality,
        "Grounding": s.grounding,
        "Financial Reasoning": s.financial_reasoning,
        "Efficiency": s.efficiency,
    }
    present = {name: value for name, value in named.items() if value is not None}
    if not present:
        return "No metrics computed (all questions failed)."
    best = max(present, key=present.get)
    worst = min(present, key=present.get)
    finding = f"Strongest in {best} ({present[best]:.2f}), weakest in {worst} ({present[worst]:.2f})."
    if analysis.issues:
        finding += f" {analysis.issues[0]}"
    return finding


class ResultsAnalyst:
    """Computes cross-experiment composite scores, rankings, ablation value, and issue flags —
    the "final synthesis pass" the `'Final Comparative Ranking and A'` sheet was designed for.

    Call `analyze()` once, after every experiment has produced its `QuestionResult` list.
    """

    def analyze(self, results_by_experiment: dict[str, list[QuestionResult]]) -> list[ExperimentAnalysis]:
        """Args:
            results_by_experiment: exp_id -> every `QuestionResult` from that experiment's run.

        Returns:
            One `ExperimentAnalysis` per key in `results_by_experiment`, with `final_rank` set
            relative to every experiment passed in (not just a subset), sorted by descending
            `overall_weighted_score` (`None` scores sort last).
        """
        raw_efficiency: dict[str, dict[str, float | None]] = {}
        analyses: dict[str, ExperimentAnalysis] = {}

        for exp_id, results in results_by_experiment.items():
            ok_results = [r for r in results if r.trace.generated_answer]
            metrics_list = [r.metrics for r in ok_results]

            scores = FamilyScores(
                answer_quality=_family_average(metrics_list, _ANSWER_QUALITY_METRICS),
                retrieval_quality=_family_average(metrics_list, _RETRIEVAL_METRICS),
                grounding=_family_average(metrics_list, _GROUNDING_METRICS),
                financial_reasoning=_family_average(metrics_list, _REASONING_METRICS),
            )
            raw_efficiency[exp_id] = {
                key: _mean([m[key] for m in metrics_list if key in m]) for key in _EFFICIENCY_METRICS
            }

            fallback_rate = _fallback_rate(ok_results)
            analysis = ExperimentAnalysis(
                exp_id=exp_id,
                experiment_name=EXPERIMENT_NAMES.get(exp_id, exp_id),
                n_questions=len(results),
                n_ok=len(ok_results),
                n_failed=len(results) - len(ok_results),
                fallback_rate=fallback_rate,
                family_scores=scores,
            )
            analysis.issues = _detect_issues(exp_id, ok_results, scores, fallback_rate)
            analyses[exp_id] = analysis

        self._apply_efficiency_scores(analyses, raw_efficiency)
        self._apply_overall_scores(analyses)
        self._apply_improvements(analyses)
        self._apply_ranking(analyses)
        for analysis in analyses.values():
            analysis.main_finding = _main_finding(analysis)

        return sorted(
            analyses.values(),
            key=lambda a: (a.overall_weighted_score is None, -(a.overall_weighted_score or 0.0)),
        )

    @staticmethod
    def _apply_efficiency_scores(
        analyses: dict[str, ExperimentAnalysis], raw_efficiency: dict[str, dict[str, float | None]]
    ) -> None:
        """Min-max normalize each efficiency metric across every experiment being compared (lower
        raw latency/tokens/cost -> higher normalized score), then average the normalized components
        into one 0-1 `efficiency` family score per experiment. Normalization is necessarily
        relative to whatever set of experiments is passed to `analyze()` in one call — re-running
        with a different subset changes the 0-1 scale, though relative ordering within that call
        does not."""
        normalized_by_experiment: dict[str, list[float]] = {exp_id: [] for exp_id in analyses}

        for metric_key in _EFFICIENCY_METRICS:
            raw_values = {
                exp_id: values[metric_key] for exp_id, values in raw_efficiency.items() if values[metric_key] is not None
            }
            if not raw_values:
                continue
            lo, hi = min(raw_values.values()), max(raw_values.values())
            for exp_id, value in raw_values.items():
                normalized = 1.0 if hi == lo else 1.0 - (value - lo) / (hi - lo)
                normalized_by_experiment[exp_id].append(normalized)

        for exp_id, normalized_values in normalized_by_experiment.items():
            analyses[exp_id].family_scores.efficiency = _mean(normalized_values)

    @staticmethod
    def _apply_overall_scores(analyses: dict[str, ExperimentAnalysis]) -> None:
        """Overall score = unweighted mean of whichever family scores this experiment has (Direct
        LLM experiments have no Retrieval family, so their overall score is averaged over fewer,
        different families than RAG experiments' — a real limitation of comparing categories this
        different, not a bug; see the module docstring / final_rank caveat."""
        for analysis in analyses.values():
            s = analysis.family_scores
            analysis.overall_weighted_score = _mean(
                [s.answer_quality, s.retrieval_quality, s.grounding, s.financial_reasoning, s.efficiency]
            )

    @staticmethod
    def _apply_improvements(analyses: dict[str, ExperimentAnalysis]) -> None:
        direct_llm_score = analyses[_BASELINE_DIRECT_LLM].overall_weighted_score if _BASELINE_DIRECT_LLM in analyses else None
        naive_rag_score = analyses[_BASELINE_NAIVE_RAG].overall_weighted_score if _BASELINE_NAIVE_RAG in analyses else None

        for exp_id, analysis in analyses.items():
            if analysis.overall_weighted_score is None:
                continue
            if direct_llm_score:
                analysis.improvement_over_direct_llm = (analysis.overall_weighted_score - direct_llm_score) / direct_llm_score
            if naive_rag_score and exp_id != _BASELINE_NAIVE_RAG:
                analysis.improvement_over_naive_rag = (analysis.overall_weighted_score - naive_rag_score) / naive_rag_score

    @staticmethod
    def _apply_ranking(analyses: dict[str, ExperimentAnalysis]) -> None:
        ranked = sorted(
            analyses.values(),
            key=lambda a: (a.overall_weighted_score is None, -(a.overall_weighted_score or 0.0)),
        )
        for rank, analysis in enumerate(ranked, start=1):
            analysis.final_rank = rank if analysis.overall_weighted_score is not None else None

    def ablation_findings(self, analyses: list[ExperimentAnalysis]) -> dict[str, str]:
        """Research Question 4: does each ablated stage (Query Refinement / Evidence Filtering /
        Verification) actually earn its cost? Compares each ablation experiment's overall score
        against EXP-11 (the full adaptive system) — a positive delta means removing that stage hurt
        performance (the stage was pulling its weight); a delta near zero or positive-for-ablation
        means that stage wasn't earning its cost in this run.

        Returns:
            exp_id (ablation) -> plain-language finding. Empty if EXP-11 or the relevant ablation
            isn't present in `analyses`.
        """
        by_id = {a.exp_id: a for a in analyses}
        findings: dict[str, str] = {}
        for ablation_id, (baseline_id, stage_name) in _ABLATION_PAIRS.items():
            if ablation_id not in by_id or baseline_id not in by_id:
                continue
            full_score = by_id[baseline_id].overall_weighted_score
            ablated_score = by_id[ablation_id].overall_weighted_score
            if full_score is None or ablated_score is None:
                continue
            delta = full_score - ablated_score
            if delta > 0.02:
                findings[ablation_id] = (
                    f"{stage_name} earns its cost: removing it dropped the overall score by "
                    f"{delta:.3f} ({baseline_id}={full_score:.3f} vs {ablation_id}={ablated_score:.3f})."
                )
            elif delta < -0.02:
                findings[ablation_id] = (
                    f"{stage_name} did NOT earn its cost in this run: the system scored {abs(delta):.3f} "
                    f"HIGHER without it ({ablation_id}={ablated_score:.3f} vs {baseline_id}={full_score:.3f}) "
                    "— worth investigating whether this stage is misconfigured or genuinely unhelpful "
                    "for this question mix."
                )
            else:
                findings[ablation_id] = (
                    f"{stage_name} made negligible difference ({delta:+.3f}) — inconclusive at this "
                    "sample size, not necessarily unhelpful."
                )
        return findings
