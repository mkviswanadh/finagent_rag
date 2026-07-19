"""Tests for analyst.py — deterministic cross-experiment synthesis, zero Groq calls."""

from __future__ import annotations

from finagent.data.schemas import EvidenceReference, FinanceBenchQuestion, PipelineTrace, QueryComplexity, QuestionResult
from finagent.results.analyst import ResultsAnalyst


def _question(qid: str = "q1") -> FinanceBenchQuestion:
    return FinanceBenchQuestion(
        question_id=qid, question="What was revenue?", reference_answer="$1",
        evidence=[EvidenceReference(doc_name="X_2022_10K", page_number=1, text="...")],
        company="X", document_type="10-K", document_name="X_2022_10K", document_year=2022,
        gics_sector="Tech", justification="", dataset_question_type="metrics-generated",
    )


def _result(
    qid: str, exp_id: str, *, answer: str = "answer", complexity: QueryComplexity = QueryComplexity.SIMPLE,
    metrics: dict | None = None,
) -> QuestionResult:
    trace = PipelineTrace(experiment_id=exp_id, question=_question(qid), complexity_used=complexity)
    trace.generated_answer = answer
    trace.mark_finished()
    return QuestionResult(trace=trace, metrics=metrics or {})


class TestFamilyScores:
    def test_answer_quality_averages_across_questions(self):
        results = {
            "EXP-01": [
                _result("q1", "EXP-01", metrics={"answer_relevance": 0.8, "f1_score": 0.6}),
                _result("q2", "EXP-01", metrics={"answer_relevance": 0.4, "f1_score": 0.2}),
            ]
        }
        analyses = ResultsAnalyst().analyze(results)
        assert analyses[0].family_scores.answer_quality == 0.5  # mean of 0.8,0.6,0.4,0.2

    def test_retrieval_quality_is_none_for_direct_llm(self):
        results = {"EXP-01": [_result("q1", "EXP-01", metrics={"answer_relevance": 0.8})]}
        analyses = ResultsAnalyst().analyze(results)
        assert analyses[0].family_scores.retrieval_quality is None

    def test_grounding_inverts_hallucination_rate(self):
        results = {"EXP-11": [_result("q1", "EXP-11", metrics={"hallucination_rate": 0.2})]}
        analyses = ResultsAnalyst().analyze(results)
        assert analyses[0].family_scores.grounding == 0.8  # 1 - 0.2


class TestFallbackRate:
    def test_detects_identical_answers_as_fallback(self):
        results = {
            "EXP-07": [
                _result("q1", "EXP-07", answer="Insufficient evidence.", metrics={"answer_relevance": 0.1}),
                _result("q2", "EXP-07", answer="Insufficient evidence.", metrics={"answer_relevance": 0.1}),
                _result("q3", "EXP-07", answer="$5 billion.", metrics={"answer_relevance": 0.9}),
            ]
        }
        analyses = ResultsAnalyst().analyze(results)
        assert analyses[0].fallback_rate == 2 / 3

    def test_flags_high_fallback_rate_as_an_issue(self):
        results = {
            "EXP-07": [_result(f"q{i}", "EXP-07", answer="Insufficient evidence.", metrics={"answer_relevance": 0.0}) for i in range(5)]
        }
        analyses = ResultsAnalyst().analyze(results)
        assert any("share the identical answer text" in issue for issue in analyses[0].issues)


class TestRankingAndImprovement:
    def test_higher_overall_score_ranks_first(self):
        results = {
            "EXP-01": [_result("q1", "EXP-01", metrics={"answer_relevance": 0.2})],
            "EXP-11": [_result("q1", "EXP-11", metrics={"answer_relevance": 0.9})],
        }
        analyses = ResultsAnalyst().analyze(results)
        assert analyses[0].exp_id == "EXP-11"
        assert analyses[0].final_rank == 1
        assert analyses[1].final_rank == 2

    def test_improvement_over_direct_llm_is_relative_delta(self):
        results = {
            "EXP-01": [_result("q1", "EXP-01", metrics={"answer_relevance": 0.5})],
            "EXP-11": [_result("q1", "EXP-11", metrics={"answer_relevance": 1.0})],
        }
        analyses = ResultsAnalyst().analyze(results)
        exp11 = next(a for a in analyses if a.exp_id == "EXP-11")
        assert exp11.improvement_over_direct_llm == 1.0  # (1.0 - 0.5) / 0.5


class TestAblationFindings:
    def test_flags_stage_that_earns_its_cost(self):
        results = {
            "EXP-11": [_result("q1", "EXP-11", metrics={"answer_relevance": 0.9})],
            "EXP-12": [_result("q1", "EXP-12", metrics={"answer_relevance": 0.5})],
        }
        analyses = ResultsAnalyst().analyze(results)
        findings = ResultsAnalyst().ablation_findings(analyses)
        assert "earns its cost" in findings["EXP-12"]

    def test_flags_stage_that_does_not_earn_its_cost(self):
        results = {
            "EXP-11": [_result("q1", "EXP-11", metrics={"answer_relevance": 0.5})],
            "EXP-13": [_result("q1", "EXP-13", metrics={"answer_relevance": 0.9})],
        }
        analyses = ResultsAnalyst().analyze(results)
        findings = ResultsAnalyst().ablation_findings(analyses)
        assert "did NOT earn its cost" in findings["EXP-13"]


class TestEfficiencyNormalization:
    def test_lower_latency_scores_higher(self):
        results = {
            "EXP-A": [_result("q1", "EXP-A", metrics={"answer_relevance": 0.5, "latency_seconds": 1.0})],
            "EXP-B": [_result("q1", "EXP-B", metrics={"answer_relevance": 0.5, "latency_seconds": 5.0})],
        }
        analyses = {a.exp_id: a for a in ResultsAnalyst().analyze(results)}
        assert analyses["EXP-A"].family_scores.efficiency == 1.0
        assert analyses["EXP-B"].family_scores.efficiency == 0.0
