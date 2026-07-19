"""Tests for reasoning_metrics.py — Numerical Accuracy, Calculation Accuracy, Multi-step Reasoning,
Explanation Completeness."""

from __future__ import annotations

from finagent.data.schemas import EvidenceReference, FinanceBenchQuestion, ReasoningOutput
from finagent.metrics.reasoning_metrics import (
    calculation_accuracy,
    explanation_completeness,
    multi_step_reasoning_score,
    numerical_accuracy,
)


class TestNumericalAccuracy:
    def test_matching_number_scores_one(self):
        assert numerical_accuracy("$198.3 billion", "198.3 billion") == 1.0

    def test_mismatched_number_scores_zero(self):
        assert numerical_accuracy("$150 billion", "198.3 billion") == 0.0

    def test_reference_with_no_numbers_scores_one_trivially(self):
        assert numerical_accuracy("some text", "no numbers here") == 1.0

    def test_generated_with_no_numbers_but_reference_has_some_scores_zero(self):
        assert numerical_accuracy("no numbers here", "198.3 billion") == 0.0

    def test_tolerates_minor_rounding_within_one_percent(self):
        assert numerical_accuracy("$198.4 billion", "$198.3 billion") == 1.0

    def test_rejects_large_deviation(self):
        assert numerical_accuracy("$250 billion", "$198.3 billion") == 0.0


class TestCalculationAccuracy:
    def test_returns_none_when_not_a_calculation_question(self):
        assert calculation_accuracy("x", "y", needs_calculation=False) is None

    def test_delegates_to_numerical_accuracy_when_calculation_needed(self):
        result = calculation_accuracy("23%", "23%", needs_calculation=True)
        assert result == 1.0


class TestMultiStepReasoningScore:
    def test_full_score_when_all_expected_evidence_cited(self):
        question = FinanceBenchQuestion(
            question_id="q1", question="Q", reference_answer="A",
            evidence=[
                EvidenceReference(doc_name="X", page_number=1, text=""),
                EvidenceReference(doc_name="X", page_number=2, text=""),
            ],
            company="X", document_type="10-K", document_name="X", document_year=2022,
            gics_sector="", justification="", dataset_question_type="metrics-generated",
        )
        reasoning_output = ReasoningOutput(
            reasoning_steps=["s1", "s2"], extracted_values={}, draft_answer="a",
            citations=["EV_001", "EV_002"], insufficient_evidence=False,
        )
        assert multi_step_reasoning_score(reasoning_output, question) == 1.0

    def test_partial_score_when_fewer_citations_than_expected(self):
        question = FinanceBenchQuestion(
            question_id="q1", question="Q", reference_answer="A",
            evidence=[
                EvidenceReference(doc_name="X", page_number=1, text=""),
                EvidenceReference(doc_name="X", page_number=2, text=""),
            ],
            company="X", document_type="10-K", document_name="X", document_year=2022,
            gics_sector="", justification="", dataset_question_type="metrics-generated",
        )
        reasoning_output = ReasoningOutput(
            reasoning_steps=["s1"], extracted_values={}, draft_answer="a",
            citations=["EV_001"], insufficient_evidence=False,
        )
        assert multi_step_reasoning_score(reasoning_output, question) == 0.5

    def test_score_never_exceeds_one_with_extra_citations(self):
        question = FinanceBenchQuestion(
            question_id="q1", question="Q", reference_answer="A",
            evidence=[EvidenceReference(doc_name="X", page_number=1, text="")],
            company="X", document_type="10-K", document_name="X", document_year=2022,
            gics_sector="", justification="", dataset_question_type="metrics-generated",
        )
        reasoning_output = ReasoningOutput(
            reasoning_steps=["s1", "s2", "s3"], extracted_values={}, draft_answer="a",
            citations=["EV_001", "EV_002", "EV_003"], insufficient_evidence=False,
        )
        assert multi_step_reasoning_score(reasoning_output, question) == 1.0


class TestExplanationCompleteness:
    def test_full_recall_when_all_reference_content_covered(self):
        assert explanation_completeness("Revenue was $198.3 billion driven by cloud growth", "$198.3 billion") == 1.0

    def test_partial_recall(self):
        score = explanation_completeness("Revenue grew", "revenue grew due to strong cloud demand")
        assert 0.0 < score < 1.0

    def test_empty_reference_scores_zero(self):
        assert explanation_completeness("some text", "") == 0.0
