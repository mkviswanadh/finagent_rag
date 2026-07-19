"""Tests for grounding.py — Faithfulness, Hallucination Rate, Evidence Coverage, Citation Correctness."""

from __future__ import annotations

from finagent.data.schemas import EvidenceReference, FinanceBenchQuestion, VerificationResult
from finagent.metrics.grounding import citation_correctness, evidence_coverage, faithfulness, hallucination_rate


def _question(evidence_pages: list[int], doc_name: str = "X_2022_10K") -> FinanceBenchQuestion:
    return FinanceBenchQuestion(
        question_id="q1", question="Q", reference_answer="A",
        evidence=[EvidenceReference(doc_name=doc_name, page_number=p, text="") for p in evidence_pages],
        company="X", document_type="10-K", document_name=doc_name, document_year=2022,
        gics_sector="", justification="", dataset_question_type="metrics-generated",
    )


class TestFaithfulness:
    def test_fully_supported_scores_one(self):
        result = VerificationResult(passed=True, unsupported_claims=[], confidence=0.95, notes="")
        assert faithfulness(result, "Revenue was $198.3 billion.") == 1.0

    def test_one_unsupported_of_two_sentences_scores_half(self):
        result = VerificationResult(passed=False, unsupported_claims=["claim"], confidence=0.6, notes="")
        answer = "Revenue was $198.3 billion. It grew due to strong cloud demand."
        assert faithfulness(result, answer) == 0.5

    def test_unsupported_count_capped_at_total_statements(self):
        """More unsupported claims than sentences must not produce a negative score."""
        result = VerificationResult(passed=False, unsupported_claims=["a", "b", "c"], confidence=0.5, notes="")
        assert faithfulness(result, "One sentence only.") == 0.0


class TestHallucinationRate:
    def test_complement_of_faithfulness(self):
        result = VerificationResult(passed=False, unsupported_claims=["claim"], confidence=0.6, notes="")
        answer = "Revenue was $198.3 billion. It grew due to strong cloud demand."
        assert hallucination_rate(result, answer) == 1.0 - faithfulness(result, answer)

    def test_fully_supported_scores_zero(self):
        result = VerificationResult(passed=True, unsupported_claims=[], confidence=0.95, notes="")
        assert hallucination_rate(result, "Revenue was $198.3 billion.") == 0.0


class TestEvidenceCoverage:
    def test_full_coverage(self, sample_evidence_item):
        question = _question([40])  # matches sample_evidence_item's page
        result = evidence_coverage(["EV_001"], [sample_evidence_item], question)
        assert result == 1.0

    def test_no_citations_scores_zero(self, sample_evidence_item):
        question = _question([40])
        assert evidence_coverage([], [sample_evidence_item], question) == 0.0

    def test_no_ground_truth_evidence_scores_zero(self, sample_evidence_item):
        question = _question([])
        assert evidence_coverage(["EV_001"], [sample_evidence_item], question) == 0.0


class TestCitationCorrectness:
    def test_correct_citation_scores_one(self, sample_evidence_item):
        # doc_name must match sample_evidence_item's source_document ("MICROSOFT_2022_10K.pdf")
        # for citation_correctness's doc+page cross-check to succeed.
        question = _question([40], doc_name="MICROSOFT_2022_10K")
        assert citation_correctness(["EV_001"], [sample_evidence_item], question) == 1.0

    def test_no_citations_scores_zero(self, sample_evidence_item):
        question = _question([40], doc_name="MICROSOFT_2022_10K")
        assert citation_correctness([], [sample_evidence_item], question) == 0.0

    def test_unresolvable_citation_id_not_counted_correct(self, sample_evidence_item):
        question = _question([40])
        result = citation_correctness(["EV_999_UNKNOWN"], [sample_evidence_item], question)
        assert result == 0.0
