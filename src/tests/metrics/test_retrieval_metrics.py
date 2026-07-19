"""Tests for retrieval_metrics.py — Context Recall/Precision, Hit@K, MRR against ground-truth pages."""

from __future__ import annotations

from finagent.data.schemas import Chunk, EvidenceItem, EvidenceReference, FinanceBenchQuestion
from finagent.metrics.retrieval_metrics import context_precision, context_recall, hit_at_k, mean_reciprocal_rank


def _question(evidence_pages: list[int]) -> FinanceBenchQuestion:
    return FinanceBenchQuestion(
        question_id="q1", question="Q", reference_answer="A",
        evidence=[EvidenceReference(doc_name="X_2022_10K", page_number=p, text="") for p in evidence_pages],
        company="X", document_type="10-K", document_name="X_2022_10K", document_year=2022,
        gics_sector="", justification="", dataset_question_type="metrics-generated",
    )


def _evidence(page: int, doc_name: str = "X_2022_10K", score: float = 0.9) -> EvidenceItem:
    chunk = Chunk(
        chunk_id=f"C{page}", company="X", year=2022, report_type="10-K", section="S",
        page_number=page, text="...", source_document=f"{doc_name}.pdf",
    )
    return EvidenceItem(evidence_id=f"EV_{page}", chunk=chunk, relevance_score=score, retrieval_query="q")


class TestContextRecall:
    def test_full_recall_when_all_evidence_pages_retrieved(self):
        question = _question([10, 20])
        retrieved = [_evidence(10), _evidence(20)]
        assert context_recall(retrieved, question) == 1.0

    def test_partial_recall(self):
        question = _question([10, 20])
        retrieved = [_evidence(10)]
        assert context_recall(retrieved, question) == 0.5

    def test_zero_recall_when_nothing_matches(self):
        question = _question([10, 20])
        retrieved = [_evidence(999)]
        assert context_recall(retrieved, question) == 0.0

    def test_wrong_document_does_not_count(self):
        question = _question([10])
        retrieved = [_evidence(10, doc_name="WRONG_COMPANY_2022_10K")]
        assert context_recall(retrieved, question) == 0.0


class TestContextPrecision:
    def test_all_retrieved_relevant(self):
        question = _question([10, 20])
        retrieved = [_evidence(10), _evidence(20)]
        assert context_precision(retrieved, question) == 1.0

    def test_half_retrieved_relevant(self):
        question = _question([10])
        retrieved = [_evidence(10), _evidence(999)]
        assert context_precision(retrieved, question) == 0.5

    def test_empty_retrieval_scores_zero(self):
        question = _question([10])
        assert context_precision([], question) == 0.0


class TestHitAtK:
    def test_hit_within_top_k(self):
        question = _question([20])
        retrieved = [_evidence(999), _evidence(20), _evidence(888)]
        assert hit_at_k(retrieved, question, k=3) == 1.0

    def test_miss_outside_top_k(self):
        question = _question([20])
        retrieved = [_evidence(999), _evidence(888), _evidence(20)]
        assert hit_at_k(retrieved, question, k=2) == 0.0


class TestMeanReciprocalRank:
    def test_first_result_relevant_gives_mrr_one(self):
        question = _question([20])
        retrieved = [_evidence(20), _evidence(999)]
        assert mean_reciprocal_rank(retrieved, question) == 1.0

    def test_second_result_relevant_gives_mrr_half(self):
        question = _question([20])
        retrieved = [_evidence(999), _evidence(20)]
        assert mean_reciprocal_rank(retrieved, question) == 0.5

    def test_no_relevant_result_gives_zero(self):
        question = _question([20])
        retrieved = [_evidence(999)]
        assert mean_reciprocal_rank(retrieved, question) == 0.0
