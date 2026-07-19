"""Tests for retrieval_metrics.py — Context Recall/Precision, Hit@K, MRR against ground-truth pages."""

from __future__ import annotations

from finagent.data.schemas import Chunk, EvidenceItem, EvidenceReference, FinanceBenchQuestion
from finagent.metrics.retrieval_metrics import (
    context_precision,
    context_recall,
    hit_at_k,
    matches_evidence_reference,
    mean_reciprocal_rank,
)


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


class TestMatchesEvidenceReference:
    """Regression coverage for the page-offset finding: FinanceBench's evidence_page_num does not
    reliably align with this codebase's raw PDF page index (verified up to a 16-page offset on a
    real pilot document) — a chunk with the right content but a "wrong" page number must still
    count as a match if it shares the evidence excerpt's own distinctive numbers."""

    def _chunk(self, *, page: int, text: str, doc_name: str = "3M_2018_10K") -> Chunk:
        return Chunk(
            chunk_id="C1", company="3M", year=2018, report_type="10-K", section="S",
            page_number=page, text=text, source_document=f"{doc_name}.pdf",
        )

    def _ref(self, *, page: int, text: str, doc_name: str = "3M_2018_10K") -> EvidenceReference:
        return EvidenceReference(doc_name=doc_name, page_number=page, text=text)

    def test_exact_page_match_is_sufficient(self):
        chunk = self._chunk(page=41, text="totally different wording with no shared numbers")
        ref = self._ref(page=41, text="net property, plant and equipment totaled $8.7 billion")
        assert matches_evidence_reference(chunk, ref) is True

    def test_number_overlap_matches_despite_page_mismatch(self):
        """The real case found in the pilot: FinanceBench says page 57, the actual content
        (matched by its distinctive numbers) is on raw PDF page 41."""
        chunk = self._chunk(
            page=41, text="net property, plant and equipment totaled $8.738 billion, up from $8.169 billion"
        )
        ref = self._ref(
            page=57, text="Property, plant and equipment - net was $8.738 billion in 2018 vs $8.169 billion in 2017"
        )
        assert matches_evidence_reference(chunk, ref) is True

    def test_wrong_document_never_matches_even_with_shared_numbers(self):
        chunk = self._chunk(page=57, text="revenue of $8.738 billion", doc_name="OTHER_COMPANY_2018_10K")
        ref = self._ref(page=57, text="revenue of $8.738 billion")
        assert matches_evidence_reference(chunk, ref) is False

    def test_single_shared_number_is_not_enough(self):
        """One coincidentally-shared number (e.g. a year) shouldn't count as a real content match."""
        chunk = self._chunk(page=10, text="In 2018 the company opened 12 new stores.")
        ref = self._ref(page=57, text="In 2018, net property, plant and equipment totaled $8.7 billion.")
        assert matches_evidence_reference(chunk, ref) is False

    def test_no_numbers_in_evidence_text_falls_back_to_page_only(self):
        chunk = self._chunk(page=10, text="qualitative narrative with no numbers")
        ref = self._ref(page=57, text="qualitative discussion with no numbers either")
        assert matches_evidence_reference(chunk, ref) is False

    def test_shared_years_and_day_numbers_alone_do_not_match(self):
        """Regression test for a real false positive found during pilot debugging: a chunk about
        share repurchases matched a capex evidence excerpt purely because both mentioned
        "December 31, 2018" (shared {2018, 2017, 3, 31}) despite having no real content overlap."""
        chunk = self._chunk(
            page=43,
            text="cash availability in the United States as of December 31, 2018 and 2017, 3 sources",
        )
        ref = self._ref(
            page=59,
            text="Consolidated Statement of Cash Flows, Years ended December 31, 2018 2017 2016",
        )
        assert matches_evidence_reference(chunk, ref) is False

    def test_distinctive_dollar_figures_still_match_despite_shared_years(self):
        """The genuine positive from the same real case: the chunk that actually contains the
        answer figure ($1,577 million capex) still matches, years present in both notwithstanding."""
        chunk = self._chunk(
            page=39,
            text="Capital Spending as of December 31, 2018 2017: United States 994 852, Total 1577 1373",
        )
        ref = self._ref(
            page=59,
            text="Consolidated Statement of Cash Flows, December 31, 2018 2017, capital expenditures 1577",
        )
        assert matches_evidence_reference(chunk, ref) is True
