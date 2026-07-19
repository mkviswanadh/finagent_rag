"""Tests for sampling.py — stratified document/question selection for pilot runs."""

from __future__ import annotations

from finagent.data.financebench_loader import FinanceBenchDocumentInfo
from finagent.data.schemas import EvidenceReference, FinanceBenchQuestion, QueryComplexity
from finagent.experiments.sampling import (
    _estimate_complexity,
    select_diversified_documents,
    select_diversified_questions,
    summarize_sample,
)


def _doc_info(doc_name: str, company: str, sector: str, year: int = 2022) -> FinanceBenchDocumentInfo:
    return FinanceBenchDocumentInfo(
        doc_name=doc_name, company=company, year=year, report_type="10-K", gics_sector=sector, doc_link=""
    )


def _question(
    qid: str, company: str, doc_name: str, sector: str, question: str = "What was revenue?",
    dataset_question_type: str = "metrics-generated", evidence_count: int = 1,
) -> FinanceBenchQuestion:
    return FinanceBenchQuestion(
        question_id=qid, question=question, reference_answer="$1",
        evidence=[EvidenceReference(doc_name=doc_name, page_number=i, text="") for i in range(evidence_count)],
        company=company, document_type="10-K", document_name=doc_name, document_year=2022,
        gics_sector=sector, justification="", dataset_question_type=dataset_question_type,
    )


class TestEstimateComplexity:
    def test_simple_lookup(self):
        q = _question("q1", "X", "X_10K", "Tech", "What was revenue in 2022?")
        assert _estimate_complexity(q) == QueryComplexity.SIMPLE

    def test_complex_why_question(self):
        q = _question("q1", "X", "X_10K", "Tech", "Why did revenue change in 2022?")
        assert _estimate_complexity(q) == QueryComplexity.COMPLEX

    def test_complex_multi_evidence(self):
        q = _question("q1", "X", "X_10K", "Tech", "What was revenue?", evidence_count=3)
        assert _estimate_complexity(q) == QueryComplexity.COMPLEX

    def test_moderate_multi_year(self):
        q = _question("q1", "X", "X_10K", "Tech", "Compare revenue between 2021 and 2022.")
        # Contains "compare" -> complex keyword; use a version without it to isolate the year signal.
        q2 = _question("q1", "X", "X_10K", "Tech", "What was revenue in 2021 and 2022?")
        assert _estimate_complexity(q2) == QueryComplexity.MODERATE

    def test_moderate_change_keyword(self):
        q = _question("q1", "X", "X_10K", "Tech", "How did revenue change from last year?")
        assert _estimate_complexity(q) == QueryComplexity.MODERATE


class TestSelectDiversifiedDocuments:
    def _build_pool(self):
        doc_info = {
            "A_2021_10K": _doc_info("A_2021_10K", "CompanyA", "Tech", 2021),
            "A_2022_10K": _doc_info("A_2022_10K", "CompanyA", "Tech", 2022),
            "A_2023_10K": _doc_info("A_2023_10K", "CompanyA", "Tech", 2023),
            "B_2022_10K": _doc_info("B_2022_10K", "CompanyB", "Healthcare", 2022),
            "C_2022_10K": _doc_info("C_2022_10K", "CompanyC", "Utilities", 2022),
            "D_2022_10K": _doc_info("D_2022_10K", "CompanyD", "Financials", 2022),
        }
        questions = [
            _question("q1", "CompanyA", "A_2021_10K", "Tech"),
            _question("q2", "CompanyA", "A_2022_10K", "Tech"),
            _question("q3", "CompanyA", "A_2023_10K", "Tech"),
            _question("q4", "CompanyB", "B_2022_10K", "Healthcare"),
            _question("q5", "CompanyC", "C_2022_10K", "Utilities"),
            _question("q6", "CompanyD", "D_2022_10K", "Financials"),
        ]
        return doc_info, questions

    def test_returns_requested_count_when_available(self):
        doc_info, questions = self._build_pool()
        result = select_diversified_documents(doc_info, questions, num_documents=4)
        assert len(result) == 4

    def test_returns_fewer_if_pool_smaller_than_requested(self):
        doc_info, questions = self._build_pool()
        result = select_diversified_documents(doc_info, questions, num_documents=100)
        assert len(result) == 6

    def test_ignores_unreferenced_documents(self):
        doc_info, questions = self._build_pool()
        doc_info["UNREFERENCED_2022_10K"] = _doc_info("UNREFERENCED_2022_10K", "Ghost", "Tech")
        result = select_diversified_documents(doc_info, questions, num_documents=100)
        assert "UNREFERENCED_2022_10K" not in result

    def test_does_not_exceed_multi_year_budget_fraction_of_a_single_company(self):
        """With a small sample size, CompanyA's 3 filings shouldn't consume the whole budget —
        breadth across other companies must still get slots."""
        doc_info, questions = self._build_pool()
        result = select_diversified_documents(doc_info, questions, num_documents=4)
        companies = {doc_info[d].company for d in result}
        assert len(companies) >= 3  # not collapsed onto just CompanyA


class TestSelectDiversifiedQuestions:
    def _build_pool(self, n_per_company: int = 1):
        questions = []
        companies = ["CompanyA", "CompanyB", "CompanyC", "CompanyD", "CompanyE"]
        for c in companies:
            for i in range(n_per_company):
                questions.append(_question(f"{c}_{i}", c, f"{c}_2022_10K", "Tech"))
        return questions

    def test_returns_requested_count_when_available(self):
        questions = self._build_pool(n_per_company=3)
        result = select_diversified_questions(questions, num_questions=8)
        assert len(result) == 8

    def test_does_not_exceed_available_pool(self):
        questions = self._build_pool(n_per_company=1)
        result = select_diversified_questions(questions, num_questions=100)
        assert len(result) == 5

    def test_prefers_company_breadth_before_repeating(self):
        """With 5 companies x 1 question each, requesting 5 should get all 5 distinct companies."""
        questions = self._build_pool(n_per_company=1)
        result = select_diversified_questions(questions, num_questions=5)
        assert len({q.company for q in result}) == 5

    def test_falls_back_to_repeats_when_target_exceeds_company_count(self):
        """Regression test: previously, once every company had one pick, selection stopped even
        with slots and candidates remaining — this must now fill the remaining slots instead."""
        questions = self._build_pool(n_per_company=3)  # 5 companies x 3 = 15 candidates
        result = select_diversified_questions(questions, num_questions=10)
        assert len(result) == 10  # not capped at 5 (the company count)

    def test_sets_assigned_complexity_on_every_returned_question(self):
        questions = self._build_pool(n_per_company=1)
        result = select_diversified_questions(questions, num_questions=5)
        assert all(q.assigned_complexity is not None for q in result)

    def test_restricts_to_allowed_documents(self):
        questions = self._build_pool(n_per_company=1)
        allowed = {"CompanyA_2022_10K", "CompanyB_2022_10K"}
        result = select_diversified_questions(questions, num_questions=10, allowed_documents=allowed)
        assert len(result) == 2
        assert {q.document_name for q in result} == allowed

    def test_no_duplicate_questions_returned(self):
        questions = self._build_pool(n_per_company=3)
        result = select_diversified_questions(questions, num_questions=10)
        ids = [q.question_id for q in result]
        assert len(ids) == len(set(ids))


class TestSummarizeSample:
    def test_summarizes_key_dimensions(self):
        questions = [
            _question("q1", "A", "A_10K", "Tech", dataset_question_type="metrics-generated"),
            _question("q2", "B", "B_10K", "Health", dataset_question_type="novel-generated", evidence_count=2),
        ]
        questions[0].assigned_complexity = QueryComplexity.SIMPLE
        questions[1].assigned_complexity = QueryComplexity.COMPLEX

        summary = summarize_sample(questions)
        assert summary["count"] == 2
        assert set(summary["companies"]) == {"A", "B"}
        assert summary["multi_evidence_count"] == 1
        assert summary["complexity_distribution"] == {"Simple": 1, "Complex": 1}
