"""Tests for financebench_loader.py — loading and joining the real FinanceBench dataset files."""

from __future__ import annotations

import pytest

from finagent.config import DOCUMENT_INFO_PATH, QA_DATASET_PATH
from finagent.data.financebench_loader import load_document_info, load_financebench_questions

requires_real_dataset = pytest.mark.skipif(
    not QA_DATASET_PATH.exists() or not DOCUMENT_INFO_PATH.exists(),
    reason="data/financebench_open_source.jsonl or financebench_document_information.jsonl not present",
)


@requires_real_dataset
class TestLoadDocumentInfo:
    def test_loads_all_documents(self):
        info = load_document_info()
        assert len(info) > 300  # 361 filings in the real dataset

    def test_report_type_normalized(self):
        info = load_document_info()
        threeM = info["3M_2015_10K"]
        assert threeM.report_type == "10-K"
        assert threeM.company == "3M"
        assert threeM.year == 2015


@requires_real_dataset
class TestLoadFinanceBenchQuestions:
    def test_loads_exactly_150_open_source_questions(self):
        questions = load_financebench_questions()
        assert len(questions) == 150

    def test_every_question_has_resolved_metadata(self):
        questions = load_financebench_questions()
        unresolved = [q for q in questions if q.document_type == "Unknown" or q.document_year is None]
        assert unresolved == []

    def test_multi_evidence_questions_exist(self):
        """Regression check: the dataset genuinely has multi-page evidence, not just single-page."""
        questions = load_financebench_questions()
        multi = [q for q in questions if len(q.evidence) > 1]
        assert len(multi) >= 30  # verified: 35 questions have 2-3 evidence excerpts

    def test_evidence_page_numbers_deduplicated_and_sorted(self):
        questions = load_financebench_questions()
        multi = next(q for q in questions if len(q.evidence) > 1)
        pages = multi.evidence_page_numbers
        assert pages == sorted(set(pages))

    def test_missing_qa_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_financebench_questions(qa_path=tmp_path / "does_not_exist.jsonl")
