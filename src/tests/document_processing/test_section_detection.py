"""Tests for section_detection.py — heading matching, TOC-page exclusion is tested separately
in test_chunking.py since it's implemented in chunking.py's `_is_toc_like_page`."""

from __future__ import annotations

from finagent.document_processing.section_detection import (
    UNCLASSIFIED_SECTION,
    detect_section,
    match_section_heading,
)


class TestMatchSectionHeading:
    def test_matches_balance_sheet_heading(self):
        assert match_section_heading("CONSOLIDATED BALANCE SHEETS") == "Consolidated Balance Sheet"

    def test_matches_risk_factors_with_item_prefix(self):
        assert match_section_heading("Item 1A. Risk Factors") == "Risk Factors"

    def test_matches_bare_risk_factors(self):
        assert match_section_heading("Risk Factors") == "Risk Factors"

    def test_matches_legal_proceedings_with_item_prefix(self):
        assert match_section_heading("Item 3. Legal Proceedings") == "Legal Proceedings"

    def test_matches_controls_and_procedures_full_heading(self):
        assert match_section_heading("Item 9A. Controls and Procedures.") == "Controls and Procedures"

    def test_matches_generic_note_number(self):
        assert match_section_heading("NOTE 6. Supplemental Income Statement Information") == \
            "Notes to Financial Statements"

    def test_matches_generic_item_with_title(self):
        assert match_section_heading("Item 10. Directors, Executive Officers and Corporate Governance.") == \
            "Other Item Disclosures"

    def test_rejects_narrative_sentence_mentioning_controls_and_procedures(self):
        """Regression test: a sentence merely mentioning the phrase must not match as a heading."""
        sentence = (
            "We concluded that the Company's disclosure controls and procedures are effective."
        )
        assert match_section_heading(sentence) is None

    def test_rejects_narrative_sentence_mentioning_notes_to_financial_statements(self):
        sentence = "The accompanying Notes to Consolidated Financial Statements are an integral part of this statement."
        assert match_section_heading(sentence) is None

    def test_rejects_empty_line(self):
        assert match_section_heading("") is None
        assert match_section_heading("   ") is None

    def test_rejects_overlong_line(self):
        long_line = "Risk Factors " + ("x" * 200)
        assert match_section_heading(long_line) is None

    def test_rejects_unrelated_text(self):
        assert match_section_heading("The quick brown fox jumps over the lazy dog.") is None


class TestDetectSection:
    def test_returns_unclassified_with_no_heading_and_no_previous(self):
        assert detect_section("just some narrative text") == UNCLASSIFIED_SECTION

    def test_carries_forward_previous_section_when_no_heading_found(self):
        assert detect_section("more narrative text", previous_section="Risk Factors") == "Risk Factors"

    def test_uses_last_heading_found_on_the_page(self):
        page_text = "Risk Factors\nsome text\nLegal Proceedings\nmore text"
        assert detect_section(page_text) == "Legal Proceedings"
