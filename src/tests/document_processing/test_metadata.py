"""Tests for metadata.py — filename parsing and report-type normalization."""

from __future__ import annotations

import pytest

from finagent.document_processing.metadata import normalize_report_type, parse_filing_metadata


class TestParseFilingMetadata:
    def test_parses_standard_10k_filename(self):
        meta = parse_filing_metadata("3M_2018_10K.pdf")
        assert meta.company == "3M"
        assert meta.year == 2018
        assert meta.report_type == "10-K"

    def test_parses_8k_with_date_suffix(self):
        meta = parse_filing_metadata("AMAZON_2022_8K_dated-2022-06-06.pdf")
        assert meta.company == "AMAZON"
        assert meta.year == 2022
        assert meta.report_type == "8-K"

    def test_parses_annualreport_variant(self):
        meta = parse_filing_metadata("WALMART_2023_annualreport.pdf")
        assert meta.company == "WALMART"
        assert meta.report_type == "Annual Report"

    def test_unrecognized_filename_falls_back_gracefully(self):
        meta = parse_filing_metadata("not_a_recognized_pattern.pdf")
        assert meta.report_type == "Unknown"
        assert meta.year == 0

    def test_empty_filename_raises(self):
        with pytest.raises(ValueError):
            parse_filing_metadata("")


class TestNormalizeReportType:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("10K", "10-K"),
            ("10k", "10-K"),
            ("10Q", "10-Q"),
            ("8K", "8-K"),
            ("Earnings", "Earnings Report"),
            ("10k_annualreport", "Annual Report"),
        ],
    )
    def test_normalizes_known_variants(self, raw, expected):
        assert normalize_report_type(raw) == expected

    def test_unrecognized_type_returned_unchanged(self):
        assert normalize_report_type("SOME_UNKNOWN_TYPE") == "SOME_UNKNOWN_TYPE"
