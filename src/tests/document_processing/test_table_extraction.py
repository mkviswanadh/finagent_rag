"""Tests for table_extraction.py — numeric-density table-row detection."""

from __future__ import annotations

from finagent.document_processing.table_extraction import extract_tables_from_page


def test_detects_balance_sheet_style_table():
    page_text = "\n".join([
        "CONSOLIDATED BALANCE SHEETS",
        "Cash and cash equivalents $ 23,646 $ 34,940",
        "Marketable securities $ 24,658 $ 27,699",
        "Total assets $ 352,755 $ 351,002",
        "Some narrative sentence that is not tabular at all here about the business.",
    ])
    blocks = extract_tables_from_page(page_text)
    assert len(blocks) == 1
    assert "Cash and cash equivalents" in blocks[0].text
    assert "Total assets" in blocks[0].text
    # The narrative sentence must not be pulled into the table block.
    assert "narrative sentence" not in blocks[0].text


def test_no_table_in_pure_narrative_page():
    page_text = "\n".join([
        "The Company operates in several business segments.",
        "Management believes the strategy will continue to support growth.",
        "Further discussion is provided in subsequent sections of this report.",
    ])
    assert extract_tables_from_page(page_text) == []


def test_single_table_like_line_below_minimum_is_not_a_block():
    """A lone numeric-dense line, with no run of 2+, should not produce a block."""
    page_text = "Intro text.\nTotal assets $ 352,755 $ 351,002\nMore narrative text follows here."
    assert extract_tables_from_page(page_text) == []


def test_bare_currency_symbols_do_not_dilute_density():
    """Regression test for the bug where lone '$' tokens diluted numeric density below threshold."""
    page_text = "\n".join([
        "Revenue $ 1,234 $ 1,111",
        "Expenses $ 987 $ 876",
    ])
    blocks = extract_tables_from_page(page_text)
    assert len(blocks) == 1


def test_empty_page():
    assert extract_tables_from_page("") == []
