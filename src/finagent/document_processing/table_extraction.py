"""Financial table extraction (Proposal Table 7.4, Step 2: "Table Extraction").

This machine has no poppler/Ghostscript, which rules out image-based table extractors
(camelot's lattice/stream-on-image modes, tabula-py's Java+Ghostscript dependency). Financial
tables in FinanceBench 10-K/10-Q filings are typically extractable as text by `pypdf` with rows
kept roughly intact (each table row on its own line, columns separated by irregular whitespace).
This module applies a text-pattern heuristic to identify and preserve those rows as distinguished
"table blocks" — rows with a high density of numeric/currency tokens — so the chunker (Proposal
Table 7.4 Step 4) can avoid splitting a table mid-row and so numerical-accuracy metrics (Proposal
Table 7.16) have a fighting chance against tables that plain paragraph chunking would mangle.

This is a best-effort heuristic ("extract... where possible", per the proposal's own wording for
this step), not a structured cell-grid parser — FinanceBench evaluation does not require
reconstructing exact table schemas, only that the numeric evidence survives chunking in a
readable, contiguous form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_NUMERIC_TOKEN = re.compile(
    r"""
    \$?\(?               # optional leading $ and/or (
    -?\d{1,3}(,\d{3})*    # integer part with thousands separators
    (\.\d+)?              # optional decimal part
    \)?%?                 # optional trailing ) and/or %
    """,
    re.VERBOSE,
)

# A line is "table-like" if at least this fraction of its whitespace-separated tokens are numeric.
# Calibrated low enough to catch typical financial-statement rows such as
# "Cash and cash equivalents $ 23,646 $ 34,940" (4 label words, 2 numeric values -> ~0.33 density
# once bare currency symbols are excluded from the denominator) without also matching ordinary
# prose, which rarely carries more than one incidental number per sentence.
_TABLE_LINE_NUMERIC_DENSITY_THRESHOLD = 0.3
# Consecutive table-like lines shorter than this are still merged into the same block (footnote
# markers / units rows are often sparse but belong inside the surrounding table).
_MIN_TABLE_BLOCK_LINES = 2


@dataclass(frozen=True)
class TableBlock:
    """A contiguous run of table-like lines detected on one page."""

    start_line_index: int
    end_line_index: int
    text: str


def _is_table_like_line(line: str) -> bool:
    # Bare currency/percent symbols (e.g. a lone "$" preceding a number on its own token, as
    # pypdf often extracts "$ 23,646") carry no signal on their own and must not dilute the
    # density denominator, or a row that is entirely dollar figures reads as mostly non-numeric.
    tokens = [tok for tok in line.split() if tok not in {"$", "%", "-", "—", "–"}]
    if not tokens:
        return False
    numeric_tokens = sum(1 for tok in tokens if _NUMERIC_TOKEN.fullmatch(tok.strip(".,;")))
    return (numeric_tokens / len(tokens)) >= _TABLE_LINE_NUMERIC_DENSITY_THRESHOLD


def extract_tables_from_page(page_text: str) -> list[TableBlock]:
    """Identify contiguous numeric-dense line runs on a page as candidate financial tables.

    Args:
        page_text: Raw text of one PDF page, as returned by `extract_pdf_pages`.

    Returns:
        A list of `TableBlock`s, each a contiguous run of table-like lines (balance sheet rows,
        income statement lines, etc.) with at least `_MIN_TABLE_BLOCK_LINES` lines. Empty if the
        page contains no detectable tabular content.
    """
    lines = page_text.splitlines()
    blocks: list[TableBlock] = []
    run_start: int | None = None

    def _flush(end_index: int) -> None:
        nonlocal run_start
        if run_start is not None and (end_index - run_start) >= _MIN_TABLE_BLOCK_LINES:
            block_lines = lines[run_start:end_index]
            blocks.append(
                TableBlock(
                    start_line_index=run_start,
                    end_line_index=end_index,
                    text="\n".join(block_lines).strip(),
                )
            )
        run_start = None

    for i, line in enumerate(lines):
        if _is_table_like_line(line):
            if run_start is None:
                run_start = i
        else:
            _flush(i)
    _flush(len(lines))

    return blocks
