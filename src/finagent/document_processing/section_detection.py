"""Financial report section detection (Proposal Table 7.4, Step 4: "Section Detection").

Identifies canonical financial report sections — "balance sheet, income statement, risk factors,
and notes" per the proposal's own wording, extended with the other sections FinanceBench filings
commonly contain (MD&A, cash flow statement, stockholders' equity, legal proceedings) — so every
`Chunk.section` (Proposal Table 7.5) is populated with a meaningful, traceable label instead of a
generic placeholder. Section labels feed directly into metadata-aware retrieval (EXP-08) and
citation correctness evaluation (Proposal Table 7.15).
"""

from __future__ import annotations

import re

UNCLASSIFIED_SECTION = "General Narrative"

# Ordered so more specific patterns are checked before generic ones (e.g. "Notes to ..." before a
# bare "Notes" match). Each entry: canonical section name -> compiled heading regex.
_SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "Consolidated Balance Sheet",
        re.compile(r"consolidated\s+balance\s+sheets?", re.IGNORECASE),
    ),
    ("Balance Sheet", re.compile(r"^\s*balance\s+sheets?\b", re.IGNORECASE)),
    (
        "Consolidated Statements of Operations",
        re.compile(
            r"consolidated\s+statements?\s+of\s+(operations|income)", re.IGNORECASE
        ),
    ),
    (
        "Income Statement",
        re.compile(r"statements?\s+of\s+income|income\s+statement", re.IGNORECASE),
    ),
    (
        "Consolidated Statements of Cash Flows",
        re.compile(r"consolidated\s+statements?\s+of\s+cash\s+flows?", re.IGNORECASE),
    ),
    ("Cash Flow Statement", re.compile(r"statements?\s+of\s+cash\s+flows?", re.IGNORECASE)),
    (
        "Stockholders' Equity",
        re.compile(r"(stock|shareholder)s?['’]?\s*equity", re.IGNORECASE),
    ),
    (
        "Notes to Financial Statements",
        re.compile(r"notes?\s+to\s+(the\s+)?(consolidated\s+)?financial\s+statements", re.IGNORECASE),
    ),
    (
        "Management's Discussion and Analysis",
        re.compile(
            r"management'?s\s+discussion\s+and\s+analysis|^\s*md&a\b", re.IGNORECASE
        ),
    ),
    ("Risk Factors", re.compile(r"^\s*risk\s+factors\b", re.IGNORECASE)),
    ("Legal Proceedings", re.compile(r"^\s*legal\s+proceedings\b", re.IGNORECASE)),
    (
        "Controls and Procedures",
        re.compile(r"controls\s+and\s+procedures", re.IGNORECASE),
    ),
    ("Business Overview", re.compile(r"^\s*item\s+1\.?\s*business\b", re.IGNORECASE)),
    # --- Generic catch-alls (checked last, i.e. lowest priority) ---
    # A 10-K's Notes to Financial Statements section typically contains 15-25+ individually
    # numbered footnotes (Income Taxes, Leases, Pension, Segment Information, ...) that cannot be
    # exhaustively enumerated by topic. Without this, the running section incorrectly keeps
    # whatever specific label (e.g. "Legal Proceedings") was last matched by name, for every note
    # afterward that happens not to match one of the named patterns above — potentially
    # mislabeling the rest of the document. Matching the numbering convention itself ("NOTE 12.",
    # "Note 3 –") catches every footnote transition regardless of its topic.
    (
        "Notes to Financial Statements",
        re.compile(r"^\s*note\s+\d{1,2}[a-z]?[.\-–—:]", re.IGNORECASE),
    ),
    # Similarly, Part II/III/IV of a 10-K (Item 9 onward: changes in accountants, controls,
    # governance, compensation, exhibits) contains items with no direct financial-QA relevance
    # that this codebase does not attempt to name individually — but an unrecognized "Item N."
    # heading must still break carry-forward from an earlier, unrelated named section.
    (
        "Other Item Disclosures",
        re.compile(r"^\s*item\s+\d{1,2}[a-z]?\.?\s*$", re.IGNORECASE),
    ),
]

_MAX_HEADING_LINE_LENGTH = 100
# Longest realistic official heading is ~12-13 words (e.g. "Management's Discussion and Analysis of
# Financial Condition and Results of Operations"); ordinary narrative sentences that merely mention
# a section by name in passing ("our disclosure controls and procedures were effective...",
# "as described in the Notes to the Consolidated Financial Statements...") almost always run
# considerably longer. Several of the section patterns above are intentionally unanchored substring
# matches (to tolerate "Item 9A. Controls and Procedures" style prefixes), so this word-count gate
# is what keeps them from false-triggering on every narrative mention of a section's name.
_MAX_HEADING_WORD_COUNT = 14
# A real heading line is essentially *just* the heading (plus maybe a short "Item 9A." prefix); a
# narrative sentence that merely mentions a section by name in passing ("...concluded that the
# Company's disclosure controls and procedures are effective.") embeds the matched phrase inside a
# much longer line. Requiring the match to cover a large share of the line's characters catches
# these short-but-narrative sentences that the word-count gate alone lets through (a 10-word
# sentence still reads as prose, not a title).
_MIN_HEADING_MATCH_COVERAGE = 0.5


def match_section_heading(line: str) -> str | None:
    """Check a single line against the canonical section heading patterns.

    This is the line-level primitive `chunking.py` uses to retag the running section mid-page
    (a heading can appear partway down a page, with the remainder of that page already belonging
    to the new section) — whole-page granularity alone would misattribute the tail of a page to
    whatever section was active at the top of it.

    Args:
        line: A single line of page text (already `.strip()`-able; leading/trailing whitespace is
            tolerated).

    Returns:
        The canonical section name if `line` matches a heading pattern AND reads like a heading
        (short, and the matched phrase dominates the line rather than being embedded in a longer
        prose sentence), else `None`.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADING_LINE_LENGTH:
        return None
    if len(stripped.split()) > _MAX_HEADING_WORD_COUNT:
        return None
    for section_name, pattern in _SECTION_PATTERNS:
        match = pattern.search(stripped)
        if not match:
            continue
        coverage = (match.end() - match.start()) / len(stripped)
        if coverage >= _MIN_HEADING_MATCH_COVERAGE:
            return section_name
    return None


def detect_section(page_text: str, previous_section: str | None = None) -> str:
    """Determine the section a page of text ends in (its dominant/carry-forward section).

    Scans every line for a heading match and keeps the *last* one found, since a page's final
    section is what should carry forward onto the next page — using only the first match would
    misclassify a page that transitions from one section to another partway through.

    Args:
        page_text: Cleaned text of one page (post `clean_text`).
        previous_section: The section assigned to the previous page in document order, used as
            the carry-forward default when no heading is found on this page at all.

    Returns:
        The canonical section name this page ends in, or `UNCLASSIFIED_SECTION` if no heading has
        ever been seen (i.e. this is the first page and it matches no pattern — typically a cover
        page).
    """
    detected: str | None = None
    for line in page_text.splitlines():
        match = match_section_heading(line)
        if match:
            detected = match

    if detected:
        return detected
    if previous_section:
        return previous_section
    return UNCLASSIFIED_SECTION
