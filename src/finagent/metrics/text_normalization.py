"""Shared text normalization for exact-match/F1/numerical-comparison metrics.

Financial answers need different normalization than general QA (SQuAD-style normalization alone
would treat "$394.3 billion" and "394.3 billion" as different strings, or "23%" and "23 percent" as
unrelated) — this module centralizes that logic so `answer_quality.py` and `reasoning_metrics.py`
don't implement subtly different normalization that would make their scores incomparable.
"""

from __future__ import annotations

import re

_PUNCTUATION_PATTERN = re.compile(r"[^\w\s.%-]")
# A period only carries meaning as a decimal point ("198.3"); a sentence-ending period stuck to the
# last word ("...was $198.3 billion.") must not survive tokenization, or "billion." and "billion"
# would score as different tokens in F1/completeness. Keep only periods with a digit on both sides.
_NON_DECIMAL_PERIOD_PATTERN = re.compile(r"(?<!\d)\.|\.(?!\d)")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_NUMBER_PATTERN = re.compile(r"-?\$?\d[\d,]*\.?\d*%?")


def normalize_answer(text: str) -> str:
    """Lowercase, strip currency/punctuation noise, and collapse whitespace for comparison.

    Args:
        text: Raw answer text (generated or reference).

    Returns:
        Normalized text suitable for token-level or exact-string comparison. Preserves digits,
        decimal points, `%`, and `-` (all numerically meaningful in financial answers) while
        stripping other punctuation (`$`, `,`, parentheses, quotes, sentence-ending periods, etc.).
    """
    text = text.lower().strip()
    text = text.replace(",", "")  # thousands separators: "1,234" -> "1234"
    text = _PUNCTUATION_PATTERN.sub(" ", text)
    text = _NON_DECIMAL_PERIOD_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def extract_numbers(text: str) -> list[float]:
    """Extract every numeric value mentioned in a text, as floats.

    Args:
        text: Raw or normalized text.

    Returns:
        All numbers found, in order of appearance, with `$`/`,`/`%` stripped before conversion.
        A trailing `%` is preserved as a value out of 100 (e.g. "23%" -> `23.0`, not `0.23`) since
        financial answers overwhelmingly state percentages this way.
    """
    numbers: list[float] = []
    for match in _NUMBER_PATTERN.finditer(text):
        token = match.group().replace("$", "").replace(",", "").replace("%", "")
        if not token or token in {"-", "."}:
            continue
        try:
            numbers.append(float(token))
        except ValueError:
            continue
    return numbers
