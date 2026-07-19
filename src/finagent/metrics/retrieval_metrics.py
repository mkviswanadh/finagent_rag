"""Evidence & Retrieval metrics (Proposal Table 7.14).

"Relevant" is determined against FinanceBench's own ground-truth evidence annotations
(`FinanceBenchQuestion.evidence`, a list of `EvidenceReference` with `doc_name` + `page_number` +
`text`) using **two** signals, not just page number:

1. Exact page match within the correct source document (the original approach).
2. Shared distinctive numeric values between the retrieved chunk and the annotated evidence excerpt's
   own text, within the correct source document.

Signal 2 exists because of a real, verified data issue: FinanceBench's `evidence_page_num` does not
reliably align with the raw sequential page index this codebase's PDF extraction produces. Checked
directly against two pilot-run documents — 3M's 2018 10-K (FinanceBench page 57 for a PP&E figure
that is actually on raw PDF page 41, a 16-page offset) and Adobe's 2022 10-K (FinanceBench page 53
for an income statement that is actually on raw PDF page 54, a 1-page offset) — the offset is
real, non-zero, and varies per document (almost certainly reflecting each filing's own front-matter/
cover-page length before its internal page numbering begins), so it cannot be corrected with a single
constant. A retrieval-quality metric that only checks exact page equality would silently
under-count correct retrievals across this entire dataset. Numeric-value overlap is a robust,
offset-independent proxy: financial evidence excerpts are dense with specific numbers unlikely to
co-occur by chance, so requiring several shared values is a reasonably strict, false-positive-resistant
check — see `metrics.text_normalization.extract_numbers` and `grounding.py`, which reuses this same
matching function for citation correctness / evidence coverage, so both families agree on "relevant."
"""

from __future__ import annotations

from finagent.data.schemas import Chunk, EvidenceItem, EvidenceReference, FinanceBenchQuestion
from finagent.metrics.text_normalization import extract_numbers

# How many of the evidence excerpt's own DISTINCTIVE numbers must also appear in the candidate
# chunk for a page-mismatched chunk to still count as relevant. Capped at the excerpt's own
# distinctive-number count for short excerpts.
MIN_SHARED_NUMBERS_FOR_TEXT_MATCH = 2

# Numbers excluded from "distinctive" status because they recur constantly across unrelated
# financial-statement boilerplate and produce false-positive matches on their own: calendar years
# ("Years ended December 31, 2018, 2017, 2016" appears near almost every financial table) and small
# integers that usually mean a day-of-month, quarter number, or footnote index rather than a
# reported financial value. Verified against a real false positive found during pilot debugging: a
# chunk about share repurchases matched a capex evidence excerpt purely because both mentioned
# "December 31, 2018" (i.e. shared {2018, 2017, 3, 31}) despite having no actual content in common.
_YEAR_RANGE = range(1900, 2101)
_SMALL_INTEGER_MAX = 31


def _is_distinctive_number(value: float) -> bool:
    if value != int(value):
        return True  # decimals (e.g. 8.7, 34.6) are inherently specific, rarely coincidental
    if int(value) in _YEAR_RANGE:
        return False
    return abs(value) > _SMALL_INTEGER_MAX


def _distinctive_numbers(text: str) -> set[float]:
    return {n for n in extract_numbers(text) if _is_distinctive_number(n)}


def _shares_distinctive_numbers(chunk_text: str, evidence_text: str) -> bool:
    """Whether `chunk_text` reproduces enough of `evidence_text`'s own distinctive numeric values
    to be the same underlying content, independent of any page-number labeling."""
    evidence_numbers = _distinctive_numbers(evidence_text)
    if not evidence_numbers:
        return False
    chunk_numbers = _distinctive_numbers(chunk_text)
    required = min(MIN_SHARED_NUMBERS_FOR_TEXT_MATCH, len(evidence_numbers))
    return len(chunk_numbers & evidence_numbers) >= required


def matches_evidence_reference(chunk: Chunk, ref: EvidenceReference) -> bool:
    """Whether `chunk` is the same underlying evidence as `ref`, by page match or number overlap.

    Args:
        chunk: A retrieved (or cited) chunk.
        ref: One of `FinanceBenchQuestion.evidence`'s annotated excerpts.

    Returns:
        `True` if `chunk` is from `ref`'s source document AND either its page number matches
        exactly or it shares enough distinctive numeric values with `ref.text` (see module
        docstring for why the page-number check alone is unreliable on this dataset).
    """
    if not chunk.source_document.startswith(ref.doc_name):
        return False
    if chunk.page_number == ref.page_number:
        return True
    return _shares_distinctive_numbers(chunk.text, ref.text)


def _is_relevant(item: EvidenceItem, question: FinanceBenchQuestion) -> bool:
    """A retrieved chunk is relevant if it matches any of the question's annotated evidence."""
    return any(matches_evidence_reference(item.chunk, ref) for ref in question.evidence)


def context_recall(retrieved: list[EvidenceItem], question: FinanceBenchQuestion) -> float:
    """Context Recall: fraction of the question's annotated evidence excerpts covered by retrieval.

    Formula (Proposal Table 7.14): `Relevant Chunks Retrieved / Total Relevant Chunks` — the
    denominator here is the number of distinct annotated evidence excerpts
    (`len(question.evidence)`), and an excerpt counts as "covered" if any retrieved chunk matches
    it (see `matches_evidence_reference`).

    Args:
        retrieved: Evidence retrieved for this question (any stage — raw retrieval or post-filter,
            depending on what the caller wants to measure).
        question: The FinanceBench question, providing the annotated evidence to match against.

    Returns:
        Recall in `[0, 1]`. `0.0` if the question has no annotated evidence (should not occur for
        FinanceBench's open-source split, but handled defensively).
    """
    if not question.evidence:
        return 0.0
    covered = sum(
        1 for ref in question.evidence if any(matches_evidence_reference(item.chunk, ref) for item in retrieved)
    )
    return covered / len(question.evidence)


def context_precision(retrieved: list[EvidenceItem], question: FinanceBenchQuestion) -> float:
    """Context Precision: fraction of retrieved chunks that are actually relevant.

    Formula (Proposal Table 7.14): `Relevant Retrieved Chunks / Total Retrieved Chunks`.

    Args:
        retrieved: Evidence retrieved for this question.
        question: The FinanceBench question, providing ground-truth evidence to match against.

    Returns:
        Precision in `[0, 1]`. `0.0` if nothing was retrieved.
    """
    if not retrieved:
        return 0.0
    relevant_count = sum(1 for item in retrieved if _is_relevant(item, question))
    return relevant_count / len(retrieved)


def hit_at_k(retrieved: list[EvidenceItem], question: FinanceBenchQuestion, k: int) -> float:
    """Hit@K: whether at least one relevant chunk appears within the top-k retrieved results.

    Formula (Proposal Table 7.14): `Queries With Correct Evidence In Top-K / Total Queries` —
    this function returns the per-question indicator (1.0 or 0.0); averaging it across questions in
    an experiment run produces the aggregate Hit@K reported per Table 7.14.

    Args:
        retrieved: Evidence retrieved for this question, in ranked order (index 0 = most relevant).
        question: The FinanceBench question, providing ground-truth evidence to match against.
        k: Number of top results to consider.

    Returns:
        `1.0` if any of the top-k results is relevant, else `0.0`.
    """
    top_k = retrieved[:k]
    return 1.0 if any(_is_relevant(item, question) for item in top_k) else 0.0


def mean_reciprocal_rank(retrieved: list[EvidenceItem], question: FinanceBenchQuestion) -> float:
    """Reciprocal rank of the first relevant chunk in this question's retrieval results.

    Formula (Proposal Table 7.14): `MRR = (1/N) * sum(1/rank_i)` — this function returns the
    per-question `1/rank_i` term (or `0.0` if no relevant chunk was retrieved at all); averaging it
    across questions in an experiment run produces the aggregate MRR.

    Args:
        retrieved: Evidence retrieved for this question, in ranked order (index 0 = rank 1).
        question: The FinanceBench question, providing ground-truth evidence to match against.

    Returns:
        `1/rank` of the first relevant result (1-indexed), or `0.0` if none of the retrieved
        results are relevant.
    """
    for rank, item in enumerate(retrieved, start=1):
        if _is_relevant(item, question):
            return 1.0 / rank
    return 0.0
