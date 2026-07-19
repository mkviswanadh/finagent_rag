"""Evidence & Retrieval metrics (Proposal Table 7.14).

"Relevant" here is defined against FinanceBench's own ground-truth evidence annotations
(`FinanceBenchQuestion.evidence`, a list of `EvidenceReference` with `doc_name` + `page_number` —
see `finagent.data.financebench_loader`): a retrieved chunk counts as relevant if it comes from the
same source document and the same page as one of the annotated evidence excerpts. This is possible
precisely because FinanceBench ships page-level evidence annotations (Proposal §7.4.1: "the dataset
supports evaluation of retrieval systems at document, page, and chunk level") — these metrics are
computed with zero Groq calls, purely from the retrieval trace and the dataset's own labels.
"""

from __future__ import annotations

from finagent.data.schemas import EvidenceItem, FinanceBenchQuestion


def _is_relevant(item: EvidenceItem, question: FinanceBenchQuestion) -> bool:
    """A retrieved chunk is relevant if it's from an annotated evidence document and page."""
    for ref in question.evidence:
        if item.chunk.source_document.startswith(ref.doc_name) and item.chunk.page_number == ref.page_number:
            return True
    return False


def context_recall(retrieved: list[EvidenceItem], question: FinanceBenchQuestion) -> float:
    """Context Recall: fraction of ground-truth evidence pages covered by retrieval.

    Formula (Proposal Table 7.14): `Relevant Chunks Retrieved / Total Relevant Chunks`, interpreted
    here at the page level — "Total Relevant Chunks" is the number of distinct annotated evidence
    pages for this question (`question.evidence_page_numbers`), and a page counts as "retrieved" if
    any retrieved chunk covers it.

    Args:
        retrieved: Evidence retrieved for this question (any stage — raw retrieval or post-filter,
            depending on what the caller wants to measure).
        question: The FinanceBench question, providing ground-truth evidence pages.

    Returns:
        Recall in `[0, 1]`. `0.0` if the question has no annotated evidence pages (should not
        occur for FinanceBench's open-source split, but handled defensively).
    """
    ground_truth_pages = set(question.evidence_page_numbers)
    if not ground_truth_pages:
        return 0.0
    covered_pages = {
        item.chunk.page_number
        for item in retrieved
        if any(item.chunk.source_document.startswith(ref.doc_name) for ref in question.evidence)
    }
    hit_pages = ground_truth_pages & covered_pages
    return len(hit_pages) / len(ground_truth_pages)


def context_precision(retrieved: list[EvidenceItem], question: FinanceBenchQuestion) -> float:
    """Context Precision: fraction of retrieved chunks that are actually relevant.

    Formula (Proposal Table 7.14): `Relevant Retrieved Chunks / Total Retrieved Chunks`.

    Args:
        retrieved: Evidence retrieved for this question.
        question: The FinanceBench question, providing ground-truth evidence pages.

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
        question: The FinanceBench question, providing ground-truth evidence pages.
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
        question: The FinanceBench question, providing ground-truth evidence pages.

    Returns:
        `1/rank` of the first relevant result (1-indexed), or `0.0` if none of the retrieved
        results are relevant.
    """
    for rank, item in enumerate(retrieved, start=1):
        if _is_relevant(item, question):
            return 1.0 / rank
    return 0.0
