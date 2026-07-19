"""Stratified document/question selection for pilot and other sub-full-scale runs.

A naive "pick the N documents with the most questions" selection (the pilot's original approach)
does not guarantee the sample actually exercises every pathway `Coding_Sheet.xlsx`'s result sheets
are structured around:

- The "Query Complexity-Wise Final Results" sheet needs real Simple/Moderate/Complex representation
  for every experiment — a sample skewed toward simple lookups leaves most of that sheet empty.
- EXP-08 (metadata-aware retrieval) only has something to prove when the corpus contains multiple
  years/filings of the *same* company — metadata filtering can't demonstrate value disambiguating a
  company that only appears once.
- EXP-09 (query rewriting) matters most for questions phrased informally relative to how a filing
  actually states the answer.
- EXP-10 (multi-query) and the adaptive system's Complex route matter most for questions needing
  more than one evidence excerpt (`len(question.evidence) > 1`).
- FinanceBench's own question-construction methodology (`dataset_question_type`:
  metrics-generated / domain-relevant / novel-generated, roughly a third each) is itself a
  diversity axis worth preserving in any subset.

This module's `select_diversified_sample` stratifies across all of these axes using only
information already in the dataset and document metadata — no Groq calls, so it costs nothing and
can be re-run freely while tuning a pilot's size.

Complexity here is estimated with a lightweight, deterministic keyword heuristic (`_estimate_complexity`)
for STRATIFICATION purposes only — it approximates, cheaply and for free, which bucket a question is
likely to land in. It is NOT the same as `QueryUnderstandingAgent.analyze`, which is the actual
live routing decision made during a real experiment run; the two can disagree, and that's fine
here, since this function's job is only to build a sample that has a reasonable *chance* of
touching all three complexity tiers, not to pre-determine routing.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from finagent.data.financebench_loader import FinanceBenchDocumentInfo
from finagent.data.schemas import FinanceBenchQuestion, QueryComplexity

# At most this fraction of a document sample is spent on multi-year depth (see
# select_diversified_documents) — the rest goes to company/sector breadth, so a small sample
# doesn't collapse into just a few over-represented companies.
MULTI_YEAR_BUDGET_FRACTION = 0.4
MAX_DOCS_PER_MULTI_YEAR_COMPANY = 3

_COMPLEX_KEYWORDS = re.compile(
    r"\b(why|explain|compare|comparison|trend|reason|factors?|drove|driving|contribut\w*)\b",
    re.IGNORECASE,
)
_MODERATE_KEYWORDS = re.compile(
    r"\b(change|increase|decrease|grow|growth|decline|year[- ]over[- ]year|yoy)\b", re.IGNORECASE
)
_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def _estimate_complexity(question: FinanceBenchQuestion) -> QueryComplexity:
    """Cheap, deterministic complexity estimate for stratification only — see module docstring."""
    text = question.question
    if _COMPLEX_KEYWORDS.search(text) or len(question.evidence) > 1:
        return QueryComplexity.COMPLEX
    distinct_years = {m.group(0) for m in _YEAR_PATTERN.finditer(text)}
    if _MODERATE_KEYWORDS.search(text) or len(distinct_years) >= 2:
        return QueryComplexity.MODERATE
    return QueryComplexity.SIMPLE


def select_diversified_documents(
    document_info: dict[str, FinanceBenchDocumentInfo],
    questions: list[FinanceBenchQuestion],
    *,
    num_documents: int,
) -> list[str]:
    """Select a document sample diversified across company, GICS sector, report type, and
    multi-year coverage of at least a few companies (for metadata-filtering to have something to
    disambiguate).

    Args:
        document_info: doc_name -> `FinanceBenchDocumentInfo`, from `load_document_info()`.
        questions: The FinanceBench questions the selected documents should be able to answer —
            only documents referenced by at least one question are eligible, since an unreferenced
            document contributes corpus noise but no evaluatable questions.
        num_documents: Target sample size.

    Returns:
        `num_documents` document names (or fewer if there aren't enough referenced documents).
        At most `MULTI_YEAR_BUDGET_FRACTION` of the sample goes to multi-year depth (a handful of
        companies with up to `MAX_DOCS_PER_MULTI_YEAR_COMPANY` filings each, so metadata filtering
        has real same-company-different-year ambiguity to resolve) — the rest is spent on breadth,
        round-robining across GICS sectors and preferring companies not already represented, so a
        small sample doesn't collapse into a handful of over-represented companies.
    """
    referenced_docs = {q.document_name for q in questions if q.document_name in document_info}

    company_docs: dict[str, list[str]] = defaultdict(list)
    for doc_name in referenced_docs:
        company_docs[document_info[doc_name].company].append(doc_name)

    selected: list[str] = []
    selected_set: set[str] = set()

    # Priority 1: a BOUNDED number of companies with multiple referenced filing years — enough for
    # EXP-08's metadata-based disambiguation to be a meaningful test, capped so multi-year depth
    # doesn't crowd out company/sector breadth in a small sample.
    multi_year_budget = max(1, round(num_documents * MULTI_YEAR_BUDGET_FRACTION))
    multi_year_companies = sorted(
        (c for c, docs in company_docs.items() if len(docs) > 1),
        key=lambda c: -len(company_docs[c]),
    )
    for company in multi_year_companies:
        if len(selected) >= multi_year_budget or len(selected) >= num_documents:
            break
        for doc_name in sorted(company_docs[company])[:MAX_DOCS_PER_MULTI_YEAR_COMPANY]:
            if len(selected) >= multi_year_budget or len(selected) >= num_documents:
                break
            if doc_name not in selected_set:
                selected.append(doc_name)
                selected_set.add(doc_name)

    # Priority 2: fill remaining slots round-robin across GICS sectors, for sector diversity. Two
    # passes: first prefer a company not yet represented at all (maximizes company breadth), then
    # — only if slots remain after every company has had a first pick — allow additional documents
    # from already-represented companies rather than leaving the sample smaller than requested.
    selected_companies = {document_info[d].company for d in selected}
    remaining = sorted(referenced_docs - selected_set)
    for prefer_new_company in (True, False):
        if len(selected) >= num_documents:
            break
        by_sector: dict[str, list[str]] = defaultdict(list)
        for doc_name in remaining:
            if doc_name in selected_set:
                continue
            by_sector[document_info[doc_name].gics_sector].append(doc_name)

        sectors = sorted(by_sector.keys())
        sector_iters = {s: iter(docs) for s, docs in by_sector.items()}
        active_sectors = set(sector_iters)
        while len(selected) < num_documents and active_sectors:
            for sector in list(active_sectors):
                try:
                    doc_name = next(sector_iters[sector])
                except StopIteration:
                    active_sectors.discard(sector)
                    continue
                company = document_info[doc_name].company
                if prefer_new_company and company in selected_companies:
                    continue  # revisit in the second (relaxed) pass instead
                if doc_name not in selected_set:
                    selected.append(doc_name)
                    selected_set.add(doc_name)
                    selected_companies.add(company)
                if len(selected) >= num_documents:
                    break

    return selected[:num_documents]


def select_diversified_questions(
    questions: list[FinanceBenchQuestion],
    *,
    num_questions: int,
    allowed_documents: set[str] | None = None,
) -> list[FinanceBenchQuestion]:
    """Select a question sample stratified across estimated complexity, multi-evidence need, and
    FinanceBench's own `dataset_question_type` construction category.

    Args:
        questions: The full candidate pool (typically already filtered to a document sample).
        num_questions: Target sample size.
        allowed_documents: If given, restrict candidates to questions whose `document_name` is in
            this set (e.g. the output of `select_diversified_documents`).

    Returns:
        Up to `num_questions` questions, distributed as evenly as possible across the 3x3 grid of
        (estimated complexity) x (`dataset_question_type`), with `assigned_complexity` set on each
        returned question from the same estimate used for stratification, so a caller can record
        results by complexity tier immediately without waiting on a live Query Understanding call.
    """
    candidates = questions
    if allowed_documents is not None:
        candidates = [q for q in candidates if q.document_name in allowed_documents]

    buckets: dict[tuple[QueryComplexity, str], list[FinanceBenchQuestion]] = defaultdict(list)
    for q in candidates:
        complexity = _estimate_complexity(q)
        q.assigned_complexity = complexity
        buckets[(complexity, q.dataset_question_type)].append(q)

    bucket_keys = sorted(buckets.keys(), key=lambda k: (k[0].value, k[1]))
    for bucket in buckets.values():
        bucket.sort(key=lambda q: q.company)

    selected: list[FinanceBenchQuestion] = []
    selected_ids: set[str] = set()
    seen_companies: set[str] = set()

    def _round_robin_pick(*, prefer_new_company: bool) -> None:
        """One or more passes round-robining across buckets, taking at most one question per
        bucket per pass, until either num_questions is met or a full pass makes no progress."""
        while len(selected) < num_questions:
            progressed = False
            for key in bucket_keys:
                if len(selected) >= num_questions:
                    break
                for candidate in buckets[key]:
                    if candidate.question_id in selected_ids:
                        continue
                    if prefer_new_company and candidate.company in seen_companies:
                        continue
                    selected.append(candidate)
                    selected_ids.add(candidate.question_id)
                    seen_companies.add(candidate.company)
                    progressed = True
                    break
            if not progressed:
                return

    # Pass 1: maximize company breadth — at most one question per company until every company in
    # the pool has had a first pick (or the target is met).
    _round_robin_pick(prefer_new_company=True)
    # Pass 2: target not yet met after every company had a first pick — relax the constraint and
    # allow additional questions from already-represented companies rather than under-filling.
    _round_robin_pick(prefer_new_company=False)

    return selected[:num_questions]


def summarize_sample(questions: list[FinanceBenchQuestion]) -> dict:
    """Human-readable diversity summary of a selected question sample, for logging/review."""
    return {
        "count": len(questions),
        "companies": sorted({q.company for q in questions}),
        "complexity_distribution": dict(Counter(q.assigned_complexity.value if q.assigned_complexity else "unknown" for q in questions)),
        "dataset_question_type_distribution": dict(Counter(q.dataset_question_type for q in questions)),
        "multi_evidence_count": sum(1 for q in questions if len(q.evidence) > 1),
        "gics_sectors": sorted({q.gics_sector for q in questions}),
    }
