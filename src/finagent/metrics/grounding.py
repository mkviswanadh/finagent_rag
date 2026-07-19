"""Grounding & Trust metrics (Proposal Table 7.15).

Faithfulness and Hallucination Rate are derived from the Verification Agent's own judgment
(`VerificationResult` — see `agents/verification.py`) rather than a separate metric-specific LLM
call: the Verification Agent already performs exactly the claim-by-claim entailment check these
metrics are defined over (Proposal Table 7.15: "Supported Statements / Total Statements"), so
re-scoring it independently would duplicate that Groq call for no benefit. "Total Statements" is
approximated as the number of sentences in the generated answer, since the agents do not currently
enumerate discrete claims individually — documented as an approximation, not exact claim counting.

Evidence Coverage and Citation Correctness are computed with zero Groq calls, purely by
cross-referencing `ReasoningOutput.citations` against the evidence list and the FinanceBench
ground-truth evidence pages (same mechanism as `retrieval_metrics.py`).
"""

from __future__ import annotations

import re

from finagent.data.schemas import EvidenceItem, FinanceBenchQuestion, VerificationResult

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def _count_sentences(text: str) -> int:
    sentences = [s for s in _SENTENCE_SPLIT_PATTERN.split(text.strip()) if s.strip()]
    return max(len(sentences), 1)


def faithfulness(verification_result: VerificationResult, generated_answer: str) -> float:
    """Faithfulness: fraction of the generated answer's statements supported by evidence.

    Formula (Proposal Table 7.15): `Supported Statements / Total Statements In Generated Answer`.
    "Total Statements" is approximated by sentence count (see module docstring); "Supported" is
    `Total - len(unsupported_claims)`, from the Verification Agent's own claim-by-claim check.

    Args:
        verification_result: Output of `VerificationAgent.verify`.
        generated_answer: The answer that was verified.

    Returns:
        Faithfulness in `[0, 1]`. `1.0` if verification passed with no unsupported claims listed.
    """
    total_statements = _count_sentences(generated_answer)
    unsupported = min(len(verification_result.unsupported_claims), total_statements)
    return (total_statements - unsupported) / total_statements


def hallucination_rate(verification_result: VerificationResult, generated_answer: str) -> float:
    """Hallucination Rate: fraction of the generated answer's statements that are unsupported.

    Formula (Proposal Table 7.15): `Unsupported Claims / Total Claims` — the complement of
    `faithfulness`, computed via the same sentence-count approximation for consistency.

    Args:
        verification_result: Output of `VerificationAgent.verify`.
        generated_answer: The answer that was verified.

    Returns:
        Hallucination rate in `[0, 1]`.
    """
    return 1.0 - faithfulness(verification_result, generated_answer)


def evidence_coverage(citations: list[str], evidence: list[EvidenceItem], question: FinanceBenchQuestion) -> float:
    """Evidence Coverage: fraction of ground-truth evidence pages actually cited in the answer.

    Formula (Proposal Table 7.15): `Evidence Points Used / Total Relevant Evidence Points` — a
    ground-truth evidence page counts as "used" if a cited evidence chunk (by ID, resolved via
    `evidence`) is from that page.

    Args:
        citations: Evidence IDs the Reasoning Agent cited (`ReasoningOutput.citations`).
        evidence: The evidence list the citation IDs index into.
        question: The FinanceBench question, providing ground-truth evidence pages.

    Returns:
        Coverage in `[0, 1]`. `0.0` if the question has no annotated evidence pages.
    """
    ground_truth_pages = set(question.evidence_page_numbers)
    if not ground_truth_pages:
        return 0.0

    evidence_by_id = {e.evidence_id: e for e in evidence}
    cited_pages = {
        evidence_by_id[cid].chunk.page_number for cid in citations if cid in evidence_by_id
    }
    return len(ground_truth_pages & cited_pages) / len(ground_truth_pages)


def citation_correctness(citations: list[str], evidence: list[EvidenceItem], question: FinanceBenchQuestion) -> float:
    """Citation Correctness: fraction of citations that actually point to ground-truth evidence.

    Formula (Proposal Table 7.15): `Correct Supporting Citations / Total Citations`.

    Args:
        citations: Evidence IDs the Reasoning Agent cited.
        evidence: The evidence list the citation IDs index into.
        question: The FinanceBench question, providing ground-truth evidence pages/documents.

    Returns:
        Correctness in `[0, 1]`. `0.0` if there are no citations to evaluate.
    """
    if not citations:
        return 0.0

    evidence_by_id = {e.evidence_id: e for e in evidence}
    correct = 0
    for cid in citations:
        item = evidence_by_id.get(cid)
        if item is None:
            continue
        if any(
            item.chunk.source_document.startswith(ref.doc_name) and item.chunk.page_number == ref.page_number
            for ref in question.evidence
        ):
            correct += 1
    return correct / len(citations)
