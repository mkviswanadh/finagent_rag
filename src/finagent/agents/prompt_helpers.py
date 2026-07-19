"""Shared prompt-construction helpers used by multiple agents.

Kept in one place so the evidence block format (which agents like Reasoning and Verification both
need to embed in their user prompts) can't silently drift between agents — a mismatched format
between what Reasoning saw and what Verification later re-checks would undermine the very
consistency the Verification Agent exists to enforce.
"""

from __future__ import annotations

from finagent.data.schemas import EvidenceItem


def format_evidence_block(evidence: list[EvidenceItem]) -> str:
    """Render retrieved/filtered evidence as a numbered, citable block for an LLM prompt.

    Args:
        evidence: Evidence items to render, in the order they should be presented (callers should
            pass evidence already sorted by relevance).

    Returns:
        A string with one evidence entry per line group, each tagged with its `evidence_id` so the
        model can cite it back (e.g. in `ReasoningOutput.citations`), and its company/year/section/
        page so grounding claims can be spot-checked against metadata.
    """
    if not evidence:
        return "(no evidence retrieved)"

    lines: list[str] = []
    for item in evidence:
        chunk = item.chunk
        lines.append(
            f"[{item.evidence_id}] {chunk.company} FY{chunk.year} — {chunk.section} "
            f"(p.{chunk.page_number}, relevance={item.relevance_score:.2f})\n{chunk.text}"
        )
    return "\n\n".join(lines)
