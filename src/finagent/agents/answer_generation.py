"""Answer Generation Agent (Proposal §7.6 Table 7.6, §7.8 step 7).

Makes **zero** additional Groq calls in the normal case. The Reasoning Agent's single structured
call already produces a grounded `draft_answer` (see `reasoning.py` module docstring for the
call-efficiency rationale) — this agent's job per Proposal Table 7.6 is "produces the final grounded
financial response", which in this implementation means finalizing that draft (attaching evidence
citations in a consistent, traceable format) rather than re-asking the LLM to generate an answer it
already produced. The two agents remain separate classes, matching the proposal's distinct 7-agent
architecture and keeping ablation/tracing boundaries clean, even though the LLM cost is concentrated
in the Reasoning Agent's call.

The one case this agent DOES fall back to a plain-text answer without evidence: when reasoning
flagged `insufficient_evidence` — in that case no fabricated grounded answer is produced.
"""

from __future__ import annotations

from finagent.data.schemas import EvidenceItem, ReasoningOutput

_INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The retrieved evidence does not contain enough information to answer this question "
    "confidently."
)


class AnswerGenerationAgent:
    """Finalizes the Reasoning Agent's draft answer into the pipeline's returned response."""

    def generate(
        self,
        reasoning_output: ReasoningOutput,
        evidence: list[EvidenceItem],
        *,
        include_citations: bool = True,
    ) -> str:
        """Produce the final answer text from reasoning output and its cited evidence.

        Args:
            reasoning_output: Output of `ReasoningAgent.reason`.
            evidence: The evidence list the reasoning was performed over — used to resolve citation
                IDs to human-readable source references (company, filing type, page).
            include_citations: Whether to append a source citation footer. Set `False` for
                experiments/ablations that don't evaluate citation correctness, to keep the answer
                text itself directly comparable to the reference answer for exact-match/F1 scoring.

        Returns:
            The finalized answer string. If `reasoning_output.insufficient_evidence` is True, a
            fixed "insufficient evidence" message is returned instead of fabricating a claim.
        """
        if reasoning_output.insufficient_evidence or not reasoning_output.draft_answer:
            return _INSUFFICIENT_EVIDENCE_MESSAGE

        if not include_citations or not reasoning_output.citations:
            return reasoning_output.draft_answer

        citation_footer = self._build_citation_footer(reasoning_output.citations, evidence)
        if not citation_footer:
            return reasoning_output.draft_answer

        return f"{reasoning_output.draft_answer}\n\nSources: {citation_footer}"

    @staticmethod
    def _build_citation_footer(citation_ids: list[str], evidence: list[EvidenceItem]) -> str:
        evidence_by_id = {e.evidence_id: e for e in evidence}
        references = []
        for citation_id in citation_ids:
            item = evidence_by_id.get(citation_id)
            if item is None:
                continue
            chunk = item.chunk
            references.append(f"{chunk.company} {chunk.report_type} FY{chunk.year} (p.{chunk.page_number})")
        return "; ".join(references)
