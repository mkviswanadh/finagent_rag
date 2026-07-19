"""Reasoning Agent (Proposal §7.6 Table 7.6, §7.8 step 6, Table 7.16).

Performs numerical interpretation and cross-section reasoning, and drafts a grounded answer, in a
**single** structured Groq call. Per the finagent-architecture skill §0 call-efficiency principle,
splitting "reason about the evidence" and "draft an answer from that reasoning" into two separate
API round-trips would just make the model restate a conclusion it already reached mid-completion —
a well-designed single prompt that asks for reasoning steps AND a draft answer in one structured
response captures the same chain-of-thought benefit at half the call cost. The Answer Generation
Agent (a separate class, matching Proposal Table 7.6's 7-agent architecture) then finalizes this
draft into the returned answer without a further Groq call — see its module docstring.
"""

from __future__ import annotations

from finagent.agents.prompt_helpers import format_evidence_block
from finagent.config import MAX_TOKENS_REASONING, Settings
from finagent.data.schemas import EvidenceItem, LLMCallRecord, QueryAnalysis, ReasoningOutput
from finagent.llm.groq_client import GroqClient

_SYSTEM_PROMPT = """You are the Reasoning Agent inside an adaptive multi-agent financial \
question-answering system. You receive a financial question and a set of evidence chunks retrieved \
from company filings (10-K, 10-Q, 8-K, earnings reports). Your job is to perform the numerical \
interpretation and cross-section reasoning needed to answer the question, and draft a grounded \
answer — you do NOT have access to any information beyond the evidence provided.

Follow these steps internally, then report them:
1. Identify which evidence chunks (by their [EV_xxx] ID) actually contain information relevant to
   the question.
2. Extract the specific numeric or factual values needed, exactly as stated in the evidence
   (preserve units, e.g. "$394.3 billion", "23%" — do not silently convert units).
3. If the question requires a calculation (difference, percentage change, ratio, sum), perform it
   explicitly and show the arithmetic in a reasoning step.
4. If the question requires combining information from multiple evidence chunks or sections, state
   explicitly how they connect.
5. Draft a concise, direct answer to the question, grounded only in the evidence above.

Critical constraints:
- Only use information present in the provided evidence. Do not use outside/prior knowledge about
  the company, even if you believe you know the answer.
- If the evidence is insufficient to answer confidently, set "insufficient_evidence" to true and
  explain what is missing in the reasoning steps, rather than guessing or extrapolating.
- Every numeric claim in your draft answer must be traceable to a specific evidence chunk.
- List the evidence IDs you actually relied on in "citations" — do not cite a chunk you did not use.

Respond with ONLY a single JSON object with exactly these keys:
- "reasoning_steps": array of strings, each one step of your reasoning (numerical interpretation,
  calculation, or cross-section connection), in order.
- "extracted_values": object mapping a short label to each numeric/factual value you extracted,
  e.g. {"fy2022_revenue": "$394.3 billion"}.
- "draft_answer": string, the concise grounded answer to the question.
- "citations": array of evidence ID strings (e.g. ["EV_001", "EV_003"]) actually used.
- "insufficient_evidence": boolean, true only if the evidence cannot support a confident answer."""


class ReasoningAgent:
    """Performs financial reasoning over filtered evidence and drafts a grounded answer."""

    def __init__(self, llm_client: GroqClient, settings: Settings | None = None) -> None:
        self._llm = llm_client
        self._settings = settings or Settings()

    def reason(
        self,
        question: str,
        evidence: list[EvidenceItem],
        query_analysis: QueryAnalysis | None = None,
    ) -> tuple[ReasoningOutput, LLMCallRecord]:
        """Reason over filtered evidence and produce a draft grounded answer.

        Args:
            question: The (possibly refined) user question.
            evidence: Filtered evidence chunks (output of `EvidenceFilteringAgent.filter`).
            query_analysis: Optional Query Understanding output — when available, its
                `question_type`/`needs_calculation` signals are surfaced to the model as a hint to
                focus its reasoning (e.g. explicitly flagging that a calculation is expected).

        Returns:
            A tuple of the parsed `ReasoningOutput` and the `LLMCallRecord` for this one call. If
            the model's JSON response is unusable after `GroqClient`'s bounded repair attempts, a
            `ReasoningOutput` with `insufficient_evidence=True` and an explanatory reasoning step is
            returned instead of raising — a degraded-but-safe result the Verification Agent will
            correctly flag as unsupported, rather than crashing the whole pipeline run.
        """
        hint = ""
        if query_analysis is not None:
            hint = (
                f"\n(Question type: {query_analysis.question_type}; "
                f"calculation required: {query_analysis.needs_calculation})"
            )

        user_prompt = (
            f"Question: {question}{hint}\n\n"
            f"Evidence:\n{format_evidence_block(evidence)}"
        )
        record = self._llm.complete_json(
            agent_name="reasoning",
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=MAX_TOKENS_REASONING,
            temperature=self._settings.temperature,
        )
        parsed = record.parsed_output

        if parsed is None:
            return (
                ReasoningOutput(
                    reasoning_steps=["Reasoning Agent returned an unparseable response."],
                    extracted_values={},
                    draft_answer="",
                    citations=[],
                    insufficient_evidence=True,
                ),
                record,
            )

        reasoning_steps = [str(s) for s in parsed.get("reasoning_steps", []) if str(s).strip()]
        extracted_values = {
            str(k): str(v) for k, v in (parsed.get("extracted_values") or {}).items()
        }
        citations = [str(c) for c in parsed.get("citations", [])]
        output = ReasoningOutput(
            reasoning_steps=reasoning_steps,
            extracted_values=extracted_values,
            draft_answer=str(parsed.get("draft_answer", "")).strip(),
            citations=citations,
            insufficient_evidence=bool(parsed.get("insufficient_evidence", False)),
        )
        return output, record
