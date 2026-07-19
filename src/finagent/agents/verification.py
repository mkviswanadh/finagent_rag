"""Verification Agent (Proposal §7.6 Table 7.6, §7.8 step 8).

Makes exactly one Groq call. Unlike Evidence Filtering (numeric threshold + dedup — no LLM
judgment needed), validating whether a *generated answer's claims* are actually supported by
evidence requires the kind of nuanced textual entailment judgment an LLM is suited for and a
simple heuristic is not (e.g. recognizing that "revenue grew" is supported by "$394.3B in 2022 vs
$365.8B in 2021" even though neither evidence sentence contains the word "grew"). This is exactly
the stage EXP-14 ablates ("without Verification Agent") to measure — the proposal expects
hallucination rate and citation correctness to visibly worsen without it (Coding_Sheet Final
Guidance Sheet, EXP-14 "Expected Understanding").
"""

from __future__ import annotations

from finagent.agents.prompt_helpers import format_evidence_block
from finagent.config import MAX_TOKENS_VERIFICATION, Settings
from finagent.data.schemas import EvidenceItem, LLMCallRecord, VerificationResult
from finagent.llm.groq_client import GroqClient

_SYSTEM_PROMPT = """You are the Verification Agent inside an adaptive multi-agent financial \
question-answering system. You receive a generated answer and the evidence chunks it was supposed \
to be grounded in. Your job is to check whether every factual and numerical claim in the answer is \
actually supported by that evidence — you are the last check before the answer is returned to the \
user.

For each distinct claim in the answer:
- Check whether the evidence directly states it, or whether it is a valid direct inference from the
  evidence (e.g. a stated percentage change computed from two stated values counts as supported).
- A claim is UNSUPPORTED if the evidence does not contain it, contradicts it, or if the answer adds
  specificity (an exact number, a named driver/cause) that the evidence does not actually provide.
- Do not use your own outside knowledge of the company to judge correctness — judge only whether the
  provided evidence supports the claim.

Respond with ONLY a single JSON object with exactly these keys:
- "passed": boolean — true only if ALL claims in the answer are supported by the evidence.
- "unsupported_claims": array of strings, each a specific claim from the answer that is not
  supported (empty array if none).
- "confidence": number between 0 and 1, your confidence in this verification judgment.
- "notes": string, one short sentence explaining the verdict."""


class VerificationAgent:
    """Validates whether a generated answer is fully supported by the evidence it cites."""

    def __init__(self, llm_client: GroqClient, settings: Settings | None = None) -> None:
        self._llm = llm_client
        self._settings = settings or Settings()

    def verify(
        self, question: str, answer: str, evidence: list[EvidenceItem]
    ) -> tuple[VerificationResult, LLMCallRecord]:
        """Check whether `answer`'s claims are supported by `evidence`.

        Args:
            question: The original user question (context for what the answer is meant to address).
            answer: The generated answer to verify (typically `AnswerGenerationAgent.generate`'s
                output, though citation footers don't affect verification — only the claims matter).
            evidence: The evidence the answer was generated from.

        Returns:
            A tuple of the parsed `VerificationResult` and the `LLMCallRecord` for this one call.
            If the model's JSON response is unusable after `GroqClient`'s bounded repair attempts,
            returns a conservative `VerificationResult(passed=False, ...)` — an unparseable
            verification response must fail closed (treat as unverified/unsupported), never fail
            open (silently treat as passed), since this is the pipeline's last safety check.
        """
        user_prompt = (
            f"Question: {question}\n\nGenerated Answer: {answer}\n\n"
            f"Evidence:\n{format_evidence_block(evidence)}"
        )
        record = self._llm.complete_json(
            agent_name="verification",
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=MAX_TOKENS_VERIFICATION,
            temperature=self._settings.temperature,
        )
        parsed = record.parsed_output

        if parsed is None:
            return (
                VerificationResult(
                    passed=False,
                    unsupported_claims=["Verification Agent returned an unparseable response."],
                    confidence=0.0,
                    notes="Verification failed closed due to an unparseable model response.",
                ),
                record,
            )

        result = VerificationResult(
            passed=bool(parsed.get("passed", False)),
            unsupported_claims=[str(c) for c in parsed.get("unsupported_claims", [])],
            confidence=float(parsed.get("confidence", 0.0) or 0.0),
            notes=str(parsed.get("notes", "")).strip(),
        )
        return result, record
