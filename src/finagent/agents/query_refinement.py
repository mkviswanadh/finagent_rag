"""Query Refinement Agent (Proposal §7.6 Table 7.6, §7.8).

Two distinct capabilities, both used by the adaptive pipeline's "Query Refinement / Multi-Query if
Needed" stage (Proposal Figure 7.6.1):

1. `refine` — rewrites a broad/ambiguous question into a retrieval-friendly form (single output,
   used when `QueryAnalysis.needs_refinement` is True).
2. `expand_multi_query` — generates several alternative phrasings to widen retrieval coverage
   (used by EXP-10 always, and by the adaptive system on Complex-route questions per Proposal
   Table 7.9's "requires evidence from multiple pages/disclosures" rule).

Each method makes exactly one Groq call — no chaining rewrite-then-expand into two calls when only
one is needed, per the finagent-architecture skill §0 call-efficiency standard.
"""

from __future__ import annotations

import logging

from finagent.agents.prompt_helpers import truncate_for_log
from finagent.config import MAX_TOKENS_MULTI_QUERY, MAX_TOKENS_REWRITE, MULTI_QUERY_VARIANT_COUNT, Settings
from finagent.data.schemas import LLMCallRecord, QueryAnalysis
from finagent.llm.groq_client import GroqClient

logger = logging.getLogger(__name__)

_REFINE_SYSTEM_PROMPT = """You are the Query Refinement Agent inside an adaptive multi-agent \
financial question-answering system. You receive a user's financial question, plus signals about \
it from the Query Understanding Agent, and rewrite it into a single clearer, retrieval-friendly \
query.

Rules:
- Preserve every specific entity already present (company name, year, financial metric) exactly —
  never drop, guess, or change one.
- Resolve vague phrasing into the specific financial statement terminology it most likely refers to
  (e.g. "how much money did they make" -> "net income" or "net sales", based on context).
- Do not answer the question. Do not add information not implied by the original question.
- Output ONLY the rewritten query text, as a single line, with no preamble, quotation marks, or
  explanation."""

_MULTI_QUERY_SYSTEM_PROMPT = """You are the Query Refinement Agent inside an adaptive multi-agent \
financial question-answering system, operating in multi-query expansion mode. You receive a \
financial question and must generate alternative phrasings of it to widen semantic search recall \
across a financial-document knowledge base.

Rules:
- Generate exactly {n} alternative queries, each phrased differently (different terminology,
  different level of specificity, or a different but equally valid way of expressing the same
  underlying information need) — not near-duplicates of each other.
- Every variant must preserve the original question's specific entities (company, year, metric)
  exactly.
- Do not answer the question in any variant.
- Respond with ONLY a single JSON object of the form {{"variants": ["...", "...", "..."]}} with
  exactly {n} strings in the array — no other keys, no prose outside the JSON."""


class QueryRefinementAgent:
    """Rewrites and/or expands financial questions to improve retrieval coverage."""

    def __init__(self, llm_client: GroqClient, settings: Settings | None = None) -> None:
        self._llm = llm_client
        self._settings = settings or Settings()

    def refine(self, question: str, query_analysis: QueryAnalysis) -> tuple[str, LLMCallRecord]:
        """Rewrite a broad/ambiguous question into a single retrieval-friendly query.

        Args:
            question: The original user question.
            query_analysis: Output of the Query Understanding Agent — used to give the rewrite
                prompt the extracted company/year/metric context so the rewrite doesn't have to
                re-derive it from scratch.

        Returns:
            A tuple of the rewritten query string and the `LLMCallRecord` for this one call. If
            the model's response is empty after stripping whitespace, the original `question` is
            returned unchanged rather than an empty string.
        """
        context_hint = (
            f"(Known context — company: {query_analysis.company or 'unknown'}, "
            f"year: {query_analysis.year or 'unknown'}, metric: {query_analysis.metric or 'unknown'})"
        )
        record = self._llm.complete(
            agent_name="query_refinement",
            system_prompt=_REFINE_SYSTEM_PROMPT,
            user_prompt=f"Original question: {question}\n{context_hint}",
            max_tokens=MAX_TOKENS_REWRITE,
            temperature=self._settings.temperature,
        )
        rewritten = record.raw_response.strip().strip('"')
        record.parsed_output = rewritten or question
        logger.info(
            "Query Refinement: rewrote %r -> %r in %.2fs, %d tokens",
            truncate_for_log(question), truncate_for_log(rewritten or question),
            record.latency_seconds, record.total_tokens,
        )
        return (rewritten or question), record

    def expand_multi_query(
        self, question: str, n: int = MULTI_QUERY_VARIANT_COUNT
    ) -> tuple[list[str], LLMCallRecord]:
        """Generate `n` alternative phrasings of a question to widen retrieval coverage.

        Args:
            question: The original (or already-refined) question to expand.
            n: Number of variants to generate. Defaults to `config.MULTI_QUERY_VARIANT_COUNT` (3),
                matching EXP-10's fixed design ("The LLM will generate 3 alternative query
                versions" — Coding_Sheet Final Guidance Sheet).

        Returns:
            A tuple of (list of `n` variant query strings, the `LLMCallRecord` for this one call).
            Falls back to `n` copies of the original `question` if the model's JSON response is
            unusable after `GroqClient`'s bounded repair attempts — this keeps downstream retrieval
            functional (querying the same text `n` times is a safe degraded mode) rather than
            raising and aborting the whole pipeline over a single malformed generation.
        """
        record = self._llm.complete_json(
            agent_name="query_refinement_multi",
            system_prompt=_MULTI_QUERY_SYSTEM_PROMPT.format(n=n),
            user_prompt=f"Question: {question}",
            max_tokens=MAX_TOKENS_MULTI_QUERY,
            temperature=self._settings.temperature,
        )
        parsed = record.parsed_output or {}
        variants = parsed.get("variants") if isinstance(parsed, dict) else None
        if not isinstance(variants, list) or not variants:
            variants = [question] * n
        else:
            variants = [str(v) for v in variants][:n]
            while len(variants) < n:
                variants.append(question)
        logger.info(
            "Query Refinement: expanded %r into %d variants in %.2fs, %d tokens",
            truncate_for_log(question), len(variants), record.latency_seconds, record.total_tokens,
        )
        return variants, record
