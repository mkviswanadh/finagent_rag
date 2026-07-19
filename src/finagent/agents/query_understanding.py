"""Query Understanding Agent (Proposal §7.6 Table 7.6, §7.7 Tables 7.7-7.9).

Responsible for identifying query intent, financial entity signals, and — critically — the
adaptive-routing complexity tier that determines how much of the rest of the pipeline runs. This
is the single most consequential classification in the whole system: an under-classified Complex
question that gets routed as Simple will silently skip refinement/multi-query/deeper evidence
gathering and produce a shallow answer, while an over-classified Simple question just costs a
little extra latency. The routing logic is therefore implemented as a hybrid: one structured Groq
call for entity extraction and an initial complexity judgment, combined with deterministic
keyword/pattern rules taken directly from Proposal Table 7.9 that can only ever *upgrade* the
complexity tier, never downgrade it — trusting the model's fluent judgment while guaranteeing the
literal routing rules from the proposal are never silently missed.
"""

from __future__ import annotations

import re

from finagent.config import MAX_TOKENS_CLASSIFICATION, Settings
from finagent.data.schemas import QueryAnalysis, QueryComplexity
from finagent.llm.groq_client import GroqClient
from finagent.data.schemas import LLMCallRecord

_SYSTEM_PROMPT = """You are the Query Understanding Agent inside an adaptive multi-agent financial \
question-answering system. Your job is to analyze one user question about a company's financial \
filings (10-K, 10-Q, 8-K, earnings report) and extract the signals the rest of the system needs to \
route and answer it correctly. You do not answer the question yourself.

Extract the following signals, exactly as defined:
- company: the company name or ticker mentioned or clearly implied, or null if none is identifiable.
- year: the single most relevant fiscal/reporting year as a 4-digit integer, or null if none is \
identifiable or if multiple years are being compared (in that case still return the primary/most \
recent year mentioned).
- metric: the specific financial metric or line item being asked about (e.g. "revenue", \
"operating income", "net income", "total assets"), or null if the question is not about a specific \
metric.
- question_type: exactly one of "lookup", "comparison", "explanation", "trend", "reasoning".
  - lookup: a single direct factual/numerical value is requested.
  - comparison: two or more values/periods are being compared.
  - explanation: the question asks "why" or for factors/drivers behind a result.
  - trend: the question asks about a pattern or direction over multiple periods.
  - reasoning: the question requires combining/deriving information not directly stated.
- needs_calculation: true if answering requires arithmetic (subtraction, percentage change, ratio, \
sum) rather than a direct lookup.
- needs_multiple_evidence_chunks: true if the answer plausibly requires information from more than \
one section, statement, page, or reporting period of the filing.
- complexity: exactly one of "Simple", "Moderate", "Complex", applying these rules in order:
  1. "Simple": exactly one company, one year, one metric, and a direct lookup with no calculation.
  2. "Moderate": involves multiple years/periods, a comparison, or requires a calculation but stays \
within a single topic/section.
  3. "Complex": asks "why"/"explain"/"compare"/"trend"/"reason", OR requires values from different \
sections or statements, OR requires evidence from multiple pages/disclosures.
- needs_refinement: true if the question as written is broad, ambiguous, uses informal or vague \
phrasing, or does not closely match how the information would be phrased in a financial filing \
(and would therefore benefit from being rewritten into a more retrieval-friendly form before \
searching the knowledge base).
- routing_rationale: one concise sentence explaining the complexity decision.

Respond with ONLY a single JSON object with exactly these keys: company, year, metric, \
question_type, needs_calculation, needs_multiple_evidence_chunks, complexity, needs_refinement, \
routing_rationale. Use JSON null (not the string "null") for unknown company/year/metric. Use JSON \
true/false (not strings) for the boolean fields."""

_COMPLEX_TRIGGER_PATTERN = re.compile(
    r"\b(why|explain|compare|comparison|trend|reason|factors?|drove|driving|contribut\w*)\b",
    re.IGNORECASE,
)
_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

_VALID_QUESTION_TYPES = {"lookup", "comparison", "explanation", "trend", "reasoning"}
_VALID_COMPLEXITIES = {c.value for c in QueryComplexity}


class QueryUnderstandingAgent:
    """Extracts financial-entity signals and assigns an adaptive-routing complexity tier."""

    def __init__(self, llm_client: GroqClient, settings: Settings | None = None) -> None:
        self._llm = llm_client
        self._settings = settings or Settings()

    def analyze(self, question: str) -> tuple[QueryAnalysis, LLMCallRecord]:
        """Classify a question's complexity and extract its financial-entity signals.

        Args:
            question: The raw user financial question.

        Returns:
            A tuple of the resolved `QueryAnalysis` (after deterministic rule reconciliation) and
            the `LLMCallRecord` documenting the single Groq call this method makes.
        """
        record = self._llm.complete_json(
            agent_name="query_understanding",
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=f"Question: {question}",
            max_tokens=MAX_TOKENS_CLASSIFICATION,
            temperature=self._settings.temperature,
        )
        parsed = record.parsed_output or {}

        question_type = str(parsed.get("question_type", "lookup")).strip().lower()
        if question_type not in _VALID_QUESTION_TYPES:
            question_type = "lookup"

        llm_complexity_raw = str(parsed.get("complexity", "Simple")).strip().title()
        llm_complexity = (
            QueryComplexity(llm_complexity_raw)
            if llm_complexity_raw in _VALID_COMPLEXITIES
            else QueryComplexity.SIMPLE
        )

        needs_calculation = bool(parsed.get("needs_calculation", False))
        needs_multiple_evidence_chunks = bool(parsed.get("needs_multiple_evidence_chunks", False))
        needs_refinement = bool(parsed.get("needs_refinement", False))

        complexity = _apply_deterministic_routing_rules(
            question=question,
            llm_complexity=llm_complexity,
            needs_calculation=needs_calculation,
            needs_multiple_evidence_chunks=needs_multiple_evidence_chunks,
        )

        year_raw = parsed.get("year")
        year = int(year_raw) if isinstance(year_raw, (int, float)) else None
        company_raw = parsed.get("company")
        company = str(company_raw) if company_raw else None
        metric_raw = parsed.get("metric")
        metric = str(metric_raw) if metric_raw else None

        analysis = QueryAnalysis(
            complexity=complexity,
            company=company,
            year=year,
            metric=metric,
            question_type=question_type,
            needs_calculation=needs_calculation,
            needs_multiple_evidence_chunks=needs_multiple_evidence_chunks,
            needs_refinement=needs_refinement,
            routing_rationale=str(parsed.get("routing_rationale", "")).strip(),
        )
        return analysis, record


def _apply_deterministic_routing_rules(
    *,
    question: str,
    llm_complexity: QueryComplexity,
    needs_calculation: bool,
    needs_multiple_evidence_chunks: bool,
) -> QueryComplexity:
    """Reconcile the LLM's complexity judgment with Proposal Table 7.9's literal routing rules.

    Only ever escalates (Simple -> Moderate -> Complex), never downgrades — an LLM that judges a
    question Complex is trusted, but a question containing an explicit "why"/"explain"/"compare"/
    "trend" trigger, multiple distinct years, or a signal for cross-section evidence is guaranteed
    to be routed at least as deep as Table 7.9 requires even if the model under-called it.
    """
    order = [QueryComplexity.SIMPLE, QueryComplexity.MODERATE, QueryComplexity.COMPLEX]
    resolved = llm_complexity

    def _escalate(target: QueryComplexity) -> None:
        nonlocal resolved
        if order.index(target) > order.index(resolved):
            resolved = target

    if _COMPLEX_TRIGGER_PATTERN.search(question):
        _escalate(QueryComplexity.COMPLEX)
    if needs_multiple_evidence_chunks:
        _escalate(QueryComplexity.COMPLEX)

    # _YEAR_PATTERN has a non-capturing-equivalent group used only to anchor the century prefix;
    # `finditer` + `.group(0)` is used (not `findall`) to recover the full 4-digit match.
    distinct_years = {m.group(0) for m in _YEAR_PATTERN.finditer(question)}
    if len(distinct_years) >= 2:
        _escalate(QueryComplexity.MODERATE)

    if needs_calculation:
        _escalate(QueryComplexity.MODERATE)

    return resolved
