"""Typed data contracts shared by every module in the FinAgent-RAG codebase.

Centralizing these dataclasses/enums is what lets the document-processing pipeline, the 7 agents,
the metrics library, and all 14 experiment runners interoperate without ad-hoc dicts drifting out
of sync. Every field here traces back to a specific table in Kasi_Research_Proposal.pdf (cited in
each docstring) so implementers can verify fidelity without re-reading the proposal.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator


class QueryComplexity(str, Enum):
    """Three-tier query complexity classification (Proposal Table 7.7, §7.7).

    SIMPLE:   direct factual/numerical lookup, one company + one year + one metric.
    MODERATE: comparison across periods, or multiple related values.
    COMPLEX:  multi-step / multi-section reasoning, explanation-oriented ("why", "explain",
              "compare", "trend"), or requires evidence from multiple pages/disclosures.
    """

    SIMPLE = "Simple"
    MODERATE = "Moderate"
    COMPLEX = "Complex"


class AgentName(str, Enum):
    """The 7 agents of the proposed architecture (Proposal Table 7.6)."""

    QUERY_UNDERSTANDING = "query_understanding"
    QUERY_REFINEMENT = "query_refinement"
    RETRIEVAL = "retrieval"
    EVIDENCE_FILTERING = "evidence_filtering"
    REASONING = "reasoning"
    ANSWER_GENERATION = "answer_generation"
    VERIFICATION = "verification"


@dataclass(frozen=True)
class Chunk:
    """A single retrieval-ready chunk of a financial document (Proposal Table 7.5).

    Attributes:
        chunk_id: Stable identifier, e.g. "AAPL_2022_AR_CH_015" (Company_Year_ReportType_CH_seq).
        company: Company name the source filing belongs to.
        year: Filing year.
        report_type: e.g. "10-K", "10-Q", "8-K", "Earnings Report".
        section: Detected section name, e.g. "Consolidated Statements of Operations".
        page_number: Page in the source PDF the chunk text was extracted from.
        text: The chunk's raw text content (~500 tokens per finagent-architecture skill §1).
        source_document: Filename of the source PDF this chunk was extracted from.
    """

    chunk_id: str
    company: str
    year: int
    report_type: str
    section: str
    page_number: int
    text: str
    source_document: str


@dataclass(frozen=True)
class EvidenceItem:
    """A retrieved (and possibly filtered) piece of evidence (Proposal Table 7.10).

    Attributes:
        evidence_id: Unique ID for this retrieval hit, e.g. "EV_001".
        chunk: The underlying `Chunk` this evidence item wraps.
        relevance_score: Similarity score from the retriever (0.0-1.0 for cosine-normalized
            embeddings).
        retrieval_query: The exact query string that retrieved this chunk (useful for tracing
            which of several multi-query variants surfaced it).
    """

    evidence_id: str
    chunk: Chunk
    relevance_score: float
    retrieval_query: str


@dataclass(frozen=True)
class EvidenceReference:
    """One ground-truth evidence excerpt as annotated in `financebench_open_source.jsonl`.

    The real dataset's `evidence` field is a *list* of these (23% of the 150 open-source questions
    have 2-3 excerpts spanning different pages, not a single page) — modeling it as a list rather
    than a single string+page_number, as the proposal's condensed Table 7.3 might suggest, is
    required for context-recall evaluation to correctly check whether retrieval covered *every*
    annotated evidence page, not just one.
    """

    doc_name: str
    page_number: int | None
    text: str


@dataclass
class FinanceBenchQuestion:
    """One FinanceBench QA record, loaded from `financebench_open_source.jsonl` and joined with
    `financebench_document_information.jsonl` (Proposal Table 7.3, §7.4.2; see
    `finagent.data.financebench_loader` for the exact field mapping from raw JSON to this schema).

    `assigned_complexity` is populated by the Query Understanding Agent's classification
    (Proposal §7.7) at run time, not part of the source dataset — kept here so a single question
    object can flow through a whole pipeline run without a second lookup structure.
    """

    question_id: str
    question: str
    reference_answer: str
    evidence: list[EvidenceReference]
    company: str
    document_type: str
    document_name: str
    document_year: int | None
    gics_sector: str
    justification: str
    dataset_question_type: str
    question_reasoning: str = ""
    assigned_complexity: QueryComplexity | None = None

    @property
    def evidence_page_numbers(self) -> list[int]:
        """Every distinct page number annotated as evidence, for context-recall scoring."""
        return sorted({e.page_number for e in self.evidence if e.page_number is not None})

    @property
    def evidence_text_combined(self) -> str:
        """All evidence excerpts concatenated, for faithfulness/verification prompts."""
        return "\n\n".join(e.text for e in self.evidence)


@dataclass
class LLMCallRecord:
    """Full trace of one Groq API call, used for both auditability and cost accounting.

    Every field required to reconstruct "what exactly was sent to the model and what came back"
    lives here, per the finagent-experiments skill §0a requirement to record raw prompts/responses
    for error analysis (Proposal §7.11 step 14), not just the final parsed answer.
    """

    agent_name: str
    model: str
    system_prompt: str
    user_prompt: str
    raw_response: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    temperature: float
    parsed_output: Any = None
    retries: int = 0
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class QueryAnalysis:
    """Output of the Query Understanding Agent (Proposal §7.6, Table 7.8, Table 7.9).

    Attributes:
        complexity: Routed complexity tier.
        company, year, metric: Extracted financial entity signals.
        question_type: One of lookup / comparison / explanation / trend / reasoning.
        needs_calculation: Whether arithmetic/comparison operations are required.
        needs_multiple_evidence_chunks: Whether info must be drawn from multiple sections/pages.
        needs_refinement: Whether the Query Refinement Agent should be invoked.
        routing_rationale: Short natural-language justification, kept for error analysis.
    """

    complexity: QueryComplexity
    company: str | None
    year: int | None
    metric: str | None
    question_type: str
    needs_calculation: bool
    needs_multiple_evidence_chunks: bool
    needs_refinement: bool
    routing_rationale: str


@dataclass
class ReasoningOutput:
    """Output of the Reasoning Agent (Proposal §7.6 Table 7.6, §7.8 step 6).

    Produced by a single structured Groq call that performs numerical interpretation and
    cross-section reasoning AND drafts a grounded answer in the same pass (finagent-architecture
    skill §0 call-efficiency principle: the model naturally reasons before answering, so asking it
    to do both in one structured response avoids a second round-trip that would just re-ask it to
    restate a conclusion it already reached).

    Attributes:
        reasoning_steps: Ordered natural-language reasoning steps (numerical interpretation,
            comparison, aggregation, or explanation logic).
        extracted_values: Numeric/factual values pulled from evidence, keyed by a short label
            (e.g. {"2022_revenue": "$394.3 billion"}) — used by numerical-accuracy scoring.
        draft_answer: The grounded natural-language answer synthesized from the reasoning.
        citations: Evidence IDs (`EvidenceItem.evidence_id`) actually relied upon in the answer —
            used for citation-correctness scoring (Proposal Table 7.15).
        insufficient_evidence: True if the agent determined the filtered evidence could not
            support a confident answer — surfaced rather than silently guessing.
    """

    reasoning_steps: list[str]
    extracted_values: dict[str, str]
    draft_answer: str
    citations: list[str]
    insufficient_evidence: bool = False


@dataclass
class VerificationResult:
    """Output of the Verification Agent (Proposal §7.6 Table 7.6, §7.8 step 8).

    Attributes:
        passed: Whether the answer is judged fully supported by the retrieved evidence.
        unsupported_claims: Specific claims in the answer the verifier could not ground in
            evidence — feeds `Hallucination Rate` (Proposal Table 7.15).
        confidence: Verifier's confidence in its own judgment, in [0, 1].
        notes: Short free-text rationale, kept for error analysis (Proposal §7.11 step 14).
    """

    passed: bool
    unsupported_claims: list[str]
    confidence: float
    notes: str


@dataclass
class PipelineTrace:
    """The complete, ordered record of one question's run through an experiment's pipeline.

    This is the unit that both the metrics library (finagent-architecture skill §7) and the
    results writer (finagent-experiments skill §3) consume — every metric and every result-sheet
    cell is derived from one or more `PipelineTrace` instances, never recomputed from scratch.
    """

    experiment_id: str
    question: FinanceBenchQuestion
    complexity_used: QueryComplexity
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    retrieved_evidence: list[EvidenceItem] = field(default_factory=list)
    filtered_evidence: list[EvidenceItem] = field(default_factory=list)
    query_variants: list[str] = field(default_factory=list)
    refined_query: str | None = None
    query_analysis: QueryAnalysis | None = None
    reasoning_output: ReasoningOutput | None = None
    generated_answer: str = ""
    verification_result: VerificationResult | None = None
    stage_timings: dict[str, float] = field(default_factory=dict)
    """Wall-clock seconds spent in each named pipeline stage (e.g. "query_understanding",
    "retrieval", "reasoning") — populated via `timed_stage`. Distinct from `llm_calls[*].latency_seconds`,
    which times only the Groq round-trip; a stage's timing also covers any local work around that
    call (e.g. building a metadata filter, merging multi-query results)."""
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def mark_finished(self) -> None:
        self.finished_at = time.time()

    @contextmanager
    def timed_stage(self, stage_name: str) -> Iterator[None]:
        """Record wall-clock time spent in a named stage into `stage_timings`.

        Usage: `with trace.timed_stage("retrieval"): evidence = retrieval_agent.retrieve(...)`.
        If the same stage name is used more than once in one trace (e.g. a stage that runs
        conditionally in a loop), the durations accumulate rather than overwrite.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.stage_timings[stage_name] = self.stage_timings.get(stage_name, 0.0) + elapsed

    @property
    def total_latency_seconds(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.llm_calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.llm_calls)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def retrieved_chunk_ids(self) -> list[str]:
        return [e.chunk.chunk_id for e in self.retrieved_evidence]

    @property
    def filtered_chunk_ids(self) -> list[str]:
        return [e.chunk.chunk_id for e in self.filtered_evidence]


@dataclass
class QuestionResult:
    """Final per-question record: the trace plus every computed metric.

    `metrics` keys match the metric function names in `finagent.metrics` exactly (e.g.
    "answer_relevance", "faithfulness", "numerical_accuracy") so the results writer can look them
    up generically instead of hardcoding per-sheet field mappings for each metric.
    """

    trace: PipelineTrace
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def experiment_id(self) -> str:
        return self.trace.experiment_id

    @property
    def complexity(self) -> QueryComplexity:
        return self.trace.complexity_used
