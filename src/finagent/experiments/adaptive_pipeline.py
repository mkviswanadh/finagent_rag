"""Parameterized RAG pipeline covering EXP-07 through EXP-14 (8 of the 14 experiments).

One class, `AdaptiveRAGPipeline`, configured by `PipelineConfig` — not eight copy-pasted files.
This is what the finagent-experiments skill §0a requires: "EXP-12/13/14 must literally reuse
EXP-11's code path with one stage's `enabled=False`, proving the ablation is real rather than a
rewritten approximation." The same reasoning extends naturally to EXP-07..10, which are just
different, simpler points on the same retrieval-strategy spectrum (see `retrieval_strategy` below).

Each `PipelineConfig` instance corresponds to exactly one experiment's pipeline string from
`Coding_Sheet.xlsx`'s Final Guidance Sheet — see `registry.py` for the 8 configured instances.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from finagent.agents.answer_generation import AnswerGenerationAgent
from finagent.agents.evidence_filtering import EvidenceFilteringAgent
from finagent.agents.query_refinement import QueryRefinementAgent
from finagent.agents.query_understanding import QueryUnderstandingAgent
from finagent.agents.reasoning import ReasoningAgent
from finagent.agents.retrieval import RetrievalAgent
from finagent.agents.verification import VerificationAgent
from finagent.config import RETRIEVAL_TOP_K, Settings
from finagent.data.schemas import (
    EvidenceItem,
    FinanceBenchQuestion,
    LLMCallRecord,
    PipelineTrace,
    QueryAnalysis,
    QueryComplexity,
)
from finagent.document_processing.vector_store import ChromaVectorStore
from finagent.experiments.base import BaseExperiment
from finagent.llm.groq_client import GroqClient

# A neutral placeholder used only for experiments that skip Query Understanding entirely (EXP-07,
# EXP-09, EXP-10 per their Coding_Sheet pipeline strings — none of them extract entities or route
# by complexity). Passed to QueryRefinementAgent methods that require a QueryAnalysis argument;
# its all-unknown fields mean the refinement prompt's context hint degrades gracefully to "unknown"
# rather than fabricating entity values that were never actually extracted for these experiments.
_NO_ANALYSIS = QueryAnalysis(
    complexity=QueryComplexity.SIMPLE,
    company=None,
    year=None,
    metric=None,
    question_type="lookup",
    needs_calculation=False,
    needs_multiple_evidence_chunks=False,
    needs_refinement=False,
    routing_rationale="Query Understanding not used by this experiment.",
)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration distinguishing EXP-07..14's pipelines — see class module docstring.

    Attributes:
        experiment_id: e.g. "EXP-07".
        experiment_name: Full name matching `Coding_Sheet.xlsx`.
        retrieval_type_label: Pre-filled "Retrieval Type" value for the Retrieval and Evidence
            Grounding sheet (finagent-experiments skill §3b) — kept here so the experiment
            definition is the single source of truth for it.
        retrieval_strategy: One of "naive" (EXP-07: unfiltered top-k, original question),
            "metadata" (EXP-08: entity-extracted metadata filter + top-k), "rewritten" (EXP-09:
            always rewrite then retrieve), "multiquery" (EXP-10: always expand to 3 variants,
            merge), or "adaptive" (EXP-11..14: routed by Query Understanding's complexity tier).
        use_query_understanding: Whether the Query Understanding Agent runs at all. `False` for
            EXP-07/09/10 (their Coding_Sheet pipeline strings have no query-understanding step).
        enable_query_refinement: Capability switch — `False` only for EXP-12 (the "without Query
            Refinement" ablation). For "adaptive" strategy, this gates whether refinement/
            multi-query can ever fire, independent of what the router would otherwise choose.
        enable_evidence_filtering: Capability switch — `False` only for EXP-13.
        enable_verification: Capability switch — `False` only for EXP-14.
        top_k: Retrieval depth. Fixed at `config.RETRIEVAL_TOP_K` (5) for every experiment.
    """

    experiment_id: str
    experiment_name: str
    retrieval_type_label: str
    retrieval_strategy: str
    use_query_understanding: bool
    enable_query_refinement: bool
    enable_evidence_filtering: bool
    enable_verification: bool
    top_k: int = RETRIEVAL_TOP_K


class AdaptiveRAGPipeline(BaseExperiment):
    """Runs one of EXP-07..14's retrieval+reasoning pipelines, per its `PipelineConfig`."""

    def __init__(
        self,
        config: PipelineConfig,
        llm_client: GroqClient,
        vector_store: ChromaVectorStore,
        settings: Settings | None = None,
    ) -> None:
        self._config = config
        self.experiment_id = config.experiment_id
        self.experiment_name = config.experiment_name
        self._settings = settings or Settings()

        self._query_understanding = QueryUnderstandingAgent(llm_client, self._settings)
        self._query_refinement = QueryRefinementAgent(llm_client, self._settings)
        self._retrieval = RetrievalAgent(vector_store)
        self._evidence_filtering = EvidenceFilteringAgent()
        self._reasoning = ReasoningAgent(llm_client, self._settings)
        self._answer_generation = AnswerGenerationAgent()
        self._verification = VerificationAgent(llm_client, self._settings)

    def _run_trace(self, question: FinanceBenchQuestion) -> PipelineTrace:
        qu_result: tuple[QueryAnalysis, LLMCallRecord] | None = None
        if self._config.use_query_understanding:
            qu_result = self._query_understanding.analyze(question.question)
        analysis = qu_result[0] if qu_result else None

        trace = PipelineTrace(
            experiment_id=self.experiment_id,
            question=question,
            complexity_used=(analysis.complexity if analysis else question.assigned_complexity)
            or QueryComplexity.SIMPLE,
        )
        if qu_result is not None:
            trace.llm_calls.append(qu_result[1])
            trace.query_analysis = analysis

        queries, refined_query = self._prepare_queries(question.question, analysis, trace)
        trace.query_variants = queries
        trace.refined_query = refined_query

        metadata_filter = None
        if self._config.retrieval_strategy in ("metadata", "adaptive") and analysis is not None:
            metadata_filter = self._retrieval.build_metadata_filter(analysis)

        retrieved: list[EvidenceItem]
        if len(queries) > 1:
            retrieved = self._retrieval.retrieve_multi(
                queries, top_k_per_query=self._config.top_k, metadata_filter=metadata_filter
            )
        else:
            retrieved = self._retrieval.retrieve(
                queries[0], top_k=self._config.top_k, metadata_filter=metadata_filter
            )
        trace.retrieved_evidence = retrieved

        if self._config.enable_evidence_filtering:
            filtered = self._evidence_filtering.filter(retrieved, max_items=self._config.top_k)
        else:
            filtered = retrieved
        trace.filtered_evidence = filtered

        reasoning_output, reasoning_call = self._reasoning.reason(question.question, filtered, analysis)
        trace.llm_calls.append(reasoning_call)
        trace.reasoning_output = reasoning_output

        answer = self._answer_generation.generate(reasoning_output, filtered)

        if self._config.enable_verification and not reasoning_output.insufficient_evidence:
            verification_result, verification_call = self._verification.verify(
                question.question, answer, filtered
            )
            trace.llm_calls.append(verification_call)
            trace.verification_result = verification_result

        trace.generated_answer = answer
        return trace

    def _prepare_queries(
        self,
        question_text: str,
        analysis: QueryAnalysis | None,
        trace: PipelineTrace,
    ) -> tuple[list[str], str | None]:
        """Decide which query string(s) to retrieve with, per `retrieval_strategy`.

        Returns:
            A tuple of (list of query strings to retrieve with — more than one triggers
            `retrieve_multi`, merging results — and the refined single-query string if one was
            produced, else `None`). Every Groq call made here is appended to `trace.llm_calls`.
        """
        strategy = self._config.retrieval_strategy

        if strategy in ("naive", "metadata"):
            return [question_text], None

        if strategy == "rewritten":
            refined, call = self._query_refinement.refine(question_text, _NO_ANALYSIS)
            trace.llm_calls.append(call)
            return [refined], refined

        if strategy == "multiquery":
            variants, call = self._query_refinement.expand_multi_query(question_text)
            trace.llm_calls.append(call)
            return variants, None

        # strategy == "adaptive" (EXP-11..14) — routed by Query Understanding's complexity tier.
        assert analysis is not None, "adaptive strategy requires use_query_understanding=True"

        if not self._config.enable_query_refinement:
            return [question_text], None  # EXP-12: capability disabled regardless of routing

        if analysis.complexity == QueryComplexity.SIMPLE:
            return [question_text], None

        if analysis.complexity == QueryComplexity.MODERATE:
            if not analysis.needs_refinement:
                return [question_text], None
            refined, call = self._query_refinement.refine(question_text, analysis)
            trace.llm_calls.append(call)
            return [refined], refined

        # COMPLEX: refine first (if the question reads as ambiguous/broad), then expand from the
        # refined form — Proposal §7.7/7.8: complex questions activate "query refinement,
        # multi-query expansion... and deeper financial reasoning" together, not one or the other.
        base_query = question_text
        refined_query: str | None = None
        if analysis.needs_refinement:
            refined_query, call = self._query_refinement.refine(question_text, analysis)
            trace.llm_calls.append(call)
            base_query = refined_query

        variants, call2 = self._query_refinement.expand_multi_query(base_query)
        trace.llm_calls.append(call2)
        return variants, refined_query
