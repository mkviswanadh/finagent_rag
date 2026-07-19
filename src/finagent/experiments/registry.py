"""The 14-experiment registry (finagent-experiments skill §1) — EXP-01 through EXP-14.

Single source of truth mapping each experiment ID to how it's constructed: the 6 Direct LLM
experiments share `DirectLLMExperiment` with a different prompt each (`direct_llm_prompts.py`);
the 8 RAG experiments share `AdaptiveRAGPipeline` with a different `PipelineConfig` each
(`adaptive_pipeline.py`). No experiment has its own bespoke runner class — this is the concrete
enforcement of "reuse, don't duplicate" for the whole 14-experiment ladder.
"""

from __future__ import annotations

from finagent.config import Settings
from finagent.document_processing.vector_store import ChromaVectorStore
from finagent.experiments import direct_llm_prompts as prompts
from finagent.experiments.adaptive_pipeline import AdaptiveRAGPipeline, PipelineConfig
from finagent.experiments.base import BaseExperiment
from finagent.experiments.direct_llm_runner import DirectLLMExperiment
from finagent.llm.groq_client import GroqClient

_DIRECT_LLM_DEFINITIONS: dict[str, tuple[str, str]] = {
    "EXP-01": ("Direct LLM with Zero-Shot Prompting", prompts.ZERO_SHOT_PROMPT),
    "EXP-02": ("Direct LLM with Role-Based Financial Analyst Prompting", prompts.ROLE_BASED_PROMPT),
    "EXP-03": ("Direct LLM with Few-Shot Prompting", prompts.FEW_SHOT_PROMPT),
    "EXP-04": ("Direct LLM with Stepwise Financial Reasoning Prompting", prompts.STEPWISE_REASONING_PROMPT),
    "EXP-05": ("Direct LLM with Self-Verification Prompting", prompts.SELF_VERIFICATION_PROMPT),
    "EXP-06": ("Direct LLM with Structured Output Prompting", prompts.STRUCTURED_OUTPUT_PROMPT),
}

_RAG_PIPELINE_CONFIGS: dict[str, PipelineConfig] = {
    "EXP-07": PipelineConfig(
        experiment_id="EXP-07",
        experiment_name="Naïve RAG using ChromaDB",
        retrieval_type_label="Direct top-k retrieval",
        retrieval_strategy="naive",
        use_query_understanding=False,
        enable_query_refinement=False,
        enable_evidence_filtering=False,
        enable_verification=False,
    ),
    "EXP-08": PipelineConfig(
        experiment_id="EXP-08",
        experiment_name="Metadata-Aware Naïve RAG using ChromaDB",
        retrieval_type_label="Metadata-filtered retrieval",
        retrieval_strategy="metadata",
        use_query_understanding=True,
        enable_query_refinement=False,
        enable_evidence_filtering=False,
        enable_verification=False,
    ),
    "EXP-09": PipelineConfig(
        experiment_id="EXP-09",
        experiment_name="Query-Rewritten RAG using ChromaDB",
        retrieval_type_label="Rewritten query retrieval",
        retrieval_strategy="rewritten",
        use_query_understanding=False,
        enable_query_refinement=True,
        enable_evidence_filtering=False,
        enable_verification=False,
    ),
    "EXP-10": PipelineConfig(
        experiment_id="EXP-10",
        experiment_name="Multi-Query RAG using ChromaDB",
        retrieval_type_label="Multiple query retrieval",
        retrieval_strategy="multiquery",
        use_query_understanding=False,
        enable_query_refinement=True,
        enable_evidence_filtering=False,
        enable_verification=False,
    ),
    "EXP-11": PipelineConfig(
        experiment_id="EXP-11",
        experiment_name="Adaptive Multi-Agent RAG using ChromaDB",
        retrieval_type_label="Adaptive routed retrieval",
        retrieval_strategy="adaptive",
        use_query_understanding=True,
        enable_query_refinement=True,
        enable_evidence_filtering=True,
        enable_verification=True,
    ),
    "EXP-12": PipelineConfig(
        experiment_id="EXP-12",
        experiment_name="Adaptive Multi-Agent RAG without Query Refinement",
        retrieval_type_label="Adaptive retrieval without refinement",
        retrieval_strategy="adaptive",
        use_query_understanding=True,
        enable_query_refinement=False,  # <- the ablation
        enable_evidence_filtering=True,
        enable_verification=True,
    ),
    "EXP-13": PipelineConfig(
        experiment_id="EXP-13",
        experiment_name="Adaptive Multi-Agent RAG without Evidence Filtering",
        retrieval_type_label="Adaptive retrieval without filtering",
        retrieval_strategy="adaptive",
        use_query_understanding=True,
        enable_query_refinement=True,
        enable_evidence_filtering=False,  # <- the ablation
        enable_verification=True,
    ),
    "EXP-14": PipelineConfig(
        experiment_id="EXP-14",
        experiment_name="Adaptive Multi-Agent RAG without Verification Agent",
        retrieval_type_label="Adaptive retrieval without verification",
        retrieval_strategy="adaptive",
        use_query_understanding=True,
        enable_query_refinement=True,
        enable_evidence_filtering=True,
        enable_verification=False,  # <- the ablation
    ),
}

ALL_EXPERIMENT_IDS = [f"EXP-{i:02d}" for i in range(1, 15)]


def list_experiment_ids() -> list[str]:
    """Return all 14 experiment IDs in canonical order (EXP-01..EXP-14)."""
    return list(ALL_EXPERIMENT_IDS)


def get_experiment(
    experiment_id: str,
    *,
    llm_client: GroqClient,
    vector_store: ChromaVectorStore | None = None,
    settings: Settings | None = None,
) -> BaseExperiment:
    """Construct the runner for one experiment.

    Args:
        experiment_id: e.g. "EXP-07".
        llm_client: Shared `GroqClient` instance — pass the same instance across all experiments in
            a run so retry/token-accounting behavior is consistent.
        vector_store: Required for EXP-07..14 (RAG experiments); ignored (may be `None`) for
            EXP-01..06 (Direct LLM experiments), which never touch ChromaDB.
        settings: Optional `Settings` override; defaults to `Settings()`.

    Returns:
        A `BaseExperiment` instance ready for `run_question`/`run_batch`.

    Raises:
        ValueError: if `experiment_id` is not one of EXP-01..EXP-14, or if a RAG experiment is
            requested without a `vector_store`.
    """
    settings = settings or Settings()

    if experiment_id in _DIRECT_LLM_DEFINITIONS:
        name, system_prompt = _DIRECT_LLM_DEFINITIONS[experiment_id]
        return DirectLLMExperiment(
            experiment_id=experiment_id,
            experiment_name=name,
            system_prompt=system_prompt,
            llm_client=llm_client,
            settings=settings,
        )

    if experiment_id in _RAG_PIPELINE_CONFIGS:
        if vector_store is None:
            raise ValueError(f"{experiment_id} requires a vector_store (it is a RAG experiment)")
        config = _RAG_PIPELINE_CONFIGS[experiment_id]
        return AdaptiveRAGPipeline(config, llm_client, vector_store, settings)

    raise ValueError(f"Unknown experiment_id: {experiment_id!r}. Expected one of {ALL_EXPERIMENT_IDS}")
