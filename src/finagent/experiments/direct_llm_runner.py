"""Direct LLM experiment runner (EXP-01 through EXP-06).

One parameterized class rather than six copy-pasted files — every Direct LLM experiment shares the
exact same pipeline shape (`Coding_Sheet.xlsx` Final Guidance Sheet: "User Question → <Prompt
Variant> → Groq Llama 3.3 → Answer"), differing only in system prompt text (see
`direct_llm_prompts.py`). Exactly one Groq call per question, no retrieval, no evidence, no
ChromaDB — `PipelineTrace.retrieved_evidence` stays empty for all six, which is what makes the
metrics library correctly skip retrieval/grounding metrics for this whole group
(`finagent.metrics.aggregate.compute_all_metrics`).
"""

from __future__ import annotations

from finagent.config import MAX_TOKENS_DIRECT_ANSWER, Settings
from finagent.data.schemas import FinanceBenchQuestion, PipelineTrace, QueryComplexity
from finagent.experiments.base import BaseExperiment
from finagent.llm.groq_client import GroqClient


class DirectLLMExperiment(BaseExperiment):
    """Runs one Direct LLM experiment (EXP-01..06): a single Groq call, no retrieval."""

    def __init__(
        self,
        experiment_id: str,
        experiment_name: str,
        system_prompt: str,
        llm_client: GroqClient,
        settings: Settings | None = None,
        max_tokens: int = MAX_TOKENS_DIRECT_ANSWER,
    ) -> None:
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self._system_prompt = system_prompt
        self._llm = llm_client
        self._settings = settings or Settings()
        self._max_tokens = max_tokens

    def _run_trace(self, question: FinanceBenchQuestion) -> PipelineTrace:
        trace = PipelineTrace(
            experiment_id=self.experiment_id,
            question=question,
            # Direct LLM experiments don't do adaptive routing at all — complexity is still
            # recorded (from the question's pre-assigned tag) purely so results can be split by
            # complexity tier in the Query Complexity-Wise Final Results sheet, not because it
            # affects this experiment's pipeline in any way.
            complexity_used=question.assigned_complexity or QueryComplexity.SIMPLE,
        )
        record = self._llm.complete(
            agent_name=self.experiment_id,
            system_prompt=self._system_prompt,
            user_prompt=question.question,
            max_tokens=self._max_tokens,
            temperature=self._settings.temperature,
        )
        trace.llm_calls.append(record)
        trace.generated_answer = record.raw_response.strip()
        return trace
