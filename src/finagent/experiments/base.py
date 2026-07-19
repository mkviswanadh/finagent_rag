"""Shared experiment runner contract (finagent-experiments skill §1/§4).

Every one of the 14 experiments implements `run_question`, producing a `QuestionResult` that
carries both the full `PipelineTrace` (raw prompts/responses, for error analysis — Proposal §7.11
step 14) and the computed metrics (`finagent.metrics.compute_all_metrics`). `run_batch` is the
entry point experiment-execution scripts and the results writer both use.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from finagent.data.schemas import FinanceBenchQuestion, PipelineTrace, QuestionResult
from finagent.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class BaseExperiment(ABC):
    """Base class every EXP-01..14 runner implements."""

    experiment_id: str
    experiment_name: str

    @abstractmethod
    def _run_trace(self, question: FinanceBenchQuestion) -> PipelineTrace:
        """Run this experiment's pipeline for one question and return the completed trace.

        Subclasses implement only this method — `run_question` wraps it with metric computation
        and consistent logging, so that step is never duplicated or subtly varied per experiment.
        """
        raise NotImplementedError

    def run_question(self, question: FinanceBenchQuestion) -> QuestionResult:
        """Run this experiment for one question, returning the trace plus every computed metric.

        Args:
            question: The FinanceBench question to answer.

        Returns:
            A `QuestionResult` combining the pipeline trace and `compute_all_metrics`' output.
        """
        trace = self._run_trace(question)
        if trace.finished_at is None:
            trace.mark_finished()
        metrics = compute_all_metrics(trace, question)
        return QuestionResult(trace=trace, metrics=metrics)

    def run_batch(self, questions: list[FinanceBenchQuestion]) -> list[QuestionResult]:
        """Run this experiment across multiple questions, in order.

        Args:
            questions: FinanceBench questions to run, ideally pre-tagged with
                `assigned_complexity` (finagent-experiments skill §3d) so results can be split by
                complexity tier afterward without re-classification.

        Returns:
            One `QuestionResult` per question, in the same order.
        """
        batch_start = time.perf_counter()
        results: list[QuestionResult] = []
        failures = 0
        for i, question in enumerate(questions, start=1):
            logger.info(
                "[%s] running question %d/%d (%s)",
                self.experiment_id, i, len(questions), question.question_id,
            )
            try:
                results.append(self.run_question(question))
            except Exception:
                failures += 1
                logger.exception(
                    "[%s] question %d/%d (%s) raised an exception — skipping, continuing batch",
                    self.experiment_id, i, len(questions), question.question_id,
                )

        elapsed = time.perf_counter() - batch_start
        total_tokens = sum(r.trace.total_tokens for r in results)
        total_calls = sum(len(r.trace.llm_calls) for r in results)
        avg_latency = (sum(r.trace.total_latency_seconds for r in results) / len(results)) if results else 0.0
        logger.info(
            "[%s] batch complete: %d/%d succeeded in %.1fs (avg %.2fs/question, %d Groq calls, "
            "%d tokens total)",
            self.experiment_id, len(results), len(questions), elapsed, avg_latency, total_calls, total_tokens,
        )
        return results
