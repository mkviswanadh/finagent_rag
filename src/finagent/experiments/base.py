"""Shared experiment runner contract (finagent-experiments skill §1/§4).

Every one of the 14 experiments implements `run_question`, producing a `QuestionResult` that
carries both the full `PipelineTrace` (raw prompts/responses, for error analysis — Proposal §7.11
step 14) and the computed metrics (`finagent.metrics.compute_all_metrics`). `run_batch` is the
entry point experiment-execution scripts and the results writer both use.
"""

from __future__ import annotations

import logging
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
        results: list[QuestionResult] = []
        for i, question in enumerate(questions, start=1):
            logger.info(
                "[%s] running question %d/%d (%s)",
                self.experiment_id, i, len(questions), question.question_id,
            )
            results.append(self.run_question(question))
        return results
