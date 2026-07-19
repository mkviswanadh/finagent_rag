"""Tests for DirectLLMExperiment — EXP-01..06's shared single-call runner."""

from __future__ import annotations

from finagent.experiments.direct_llm_runner import DirectLLMExperiment


def test_run_question_makes_exactly_one_call(mock_groq_client, sample_question):
    mock_groq_client.set_response("EXP-01", "Microsoft's revenue in FY2022 was $198.3 billion.")
    experiment = DirectLLMExperiment(
        experiment_id="EXP-01", experiment_name="Direct LLM with Zero-Shot Prompting",
        system_prompt="You are a financial assistant.", llm_client=mock_groq_client,
    )
    result = experiment.run_question(sample_question)

    assert mock_groq_client.call_log == ["EXP-01"]
    assert "198.3 billion" in result.trace.generated_answer


def test_run_question_has_no_retrieved_evidence(mock_groq_client, sample_question):
    """Direct LLM experiments never touch retrieval — this is what makes compute_all_metrics
    correctly skip retrieval/grounding metrics for this whole group."""
    mock_groq_client.set_response("EXP-02", "some answer")
    experiment = DirectLLMExperiment(
        experiment_id="EXP-02", experiment_name="Direct LLM with Role-Based Financial Analyst Prompting",
        system_prompt="role prompt", llm_client=mock_groq_client,
    )
    result = experiment.run_question(sample_question)
    assert result.trace.retrieved_evidence == []


def test_run_batch_processes_all_questions_in_order(mock_groq_client, sample_question):
    mock_groq_client.set_response("EXP-01", "an answer")
    experiment = DirectLLMExperiment(
        experiment_id="EXP-01", experiment_name="Direct LLM with Zero-Shot Prompting",
        system_prompt="prompt", llm_client=mock_groq_client,
    )
    results = experiment.run_batch([sample_question, sample_question])
    assert len(results) == 2
    assert mock_groq_client.call_log == ["EXP-01", "EXP-01"]


def test_complexity_used_falls_back_to_question_assigned_complexity(mock_groq_client, sample_question):
    mock_groq_client.set_response("EXP-01", "an answer")
    experiment = DirectLLMExperiment(
        experiment_id="EXP-01", experiment_name="Direct LLM with Zero-Shot Prompting",
        system_prompt="prompt", llm_client=mock_groq_client,
    )
    result = experiment.run_question(sample_question)
    assert result.trace.complexity_used == sample_question.assigned_complexity
