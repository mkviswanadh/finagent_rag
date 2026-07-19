"""Tests for registry.py — all 14 experiments construct and run correctly with the expected,
exact Groq call pattern per experiment. This is the test that proves the ablation experiments
(EXP-12/13/14) genuinely reuse EXP-11's code path rather than approximating it."""

from __future__ import annotations

import pytest

from finagent.data.schemas import EvidenceReference, FinanceBenchQuestion, QueryComplexity
from finagent.experiments.registry import ALL_EXPERIMENT_IDS, get_experiment, list_experiment_ids


@pytest.fixture
def scripted_client(mock_groq_client):
    """A MockGroqClient scripted to answer any of the 7-agent-framework call types, plus every
    Direct LLM experiment ID, so every one of the 14 experiments can run end-to-end."""
    for exp_id in [f"EXP-{i:02d}" for i in range(1, 7)]:
        mock_groq_client.set_response(exp_id, "Microsoft's revenue in FY2022 was $198.3 billion.")

    mock_groq_client.set_response("query_understanding", {
        "company": "Microsoft", "year": 2022, "metric": "revenue", "question_type": "lookup",
        "needs_calculation": False, "needs_multiple_evidence_chunks": False, "complexity": "Simple",
        # needs_refinement=True so Moderate/Complex-routed tests exercise the deepest call path
        # (refine + multi-query) — Simple-routed questions ignore this flag entirely regardless
        # (see adaptive_pipeline._prepare_queries), so this doesn't affect Simple-route tests.
        "needs_refinement": True, "routing_rationale": "mock",
    })
    mock_groq_client.set_response("query_refinement", "What was Microsoft's total revenue in FY2022?")
    mock_groq_client.set_response("query_refinement_multi", {
        "variants": ["MSFT revenue 2022", "Microsoft FY22 revenue", "Microsoft 2022 total revenue"]
    })
    mock_groq_client.set_response("reasoning", {
        "reasoning_steps": ["EV_001 states revenue was $198.3 billion"],
        "extracted_values": {"revenue": "$198.3 billion"},
        "draft_answer": "Microsoft's revenue in fiscal year 2022 was $198.3 billion.",
        "citations": ["EV_001"], "insufficient_evidence": False,
    })
    mock_groq_client.set_response("verification", {
        "passed": True, "unsupported_claims": [], "confidence": 0.9, "notes": "ok",
    })
    return mock_groq_client


def test_list_experiment_ids_returns_all_14_in_order():
    assert list_experiment_ids() == [f"EXP-{i:02d}" for i in range(1, 15)]
    assert list_experiment_ids() == ALL_EXPERIMENT_IDS


def test_get_experiment_unknown_id_raises(mock_groq_client):
    with pytest.raises(ValueError):
        get_experiment("EXP-99", llm_client=mock_groq_client)


def test_rag_experiment_without_vector_store_raises(mock_groq_client):
    with pytest.raises(ValueError):
        get_experiment("EXP-07", llm_client=mock_groq_client, vector_store=None)


@pytest.mark.parametrize("exp_id", [f"EXP-{i:02d}" for i in range(1, 7)])
def test_direct_llm_experiment_makes_exactly_one_call(exp_id, scripted_client, sample_question):
    experiment = get_experiment(exp_id, llm_client=scripted_client)
    experiment.run_question(sample_question)
    assert scripted_client.call_log == [exp_id]


@pytest.mark.parametrize("exp_id,expected_calls", [
    ("EXP-07", ["reasoning"]),
    ("EXP-08", ["query_understanding", "reasoning"]),
    ("EXP-09", ["query_refinement", "reasoning"]),
    ("EXP-10", ["query_refinement_multi", "reasoning"]),
])
def test_fixed_strategy_rag_experiment_call_pattern(exp_id, expected_calls, scripted_client, populated_vector_store, sample_question):
    experiment = get_experiment(exp_id, llm_client=scripted_client, vector_store=populated_vector_store)
    experiment.run_question(sample_question)
    assert scripted_client.call_log == expected_calls


def _complex_question() -> FinanceBenchQuestion:
    return FinanceBenchQuestion(
        question_id="q_complex", question="Why did Microsoft's revenue increase in fiscal year 2022?",
        reference_answer="$198.3 billion",
        evidence=[EvidenceReference(doc_name="MICROSOFT_2022_10K", page_number=40, text="...")],
        company="Microsoft", document_type="10-K", document_name="MICROSOFT_2022_10K", document_year=2022,
        gics_sector="Technology", justification="", dataset_question_type="metrics-generated",
        assigned_complexity=QueryComplexity.COMPLEX,
    )


class TestAdaptiveExperimentsAblationsAreReal:
    """The key claim being tested: EXP-12/13/14 reuse EXP-11's exact code path with one stage
    disabled, proven by observing the actual Groq calls made for the same Complex-routed question."""

    def test_exp11_full_system_makes_all_5_calls_on_complex_question(self, scripted_client, populated_vector_store):
        experiment = get_experiment("EXP-11", llm_client=scripted_client, vector_store=populated_vector_store)
        experiment.run_question(_complex_question())
        assert scripted_client.call_log == [
            "query_understanding", "query_refinement", "query_refinement_multi", "reasoning", "verification",
        ]

    def test_exp12_skips_refinement_even_on_complex_question(self, scripted_client, populated_vector_store):
        """This is the core ablation proof: EXP-12 disables refinement, so a Complex question that
        would normally trigger refine+multi-query in EXP-11 does NOT trigger them here."""
        experiment = get_experiment("EXP-12", llm_client=scripted_client, vector_store=populated_vector_store)
        experiment.run_question(_complex_question())
        assert scripted_client.call_log == ["query_understanding", "reasoning", "verification"]

    def test_exp13_same_calls_as_exp11_since_filtering_is_free(self, scripted_client, populated_vector_store):
        """Evidence filtering costs zero Groq calls either way — EXP-13's ablation shows up in
        which chunks reach reasoning, not in call count."""
        experiment = get_experiment("EXP-13", llm_client=scripted_client, vector_store=populated_vector_store)
        experiment.run_question(_complex_question())
        assert scripted_client.call_log == [
            "query_understanding", "query_refinement", "query_refinement_multi", "reasoning", "verification",
        ]

    def test_exp14_skips_verification_call(self, scripted_client, populated_vector_store):
        experiment = get_experiment("EXP-14", llm_client=scripted_client, vector_store=populated_vector_store)
        experiment.run_question(_complex_question())
        assert scripted_client.call_log == [
            "query_understanding", "query_refinement", "query_refinement_multi", "reasoning",
        ]
        assert "verification" not in scripted_client.call_log


def test_simple_question_uses_shortest_adaptive_path(scripted_client, populated_vector_store, sample_question):
    """A Simple-routed question through the full EXP-11 system should skip refinement/multi-query
    entirely — proving adaptive routing actually reduces cost for easy questions, not just ablations."""
    experiment = get_experiment("EXP-11", llm_client=scripted_client, vector_store=populated_vector_store)
    experiment.run_question(sample_question)
    assert scripted_client.call_log == ["query_understanding", "reasoning", "verification"]


@pytest.mark.parametrize("exp_id", ALL_EXPERIMENT_IDS)
def test_every_experiment_produces_a_nonempty_answer(exp_id, scripted_client, populated_vector_store, sample_question):
    vector_store = populated_vector_store if exp_id not in [f"EXP-{i:02d}" for i in range(1, 7)] else None
    experiment = get_experiment(exp_id, llm_client=scripted_client, vector_store=vector_store)
    result = experiment.run_question(sample_question)
    assert result.trace.generated_answer.strip() != ""
    assert result.trace.experiment_id == exp_id
