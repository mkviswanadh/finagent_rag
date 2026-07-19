"""Tests for AnswerGenerationAgent — zero-LLM-call finalization of the Reasoning Agent's draft."""

from __future__ import annotations

from finagent.agents.answer_generation import AnswerGenerationAgent
from finagent.data.schemas import ReasoningOutput


class TestGenerate:
    def test_appends_citation_footer_by_default(self, sample_evidence_item):
        reasoning_output = ReasoningOutput(
            reasoning_steps=["step"], extracted_values={}, draft_answer="Revenue was $198.3 billion.",
            citations=["EV_001"], insufficient_evidence=False,
        )
        agent = AnswerGenerationAgent()
        answer = agent.generate(reasoning_output, [sample_evidence_item])

        assert "Revenue was $198.3 billion." in answer
        assert "Sources:" in answer
        assert "Microsoft" in answer

    def test_omits_citations_when_disabled(self, sample_evidence_item):
        reasoning_output = ReasoningOutput(
            reasoning_steps=["step"], extracted_values={}, draft_answer="Revenue was $198.3 billion.",
            citations=["EV_001"], insufficient_evidence=False,
        )
        agent = AnswerGenerationAgent()
        answer = agent.generate(reasoning_output, [sample_evidence_item], include_citations=False)
        assert answer == "Revenue was $198.3 billion."

    def test_insufficient_evidence_returns_fixed_message_not_fabricated_answer(self, sample_evidence_item):
        reasoning_output = ReasoningOutput(
            reasoning_steps=["no evidence found"], extracted_values={}, draft_answer="",
            citations=[], insufficient_evidence=True,
        )
        agent = AnswerGenerationAgent()
        answer = agent.generate(reasoning_output, [sample_evidence_item])
        assert "does not contain enough information" in answer

    def test_empty_draft_answer_treated_as_insufficient(self, sample_evidence_item):
        reasoning_output = ReasoningOutput(
            reasoning_steps=[], extracted_values={}, draft_answer="", citations=[],
            insufficient_evidence=False,  # not flagged, but draft is empty anyway
        )
        agent = AnswerGenerationAgent()
        answer = agent.generate(reasoning_output, [sample_evidence_item])
        assert "does not contain enough information" in answer

    def test_unresolvable_citation_id_omitted_gracefully(self, sample_evidence_item):
        reasoning_output = ReasoningOutput(
            reasoning_steps=["step"], extracted_values={}, draft_answer="Answer text.",
            citations=["EV_999_DOES_NOT_EXIST"], insufficient_evidence=False,
        )
        agent = AnswerGenerationAgent()
        answer = agent.generate(reasoning_output, [sample_evidence_item])
        # No footer added since the citation couldn't be resolved to a real evidence item.
        assert answer == "Answer text."

    def test_makes_no_llm_calls(self):
        agent = AnswerGenerationAgent()
        assert not hasattr(agent, "_llm")
