"""Tests for ReasoningAgent — combined reasoning + draft-answer single call."""

from __future__ import annotations

from finagent.agents.reasoning import ReasoningAgent


class TestReason:
    def test_produces_grounded_draft_answer(self, mock_groq_client, sample_evidence_item, sample_question):
        mock_groq_client.set_response("reasoning", {
            "reasoning_steps": ["EV_001 states revenue was $198.3 billion in FY2022."],
            "extracted_values": {"fy2022_revenue": "$198.3 billion"},
            "draft_answer": "Microsoft's revenue in fiscal year 2022 was $198.3 billion.",
            "citations": ["EV_001"],
            "insufficient_evidence": False,
        })
        agent = ReasoningAgent(mock_groq_client)
        output, record = agent.reason(sample_question.question, [sample_evidence_item])

        assert "198.3 billion" in output.draft_answer
        assert output.citations == ["EV_001"]
        assert output.insufficient_evidence is False
        assert record.agent_name == "reasoning"

    def test_makes_exactly_one_call(self, mock_groq_client, sample_evidence_item, sample_question):
        mock_groq_client.set_response("reasoning", {
            "reasoning_steps": [], "extracted_values": {}, "draft_answer": "x",
            "citations": [], "insufficient_evidence": False,
        })
        agent = ReasoningAgent(mock_groq_client)
        agent.reason(sample_question.question, [sample_evidence_item])
        assert mock_groq_client.call_log == ["reasoning"]

    def test_flags_insufficient_evidence(self, mock_groq_client, sample_question):
        mock_groq_client.set_response("reasoning", {
            "reasoning_steps": ["No evidence contains the requested figure."],
            "extracted_values": {}, "draft_answer": "", "citations": [],
            "insufficient_evidence": True,
        })
        agent = ReasoningAgent(mock_groq_client)
        output, _ = agent.reason(sample_question.question, [])
        assert output.insufficient_evidence is True

    def test_unparseable_response_returns_safe_fallback(self, mock_groq_client, sample_evidence_item, sample_question):
        """Simulates GroqClient.complete_json exhausting its bounded JSON-repair attempts and
        giving up (parsed_output=None) — must not crash, must degrade safely."""

        def unparseable_complete_json(**kwargs):
            from finagent.data.schemas import LLMCallRecord
            return LLMCallRecord(
                agent_name=kwargs["agent_name"], model=mock_groq_client.model,
                system_prompt=kwargs["system_prompt"], user_prompt=kwargs["user_prompt"],
                raw_response="not valid json", input_tokens=50, output_tokens=10,
                latency_seconds=0.001, temperature=0.0, parsed_output=None,
            )

        mock_groq_client.complete_json = unparseable_complete_json
        agent = ReasoningAgent(mock_groq_client)
        output, _ = agent.reason(sample_question.question, [sample_evidence_item])

        assert output.insufficient_evidence is True
        assert output.draft_answer == ""
