"""Tests for VerificationAgent — entailment check against evidence, fails closed on parse errors."""

from __future__ import annotations

from finagent.agents.verification import VerificationAgent


class TestVerify:
    def test_passes_when_answer_is_supported(self, mock_groq_client, sample_evidence_item, sample_question):
        mock_groq_client.set_response("verification", {
            "passed": True, "unsupported_claims": [], "confidence": 0.95,
            "notes": "The figure is directly stated in the evidence.",
        })
        agent = VerificationAgent(mock_groq_client)
        result, record = agent.verify(
            sample_question.question, "Microsoft's revenue was $198.3 billion.", [sample_evidence_item]
        )

        assert result.passed is True
        assert result.unsupported_claims == []
        assert result.confidence == 0.95
        assert record.agent_name == "verification"

    def test_fails_when_answer_has_unsupported_claims(self, mock_groq_client, sample_evidence_item, sample_question):
        mock_groq_client.set_response("verification", {
            "passed": False, "unsupported_claims": ["the $250 billion figure is not in the evidence"],
            "confidence": 0.8, "notes": "evidence states $198.3 billion, not $250 billion",
        })
        agent = VerificationAgent(mock_groq_client)
        result, _ = agent.verify(
            sample_question.question, "Microsoft's revenue was $250 billion.", [sample_evidence_item]
        )
        assert result.passed is False
        assert len(result.unsupported_claims) == 1

    def test_makes_exactly_one_call(self, mock_groq_client, sample_evidence_item, sample_question):
        mock_groq_client.set_response("verification", {
            "passed": True, "unsupported_claims": [], "confidence": 0.9, "notes": "",
        })
        agent = VerificationAgent(mock_groq_client)
        agent.verify(sample_question.question, "answer", [sample_evidence_item])
        assert mock_groq_client.call_log == ["verification"]

    def test_unparseable_response_fails_closed(self, mock_groq_client, sample_evidence_item, sample_question):
        """An unparseable verification response must be treated as NOT passed (fail closed), never
        silently treated as passed (fail open) — this is the pipeline's last safety check."""

        def broken_complete_json(**kwargs):
            from finagent.data.schemas import LLMCallRecord
            return LLMCallRecord(
                agent_name=kwargs["agent_name"], model=mock_groq_client.model,
                system_prompt=kwargs["system_prompt"], user_prompt=kwargs["user_prompt"],
                raw_response="not valid json", input_tokens=50, output_tokens=10,
                latency_seconds=0.001, temperature=0.0, parsed_output=None,
            )

        mock_groq_client.complete_json = broken_complete_json
        agent = VerificationAgent(mock_groq_client)
        result, _ = agent.verify(sample_question.question, "answer", [sample_evidence_item])

        assert result.passed is False
        assert result.confidence == 0.0
