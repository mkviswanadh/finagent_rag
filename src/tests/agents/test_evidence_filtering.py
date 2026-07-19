"""Tests for EvidenceFilteringAgent — zero-LLM-call relevance threshold + near-duplicate dedup."""

from __future__ import annotations

from finagent.agents.evidence_filtering import EvidenceFilteringAgent
from finagent.data.schemas import Chunk, EvidenceItem


def _evidence(evidence_id: str, text: str, score: float) -> EvidenceItem:
    chunk = Chunk(
        chunk_id=evidence_id, company="X", year=2022, report_type="10-K", section="S",
        page_number=1, text=text, source_document="X.pdf",
    )
    return EvidenceItem(evidence_id=evidence_id, chunk=chunk, relevance_score=score, retrieval_query="q")


class TestFilter:
    def test_drops_below_min_relevance(self):
        agent = EvidenceFilteringAgent()
        evidence = [_evidence("E1", "strong match text", 0.8), _evidence("E2", "weak match text", 0.1)]
        result = agent.filter(evidence, min_relevance=0.25)
        assert [e.evidence_id for e in result] == ["E1"]

    def test_keeps_all_above_threshold(self):
        agent = EvidenceFilteringAgent()
        evidence = [_evidence("E1", "text one about revenue", 0.9), _evidence("E2", "text two about expenses", 0.6)]
        result = agent.filter(evidence, min_relevance=0.25)
        assert len(result) == 2

    def test_dedups_near_identical_text(self):
        agent = EvidenceFilteringAgent()
        text = "Revenue increased to $198.3 billion in fiscal year 2022 driven by cloud growth."
        evidence = [_evidence("E1", text, 0.9), _evidence("E2", text, 0.7)]  # identical text, different score
        result = agent.filter(evidence)
        assert len(result) == 1
        assert result[0].evidence_id == "E1"  # higher-relevance duplicate is kept

    def test_keeps_distinct_texts(self):
        agent = EvidenceFilteringAgent()
        evidence = [
            _evidence("E1", "Revenue increased significantly this fiscal year.", 0.9),
            _evidence("E2", "Operating expenses decreased due to cost cutting measures.", 0.8),
        ]
        result = agent.filter(evidence)
        assert len(result) == 2

    def test_respects_max_items_cap(self):
        agent = EvidenceFilteringAgent()
        topics = [
            "Revenue increased due to strong cloud demand.",
            "Operating expenses decreased following cost cutting.",
            "The company completed a major acquisition this year.",
            "Litigation reserves were adjusted for pending lawsuits.",
            "Inventory levels rose ahead of the holiday season.",
            "Debt was refinanced at a lower interest rate.",
            "Headcount grew across engineering and sales teams.",
            "A new product line launched in international markets.",
            "Currency fluctuations affected reported earnings.",
            "Capital expenditures funded new manufacturing facilities.",
        ]
        evidence = [_evidence(f"E{i}", topics[i], 0.9 - i * 0.01) for i in range(10)]
        result = agent.filter(evidence, max_items=3)
        assert len(result) == 3

    def test_sorted_by_descending_relevance(self):
        agent = EvidenceFilteringAgent()
        evidence = [
            _evidence("E1", "text alpha content", 0.5),
            _evidence("E2", "text beta content", 0.9),
            _evidence("E3", "text gamma content", 0.7),
        ]
        result = agent.filter(evidence)
        assert [e.evidence_id for e in result] == ["E2", "E3", "E1"]

    def test_empty_input(self):
        assert EvidenceFilteringAgent().filter([]) == []

    def test_makes_no_llm_calls(self):
        agent = EvidenceFilteringAgent()
        assert not hasattr(agent, "_llm")
