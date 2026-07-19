"""Shared pytest fixtures: a scripted mock GroqClient, sample data factories, and a temp vector store.

No test in this suite makes a real Groq API call — `MockGroqClient` implements the same
`complete`/`complete_json` interface as `finagent.llm.groq_client.GroqClient` (duck-typed, agents
only call these two methods) and returns pre-scripted responses keyed by `agent_name`. This lets
the full 7-agent orchestration and all 14 experiments be tested deterministically and for free.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from finagent.data.schemas import (
    Chunk,
    EvidenceItem,
    EvidenceReference,
    FinanceBenchQuestion,
    LLMCallRecord,
    QueryComplexity,
)
from finagent.document_processing.vector_store import ChromaVectorStore


class MockGroqClient:
    """Scripted stand-in for `GroqClient`. Script keys are `agent_name`; values are either a raw
    string (for `.complete`) or a dict (auto-serialized to JSON, for `.complete_json`)."""

    def __init__(self, script: dict[str, str | dict] | None = None) -> None:
        self.script = script or {}
        self.call_log: list[str] = []
        self.model = "mock-llama-3.3-70b-versatile"

    def set_response(self, agent_name: str, response: str | dict) -> None:
        self.script[agent_name] = response

    def _record(self, agent_name, system_prompt, user_prompt, raw_response, parsed_output) -> LLMCallRecord:
        self.call_log.append(agent_name)
        return LLMCallRecord(
            agent_name=agent_name,
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_response=raw_response,
            input_tokens=100,
            output_tokens=50,
            latency_seconds=0.001,
            temperature=0.0,
            parsed_output=parsed_output,
        )

    def complete(self, *, agent_name, system_prompt, user_prompt, max_tokens, temperature=0.0, response_format_json=False):
        response = self.script.get(agent_name, "Mock response.")
        raw = json.dumps(response) if isinstance(response, dict) else str(response)
        return self._record(agent_name, system_prompt, user_prompt, raw, None)

    def complete_json(self, *, agent_name, system_prompt, user_prompt, max_tokens, temperature=0.0):
        response = self.script.get(agent_name, {})
        if not isinstance(response, dict):
            raise ValueError(f"complete_json script for {agent_name!r} must be a dict, got {type(response)}")
        raw = json.dumps(response)
        return self._record(agent_name, system_prompt, user_prompt, raw, response)


@pytest.fixture
def mock_groq_client() -> MockGroqClient:
    """A `MockGroqClient` with no responses scripted — tests populate it via `.set_response`."""
    return MockGroqClient()


@pytest.fixture
def sample_chunk() -> Chunk:
    """One realistic chunk: Microsoft FY2022 revenue, from the Income Statement section."""
    return Chunk(
        chunk_id="MSFT_2022_10K_CH_001",
        company="Microsoft",
        year=2022,
        report_type="10-K",
        section="Consolidated Statements of Income",
        page_number=40,
        text=(
            "Revenue increased $16.5 billion or 18% to $198.3 billion in fiscal year 2022, "
            "driven by growth in Server products and cloud services."
        ),
        source_document="MICROSOFT_2022_10K.pdf",
    )


@pytest.fixture
def sample_evidence_item(sample_chunk: Chunk) -> EvidenceItem:
    return EvidenceItem(
        evidence_id="EV_001", chunk=sample_chunk, relevance_score=0.9, retrieval_query="mock query"
    )


@pytest.fixture
def sample_question() -> FinanceBenchQuestion:
    """A Simple-complexity question matching `sample_chunk`'s content, with ground-truth evidence."""
    return FinanceBenchQuestion(
        question_id="q_test_001",
        question="What was Microsoft's revenue in fiscal year 2022?",
        reference_answer="$198.3 billion",
        evidence=[
            EvidenceReference(
                doc_name="MICROSOFT_2022_10K", page_number=40, text="Revenue was $198.3 billion"
            )
        ],
        company="Microsoft",
        document_type="10-K",
        document_name="MICROSOFT_2022_10K",
        document_year=2022,
        gics_sector="Technology",
        justification="",
        dataset_question_type="metrics-generated",
        assigned_complexity=QueryComplexity.SIMPLE,
    )


@pytest.fixture
def temp_vector_store():
    """A ChromaDB vector store in a throwaway temp directory, cleaned up after the test."""
    tmp_dir = tempfile.mkdtemp(prefix="finagent_test_chroma_")
    store = ChromaVectorStore(persist_dir=tmp_dir, collection_name="test_collection")
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def populated_vector_store(temp_vector_store: ChromaVectorStore, sample_chunk: Chunk):
    """A `temp_vector_store` pre-loaded with `sample_chunk`."""
    temp_vector_store.add_chunks([sample_chunk])
    return temp_vector_store


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"

requires_real_pdfs = pytest.mark.skipif(
    not PDF_DIR.exists() or not any(PDF_DIR.glob("*.pdf")),
    reason="data/pdfs/ (FinanceBench PDF corpus) not present in this environment",
)
