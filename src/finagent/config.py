"""Fixed experimental configuration for FinAgent-RAG.

Every constant in this module corresponds to a row in `Coding_Sheet.xlsx`'s "Common Metrics"
sheet and to Table 7.12 ("Controlled Experimental Setup") of Kasi_Research_Proposal.pdf §7.9.
These values are held identical across all 14 experiments (EXP-01..EXP-14) so that any measured
performance difference is attributable to pipeline architecture alone, not to a changed dataset,
model, chunking strategy, or retrieval depth.

Do not hardcode any of these values elsewhere in the codebase — import them from here. Overriding
a value (e.g. for a quick local test) must go through the `Settings` dataclass's constructor
arguments or environment variables, never a copy-pasted literal in an agent/experiment file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Layout mirrors the upstream patronus-ai/financebench repo exactly (see
# https://github.com/patronus-ai/financebench), moved under this project's data/ directory rather
# than nested inside the src/finagent package — see finagent-architecture skill for why source and
# ~680MB of dataset PDFs must not share a directory tree.
PDF_DIR = DATA_DIR / "pdfs"
QA_DATASET_PATH = DATA_DIR / "financebench_open_source.jsonl"
"""The 150-question open-source QA split (real field names: question, answer, evidence, company,
doc_name, question_type, justification, financebench_id — see `financebench_loader` for mapping)."""
DOCUMENT_INFO_PATH = DATA_DIR / "financebench_document_information.jsonl"
"""361-filing metadata table (doc_name, company, doc_type, doc_period, gics_sector, doc_link) —
joined against `QA_DATASET_PATH` by `doc_name` to resolve each question's document_type/year."""
BASELINE_RESULTS_DIR = DATA_DIR / "baseline_results"
"""Patronus' own published baseline results (GPT-4, Claude-2, Llama-2 variants) — reference data
only, not produced or consumed by this project's own 14-experiment pipeline."""

CHROMA_PERSIST_DIR = DATA_DIR / "chroma_store"
RUN_TRACES_DIR = DATA_DIR / "run_traces"
RESULTS_WORKBOOK_ORIGINAL = PROJECT_ROOT / "Coding_Sheet.xlsx"
RESULTS_WORKBOOK_COPY = PROJECT_ROOT / "Coding_Sheet_RESULTS.xlsx"

# ---------------------------------------------------------------------------
# Fixed controlled setup (Proposal Table 7.12 / Coding_Sheet "Common Metrics")
# ---------------------------------------------------------------------------

GROQ_MODEL = "llama-3.3-70b-versatile"
"""LLM held fixed across all 14 experiments (Common Metrics: 'Groq Llama 3.3 70B Versatile')."""

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
"""Sentence-transformers embedding model, fixed across all RAG experiments (EXP-07..14)."""

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 100
RETRIEVAL_TOP_K = 5
LLM_TEMPERATURE = 0.0
"""Deterministic decoding — required so repeated runs / ablations are comparable."""

CHROMA_COLLECTION_NAME = "financebench_chunks"

# Groq pricing for llama-3.3-70b-versatile, USD per million tokens, as published on
# https://groq.com/pricing/ (checked at implementation time — prices change; update here if it
# drifts, rather than hardcoding a stale rate inline wherever cost is computed).
GROQ_PRICE_PER_MILLION_INPUT_TOKENS_USD = 0.59
GROQ_PRICE_PER_MILLION_OUTPUT_TOKENS_USD = 0.79

# Multi-query expansion width used by EXP-10 and, when the adaptive route selects it, EXP-11/12/13/14.
MULTI_QUERY_VARIANT_COUNT = 3

# Bounded retry policy for transient Groq failures (rate limit / timeout / 5xx).
GROQ_MAX_RETRIES = 4
GROQ_RETRY_MIN_WAIT_SECONDS = 1.0
GROQ_RETRY_MAX_WAIT_SECONDS = 20.0

# Bounded retries for repairing a malformed structured-output (JSON) response before giving up.
JSON_REPAIR_MAX_ATTEMPTS = 2

# Default max_tokens ceilings per call type — sized to the expected output, not left unbounded,
# per the Groq API-efficiency standard in the finagent-architecture skill §0.
MAX_TOKENS_CLASSIFICATION = 400
MAX_TOKENS_REWRITE = 200
MAX_TOKENS_MULTI_QUERY = 300
MAX_TOKENS_REASONING = 700
MAX_TOKENS_ANSWER = 500
MAX_TOKENS_VERIFICATION = 400
MAX_TOKENS_DIRECT_ANSWER = 600


@dataclass(frozen=True)
class Settings:
    """Runtime settings bundle, resolved once per process and threaded through explicitly.

    Reading `GROQ_API_KEY` happens lazily (only when an LLM call is actually made) so that
    modules which don't touch the network — document processing, metrics, results writing —
    can be imported and unit-tested without an API key present.
    """

    groq_api_key: str | None = field(default_factory=lambda: os.environ.get("GROQ_API_KEY"))
    groq_model: str = GROQ_MODEL
    embedding_model_name: str = EMBEDDING_MODEL_NAME
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS
    retrieval_top_k: int = RETRIEVAL_TOP_K
    temperature: float = LLM_TEMPERATURE
    chroma_persist_dir: Path = CHROMA_PERSIST_DIR
    chroma_collection_name: str = CHROMA_COLLECTION_NAME

    def require_api_key(self) -> str:
        """Return the Groq API key or raise a clear, actionable error if it's missing.

        Raises:
            RuntimeError: if `GROQ_API_KEY` is not set in the environment or a `.env` file.
        """
        if not self.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to a .env file in the project root "
                "(GROQ_API_KEY=gsk_...) or export it in the shell environment before "
                "running any experiment that calls the LLM."
            )
        return self.groq_api_key


DEFAULT_SETTINGS = Settings()
