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

ARCHIVE_DIR = PROJECT_ROOT / "archive"
"""Timestamped snapshots of results/report files, written by `results.archive.archive_file` before
each run overwrites them — see that module for why (a run that produces worse results than the last
one, or crashes partway through, must not silently destroy the previous good output)."""

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

# Groq free-tier limits for llama-3.3-70b-versatile, as published at
# console.groq.com/docs/rate-limits (confirmed against this project's own account, 2026). Limits
# apply per ORGANIZATION, not per key — Groq's own docs state this explicitly — so these are only
# meaningful per key when each key genuinely belongs to a separate account/organization. A small
# safety margin (not the exact published number) is used so the pool proactively rotates away from
# a key before it actually 429s, rather than relying on hitting the wall and retrying.
GROQ_FREE_TIER_RPM = 30
GROQ_FREE_TIER_RPD = 1000
GROQ_FREE_TIER_TPM = 12000
GROQ_FREE_TIER_TPD = 100000
GROQ_KEY_SAFETY_MARGIN = 0.9  # treat a key as exhausted at 90% of its published daily cap
GROQ_KEY_USAGE_STATE_PATH = DATA_DIR / "groq_key_usage.json"
"""Persistent per-key daily usage tracker (git-ignored — see .gitignore) — survives process
restarts so a multi-day paced run doesn't lose track of how much of each key's daily budget has
already been spent."""

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


def _parse_groq_api_keys() -> list[str]:
    """Read one or more Groq API keys from the environment.

    `GROQ_API_KEYS` (comma-separated) takes precedence, for a multi-account key pool
    (`llm.key_pool.GroqKeyPool`) — each key must belong to a *separate* Groq account/organization
    to actually add quota, since Groq's free-tier limits apply per organization, not per key (see
    `GROQ_FREE_TIER_*` above). Falls back to the single `GROQ_API_KEY` for the common single-key case.
    """
    multi = os.environ.get("GROQ_API_KEYS")
    if multi:
        return [k.strip() for k in multi.split(",") if k.strip()]
    single = os.environ.get("GROQ_API_KEY")
    return [single] if single else []


@dataclass(frozen=True)
class Settings:
    """Runtime settings bundle, resolved once per process and threaded through explicitly.

    Reading Groq API keys happens lazily (only when an LLM call is actually made) so that
    modules which don't touch the network — document processing, metrics, results writing —
    can be imported and unit-tested without an API key present.
    """

    groq_api_keys: list[str] = field(default_factory=_parse_groq_api_keys)
    groq_model: str = GROQ_MODEL
    embedding_model_name: str = EMBEDDING_MODEL_NAME
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS
    retrieval_top_k: int = RETRIEVAL_TOP_K
    temperature: float = LLM_TEMPERATURE
    chroma_persist_dir: Path = CHROMA_PERSIST_DIR
    chroma_collection_name: str = CHROMA_COLLECTION_NAME

    @property
    def groq_api_key(self) -> str | None:
        """The first configured key, for callers that only need a single key (e.g. `require_api_key`)."""
        return self.groq_api_keys[0] if self.groq_api_keys else None

    def require_api_key(self) -> str:
        """Return the first Groq API key or raise a clear, actionable error if none is configured.

        Raises:
            RuntimeError: if neither `GROQ_API_KEY` nor `GROQ_API_KEYS` is set.
        """
        if not self.groq_api_keys:
            raise RuntimeError(
                "No Groq API key is set. Add GROQ_API_KEY=gsk_... (single key) or "
                "GROQ_API_KEYS=gsk_key1,gsk_key2,... (multi-account pool) to a .env file in the "
                "project root, or export it in the shell environment, before running any "
                "experiment that calls the LLM."
            )
        return self.groq_api_keys[0]

    def require_api_keys(self) -> list[str]:
        """Return all configured Groq API keys, for `GroqKeyPool`. Same error as `require_api_key`."""
        self.require_api_key()
        return self.groq_api_keys


DEFAULT_SETTINGS = Settings()
