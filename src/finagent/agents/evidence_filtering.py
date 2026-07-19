"""Evidence Filtering Agent (Proposal §7.6 Table 7.6, §7.8).

Deliberately rule-based, not LLM-based — makes **zero** Groq calls. The proposal's own framing of
this agent's role ("Removes weak, noisy, or redundant evidence before reasoning" — Table 7.6) is a
filtering/deduplication operation, not a judgment call requiring language understanding: relevance
is already scored numerically by the retriever, and redundancy between two chunks is a text-overlap
property, not something that benefits from an LLM's opinion. Per the finagent-architecture skill §0
call-efficiency standard, this is exactly the kind of stage that should NOT cost an API call —
EXP-13 (ablation without evidence filtering) is meaningful specifically because this stage does real
work for free, and removing it should measurably increase hallucination/noise, not just remove an
LLM call.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher

from finagent.config import RETRIEVAL_TOP_K
from finagent.data.schemas import EvidenceItem

logger = logging.getLogger(__name__)

DEFAULT_MIN_RELEVANCE = 0.25
DEFAULT_DEDUP_SIMILARITY_THRESHOLD = 0.92


class EvidenceFilteringAgent:
    """Removes weak and near-duplicate evidence chunks before reasoning."""

    def filter(
        self,
        evidence: list[EvidenceItem],
        *,
        min_relevance: float = DEFAULT_MIN_RELEVANCE,
        max_items: int = RETRIEVAL_TOP_K,
        dedup_similarity_threshold: float = DEFAULT_DEDUP_SIMILARITY_THRESHOLD,
    ) -> list[EvidenceItem]:
        """Filter retrieved evidence down to the strongest, non-redundant subset.

        Args:
            evidence: Retrieved evidence, any order.
            min_relevance: Chunks scoring below this are dropped outright — a weak semantic match
                is more likely to introduce noise than support grounded reasoning.
            max_items: Hard cap on the number of chunks returned, applied after relevance/dedup
                filtering, keeping the strongest-scoring survivors. Defaults to the fixed
                experimental top-k (5) so filtering never exceeds what the reasoning stage expects.
            dedup_similarity_threshold: Two chunks with a `SequenceMatcher` text-similarity ratio at
                or above this are considered near-duplicates; only the higher-relevance one is kept.
                Near-duplicate chunks commonly arise from chunk-overlap (Proposal §7.5's 100-token
                overlap) or from multi-query retrieval surfacing the same passage from slightly
                different angles.

        Returns:
            Evidence items sorted by descending relevance, deduplicated, thresholded, and capped —
            ready for the Reasoning Agent.
        """
        strong_evidence = [e for e in evidence if e.relevance_score >= min_relevance]
        strong_evidence.sort(key=lambda e: e.relevance_score, reverse=True)

        deduped: list[EvidenceItem] = []
        for candidate in strong_evidence:
            if any(
                self._text_similarity(candidate.chunk.text, kept.chunk.text) >= dedup_similarity_threshold
                for kept in deduped
            ):
                continue
            deduped.append(candidate)

        result = deduped[:max_items]
        logger.info(
            "Evidence Filtering: %d retrieved -> %d survive (relevance>=%.2f, dedup>=%.2f, cap=%d)",
            len(evidence), len(result), min_relevance, dedup_similarity_threshold, max_items,
        )
        return result

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()
