"""Retrieval Agent (Proposal §7.6 Table 7.6, §7.8).

Makes **zero** Groq API calls — retrieval is pure vector-database search against the shared
ChromaDB collection (Proposal §7.6: "searches the financial knowledge base using embeddings and
metadata"). This is deliberate: retrieval is a search operation, not a generation task, so it costs
nothing in LLM-call budget regardless of how many queries or how large `top_k` is — the
finagent-architecture skill §0 call-efficiency standard is about minimizing *Groq* calls, and this
agent shows the ceiling case of that (zero, by design, not by optimization).
"""

from __future__ import annotations

from typing import Any

from finagent.config import RETRIEVAL_TOP_K
from finagent.data.schemas import EvidenceItem, QueryAnalysis
from finagent.document_processing.vector_store import ChromaVectorStore


class RetrievalAgent:
    """Searches the shared financial knowledge base, with optional metadata filtering."""

    def __init__(self, vector_store: ChromaVectorStore) -> None:
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = RETRIEVAL_TOP_K,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[EvidenceItem]:
        """Retrieve the top-k most relevant chunks for a single query.

        Args:
            query: The (possibly refined) query text.
            top_k: Number of chunks to retrieve. Defaults to the fixed experimental value (5).
            metadata_filter: Optional ChromaDB `where` clause, e.g. `{"company": "Apple"}` — pass
                `None` for unfiltered semantic search (EXP-07's "Naïve RAG"), or the output of
                `build_metadata_filter` for metadata-aware retrieval (EXP-08, and the adaptive
                system when the Query Understanding Agent has confident entity extraction).

        Returns:
            Evidence items ordered by descending relevance.
        """
        return self._vector_store.query(query, top_k=top_k, metadata_filter=metadata_filter)

    def retrieve_multi(
        self,
        queries: list[str],
        *,
        top_k_per_query: int = RETRIEVAL_TOP_K,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[EvidenceItem]:
        """Retrieve and merge results across multiple query variants (EXP-10, Multi-Query RAG).

        Args:
            queries: Multiple phrasings of the same underlying information need (typically from
                `QueryRefinementAgent.expand_multi_query`).
            top_k_per_query: Chunks to retrieve per individual query before merging.
            metadata_filter: See `retrieve`.

        Returns:
            The union of all queries' results, deduplicated by `chunk_id` (keeping the
            highest-scoring `EvidenceItem` for each chunk — the same chunk can legitimately surface
            from more than one query variant), sorted by descending relevance score. This directly
            implements the Coding_Sheet EXP-10 pipeline step "Merge Chunks" — dedup is essential
            here since without it, a chunk relevant to multiple variants would appear multiple
            times and skew downstream evidence-filtering/reasoning toward over-weighting it.
        """
        best_by_chunk_id: dict[str, EvidenceItem] = {}
        for query in queries:
            for item in self.retrieve(query, top_k=top_k_per_query, metadata_filter=metadata_filter):
                existing = best_by_chunk_id.get(item.chunk.chunk_id)
                if existing is None or item.relevance_score > existing.relevance_score:
                    best_by_chunk_id[item.chunk.chunk_id] = item

        return sorted(best_by_chunk_id.values(), key=lambda e: e.relevance_score, reverse=True)

    @staticmethod
    def build_metadata_filter(query_analysis: QueryAnalysis) -> dict[str, Any] | None:
        """Build a ChromaDB `where` clause from confidently-extracted query signals.

        Args:
            query_analysis: Output of the Query Understanding Agent.

        Returns:
            A `where` clause restricting retrieval to the extracted company and/or year (Proposal
            §7.6: "metadata such as report type, filing year, section name, and page reference"),
            or `None` if neither company nor year was confidently extracted — retrieval should fall
            back to unfiltered semantic search rather than filtering on an empty/absent value, which
            would otherwise silently return zero results.

        Note:
            ChromaDB's `where` syntax requires `$and` for multi-condition filters; a single
            condition is passed as a plain one-key dict.
        """
        conditions: list[dict[str, Any]] = []
        if query_analysis.company:
            conditions.append({"company": query_analysis.company})
        if query_analysis.year:
            conditions.append({"year": query_analysis.year})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
