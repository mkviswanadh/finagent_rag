"""ChromaDB-backed vector store (Proposal Table 7.4, Step 6/7: embedding + storage).

Wraps a single ChromaDB collection shared by every RAG experiment (EXP-07..EXP-14). Embeddings use
`config.EMBEDDING_MODEL_NAME` (all-MiniLM-L6-v2, sentence-transformers) — fixed across all RAG
experiments per finagent-architecture skill §1, so retrieval differences between experiments are
attributable to query/filtering strategy, never to embedding drift.

Embedding generation is batched at ingestion time (one `collection.add(...)` call per document's
chunks, not one call per chunk) per the Groq-API-efficiency standard's batching principle extended
to the embedding backend (finagent-architecture skill §0) — sentence-transformers batches
internally when handed a list, so a single `add` call amortizes model-forward-pass overhead across
every chunk in the document.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from finagent.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME
from finagent.data.schemas import Chunk, EvidenceItem


class ChromaVectorStore:
    """Persistent ChromaDB collection of FinanceBench chunks with metadata-filtered retrieval."""

    def __init__(
        self,
        persist_dir: str | Path = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and store chunks in a single batched call.

        Args:
            chunks: Chunks produced by `chunk_document`. Re-adding a `chunk_id` that already
                exists overwrites it (Chroma `upsert` semantics via `add` + existing-id handling
                is avoided here by using `upsert`, so re-running ingestion is idempotent).
        """
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "company": c.company,
                    "year": c.year,
                    "report_type": c.report_type,
                    "section": c.section,
                    "page_number": c.page_number,
                    "source_document": c.source_document,
                }
                for c in chunks
            ],
        )

    def count(self) -> int:
        """Return the number of chunks currently stored in the collection."""
        return self._collection.count()

    def query(
        self,
        query_text: str,
        *,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[EvidenceItem]:
        """Retrieve the top-k most semantically similar chunks to a query.

        Args:
            query_text: The (possibly rewritten) query string to embed and search with.
            top_k: Number of results to return (fixed to `config.RETRIEVAL_TOP_K` = 5 in normal
                experiment runs).
            metadata_filter: Optional Chroma `where` clause, e.g.
                `{"company": "Apple", "year": 2022}`, used by metadata-aware retrieval (EXP-08) and
                by the adaptive system's Retrieval Agent when the Query Understanding Agent has
                extracted a confident company/year. `None` performs unfiltered semantic search
                (EXP-07 and any route that doesn't have confident metadata).

        Returns:
            `EvidenceItem`s ordered by descending relevance (nearest first), each wrapping the
            matched `Chunk` and a cosine-similarity-derived relevance score in `[0, 1]`.
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(top_k, self._collection.count()),
            where=metadata_filter or None,
        )

        evidence: list[EvidenceItem] = []
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i, chunk_id in enumerate(ids):
            meta = metadatas[i]
            chunk = Chunk(
                chunk_id=chunk_id,
                company=meta["company"],
                year=meta["year"],
                report_type=meta["report_type"],
                section=meta["section"],
                page_number=meta["page_number"],
                text=documents[i],
                source_document=meta["source_document"],
            )
            # Cosine distance in [0, 2] -> similarity in [0, 1]; clamp for numerical safety.
            similarity = max(0.0, min(1.0, 1.0 - (distances[i] / 2.0)))
            evidence.append(
                EvidenceItem(
                    evidence_id=f"EV_{len(evidence) + 1:03d}",
                    chunk=chunk,
                    relevance_score=round(similarity, 4),
                    retrieval_query=query_text,
                )
            )

        return evidence
