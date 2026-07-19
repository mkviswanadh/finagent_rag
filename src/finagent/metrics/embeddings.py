"""Shared embedding backend for embedding-based metrics (Proposal Tables 7.13).

Answer Relevance and Semantic Similarity are both cosine-similarity-over-embeddings metrics
(Proposal Table 7.13's own formulas: `Q·A/(|Q||A|)` and `Eg·Er/(|Eg||Er|)`). Computing them via the
same local `all-MiniLM-L6-v2` model already used for retrieval (`config.EMBEDDING_MODEL_NAME`) —
rather than an LLM-judge call — means these two metrics cost **zero** additional Groq calls per
question, which matters a great deal at 150 questions × 14 experiments.

The model is loaded once per process (module-level singleton) since construction is the expensive
part; encoding is fast once loaded.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from finagent.config import EMBEDDING_MODEL_NAME

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def cosine_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts' embeddings.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Cosine similarity in `[-1, 1]` (in practice `[0, 1]` for typical natural-language text with
        this model); `0.0` if either text is empty after stripping, since an empty string has no
        meaningful embedding to compare.
    """
    if not text_a.strip() or not text_b.strip():
        return 0.0
    model = _get_model()
    embeddings = model.encode([text_a, text_b])
    a, b = embeddings[0], embeddings[1]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
