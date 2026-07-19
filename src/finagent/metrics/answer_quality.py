"""Answer Quality metrics (Proposal Table 7.13).

All four metrics here are computed without any Groq call — Answer Relevance and Semantic
Similarity via local embeddings (`metrics.embeddings`), Exact Match and F1 via normalized string/
token comparison. This is the "Answer Quality" family reported in every experiment's `Overall
Performance Results of` row (finagent-experiments skill §3a).
"""

from __future__ import annotations

from finagent.metrics.embeddings import cosine_similarity
from finagent.metrics.text_normalization import normalize_answer


def answer_relevance(question: str, generated_answer: str) -> float:
    """Answer Relevance: cosine similarity between the question and the generated answer.

    Formula (Proposal Table 7.13): `Q·A / (|Q||A|)`.

    Args:
        question: The original user question.
        generated_answer: The system's generated answer.

    Returns:
        Cosine similarity in `[0, 1]` (typical range for this embedding model on natural text).
    """
    return cosine_similarity(question, generated_answer)


def exact_match(generated_answer: str, reference_answer: str) -> float:
    """Exact Match: 1.0 if the normalized generated answer equals the normalized reference, else 0.0.

    Formula (Proposal Table 7.13): `1` if generated == reference else `0`. Normalization (see
    `text_normalization.normalize_answer`) is applied first so "$394.3 billion" and "394.3 billion"
    are treated as the same answer for this metric's purposes — the proposal's own example use case
    is "especially for numerical queries", where currency/comma formatting differences shouldn't
    count as a mismatch.

    Args:
        generated_answer: The system's generated answer.
        reference_answer: The FinanceBench ground-truth answer.

    Returns:
        `1.0` or `0.0`.
    """
    return 1.0 if normalize_answer(generated_answer) == normalize_answer(reference_answer) else 0.0


def f1_score(generated_answer: str, reference_answer: str) -> float:
    """F1-Score: token-level overlap between generated and reference answers.

    Formula (Proposal Table 7.13): `2 * Precision * Recall / (Precision + Recall)`, computed over
    normalized whitespace-tokenized bag-of-words (SQuAD-style), which tolerates the generated
    answer containing extra grounding/citation text around the core answer.

    Args:
        generated_answer: The system's generated answer.
        reference_answer: The FinanceBench ground-truth answer.

    Returns:
        F1 in `[0, 1]`. `0.0` if either answer normalizes to no tokens.
    """
    gen_tokens = normalize_answer(generated_answer).split()
    ref_tokens = normalize_answer(reference_answer).split()
    if not gen_tokens or not ref_tokens:
        return 0.0

    gen_counts: dict[str, int] = {}
    for tok in gen_tokens:
        gen_counts[tok] = gen_counts.get(tok, 0) + 1
    ref_counts: dict[str, int] = {}
    for tok in ref_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1

    overlap = sum(min(gen_counts.get(tok, 0), count) for tok, count in ref_counts.items())
    if overlap == 0:
        return 0.0

    precision = overlap / len(gen_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def semantic_similarity(generated_answer: str, reference_answer: str) -> float:
    """Semantic Similarity: cosine similarity between generated and reference answer embeddings.

    Formula (Proposal Table 7.13): `Eg·Er / (|Eg||Er|)`.

    Args:
        generated_answer: The system's generated answer.
        reference_answer: The FinanceBench ground-truth answer.

    Returns:
        Cosine similarity in `[0, 1]`.
    """
    return cosine_similarity(generated_answer, reference_answer)
