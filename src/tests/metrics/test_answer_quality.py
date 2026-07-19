"""Tests for answer_quality.py — Answer Relevance, Exact Match, F1, Semantic Similarity."""

from __future__ import annotations

from finagent.metrics.answer_quality import answer_relevance, exact_match, f1_score, semantic_similarity


class TestExactMatch:
    def test_identical_answers(self):
        assert exact_match("$198.3 billion", "$198.3 billion") == 1.0

    def test_currency_and_comma_formatting_ignored(self):
        assert exact_match("$198.3 billion", "198.3 billion") == 1.0
        assert exact_match("$1,234", "1234") == 1.0

    def test_case_insensitive(self):
        assert exact_match("Revenue Grew", "revenue grew") == 1.0

    def test_different_values(self):
        assert exact_match("$150 billion", "$198.3 billion") == 0.0

    def test_sentence_ending_period_does_not_break_match(self):
        assert exact_match("$198.3 billion.", "$198.3 billion") == 1.0


class TestF1Score:
    def test_identical_answers_score_one(self):
        assert f1_score("revenue was $198.3 billion", "revenue was $198.3 billion") == 1.0

    def test_completely_different_scores_zero(self):
        assert f1_score("apples and oranges", "revenue was strong") == 0.0

    def test_partial_overlap_between_zero_and_one(self):
        score = f1_score(
            "Microsoft revenue in fiscal 2022 was $198.3 billion.", "$198.3 billion"
        )
        assert 0.0 < score < 1.0

    def test_empty_generated_answer_scores_zero(self):
        assert f1_score("", "some reference") == 0.0

    def test_empty_reference_scores_zero(self):
        assert f1_score("some generated text", "") == 0.0


class TestSemanticSimilarity:
    def test_similar_meaning_scores_high(self):
        score = semantic_similarity(
            "Revenue grew to $198.3 billion", "Revenue increased to $198.3 billion"
        )
        assert score > 0.7

    def test_unrelated_texts_score_lower(self):
        similar = semantic_similarity("Revenue grew to $198.3 billion", "Revenue increased to $198.3 billion")
        unrelated = semantic_similarity("Revenue grew to $198.3 billion", "The weather was sunny today")
        assert unrelated < similar

    def test_empty_text_scores_zero(self):
        assert semantic_similarity("", "some text") == 0.0


class TestAnswerRelevance:
    def test_relevant_answer_scores_positively(self):
        score = answer_relevance(
            "What was Microsoft's revenue in 2022?",
            "Microsoft's revenue in fiscal year 2022 was $198.3 billion.",
        )
        assert score > 0.5

    def test_empty_answer_scores_zero(self):
        assert answer_relevance("What was the revenue?", "") == 0.0
