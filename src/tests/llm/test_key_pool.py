"""Tests for key_pool.py — multi-account daily-budget tracking and key selection."""

from __future__ import annotations

import json

import pytest

from finagent.llm.key_pool import GroqKeyPool


def test_requires_at_least_one_key(tmp_path):
    with pytest.raises(ValueError):
        GroqKeyPool([], state_path=tmp_path / "usage.json")


def test_dedups_repeated_keys(tmp_path):
    pool = GroqKeyPool(["key1", "key1", "key2"], state_path=tmp_path / "usage.json")
    assert pool.size == 2


def test_select_key_returns_key_with_most_remaining_budget(tmp_path):
    pool = GroqKeyPool(["key1", "key2"], daily_token_budget=1000, state_path=tmp_path / "usage.json")
    pool.record_usage("key1", 800)  # key1 now has less remaining budget than key2
    selected = pool.select_key()
    assert selected == "key2"


def test_select_key_excludes_specified_keys(tmp_path):
    pool = GroqKeyPool(["key1", "key2"], daily_token_budget=1000, state_path=tmp_path / "usage.json")
    selected = pool.select_key(exclude=frozenset({"key1"}))
    assert selected == "key2"


def test_select_key_returns_none_when_all_excluded(tmp_path):
    pool = GroqKeyPool(["key1", "key2"], daily_token_budget=1000, state_path=tmp_path / "usage.json")
    selected = pool.select_key(exclude=frozenset({"key1", "key2"}))
    assert selected is None


def test_select_key_returns_none_when_all_exhausted(tmp_path):
    pool = GroqKeyPool(["key1", "key2"], daily_token_budget=1000, state_path=tmp_path / "usage.json")
    pool.mark_exhausted("key1")
    pool.mark_exhausted("key2")
    assert pool.select_key() is None


def test_mark_exhausted_removes_key_from_selection(tmp_path):
    pool = GroqKeyPool(["key1", "key2"], daily_token_budget=1000, state_path=tmp_path / "usage.json")
    pool.mark_exhausted("key1")
    assert pool.select_key() == "key2"


def test_remaining_budget_decreases_with_usage(tmp_path):
    pool = GroqKeyPool(["key1"], daily_token_budget=1000, daily_request_budget=10,
                        safety_margin=1.0, state_path=tmp_path / "usage.json")
    tokens_before, requests_before = pool.remaining_budget("key1")
    pool.record_usage("key1", 100)
    tokens_after, requests_after = pool.remaining_budget("key1")
    assert tokens_after == tokens_before - 100
    assert requests_after == requests_before - 1


def test_safety_margin_reduces_effective_budget(tmp_path):
    pool = GroqKeyPool(["key1"], daily_token_budget=1000, safety_margin=0.9, state_path=tmp_path / "usage.json")
    tokens_left, _ = pool.remaining_budget("key1")
    assert tokens_left == 900


def test_usage_persists_across_pool_instances(tmp_path):
    state_path = tmp_path / "usage.json"
    pool1 = GroqKeyPool(["key1"], daily_token_budget=1000, safety_margin=1.0, state_path=state_path)
    pool1.record_usage("key1", 300)

    pool2 = GroqKeyPool(["key1"], daily_token_budget=1000, safety_margin=1.0, state_path=state_path)
    tokens_left, _ = pool2.remaining_budget("key1")
    assert tokens_left == 700


def test_state_file_never_stores_raw_key(tmp_path):
    state_path = tmp_path / "usage.json"
    pool = GroqKeyPool(["super-secret-key-value"], state_path=state_path)
    pool.record_usage("super-secret-key-value", 100)

    raw_content = state_path.read_text(encoding="utf-8")
    assert "super-secret-key-value" not in raw_content


def test_usage_resets_on_new_day(tmp_path):
    state_path = tmp_path / "usage.json"
    # Simulate a stale entry from yesterday.
    from finagent.llm.key_pool import _hash_key
    key_hash = _hash_key("key1")
    state_path.write_text(json.dumps({
        key_hash: {"date": "2020-01-01", "tokens_used": 99999, "requests_used": 999}
    }), encoding="utf-8")

    pool = GroqKeyPool(["key1"], daily_token_budget=1000, safety_margin=1.0, state_path=state_path)
    tokens_left, _ = pool.remaining_budget("key1")
    assert tokens_left == 1000  # fully reset, not still showing yesterday's near-exhausted usage


def test_status_returns_masked_key_identifiers(tmp_path):
    pool = GroqKeyPool(["abcd1234wxyz"], state_path=tmp_path / "usage.json")
    status = pool.status()
    assert len(status) == 1
    assert status[0]["key"] == "...wxyz"
    assert "abcd1234wxyz" not in str(status)
