"""Tests for groq_client.py's multi-key rotation: a rate-limit error on one key must trigger an
immediate retry on the next key, not a backoff-and-wait on the same exhausted one."""

from __future__ import annotations

import unittest.mock as um
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from groq import RateLimitError

import finagent.llm.groq_client as gc_module
from finagent.config import Settings
from finagent.llm.groq_client import GroqCallError, GroqClient
from finagent.llm.key_pool import GroqKeyPool


def _fake_response(content: str = "answer text", input_tokens: int = 100, output_tokens: int = 50):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens),
    )


def _rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return RateLimitError("rate limit exceeded", response=response, body=None)


class TestSingleKeyMode:
    def test_successful_call_returns_record(self):
        def constructor(api_key, **kwargs):
            m = MagicMock()
            m.chat.completions.create.return_value = _fake_response("hello world")
            return m

        settings = Settings(groq_api_keys=["key1"])
        client = GroqClient(settings)

        with um.patch.object(gc_module, "Groq", side_effect=constructor):
            record = client.complete(agent_name="test", system_prompt="s", user_prompt="u", max_tokens=100)

        assert record.raw_response == "hello world"
        assert record.input_tokens == 100
        assert record.output_tokens == 50
        assert client.key_pool is None  # single key -> no pool at all

    def test_no_key_configured_raises_clear_error(self):
        settings = Settings(groq_api_keys=[])
        client = GroqClient(settings)
        with pytest.raises(Exception):
            client.complete(agent_name="test", system_prompt="s", user_prompt="u", max_tokens=100)


class TestMultiKeyRotation:
    def test_rate_limit_on_first_key_rotates_to_second(self, tmp_path):
        mocks_by_key: dict[str, MagicMock] = {}

        def constructor(api_key, **kwargs):
            m = MagicMock()
            if api_key == "key1":
                m.chat.completions.create.side_effect = _rate_limit_error()
            else:
                m.chat.completions.create.return_value = _fake_response("answer from key2")
            mocks_by_key[api_key] = m
            return m

        settings = Settings(groq_api_keys=["key1", "key2"])
        pool = GroqKeyPool(settings.groq_api_keys, state_path=tmp_path / "usage.json")
        client = GroqClient(settings, key_pool=pool)

        with um.patch.object(gc_module, "Groq", side_effect=constructor):
            record = client.complete(agent_name="test", system_prompt="s", user_prompt="u", max_tokens=100)

        assert record.raw_response == "answer from key2"
        # key1's mock should have been tried (and failed) before key2 succeeded.
        assert "key1" in mocks_by_key
        assert "key2" in mocks_by_key

    def test_all_keys_rate_limited_raises_groq_call_error(self, tmp_path):
        def constructor(api_key, **kwargs):
            m = MagicMock()
            m.chat.completions.create.side_effect = _rate_limit_error()
            return m

        settings = Settings(groq_api_keys=["key1", "key2"])
        pool = GroqKeyPool(settings.groq_api_keys, state_path=tmp_path / "usage.json")
        client = GroqClient(settings, key_pool=pool)

        with um.patch.object(gc_module, "Groq", side_effect=constructor):
            with pytest.raises(GroqCallError):
                client.complete(agent_name="test", system_prompt="s", user_prompt="u", max_tokens=100)

    def test_successful_call_records_usage_against_pool(self, tmp_path):
        def constructor(api_key, **kwargs):
            m = MagicMock()
            m.chat.completions.create.return_value = _fake_response("ok", input_tokens=200, output_tokens=100)
            return m

        settings = Settings(groq_api_keys=["key1", "key2"])
        pool = GroqKeyPool(settings.groq_api_keys, state_path=tmp_path / "usage.json")
        client = GroqClient(settings, key_pool=pool)
        tokens_before_1, _ = client.key_pool.remaining_budget("key1")
        tokens_before_2, _ = client.key_pool.remaining_budget("key2")

        with um.patch.object(gc_module, "Groq", side_effect=constructor):
            client.complete(agent_name="test", system_prompt="s", user_prompt="u", max_tokens=100)

        # Whichever key select_key() picked should now show exactly 300 fewer tokens remaining.
        tokens_after_1, _ = client.key_pool.remaining_budget("key1")
        tokens_after_2, _ = client.key_pool.remaining_budget("key2")
        used_key1 = tokens_before_1 - tokens_after_1
        used_key2 = tokens_before_2 - tokens_after_2
        assert (used_key1, used_key2) in [(300, 0), (0, 300)]

    def test_key_pool_exposed_when_multiple_keys_configured(self):
        settings = Settings(groq_api_keys=["key1", "key2", "key3"])
        client = GroqClient(settings)
        assert client.key_pool is not None
        assert client.key_pool.size == 3
