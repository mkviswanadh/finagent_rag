"""Optimized Groq API client shared by every agent and direct-LLM experiment.

Design goals (finagent-architecture skill §0, "Groq API call efficiency is a first-class design
constraint"):

1. One thin, well-tested call path (`complete` / `complete_json`) used by every agent, so retry,
   token accounting, and error handling are implemented exactly once instead of once per agent.
2. Bounded, exponential-backoff retries on transient failures only (rate limit / timeout / 5xx) —
   never on a merely-unexpected-content response, which is handled separately via the JSON repair
   path in `complete_json`.
3. Every call returns an `LLMCallRecord` carrying the full prompt/response and token usage, so
   callers never need a separate accounting pass to populate `Token Usage` / `Cost per Answer`
   (Proposal Table 7.17).
4. `temperature` defaults to `config.LLM_TEMPERATURE` (0.0) and every call requires an explicit,
   right-sized `max_tokens` — no unbounded generations.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from groq import APIConnectionError, APIStatusError, APITimeoutError, Groq, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from finagent.config import (
    GROQ_MAX_RETRIES,
    GROQ_RETRY_MAX_WAIT_SECONDS,
    GROQ_RETRY_MIN_WAIT_SECONDS,
    JSON_REPAIR_MAX_ATTEMPTS,
    LLM_TEMPERATURE,
    Settings,
)
from finagent.data.schemas import LLMCallRecord

logger = logging.getLogger(__name__)

_TRANSIENT_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError)


class GroqCallError(RuntimeError):
    """Raised when a Groq call fails after exhausting all retries, or returns unusable content."""


class GroqClient:
    """Thin, accountable wrapper around the Groq chat-completions API.

    Every agent in `finagent.agents` receives a `GroqClient` instance rather than constructing its
    own `groq.Groq` client — this is what makes call-count auditing and a single shared retry
    policy possible.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._client: Groq | None = None

    @property
    def model(self) -> str:
        return self._settings.groq_model

    def _client_lazy(self) -> Groq:
        if self._client is None:
            self._client = Groq(api_key=self._settings.require_api_key())
        return self._client

    def complete(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float = LLM_TEMPERATURE,
        response_format_json: bool = False,
    ) -> LLMCallRecord:
        """Issue one chat-completion call and return a fully populated `LLMCallRecord`.

        Args:
            agent_name: Which agent/experiment stage is making this call, e.g.
                "query_understanding" or "exp01_zero_shot" — stored on the record for tracing
                and for per-agent cost breakdowns.
            system_prompt: The system-role instruction (agent role, task, constraints, output
                format). Must be the complete, final prompt text — no placeholders.
            user_prompt: The user-role content (the question plus any injected context/evidence).
            max_tokens: Explicit output token ceiling. Callers must size this to the expected
                output (see `config.MAX_TOKENS_*` constants) rather than passing an arbitrary
                large number.
            temperature: Defaults to the fixed experimental temperature (0.0). Only override for
                deliberate, documented reasons — never to "get more creative answers".
            response_format_json: If True, requests Groq's JSON object response mode. Combine
                with `complete_json` (which also validates/repairs the JSON) rather than calling
                this directly for structured-output agents.

        Returns:
            An `LLMCallRecord` with the raw response text and token usage populated. Callers that
            need structured output should use `complete_json` instead, which parses and
            bounded-retries on invalid JSON on top of this method's transient-error retries.

        Raises:
            GroqCallError: if all retries are exhausted without a usable response.
        """
        client = self._client_lazy()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs: dict[str, Any] = dict(
            model=self._settings.groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format_json:
            kwargs["response_format"] = {"type": "json_object"}

        retries_used = 0

        @retry(
            retry=retry_if_exception_type(_TRANSIENT_ERRORS),
            stop=stop_after_attempt(GROQ_MAX_RETRIES),
            wait=wait_exponential(
                multiplier=GROQ_RETRY_MIN_WAIT_SECONDS, max=GROQ_RETRY_MAX_WAIT_SECONDS
            ),
            reraise=True,
        )
        def _call() -> Any:
            nonlocal retries_used
            try:
                return client.chat.completions.create(**kwargs)
            except _TRANSIENT_ERRORS:
                retries_used += 1
                raise

        start = time.perf_counter()
        try:
            response = _call()
        except _TRANSIENT_ERRORS as exc:
            raise GroqCallError(
                f"Groq call for agent '{agent_name}' failed after {GROQ_MAX_RETRIES} attempts: {exc}"
            ) from exc
        latency = time.perf_counter() - start

        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        return LLMCallRecord(
            agent_name=agent_name,
            model=self._settings.groq_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_response=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
            temperature=temperature,
            retries=retries_used,
        )

    def complete_json(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float = LLM_TEMPERATURE,
    ) -> LLMCallRecord:
        """Issue a structured-output call and return a record with `parsed_output` populated.

        Requests Groq's native JSON object mode first. If the model still returns invalid JSON
        (rare, but the 70B model is not infallible per finagent-architecture skill §0), makes up
        to `config.JSON_REPAIR_MAX_ATTEMPTS` bounded follow-up calls that show the model its own
        broken output and ask it to fix it — this is far cheaper than a full retry from scratch
        and keeps the common case at exactly one API call.

        Args:
            agent_name: See `complete`.
            system_prompt: Must explicitly instruct the model to respond with a single JSON
                object matching the caller's expected schema (the caller is responsible for
                describing that schema in the prompt text).
            user_prompt: See `complete`.
            max_tokens: See `complete`.
            temperature: See `complete`.

        Returns:
            The `LLMCallRecord` from the final (successful or exhausted) attempt, with
            `parsed_output` set to the parsed `dict` on success or `None` if parsing never
            succeeded — callers must check for `None` and apply a documented fallback rather than
            assuming success.
        """
        record = self.complete(
            agent_name=agent_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format_json=True,
        )
        parsed = _try_parse_json(record.raw_response)
        if parsed is not None:
            record.parsed_output = parsed
            return record

        repair_prompt = user_prompt
        for attempt in range(1, JSON_REPAIR_MAX_ATTEMPTS + 1):
            repair_prompt = (
                f"{repair_prompt}\n\n"
                f"Your previous response was not valid JSON:\n{record.raw_response}\n\n"
                "Respond again with ONLY a single valid JSON object, no prose, no markdown "
                "code fences."
            )
            logger.warning(
                "agent=%s produced invalid JSON, repair attempt %d/%d",
                agent_name,
                attempt,
                JSON_REPAIR_MAX_ATTEMPTS,
            )
            record = self.complete(
                agent_name=agent_name,
                system_prompt=system_prompt,
                user_prompt=repair_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format_json=True,
            )
            parsed = _try_parse_json(record.raw_response)
            if parsed is not None:
                record.parsed_output = parsed
                return record

        record.parsed_output = None
        return record


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON parse, tolerant of a leading/trailing markdown code fence."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None
