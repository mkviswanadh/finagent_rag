"""Multi-account Groq API key pool with persistent daily-budget tracking.

Groq's rate limits apply per organization, not per API key (per Groq's own docs) — so this pool
only adds real quota when each key genuinely belongs to a *separate* Groq account. Given that,
`GroqClient` uses this to: (1) always issue the next call on whichever key currently has the most
remaining daily token budget, and (2) on a rate-limit error from one key, immediately retry on a
different key rather than backing off and waiting on the same exhausted one.

Usage is tracked per key (identified by a short hash, never the raw key, even in the local
git-ignored state file) and persisted to disk so a multi-day paced run started fresh each morning
still knows how much of each key's daily budget was already spent in earlier processes today.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from finagent.config import (
    GROQ_FREE_TIER_RPD,
    GROQ_FREE_TIER_TPD,
    GROQ_KEY_SAFETY_MARGIN,
    GROQ_KEY_USAGE_STATE_PATH,
)

logger = logging.getLogger(__name__)


def _hash_key(api_key: str) -> str:
    """Short, irreversible identifier for an API key — never store the raw key on disk."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


@dataclass
class _KeyUsage:
    date_str: str
    tokens_used: int = 0
    requests_used: int = 0


class GroqKeyPool:
    """Tracks daily token/request usage across multiple Groq API keys and picks the best one."""

    def __init__(
        self,
        api_keys: list[str],
        *,
        daily_token_budget: int = GROQ_FREE_TIER_TPD,
        daily_request_budget: int = GROQ_FREE_TIER_RPD,
        safety_margin: float = GROQ_KEY_SAFETY_MARGIN,
        state_path: Path = GROQ_KEY_USAGE_STATE_PATH,
    ) -> None:
        if not api_keys:
            raise ValueError("GroqKeyPool requires at least one API key")
        self._api_keys = list(dict.fromkeys(api_keys))  # de-dup while preserving order
        self._daily_token_budget = int(daily_token_budget * safety_margin)
        self._daily_request_budget = int(daily_request_budget * safety_margin)
        self._state_path = Path(state_path)
        self._lock = threading.Lock()
        self._usage: dict[str, _KeyUsage] = self._load_state()

    @property
    def size(self) -> int:
        """Number of distinct API keys in the pool."""
        return len(self._api_keys)

    def _load_state(self) -> dict[str, _KeyUsage]:
        if not self._state_path.exists():
            return {}
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read %s; starting with fresh key-usage state", self._state_path)
            return {}
        return {
            key_hash: _KeyUsage(
                date_str=entry["date"], tokens_used=entry["tokens_used"], requests_used=entry["requests_used"]
            )
            for key_hash, entry in raw.items()
        }

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            key_hash: {"date": u.date_str, "tokens_used": u.tokens_used, "requests_used": u.requests_used}
            for key_hash, u in self._usage.items()
        }
        self._state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _usage_for(self, api_key: str) -> _KeyUsage:
        key_hash = _hash_key(api_key)
        today = date.today().isoformat()
        usage = self._usage.get(key_hash)
        if usage is None or usage.date_str != today:
            usage = _KeyUsage(date_str=today)
            self._usage[key_hash] = usage
        return usage

    def remaining_budget(self, api_key: str) -> tuple[int, int]:
        """Return (tokens remaining, requests remaining) today for `api_key`, after the safety margin."""
        usage = self._usage_for(api_key)
        tokens_left = max(0, self._daily_token_budget - usage.tokens_used)
        requests_left = max(0, self._daily_request_budget - usage.requests_used)
        return tokens_left, requests_left

    def record_usage(self, api_key: str, tokens: int) -> None:
        """Record a successful call's token usage against `api_key`'s daily budget."""
        with self._lock:
            usage = self._usage_for(api_key)
            usage.tokens_used += tokens
            usage.requests_used += 1
            self._save_state()

    def mark_exhausted(self, api_key: str) -> None:
        """Force `api_key` to read as exhausted for the rest of today (e.g. after a 429), so
        `select_key` stops offering it even though our own tracked usage hadn't predicted the cap."""
        with self._lock:
            usage = self._usage_for(api_key)
            usage.tokens_used = self._daily_token_budget
            usage.requests_used = self._daily_request_budget
            self._save_state()

    def select_key(self, exclude: frozenset[str] = frozenset()) -> str | None:
        """Return the API key with the most remaining daily token budget, excluding `exclude`.

        Args:
            exclude: Raw API keys to skip (e.g. ones that just failed with a rate-limit error on
                this same call, so a retry shouldn't immediately pick them again).

        Returns:
            The best available key, or `None` if every key in the pool is exhausted for today.
        """
        candidates = [k for k in self._api_keys if k not in exclude]
        if not candidates:
            return None
        best = max(candidates, key=lambda k: self.remaining_budget(k)[0])
        tokens_left, requests_left = self.remaining_budget(best)
        if tokens_left <= 0 or requests_left <= 0:
            return None
        return best

    def status(self) -> list[dict]:
        """Human-readable per-key status, for logging/debugging a multi-day run's progress."""
        return [
            {
                "key": f"...{api_key[-4:]}",
                "tokens_remaining": self.remaining_budget(api_key)[0],
                "requests_remaining": self.remaining_budget(api_key)[1],
            }
            for api_key in self._api_keys
        ]
