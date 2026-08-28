"""Retry/backoff and rate-limit-aware pacing for any `LLMClient`.

Built directly from real failures observed live-testing this module (see
`README.md`'s "Latency (live-verified -- and a reliability problem)"
section): a transient `503 UNAVAILABLE`, a hard `429 RESOURCE_EXHAUSTED`
after the free tier's per-minute limit, and `ConnectError` failures that
needed several retries to clear. None of this existed in the module
before those failures were observed live -- `service.py` previously just
caught, logged, and skipped a failed alert with no retry at all.

Two complementary mechanisms, not one:

1. **Reactive retry with backoff** -- on any exception from the wrapped
   client, wait and retry, up to `max_retries` times. When the API's own
   error response names a specific retry delay (Gemini's 429 responses
   include a `RetryInfo.retryDelay`, e.g. "21s" -- verified against a
   real 429 caught live, not assumed from documentation), that delay is
   used instead of the generic exponential backoff, since the server is
   telling you exactly how long it wants you to wait.
2. **Proactive pacing** -- `min_interval_s` enforces a minimum gap
   between calls regardless of whether the last one failed, so a known
   per-minute rate limit (5 requests/minute observed live on Gemini's
   free tier for at least one model) can be avoided by construction
   instead of always being hit and recovered from reactively.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .llm_client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


def _extract_retry_delay_s(exc: Exception) -> Optional[float]:
    """Pulls a server-suggested retry delay out of an exception if one is
    present. Handles google-genai's `APIError.details` shape (a dict with
    `error.details[].retryDelay`, e.g. `"21s"`) -- verified against a real
    429 response caught live. Returns None (caller falls back to
    exponential backoff) for any other exception shape, including
    non-API errors like connection failures.
    """
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return None
    error_details = details.get("error", {}).get("details", [])
    if not isinstance(error_details, list):
        return None
    for entry in error_details:
        if isinstance(entry, dict) and "retryDelay" in entry:
            match = re.match(r"^([\d.]+)s?$", str(entry["retryDelay"]).strip())
            if match:
                return float(match.group(1))
    return None


@dataclass
class RetryingLLMClient(LLMClient):
    """Wraps another `LLMClient` with retry/backoff and rate-limit pacing.

    `inner`: the real client to wrap (e.g. `GeminiLLMClient`). Wrapping
    rather than baking retry logic into each concrete client keeps
    `AnthropicLLMClient`/`GeminiLLMClient` focused on request/response
    shape only, and makes the retry policy itself independently testable
    against a fake inner client (see `test_tier2_retry.py`) without
    needing a real API.
    """

    inner: LLMClient
    max_retries: int = 4
    base_backoff_s: float = 15.0
    min_interval_s: float = 0.0
    _last_call_monotonic: Optional[float] = field(default=None, init=False, repr=False)

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self._wait_for_pacing()
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.inner.complete(system_prompt, user_prompt)
                self._last_call_monotonic = time.monotonic()
                return response
            except Exception as exc:  # noqa: BLE001 - any transport/API failure is retryable here
                last_exc = exc
                self._last_call_monotonic = time.monotonic()
                if attempt >= self.max_retries:
                    break
                delay = _extract_retry_delay_s(exc)
                if delay is None:
                    delay = self.base_backoff_s * attempt
                logger.warning(
                    "LLM call failed (attempt %d/%d, %s); retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)

        assert last_exc is not None
        raise last_exc

    def _wait_for_pacing(self) -> None:
        if self.min_interval_s <= 0 or self._last_call_monotonic is None:
            return
        elapsed = time.monotonic() - self._last_call_monotonic
        remaining = self.min_interval_s - elapsed
        if remaining > 0:
            logger.debug("Pacing: waiting %.1fs before next LLM call", remaining)
            time.sleep(remaining)
