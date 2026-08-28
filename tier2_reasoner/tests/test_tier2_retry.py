from unittest.mock import patch

import pytest

from ids_tier2.llm_client import LLMResponse
from ids_tier2.retry import RetryingLLMClient, _extract_retry_delay_s


class _FlakyClient:
    """Fails with a given exception the first `n_failures` calls, then
    succeeds. Records every call for assertions."""

    def __init__(self, n_failures: int, exc_factory=lambda: RuntimeError("boom")):
        self.n_failures = n_failures
        self.exc_factory = exc_factory
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.calls += 1
        if self.calls <= self.n_failures:
            raise self.exc_factory()
        return LLMResponse(text="ok", latency_ms=1.0, model_name="flaky")


class _AlwaysFailsClient:
    def __init__(self, exc_factory=lambda: RuntimeError("boom")):
        self.exc_factory = exc_factory
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.calls += 1
        raise self.exc_factory()


class _FakeAPIErrorWithRetryDelay(Exception):
    """Mimics google.genai.errors.APIError's shape: a `.details` dict with
    error.details[].retryDelay, without depending on the real SDK."""

    def __init__(self, retry_delay: str):
        super().__init__("429 RESOURCE_EXHAUSTED")
        self.details = {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay},
                ],
            }
        }


def test_succeeds_immediately_with_no_retry_needed():
    inner = _FlakyClient(n_failures=0)
    client = RetryingLLMClient(inner=inner)
    with patch("ids_tier2.retry.time.sleep") as mock_sleep:
        response = client.complete("sys", "user")
    assert response.text == "ok"
    assert inner.calls == 1
    mock_sleep.assert_not_called()


def test_retries_and_succeeds_after_transient_failures():
    inner = _FlakyClient(n_failures=2)
    client = RetryingLLMClient(inner=inner, max_retries=4, base_backoff_s=1.0)
    with patch("ids_tier2.retry.time.sleep") as mock_sleep:
        response = client.complete("sys", "user")
    assert response.text == "ok"
    assert inner.calls == 3  # failed twice, succeeded on 3rd
    assert mock_sleep.call_count == 2


def test_gives_up_and_raises_after_max_retries():
    inner = _AlwaysFailsClient()
    client = RetryingLLMClient(inner=inner, max_retries=3, base_backoff_s=0.1)
    with patch("ids_tier2.retry.time.sleep"):
        with pytest.raises(RuntimeError):
            client.complete("sys", "user")
    assert inner.calls == 3


def test_backoff_delay_grows_with_attempt_number_when_no_server_hint():
    inner = _FlakyClient(n_failures=3)
    client = RetryingLLMClient(inner=inner, max_retries=5, base_backoff_s=10.0)
    with patch("ids_tier2.retry.time.sleep") as mock_sleep:
        client.complete("sys", "user")
    # base_backoff_s * attempt for attempts 1, 2, 3
    assert [call.args[0] for call in mock_sleep.call_args_list] == [10.0, 20.0, 30.0]


def test_uses_server_suggested_retry_delay_when_available():
    inner = _FlakyClient(n_failures=1, exc_factory=lambda: _FakeAPIErrorWithRetryDelay("21s"))
    client = RetryingLLMClient(inner=inner, max_retries=3, base_backoff_s=999.0)
    with patch("ids_tier2.retry.time.sleep") as mock_sleep:
        client.complete("sys", "user")
    # 21.0 from the server, not base_backoff_s * attempt (which would be 999.0)
    mock_sleep.assert_called_once_with(21.0)


def test_extract_retry_delay_parses_seconds_string():
    exc = _FakeAPIErrorWithRetryDelay("7.5s")
    assert _extract_retry_delay_s(exc) == pytest.approx(7.5)


def test_extract_retry_delay_returns_none_for_exceptions_without_details():
    assert _extract_retry_delay_s(RuntimeError("plain error")) is None
    assert _extract_retry_delay_s(ConnectionError("connect failed")) is None


def test_pacing_waits_minimum_interval_between_calls():
    inner = _FlakyClient(n_failures=0)
    client = RetryingLLMClient(inner=inner, min_interval_s=30.0)

    # monotonic() is called: (a) after call 1 succeeds, to record last-call
    # time; (b) at the start of call 2's pacing check; (c) after call 2
    # succeeds, to record its own last-call time.
    with patch("ids_tier2.retry.time.sleep") as mock_sleep, patch("ids_tier2.retry.time.monotonic") as mock_mono:
        mock_mono.side_effect = [0.0, 5.0, 5.0]
        client.complete("sys", "user")  # call 1: no prior call yet, no pacing check at all
        assert mock_sleep.call_count == 0

        client.complete("sys", "user")  # call 2: pacing check sees elapsed=5.0, waits 25.0
        mock_sleep.assert_called_once()
        waited = mock_sleep.call_args.args[0]
        assert waited == pytest.approx(25.0)  # 30s interval - 5s elapsed = 25s


def test_pacing_does_not_wait_if_disabled():
    inner = _FlakyClient(n_failures=0)
    client = RetryingLLMClient(inner=inner, min_interval_s=0.0)
    with patch("ids_tier2.retry.time.sleep") as mock_sleep:
        client.complete("sys", "user")
        client.complete("sys", "user")
    mock_sleep.assert_not_called()
