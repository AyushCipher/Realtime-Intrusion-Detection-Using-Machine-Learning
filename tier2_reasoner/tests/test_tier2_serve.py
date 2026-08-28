from ids_tier2.llm_client import AnthropicLLMClient, GeminiLLMClient, StubLLMClient
from ids_tier2.retry import RetryingLLMClient
from ids_tier2.serve import build_llm_client, parse_args


def test_default_llm_is_stub_unwrapped():
    args = parse_args([])
    client = build_llm_client(args)
    assert isinstance(client, StubLLMClient)


def test_gemini_llm_is_wrapped_in_retry_by_default(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    args = parse_args(["--llm", "gemini"])
    client = build_llm_client(args)
    assert isinstance(client, RetryingLLMClient)
    assert isinstance(client.inner, GeminiLLMClient)


def test_anthropic_llm_is_wrapped_in_retry_by_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")
    args = parse_args(["--llm", "anthropic"])
    client = build_llm_client(args)
    assert isinstance(client, RetryingLLMClient)
    assert isinstance(client.inner, AnthropicLLMClient)


def test_no_retry_flag_disables_wrapping(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    args = parse_args(["--llm", "gemini", "--no-retry"])
    client = build_llm_client(args)
    assert isinstance(client, GeminiLLMClient)
    assert not isinstance(client, RetryingLLMClient)


def test_retry_settings_are_forwarded_from_cli_args(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    args = parse_args(["--llm", "gemini", "--max-retries", "7", "--base-backoff-s", "3.5", "--min-interval-s", "9.0"])
    client = build_llm_client(args)
    assert client.max_retries == 7
    assert client.base_backoff_s == 3.5
    assert client.min_interval_s == 9.0


def test_default_min_interval_matches_observed_free_tier_rate_limit():
    # 5 requests/minute observed live -> 60/5 = 12s spacing; see serve.py's
    # --min-interval-s help text and the README's Latency section.
    args = parse_args(["--llm", "gemini"])
    assert args.min_interval_s == 12.0


def test_stub_llm_is_never_wrapped_even_with_retry_flags_set():
    args = parse_args(["--llm", "stub", "--max-retries", "10"])
    client = build_llm_client(args)
    assert isinstance(client, StubLLMClient)
