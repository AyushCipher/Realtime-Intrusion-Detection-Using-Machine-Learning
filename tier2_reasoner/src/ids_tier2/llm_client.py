"""LLM client abstraction: a deterministic stub (for tests, and for any
environment without API access) and a real hosted-model client both
satisfy the same interface, so `reasoner.py` never has to know which one
it's talking to -- the same `--use-stub`-style pattern the other three
modules already use for Kafka (see e.g. `ids_ml.serve`'s `--use-stub`).
"""

from __future__ import annotations

import abc
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class LLMResponse:
    text: str
    latency_ms: float
    model_name: str


class LLMClient(abc.ABC):
    @abc.abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        raise NotImplementedError


class StubLLMClient(LLMClient):
    """Deterministic, offline stand-in -- no network call, no API key.

    Default behavior: pulls the first "Txxxx"-shaped MITRE ATT&CK
    technique ID (and, if present, the parenthesized name right after it
    -- matching how `reasoner.py` formats retrieved context into the
    prompt) out of the user prompt, and echoes a templated, validly-
    structured JSON response built around it. Pass `fixed_response` to
    return an exact string instead, for tests that need to exercise the
    reasoner's malformed/unparseable-response handling.
    """

    def __init__(self, fixed_response: Optional[str] = None, simulated_latency_ms: float = 5.0) -> None:
        self.fixed_response = fixed_response
        self.simulated_latency_ms = simulated_latency_ms
        self.calls: List[Tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.calls.append((system_prompt, user_prompt))
        t0 = time.perf_counter()

        if self.fixed_response is not None:
            text = self.fixed_response
        else:
            id_match = re.search(r"T\d{4}(?:\.\d{3})?", user_prompt)
            technique_id = id_match.group(0) if id_match else ""
            technique_name = ""
            if technique_id:
                name_match = re.search(rf"{re.escape(technique_id)}\s*\(([^)]+)\)", user_prompt)
                if name_match:
                    technique_name = name_match.group(1)
            text = (
                '{"suspected_technique_id": "%s", "suspected_technique_name": "%s", '
                '"risk_explanation": "Traffic pattern matches indicators associated with %s.", '
                '"recommended_action": "Review the flagged flow and correlate with recent alerts from the same source IP."}'
            ) % (technique_id, technique_name, technique_name or "an unrecognized pattern")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0 + self.simulated_latency_ms
        return LLMResponse(text=text, latency_ms=elapsed_ms, model_name="stub-v1")


class AnthropicLLMClient(LLMClient):
    """Real hosted-model client via the `anthropic` package.

    Requires `pip install anthropic` (already a listed dependency here)
    and an API key -- passed explicitly or read from the
    `ANTHROPIC_API_KEY` environment variable by the SDK itself.
    Connectivity and request wiring were confirmed live (real key, real
    request, real 400 response) -- the account behind that key had no
    credit balance, so no successful completion has been observed yet.
    See the module README's known-limitations section for the current
    status before trusting this in production without testing it
    yourself first with a funded account.
    """

    def __init__(self, model: str = "claude-sonnet-5", api_key: Optional[str] = None, max_tokens: int = 512) -> None:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - exercised only when the package is absent
            raise ImportError("AnthropicLLMClient requires the 'anthropic' package: pip install anthropic") from e
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        t0 = time.perf_counter()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        return LLMResponse(text=text, latency_ms=elapsed_ms, model_name=self.model)


class GeminiLLMClient(LLMClient):
    """Real hosted-model client via Google's unified `google-genai` SDK.

    Requires `pip install google-genai` (listed in requirements.txt) and
    an API key from https://aistudio.google.com/apikey -- passed
    explicitly or read from the `GEMINI_API_KEY` environment variable by
    the SDK itself. Defaults to a Flash model since, per Google AI
    Studio's free-tier documentation, Flash models are free-tier-eligible
    while Pro models require billing enabled even on a free-tier key --
    added as a lower-friction alternative to `AnthropicLLMClient` (Google
    AI Studio keys work on the free tier immediately with no payment
    method, unlike the Anthropic Console).

    The default model name has already had to change once during this
    project's own live testing: `gemini-2.5-flash` (this module's
    original default, and what the first live-verified results in
    `tier2_reasoner/README.md` were measured against) returned a live 404
    ("no longer available to new users") on a second, newer test account,
    which pointed to `gemini-3.6-flash` as the replacement -- notably
    faster in practice (~2.6s vs. ~5.75s median, one call). Google's Flash
    model lineup moves; if this default 404s for you, the error message
    itself names the current replacement.
    """

    def __init__(self, model: str = "gemini-3.6-flash", api_key: Optional[str] = None) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:  # pragma: no cover - exercised only when the package is absent
            raise ImportError("GeminiLLMClient requires the 'google-genai' package: pip install google-genai") from e
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self._types = types
        self.model = model

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        t0 = time.perf_counter()
        response = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=self._types.GenerateContentConfig(system_instruction=system_prompt),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return LLMResponse(text=response.text or "", latency_ms=elapsed_ms, model_name=self.model)
