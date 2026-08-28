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
    `ANTHROPIC_API_KEY` environment variable by the SDK itself. **Neither
    a key nor live network access to the API was available in the
    environment this module was built in**, so this path is implemented
    and unit-tested for its request/response wiring (see
    `test_tier2_llm_client.py`, which mocks the SDK client) but has not
    been exercised against a real API call -- see the module README's
    known-limitations section before trusting this in production without
    testing it yourself first.
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
