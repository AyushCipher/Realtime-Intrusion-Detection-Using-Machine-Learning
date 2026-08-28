from unittest.mock import MagicMock, patch

import pytest

from ids_tier2.llm_client import AnthropicLLMClient, GeminiLLMClient, LLMResponse, StubLLMClient


def test_stub_llm_client_extracts_technique_from_prompt():
    client = StubLLMClient()
    prompt = "- T1110 (Brute Force) [Credential Access]: some description"
    response = client.complete("system", prompt)

    assert isinstance(response, LLMResponse)
    assert response.model_name == "stub-v1"
    assert response.latency_ms > 0
    assert '"suspected_technique_id": "T1110"' in response.text
    assert '"suspected_technique_name": "Brute Force"' in response.text


def test_stub_llm_client_handles_no_technique_in_prompt():
    client = StubLLMClient()
    response = client.complete("system", "No reference techniques were retrieved for this alert.")
    assert '"suspected_technique_id": ""' in response.text


def test_stub_llm_client_fixed_response_overrides_extraction():
    client = StubLLMClient(fixed_response="not json at all")
    response = client.complete("system", "- T1110 (Brute Force): text")
    assert response.text == "not json at all"


def test_stub_llm_client_records_calls():
    client = StubLLMClient()
    client.complete("sys1", "user1")
    client.complete("sys2", "user2")
    assert client.calls == [("sys1", "user1"), ("sys2", "user2")]


def test_stub_llm_client_response_is_valid_json():
    import json

    client = StubLLMClient()
    response = client.complete("system", "- T1046 (Network Service Discovery): text")
    parsed = json.loads(response.text)
    assert parsed["suspected_technique_id"] == "T1046"


def test_anthropic_client_requires_the_package_or_raises_clear_error():
    # This project's environment does have `anthropic` installed, so this
    # tests the wiring instead by forcing the import to fail.
    with patch.dict("sys.modules", {"anthropic": None}):
        with pytest.raises(ImportError):
            AnthropicLLMClient()


def test_anthropic_client_calls_messages_create_and_parses_text_blocks():
    mock_anthropic_module = MagicMock()
    mock_client_instance = MagicMock()
    mock_anthropic_module.Anthropic.return_value = mock_client_instance

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = '{"suspected_technique_id": "T1190"}'
    mock_response = MagicMock()
    mock_response.content = [text_block]
    mock_client_instance.messages.create.return_value = mock_response

    with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
        client = AnthropicLLMClient(model="claude-sonnet-5", api_key="test-key")
        response = client.complete("system prompt", "user prompt")

    mock_client_instance.messages.create.assert_called_once()
    call_kwargs = mock_client_instance.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-5"
    assert call_kwargs["system"] == "system prompt"
    assert call_kwargs["messages"] == [{"role": "user", "content": "user prompt"}]
    assert response.text == '{"suspected_technique_id": "T1190"}'
    assert response.model_name == "claude-sonnet-5"
    assert response.latency_ms >= 0


def test_gemini_client_requires_the_package_or_raises_clear_error():
    # google-genai IS installed in this project's environment (needed for
    # the live-verified path), so this tests the wiring by forcing the
    # import to fail rather than actually uninstalling anything.
    with patch.dict("sys.modules", {"google": None, "google.genai": None}):
        with pytest.raises(ImportError):
            GeminiLLMClient()


def test_gemini_client_calls_generate_content_and_returns_text():
    from google import genai

    mock_response = MagicMock()
    mock_response.text = '{"suspected_technique_id": "T1190"}'
    mock_instance = MagicMock()
    mock_instance.models.generate_content.return_value = mock_response

    with patch.object(genai, "Client", return_value=mock_instance) as mock_client_cls:
        client = GeminiLLMClient(model="gemini-2.5-flash", api_key="test-key")
        response = client.complete("system prompt", "user prompt")

    mock_client_cls.assert_called_once_with(api_key="test-key")
    mock_instance.models.generate_content.assert_called_once()
    call_kwargs = mock_instance.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert call_kwargs["contents"] == "user prompt"
    assert call_kwargs["config"].system_instruction == "system prompt"
    assert response.text == '{"suspected_technique_id": "T1190"}'
    assert response.model_name == "gemini-2.5-flash"
    assert response.latency_ms >= 0


def test_gemini_client_handles_empty_response_text():
    from google import genai

    mock_response = MagicMock()
    mock_response.text = None
    mock_instance = MagicMock()
    mock_instance.models.generate_content.return_value = mock_response

    with patch.object(genai, "Client", return_value=mock_instance):
        client = GeminiLLMClient(api_key="test-key")
        response = client.complete("system", "user")

    assert response.text == ""
