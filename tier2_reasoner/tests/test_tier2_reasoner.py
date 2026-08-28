from ids_tier2.llm_client import StubLLMClient
from ids_tier2.reasoner import Tier2Reasoner, build_user_prompt

_ALERT = {
    "alert_id": "a-1",
    "flow_id": "f-1",
    "stage2_predicted_class": "Brute Force",
    "stage2_confidence": 0.82,
    "severity": "medium",
    "unknown_mass": 0.65,
    "explanation": [{"feature": "syn_flag_count", "value": 1.0, "shap_value": 0.3}],
    "escalated": True,
}


def test_explain_with_rag_retrieves_and_grounds_response():
    client = StubLLMClient()
    reasoner = Tier2Reasoner(client, use_rag=True)
    explanation = reasoner.explain(_ALERT)

    assert explanation.rag_enabled is True
    assert "T1110" in explanation.retrieved_technique_ids
    assert explanation.suspected_technique_id == "T1110"
    assert explanation.model_version == "stub-v1"
    assert explanation.llm_latency_ms > 0


def test_explain_without_rag_retrieves_nothing():
    client = StubLLMClient()
    reasoner = Tier2Reasoner(client, use_rag=False)
    explanation = reasoner.explain(_ALERT)

    assert explanation.rag_enabled is False
    assert explanation.retrieved_technique_ids == []
    # Stub has nothing to extract without retrieved context in the prompt
    assert explanation.suspected_technique_id == ""


def test_explain_handles_malformed_llm_response_gracefully():
    client = StubLLMClient(fixed_response="this is not json")
    reasoner = Tier2Reasoner(client, use_rag=True)
    explanation = reasoner.explain(_ALERT)

    assert explanation.risk_explanation == "this is not json"
    assert explanation.suspected_technique_id == ""
    assert explanation.recommended_action == ""


def test_explain_strips_markdown_code_fence_around_json():
    # Regression test: a real live Gemini 2.5 Flash call wrapped an
    # otherwise-correct JSON response in a ```json fence despite the
    # system prompt asking for raw JSON only -- this used to fall through
    # to the degraded (unparsed) fallback and silently discard a good
    # structured response.
    fenced = (
        '```json\n'
        '{"suspected_technique_id": "T1110", "suspected_technique_name": "Brute Force", '
        '"risk_explanation": "text", "recommended_action": "block the source IP"}\n'
        '```'
    )
    client = StubLLMClient(fixed_response=fenced)
    reasoner = Tier2Reasoner(client, use_rag=True)
    explanation = reasoner.explain(_ALERT)

    assert explanation.suspected_technique_id == "T1110"
    assert explanation.suspected_technique_name == "Brute Force"
    assert explanation.recommended_action == "block the source IP"


def test_explain_strips_bare_code_fence_without_language_tag():
    fenced = '```\n{"suspected_technique_id": "T1046"}\n```'
    client = StubLLMClient(fixed_response=fenced)
    reasoner = Tier2Reasoner(client, use_rag=True)
    explanation = reasoner.explain(_ALERT)
    assert explanation.suspected_technique_id == "T1046"


def test_build_user_prompt_includes_confidence_and_severity():
    prompt = build_user_prompt(_ALERT, [])
    assert "Brute Force" in prompt
    assert "0.82" in prompt
    assert "medium" in prompt


def test_build_user_prompt_includes_shap_features():
    prompt = build_user_prompt(_ALERT, [])
    assert "syn_flag_count" in prompt


def test_build_user_prompt_notes_absence_of_retrieval():
    prompt = build_user_prompt(_ALERT, [])
    assert "No reference techniques were retrieved" in prompt


def test_explain_handles_missing_optional_alert_fields():
    # A minimal alert (no explanation/unknown_mass) shouldn't crash prompt building.
    minimal_alert = {"alert_id": "a-2", "stage2_predicted_class": "PortScan", "stage2_confidence": 0.5, "severity": "low"}
    client = StubLLMClient()
    reasoner = Tier2Reasoner(client, use_rag=True)
    explanation = reasoner.explain(minimal_alert)
    assert explanation.suspected_technique_id == "T1046"
