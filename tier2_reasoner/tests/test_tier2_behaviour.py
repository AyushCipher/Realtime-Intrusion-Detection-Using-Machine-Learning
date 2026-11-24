"""Behaviour-keyed retrieval: describing a flow from its SHAP features and
retrieving ATT&CK context from that description when the predicted class
cannot be trusted."""

import pytest

from ids_tier2.behaviour import describe_flow
from ids_tier2.llm_client import StubLLMClient
from ids_tier2.reasoner import Tier2Reasoner, build_user_prompt
from ids_tier2.retrieval import TechniqueRetriever


def _alert(features, predicted="BENIGN", escalated=True):
    return {
        "stage2_predicted_class": predicted,
        "stage2_confidence": 0.9,
        "severity": "high",
        "unknown_mass": 0.5,
        "escalated": escalated,
        "explanation": [{"feature": f, "value": v, "shap_value": 0.1} for f, v in features],
    }


SCAN_LIKE = [("flow_duration", 0.0001), ("total_fwd_packets", 1), ("total_fwd_bytes", 0), ("syn_flag_count", 1), ("rst_flag_count", 1)]
FLOOD_LIKE = [("flow_packets_per_sec", 50000), ("flow_bytes_per_sec", 5e6), ("flow_iat_min", 1e-5), ("total_bwd_bytes", 10)]


def test_describe_flow_maps_features_to_phrases_in_order_without_duplicates():
    text = describe_flow(_alert(SCAN_LIKE))
    assert text.split(", ")[:2] == ["very short connection", "very few packets"]
    assert "no payload" in text and "syn flag" in text and "reset flag" in text
    assert len(text.split(", ")) == len(set(text.split(", ")))


def test_describe_flow_ignores_zero_flags_unknown_features_and_missing_values():
    alert = _alert([("syn_flag_count", 0), ("made_up_feature", 3), ("flow_duration", None)])
    assert describe_flow(alert) == ""
    assert describe_flow({"explanation": None}) == ""


@pytest.mark.parametrize("features,expected", [(SCAN_LIKE, "T1046"), (FLOOD_LIKE, "T1498")])
def test_retrieve_by_behaviour_ranks_matching_technique_first(features, expected):
    results = TechniqueRetriever().retrieve_by_behaviour(describe_flow(_alert(features)))
    assert results and results[0].entry.technique_id == expected


def test_retrieve_by_behaviour_empty_description_returns_nothing():
    assert TechniqueRetriever().retrieve_by_behaviour("") == []


def test_auto_mode_uses_behaviour_for_escalated_alert_and_class_otherwise():
    reasoner = Tier2Reasoner(StubLLMClient(), retrieval_mode="auto")
    escalated = reasoner.explain(_alert(SCAN_LIKE, predicted="BENIGN", escalated=True))
    assert escalated.retrieved_technique_ids[0] == "T1046"

    confident = reasoner.explain(_alert(SCAN_LIKE, predicted="Brute Force", escalated=False))
    assert confident.retrieved_technique_ids[0] == "T1110"


def test_auto_mode_falls_back_to_class_when_no_behaviour_phrase_applies():
    reasoner = Tier2Reasoner(StubLLMClient(), retrieval_mode="auto")
    result = reasoner.explain(_alert([("made_up_feature", 1)], predicted="PortScan", escalated=True))
    assert result.retrieved_technique_ids[0] == "T1046"


def test_class_mode_is_the_default_and_unchanged():
    reasoner = Tier2Reasoner(StubLLMClient())
    assert reasoner.retrieval_mode == "class"
    result = reasoner.explain(_alert(SCAN_LIKE, predicted="Brute Force", escalated=True))
    assert result.retrieved_technique_ids[0] == "T1110"


def test_prompt_flags_untrusted_class_only_when_behaviour_used():
    retrieved = TechniqueRetriever().retrieve_by_behaviour(describe_flow(_alert(SCAN_LIKE)))
    with_behaviour = build_user_prompt(_alert(SCAN_LIKE), retrieved, behaviour="very short connection")
    assert "may be wrong" in with_behaviour and "very short connection" in with_behaviour
    assert "may be wrong" not in build_user_prompt(_alert(SCAN_LIKE), retrieved)


def test_invalid_retrieval_mode_rejected():
    with pytest.raises(ValueError):
        Tier2Reasoner(StubLLMClient(), retrieval_mode="vibes")
