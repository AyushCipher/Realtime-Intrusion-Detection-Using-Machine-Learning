"""Describe what a flow DOES, for retrieval when its predicted class is wrong.

Tier 2 retrieves ATT&CK context keyed on Stage 2's predicted class. That
works when the class is right, but the alerts Tier 2 most needs to explain
are the ones escalated as unfamiliar, and for those the class is wrong by
construction: on real escalated alerts from withheld families, every
predicted class was wrong (84% said BENIGN) and class-keyed grounding scored
0/100 (docs/real_data_findings.md, section 7a).

A data-driven fix cannot work: the family was never in training, so nothing
learned from training data maps it to a technique. What remains is domain
knowledge. `describe_flow` turns an alert's top SHAP features and their
values into phrases from a small controlled vocabulary ("very short
connection", "reset flag", "high packet rate", ...), and each knowledge-base
entry carries a `signature` in the same vocabulary, so TF-IDF can match
behaviour to technique.

Thresholds are fixed from units (seconds, bytes, packets per second), not
fitted to any dataset. Only the top SHAP features are available in an alert,
so a description covers at most five behaviours; flows with uninformative
top features get a thin or empty description and fall back to class-keyed
retrieval in the reasoner.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _phrases(feature: str, value: float) -> List[str]:
    f = feature
    v = float(value)
    if f == "flow_duration":
        if v < 0.01:
            return ["very short connection"]
        if v < 1.0:
            return ["short connection"]
        if v > 60.0:
            return ["long duration", "connection kept open"]
        return ["moderate duration"]
    if f == "total_fwd_packets":
        if v <= 3:
            return ["very few packets"]
        if v <= 10:
            return ["few packets"]
        return ["many packets"] if v > 100 else []
    if f == "total_bwd_packets":
        return ["no server response"] if v == 0 else ["server replies"]
    if f == "total_fwd_bytes":
        if v == 0:
            return ["no payload"]
        if v < 200:
            return ["small payload"]
        return ["large client payload"] if v > 1000 else []
    if f == "total_bwd_bytes":
        if v == 0:
            return ["no server response"]
        return ["little server response"] if v < 200 else ["server replies"]
    if f.startswith("fwd_packet_length_") and not f.endswith("_std"):
        if v == 0:
            return ["no payload"]
        if v < 100:
            return ["small packets"]
        return ["large forward packets"] if v > 500 else []
    if f.startswith("bwd_packet_length_") and not f.endswith("_std"):
        return ["no server response"] if v == 0 else ["server replies"]
    if f == "flow_bytes_per_sec":
        if v > 1e5:
            return ["high byte rate"]
        return ["low byte rate"] if v < 1e3 else []
    if f == "flow_packets_per_sec":
        if v > 1000:
            return ["high packet rate"]
        return ["slow"] if v < 10 else []
    if f.endswith("_iat_min") or f.endswith("_iat_mean"):
        if v < 0.001:
            return ["short gaps between packets"]
        return ["slow"] if v > 1.0 else []
    if f.endswith("_flag_count") and v > 0:
        return {
            "syn_flag_count": ["syn flag"],
            "rst_flag_count": ["reset flag"],
            "fin_flag_count": ["fin flag", "connection closed"],
            "psh_flag_count": ["push flag"],
        }.get(f, [])
    return []


def describe_flow(alert: Dict[str, Any], max_features: int = 5) -> str:
    """Comma-separated behaviour phrases for the alert's top SHAP features,
    de-duplicated, in attribution order. Empty when none of the top
    features maps to a phrase."""
    phrases: List[str] = []
    for item in (alert.get("explanation") or [])[:max_features]:
        name, value = item.get("feature"), item.get("value")
        if name is None or value is None:
            continue
        for p in _phrases(str(name), value):
            if p not in phrases:
                phrases.append(p)
    return ", ".join(phrases)
