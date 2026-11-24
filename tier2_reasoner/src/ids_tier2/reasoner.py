"""Ties retrieval (`retrieval.py`), an LLM client (`llm_client.py`), and
an alert's own fields into a structured `Explanation` -- the actual
"reasoning" step this module exists to do.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .behaviour import describe_flow
from .llm_client import LLMClient
from .retrieval import RetrievalResult, TechniqueRetriever

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a security analyst assistant helping triage network intrusion "
    "detection alerts. You will be given an alert's predicted attack category, "
    "confidence, and (if available) reference MITRE ATT&CK technique "
    "descriptions retrieved for that category. Respond with ONLY a JSON object "
    "with exactly these keys: suspected_technique_id, suspected_technique_name, "
    "risk_explanation, recommended_action. If no reference technique is "
    "provided or none plausibly matches, leave the technique_id/name fields as "
    "empty strings and say so in risk_explanation rather than inventing one."
)


@dataclass
class Explanation:
    suspected_technique_id: str
    suspected_technique_name: str
    risk_explanation: str
    recommended_action: str
    retrieved_technique_ids: List[str]
    rag_enabled: bool
    llm_latency_ms: float
    model_version: str


def build_user_prompt(alert: Dict[str, Any], retrieved: List[RetrievalResult], behaviour: str = "") -> str:
    confidence = alert.get("stage2_confidence") or 0.0
    unknown_mass = alert.get("unknown_mass") or 0.0
    lines = [
        f"Alert: predicted category={alert.get('stage2_predicted_class')}, "
        f"confidence={confidence:.2f}, severity={alert.get('severity')}, "
        f"unknown_mass={unknown_mass:.2f}."
    ]

    top_features = alert.get("explanation") or []
    if top_features:
        feat_str = ", ".join(f"{f.get('feature')}={f.get('value')}" for f in top_features[:5])
        lines.append(f"Top contributing features (SHAP): {feat_str}.")

    if behaviour:
        # Behaviour-keyed retrieval is only used when the predicted category
        # is untrustworthy (an escalated, unfamiliar flow), so say so: the
        # model should weigh the flow's behaviour over the label.
        lines.append(
            "This flow was escalated as unlike any known category, so the predicted category may be wrong. "
            f"Observed behaviour: {behaviour}. The reference techniques below were retrieved by matching this "
            "behaviour, not the predicted category."
        )

    if retrieved:
        lines.append("Reference techniques (retrieved, ranked by relevance):")
        for r in retrieved:
            lines.append(f"- {r.entry.technique_id} ({r.entry.name}) [{r.entry.tactic}]: {r.entry.description}")
    else:
        lines.append("No reference techniques were retrieved for this alert.")

    lines.append("Respond with ONLY the JSON object described in the system prompt -- no other text.")
    return "\n".join(lines)


class Tier2Reasoner:
    def __init__(
        self,
        llm_client: LLMClient,
        retriever: Optional[TechniqueRetriever] = None,
        use_rag: bool = True,
        top_k: int = 3,
        min_relative_score: float = 0.5,
        retrieval_mode: str = "class",
    ) -> None:
        self.llm_client = llm_client
        self.retriever = retriever or TechniqueRetriever()
        self.use_rag = use_rag
        self.top_k = top_k
        # Retrieved techniques scoring below this fraction of the best match are dropped before
        # prompting. Weak matches act as distractors: in the H3 v2 live run, Web Attack alerts
        # retrieved T1190 (score 0.24) plus T1499 (0.08), and the model picked T1499 in 19 of 20
        # trials. 0.0 keeps every top_k result (the previous behaviour).
        self.min_relative_score = min_relative_score
        # "class": key retrieval on stage2_predicted_class (the original
        # behaviour). "behaviour": key it on behaviour.describe_flow(alert).
        # "auto": behaviour for flows whose class cannot be trusted (escalated
        # by the open-set gate, or predicted BENIGN yet alerting), class
        # otherwise, falling back to class when no behaviour phrase applies.
        if retrieval_mode not in ("class", "behaviour", "auto"):
            raise ValueError("retrieval_mode must be 'class', 'behaviour' or 'auto'")
        self.retrieval_mode = retrieval_mode

    def explain(self, alert: Dict[str, Any]) -> Explanation:
        """Retrieves grounding context (unless `use_rag=False` -- the H3
        ablation switch: RAG vs. no-RAG explanation quality), builds a
        prompt, calls the LLM, and parses the result. Never raises on a
        malformed LLM response -- falls back to the raw text as
        `risk_explanation` with empty technique fields rather than
        dropping the alert, since a degraded explanation is still more
        useful to an analyst than none.
        """
        retrieved: List[RetrievalResult] = []
        behaviour = ""
        if self.use_rag:
            retrieved, behaviour = self._retrieve(alert)
            if retrieved and retrieved[0].score > 0:
                floor = self.min_relative_score * retrieved[0].score
                retrieved = [r for r in retrieved if r.score >= floor]

        user_prompt = build_user_prompt(alert, retrieved, behaviour=behaviour)
        response = self.llm_client.complete(SYSTEM_PROMPT, user_prompt)
        parsed = self._parse_response(response.text)

        return Explanation(
            suspected_technique_id=parsed.get("suspected_technique_id", ""),
            suspected_technique_name=parsed.get("suspected_technique_name", ""),
            risk_explanation=parsed.get("risk_explanation", response.text),
            recommended_action=parsed.get("recommended_action", ""),
            retrieved_technique_ids=[r.entry.technique_id for r in retrieved],
            rag_enabled=self.use_rag,
            llm_latency_ms=response.latency_ms,
            model_version=response.model_name,
        )

    def _retrieve(self, alert: Dict[str, Any]):
        """Returns (results, behaviour description used, or "")."""
        class_untrusted = bool(alert.get("escalated")) or alert.get("stage2_predicted_class") == "BENIGN"
        if self.retrieval_mode == "behaviour" or (self.retrieval_mode == "auto" and class_untrusted):
            behaviour = describe_flow(alert)
            results = self.retriever.retrieve_by_behaviour(behaviour, top_k=self.top_k)
            if results or self.retrieval_mode == "behaviour":
                return results, behaviour if results else ""
        query = str(alert.get("stage2_predicted_class", ""))
        return self.retriever.retrieve(query, top_k=self.top_k), ""

    @staticmethod
    def _parse_response(text: str) -> Dict[str, str]:
        # Strip a leading/trailing markdown code fence (```json ... ``` or
        # ``` ... ```) before parsing -- a common LLM habit even when the
        # system prompt explicitly asks for raw JSON only. Caught live: a
        # real Gemini 2.5 Flash response wrapped otherwise-correct JSON in
        # a ```json fence, which fell through to the degraded fallback
        # below and silently discarded a perfectly good structured
        # response until this was added.
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                raise ValueError("expected a JSON object")
            return {k: str(v) for k, v in parsed.items()}
        except (json.JSONDecodeError, ValueError):
            logger.warning("LLM response was not valid JSON; falling back to raw text as risk_explanation")
            return {"risk_explanation": text}
