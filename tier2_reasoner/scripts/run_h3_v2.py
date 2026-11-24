"""H3 v2: a larger, fairer test of whether retrieval grounding helps Tier 2.

Differences from run_h3_ablation.py (which stays as the original pilot):
  * Three conditions through one shared, paced client:
      rag            retrieval-grounded (the system as built)
      no_rag_free    NO retrieval, but the model is told to name its best-guess
                     ATT&CK technique ID anyway -- a fair parametric-knowledge
                     baseline. The original no-RAG condition was instructed not
                     to name a technique, so its ~0% was close to guaranteed.
      no_rag_strict  the original no-RAG condition (kept for comparison).
  * More trials, and alert fields vary per trial (confidence, feature values,
    addresses) so prompts are not byte-identical copies.
  * Results are appended to a JSONL file after every call and the run resumes
    from it, so a quota failure part-way does not lose finished calls.
  * Reports exact McNemar (paired) and Wilson 95% intervals, plus the number of
    failed calls per condition (failures are reported, not hidden).

Needs GEMINI_API_KEY in the environment (never pass it on the command line or
commit it). Real API calls consume quota.

Usage:
    python -m scripts.run_h3_v2 --trials-per-category 20 --model gemini-3.8-flash --output h3_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from scipy.stats import binomtest

from ids_tier2.knowledge_base import KNOWLEDGE_BASE, categories_covered
from ids_tier2.llm_client import GeminiLLMClient, LLMClient, LLMResponse
from ids_tier2.reasoner import Tier2Reasoner
from ids_tier2.retry import RetryingLLMClient

logger = logging.getLogger("ids_tier2.scripts.run_h3_v2")

EXPECTED: Dict[str, set] = {}
for _e in KNOWLEDGE_BASE:
    EXPECTED.setdefault(_e.category, set()).add(_e.technique_id)

FREE_PROMPT = (
    "You are a security analyst assistant helping triage network intrusion detection alerts. "
    "You are given an alert's predicted attack category, confidence and top contributing features. "
    "Respond with ONLY a JSON object with exactly these keys: suspected_technique_id, "
    "suspected_technique_name, risk_explanation, recommended_action. Always give your single best-guess "
    "MITRE ATT&CK technique ID (for example T1110) for this alert, even without reference material."
)


class _PromptOverride(LLMClient):
    """Replaces the system prompt, so the no_rag_free condition shares the paced client."""

    def __init__(self, inner: LLMClient, system_prompt: str) -> None:
        self.inner, self.system_prompt = inner, system_prompt

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        return self.inner.complete(self.system_prompt, user_prompt)


def build_alert(category: str, trial: int) -> Dict[str, Any]:
    rng = random.Random(f"{category}-{trial}")
    return {
        "alert_id": f"h3v2-{category}-{trial}", "flow_id": f"flow-{trial}",
        "src_ip": f"10.{rng.randint(0, 9)}.{rng.randint(0, 9)}.{rng.randint(2, 250)}",
        "dst_ip": f"10.0.0.{rng.randint(2, 250)}", "dst_port": rng.choice([22, 80, 443, 445, 3389, 8080]),
        "protocol": 6, "stage2_predicted_class": category,
        "stage2_confidence": round(rng.uniform(0.35, 0.99), 2), "severity": rng.choice(["medium", "high"]),
        "explanation": [
            {"feature": "flow_duration", "value": round(rng.uniform(0.01, 30), 3), "shap_value": round(rng.uniform(0.1, 0.9), 2)},
            {"feature": "flow_packets_per_sec", "value": round(rng.uniform(1, 5000), 1), "shap_value": round(rng.uniform(0.1, 0.9), 2)},
            {"feature": "syn_flag_count", "value": rng.randint(0, 40), "shap_value": round(rng.uniform(0.0, 0.6), 2)},
        ],
        "unknown_mass": round(rng.uniform(0.1, 0.9), 2), "escalated": True, "escalation_trigger": "openset",
    }


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials-per-category", type=int, default=20)
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--min-interval-s", type=float, default=4.0)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--base-backoff-s", type=float, default=10.0)
    ap.add_argument("--trial-offset", type=int, default=0, help="first trial index; use a fresh range to re-test on alerts not seen before")
    ap.add_argument("--min-relative-score", type=float, default=0.5, help="Tier2Reasoner retrieval floor (0 = keep all top_k)")
    ap.add_argument("--output", default="h3_v2.jsonl")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    shared = RetryingLLMClient(
        inner=GeminiLLMClient(model=args.model, api_key=os.environ["GEMINI_API_KEY"]),
        max_retries=args.max_retries, base_backoff_s=args.base_backoff_s, min_interval_s=args.min_interval_s,
    )
    reasoners = {
        "rag": Tier2Reasoner(shared, use_rag=True, min_relative_score=args.min_relative_score),
        "no_rag_free": Tier2Reasoner(_PromptOverride(shared, FREE_PROMPT), use_rag=False),
        "no_rag_strict": Tier2Reasoner(shared, use_rag=False),
    }

    out = Path(args.output)
    done = set()
    rows: List[dict] = []
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            rows.append(r)
            done.add((r["category"], r["trial"], r["condition"]))
        logger.info("Resuming: %d calls already recorded", len(done))

    cats = categories_covered()
    logger.info("model=%s categories=%s plan=%d calls", args.model, cats, len(cats) * args.trials_per_category * 3)
    with out.open("a") as f:
        for trial in range(args.trial_offset, args.trial_offset + args.trials_per_category):
            for cat in cats:
                for cond, reasoner in reasoners.items():
                    if (cat, trial, cond) in done:
                        continue
                    alert = build_alert(cat, trial)
                    rec = {"category": cat, "trial": trial, "condition": cond, "completed": False,
                           "technique_id": "", "correct": False, "error": ""}
                    try:
                        exp = reasoner.explain(alert)
                        rec.update(completed=True, technique_id=exp.suspected_technique_id,
                                   correct=exp.suspected_technique_id in EXPECTED[cat], latency_ms=exp.llm_latency_ms)
                    except Exception as exc:  # noqa: BLE001 - a failed call is data, not a crash
                        rec["error"] = str(exc)[:200]
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    rows.append(rec)
            logger.info("trial %d/%d complete (%d calls recorded)", trial + 1, args.trials_per_category, len(rows))

    summary: Dict[str, Any] = {"model": args.model, "trials_per_category": args.trials_per_category, "conditions": {}}
    for cond in reasoners:
        rs = [r for r in rows if r["condition"] == cond]
        comp = [r for r in rs if r["completed"]]
        k = sum(r["correct"] for r in comp)
        summary["conditions"][cond] = {
            "calls": len(rs), "failed": len(rs) - len(comp), "correct": k, "completed": len(comp),
            "accuracy": round(k / len(comp), 3) if comp else None, "wilson95": wilson(k, len(comp)),
        }
    by = {(r["category"], r["trial"], r["condition"]): r for r in rows}
    for other in ("no_rag_free", "no_rag_strict"):
        b = c = 0
        for (cat, trial, cond), r in by.items():
            if cond != "rag":
                continue
            o = by.get((cat, trial, other))
            if o and r["completed"] and o["completed"]:
                b += int(r["correct"] and not o["correct"])
                c += int((not r["correct"]) and o["correct"])
        p = float(binomtest(b, b + c, 0.5).pvalue) if b + c else None
        summary[f"mcnemar_rag_vs_{other}"] = {"b_rag_only": b, "c_other_only": c, "p": p}
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
