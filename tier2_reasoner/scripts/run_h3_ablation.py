"""H3: does RAG grounding actually improve Tier 2's explanation, measured
as "did it name the correct MITRE technique for this alert's category"?

`tier2_reasoner/README.md`'s "RAG ablation" section has an informal n=9
(RAG) / n=3 (no-RAG) live result from ad-hoc single calls -- real, but not
the paired, statistically-powered comparison H1/H2 got in `ml/`. This
script is that comparison: for each category `knowledge_base.py` covers,
run `--trials-per-category` paired (RAG, no-RAG) trials against a real
LLM, using this project's own retry/backoff/pacing (`RetryingLLMClient`)
so a live batch survives the same rate-limit/reliability conditions that
broke an earlier unpaced batch (see the README's Latency section) --
this script doubles as that retry logic's live re-test, not just H3.

Metric (applied identically to both conditions, not two different
rubrics): did `Explanation.suspected_technique_id` match one of the
category's true technique ID(s) in `knowledge_base.KNOWLEDGE_BASE`
(DoS/DDoS maps to two techniques, T1498/T1499 -- either counts).
No-RAG is *expected* to score near zero under this metric (the system
prompt instructs it not to invent a technique ID absent retrieved
grounding), which is exactly what a real RAG effect looks like under a
single, fair, shared rubric -- not evidence the metric is unfair to it.

A single `RetryingLLMClient` is shared by both the RAG and no-RAG
`Tier2Reasoner` instances so `min_interval_s` pacing is enforced against
the *combined* call rate, not per-condition independently (two
independently-paced clients could together exceed the real per-minute
quota even though each looks correctly paced on its own).

Usage:
    python -m scripts.run_h3_ablation --trials-per-category 3 --output h3_results.json

Requires GEMINI_API_KEY (or --anthropic with ANTHROPIC_API_KEY) in the
environment. Real API calls, real cost/quota usage -- see the README's
free-tier rate limit (5 requests/minute observed for at least one model)
before raising --trials-per-category or lowering --min-interval-s.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from scipy.stats import binomtest

from ids_tier2.knowledge_base import KNOWLEDGE_BASE, categories_covered
from ids_tier2.llm_client import AnthropicLLMClient, GeminiLLMClient
from ids_tier2.reasoner import Tier2Reasoner
from ids_tier2.retry import RetryingLLMClient

logger = logging.getLogger("ids_tier2.scripts.run_h3_ablation")

# category -> acceptable true technique ID(s)
_EXPECTED_TECHNIQUE_IDS: Dict[str, set] = {}
for _entry in KNOWLEDGE_BASE:
    _EXPECTED_TECHNIQUE_IDS.setdefault(_entry.category, set()).add(_entry.technique_id)


def _build_alert(category: str, trial: int) -> Dict[str, Any]:
    """A minimal, realistic escalated-alert dict for `category` -- only
    the fields build_user_prompt actually reads matter for this ablation;
    the rest are filled in for schema-shape realism, not exercised.
    """
    return {
        "alert_id": f"h3-{category.lower().replace(' ', '-').replace('/', '-')}-{trial}",
        "flow_id": f"flow-h3-{trial}",
        "src_ip": "10.0.0.5",
        "dst_ip": "10.0.0.9",
        "dst_port": 443,
        "protocol": 6,
        "stage2_predicted_class": category,
        "stage2_confidence": 0.91,
        "severity": "high",
        "explanation": [
            {"feature": "flow_duration", "value": 0.12, "shap_value": 0.41},
            {"feature": "flow_packets_per_sec", "value": 220.0, "shap_value": 0.33},
        ],
        "unknown_mass": 0.35,
        "escalated": True,
        "escalation_trigger": "openset",
    }


@dataclass
class TrialResult:
    category: str
    condition: str  # "rag" | "no_rag"
    trial: int
    completed: bool
    suspected_technique_id: str
    correct: bool
    latency_ms: float
    error: str


def _run_one(reasoner: Tier2Reasoner, category: str, condition: str, trial: int) -> TrialResult:
    alert = _build_alert(category, trial)
    expected = _EXPECTED_TECHNIQUE_IDS[category]
    try:
        explanation = reasoner.explain(alert)
        correct = explanation.suspected_technique_id in expected
        result = TrialResult(
            category=category,
            condition=condition,
            trial=trial,
            completed=True,
            suspected_technique_id=explanation.suspected_technique_id,
            correct=correct,
            latency_ms=explanation.llm_latency_ms,
            error="",
        )
        logger.info(
            "%s/%s trial %d: technique_id=%r correct=%s (%.0fms)",
            category, condition, trial, explanation.suspected_technique_id, correct, explanation.llm_latency_ms,
        )
        return result
    except Exception as exc:  # noqa: BLE001 - record and continue, this is a live-reliability probe too
        logger.warning("%s/%s trial %d FAILED: %s", category, condition, trial, exc)
        return TrialResult(
            category=category, condition=condition, trial=trial, completed=False,
            suspected_technique_id="", correct=False, latency_ms=float("nan"), error=str(exc),
        )


def mcnemar_exact(b: int, c: int) -> Dict[str, Any]:
    """Exact McNemar test (binomial, not chi-square -- appropriate for the
    small discordant-pair counts a live LLM batch like this produces) on
    the paired (rag_correct, no_rag_correct) outcomes. b = rag correct &
    no_rag wrong; c = rag wrong & no_rag correct.
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": float("nan"), "note": "no discordant pairs"}
    p = binomtest(b, n, 0.5).pvalue
    return {"b": b, "c": c, "n_discordant": n, "p_value": float(p), "note": ""}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials-per-category", type=int, default=3)
    parser.add_argument("--categories", nargs="+", default=None, help="Default: all categories_covered()")
    parser.add_argument("--anthropic", action="store_true", help="Use AnthropicLLMClient instead of Gemini")
    parser.add_argument("--model", default=None)
    parser.add_argument("--min-interval-s", type=float, default=13.0, help="Proactive pacing floor between calls")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--base-backoff-s", type=float, default=15.0)
    parser.add_argument("--output", default="h3_results.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s")

    if args.anthropic:
        api_key = os.environ["ANTHROPIC_API_KEY"]
        inner = AnthropicLLMClient(api_key=api_key, **({"model": args.model} if args.model else {}))
    else:
        api_key = os.environ["GEMINI_API_KEY"]
        inner = GeminiLLMClient(api_key=api_key, **({"model": args.model} if args.model else {}))

    shared_client = RetryingLLMClient(
        inner=inner, max_retries=args.max_retries, base_backoff_s=args.base_backoff_s, min_interval_s=args.min_interval_s
    )
    rag_reasoner = Tier2Reasoner(shared_client, use_rag=True)
    no_rag_reasoner = Tier2Reasoner(shared_client, use_rag=False)

    categories = args.categories or categories_covered()
    logger.info("Categories: %s", categories)
    logger.info(
        "Plan: %d categories x 2 conditions x %d trials = %d live LLM calls, min_interval_s=%.1f",
        len(categories), args.trials_per_category, len(categories) * 2 * args.trials_per_category, args.min_interval_s,
    )

    start = time.monotonic()
    results: List[TrialResult] = []
    for category in categories:
        for trial in range(args.trials_per_category):
            results.append(_run_one(rag_reasoner, category, "rag", trial))
            results.append(_run_one(no_rag_reasoner, category, "no_rag", trial))
    elapsed_s = time.monotonic() - start

    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info("Wrote %d raw trial results to %s (elapsed %.0fs)", len(results), args.output, elapsed_s)

    # --- Summary ---
    for condition in ("rag", "no_rag"):
        completed = [r for r in results if r.condition == condition and r.completed]
        correct = [r for r in completed if r.correct]
        n_total = sum(1 for r in results if r.condition == condition)
        logger.info(
            "%s: %d/%d completed, %d/%d completed calls correct (%.1f%%)",
            condition, len(completed), n_total, len(correct), len(completed) or 1,
            100.0 * len(correct) / len(completed) if completed else float("nan"),
        )

    # Paired McNemar on trials where BOTH conditions completed for that (category, trial).
    by_key = {(r.category, r.trial): {} for r in results}
    for r in results:
        by_key[(r.category, r.trial)][r.condition] = r
    b = c = 0
    n_pairs = 0
    for pair in by_key.values():
        if "rag" not in pair or "no_rag" not in pair:
            continue
        if not (pair["rag"].completed and pair["no_rag"].completed):
            continue
        n_pairs += 1
        if pair["rag"].correct and not pair["no_rag"].correct:
            b += 1
        elif not pair["rag"].correct and pair["no_rag"].correct:
            c += 1
    mcnemar = mcnemar_exact(b, c)
    logger.info("Paired trials (both conditions completed): %d", n_pairs)
    logger.info("McNemar exact test: %s", mcnemar)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
