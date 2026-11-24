"""H3 on REAL escalated alerts (ml/scripts/export_escalated_alerts.py), not synthetic ones.

Each alert is what the deployed cascade escalated for a flow from an attack family it
was trained without, with real TreeSHAP attributions; its predicted class is therefore
usually wrong. Correctness is judged against the flow's TRUE family's ATT&CK
technique(s) from the knowledge base. Conditions through one paced client:
  rag            retrieval grounding as built (retrieval is keyed on the predicted class)
  no_rag_free    no retrieval, model told to give its best-guess technique
  rag_behaviour  retrieval keyed on the flow's behaviour (Tier2Reasoner
                 retrieval_mode="auto"; see ids_tier2/behaviour.py)
Pick conditions with --conditions; already-completed (alert, condition) pairs in
--output are skipped, so pointing --output at a copy of an earlier run adds only the
new condition and keeps the comparison paired.
The fields `true_family` and `heldout_family` are removed before the alert reaches the
reasoner. Writes a JSONL of calls (resumable) and a .summary.json.

Needs GEMINI_API_KEY in the environment (never on the command line or in a file) for
--provider gemini (Gemini and Gemma models); --provider ollama uses a local Ollama server.

Usage:
    python -m scripts.run_h3_real --alerts C:/nids_data/real_alerts.jsonl --output h3_real.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from scipy.stats import binomtest

from ids_tier2.llm_client import GeminiLLMClient, OllamaLLMClient
from ids_tier2.reasoner import Tier2Reasoner
from ids_tier2.retry import RetryingLLMClient
from scripts.run_h3_v2 import EXPECTED, FREE_PROMPT, _PromptOverride, wilson

logger = logging.getLogger("ids_tier2.scripts.run_h3_real")
HIDDEN = ("true_family", "heldout_family")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--alerts", required=True)
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--provider", choices=["gemini", "ollama"], default="gemini")
    ap.add_argument("--min-interval-s", type=float, default=2.0)
    ap.add_argument("--output", default="h3_real.jsonl")
    ap.add_argument("--conditions", default="rag,no_rag_free", help="comma-separated subset of rag,no_rag_free,rag_behaviour")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    inner = (OllamaLLMClient(model=args.model) if args.provider == "ollama"
             else GeminiLLMClient(model=args.model, api_key=os.environ["GEMINI_API_KEY"]))
    shared = RetryingLLMClient(inner=inner,
                               max_retries=4, base_backoff_s=10.0, min_interval_s=args.min_interval_s)
    all_reasoners = {"rag": Tier2Reasoner(shared, use_rag=True),
                     "no_rag_free": Tier2Reasoner(_PromptOverride(shared, FREE_PROMPT), use_rag=False),
                     "rag_behaviour": Tier2Reasoner(shared, use_rag=True, retrieval_mode="auto")}
    reasoners = {c: all_reasoners[c] for c in args.conditions.split(",") if c}
    alerts = [json.loads(l) for l in open(args.alerts)]
    out = Path(args.output)
    rows = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
    done = {(r["alert_id"], r["condition"]) for r in rows}
    with out.open("a") as f:
        for a in alerts:
            truth = a["true_family"]
            clean = {k: v for k, v in a.items() if k not in HIDDEN}
            for cond, reasoner in reasoners.items():
                if (a["alert_id"], cond) in done:
                    continue
                rec = {"alert_id": a["alert_id"], "true_family": truth, "predicted": a["stage2_predicted_class"],
                       "condition": cond, "completed": False, "technique_id": "", "correct": False, "error": ""}
                try:
                    exp = reasoner.explain(clean)
                    rec.update(completed=True, technique_id=exp.suspected_technique_id,
                               correct=exp.suspected_technique_id in EXPECTED.get(truth, set()))
                except Exception as exc:  # noqa: BLE001 - a failed call is data
                    rec["error"] = str(exc)[:200]
                f.write(json.dumps(rec) + "\n")
                f.flush()
                rows.append(rec)

    present = [c for c in all_reasoners if any(r["condition"] == c for r in rows)]
    summary = {"model": args.model, "n_alerts": len(alerts), "conditions": {}, "per_family": {}}
    for cond in present:
        comp = [r for r in rows if r["condition"] == cond and r["completed"]]
        k = sum(r["correct"] for r in comp)
        summary["conditions"][cond] = {"completed": len(comp), "failed": sum(1 for r in rows if r["condition"] == cond and not r["completed"]),
                                       "correct": k, "accuracy": round(k / len(comp), 3) if comp else None, "wilson95": wilson(k, len(comp))}
    for fam in sorted({r["true_family"] for r in rows}):
        summary["per_family"][fam] = {c: f"{sum(r['correct'] for r in rows if r['true_family'] == fam and r['condition'] == c and r['completed'])}/"
                                         f"{sum(1 for r in rows if r['true_family'] == fam and r['condition'] == c and r['completed'])}" for c in present}
    by = {(r["alert_id"], r["condition"]): r for r in rows}
    for a_cond, b_cond in (("rag", "no_rag_free"), ("rag_behaviour", "no_rag_free"), ("rag_behaviour", "rag")):
        if a_cond not in present or b_cond not in present:
            continue
        b = c = 0
        for (aid, cond), r in by.items():
            o = by.get((aid, b_cond))
            if cond == a_cond and o and r["completed"] and o["completed"]:
                b += int(r["correct"] and not o["correct"])
                c += int(o["correct"] and not r["correct"])
        summary[f"mcnemar_{a_cond}_vs_{b_cond}"] = {f"b_{a_cond}_only": b, f"c_{b_cond}_only": c,
                                                   "p": float(binomtest(b, b + c, 0.5).pvalue) if b + c else None}
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
