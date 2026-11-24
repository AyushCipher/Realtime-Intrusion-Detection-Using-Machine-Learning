"""Offline retrieval check on real escalated alerts: does Tier 2's retrieval
put the TRUE family's ATT&CK technique in front of the LLM? No LLM is
called, so this isolates retrieval from the model.

Input: the JSONL written by ml/scripts/export_escalated_alerts.py (each line
an alert as the scoring service publishes it, plus `true_family`, which the
reasoner never sees). A technique counts as correct if its knowledge-base
`category` equals the true family (DoS/DDoS has two such techniques).

Metrics, per retrieval mode (class, behaviour, auto):
  hit@1        -- the top retrieved technique is correct;
  hit@sent     -- a correct technique is among those actually passed to the
                  LLM (top_k=3 after the reasoner's relative-score filter);
  n_sent       -- mean number of techniques passed (more = easier hit@sent);
  empty        -- alerts for which nothing was retrieved.
Random baseline: hit@1 = m/N, hit@3 = 1 - C(N-m,3)/C(N,3) for a family with
m of the N knowledge-base entries.

Usage:
    python tier2_reasoner/scripts/eval_retrieval_real.py --alerts C:/nids_data/real_alerts.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ids_tier2.behaviour import describe_flow  # noqa: E402
from ids_tier2.knowledge_base import KNOWLEDGE_BASE  # noqa: E402
from ids_tier2.llm_client import StubLLMClient  # noqa: E402
from ids_tier2.reasoner import Tier2Reasoner  # noqa: E402

MODES = ("class", "behaviour", "auto")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--alerts", required=True)
    p.add_argument("--out", default=None, help="JSON summary path (default: <alerts>.retrieval.json)")
    args = p.parse_args(argv)

    alerts = [json.loads(line) for line in Path(args.alerts).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_category = defaultdict(set)
    for e in KNOWLEDGE_BASE:
        by_category[e.category].add(e.technique_id)
    n_kb = len(KNOWLEDGE_BASE)

    reasoners = {m: Tier2Reasoner(StubLLMClient(), retrieval_mode=m) for m in MODES}
    rows = []
    for a in alerts:
        truth = by_category.get(a["true_family"], set())
        hidden = {k: v for k, v in a.items() if k not in ("true_family", "heldout_family")}
        row = {"true_family": a["true_family"], "predicted": a.get("stage2_predicted_class"), "description": describe_flow(hidden)}
        for m, r in reasoners.items():
            sent = r.explain(hidden).retrieved_technique_ids
            row[m] = {"sent": sent, "hit1": bool(sent) and sent[0] in truth, "hit_sent": any(t in truth for t in sent)}
        m_true = len(truth)
        row["random_hit1"] = m_true / n_kb
        row["random_hit3"] = 1 - comb(n_kb - m_true, 3) / comb(n_kb, 3)
        rows.append(row)

    def agg(subset):
        out = {"n": len(subset)}
        for m in MODES:
            out[m] = {
                "hit@1": sum(r[m]["hit1"] for r in subset),
                "hit@sent": sum(r[m]["hit_sent"] for r in subset),
                "mean_n_sent": round(sum(len(r[m]["sent"]) for r in subset) / max(len(subset), 1), 2),
                "empty": sum(not r[m]["sent"] for r in subset),
            }
        out["random_expected_hit@1"] = round(sum(r["random_hit1"] for r in subset), 1)
        out["random_expected_hit@3"] = round(sum(r["random_hit3"] for r in subset), 1)
        out["empty_behaviour_description"] = sum(not r["description"] for r in subset)
        return out

    families = sorted({r["true_family"] for r in rows})
    summary = {
        "alerts": args.alerts,
        "n_alerts": len(rows),
        "overall": agg(rows),
        "per_family": {f: agg([r for r in rows if r["true_family"] == f]) for f in families},
        "top_behaviour_phrases_per_family": {
            f: sorted(
                {ph for r in rows if r["true_family"] == f for ph in r["description"].split(", ") if ph},
                key=lambda ph: -sum(ph in r["description"].split(", ") for r in rows if r["true_family"] == f),
            )[:6]
            for f in families
        },
    }
    out = Path(args.out) if args.out else Path(args.alerts + ".retrieval.json")
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("n_alerts", "overall", "per_family")}, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
