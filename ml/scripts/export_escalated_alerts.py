"""Export REAL escalated alerts from the cascade, for the Tier-2 (H3) study.

For each attack family F, the deployed cascade (hybrid Stage 1 with conformal recall
control, XGBoost Stage 2, kNN escalation gate at the escalation budget) is trained
WITHOUT F, then scores held-out known flows and F's flows. Every flow the cascade
escalates becomes an alert exactly as the scoring service would publish it
(pipeline.build_alert, with TreeSHAP attributions from explainability.py), plus two
fields the reasoner never sees: `true_family` (the flow's real label) and
`heldout_family` (the family withheld when training this cascade). Up to
--per-family alerts are sampled per true family.

Because the withheld family is unknown to Stage 2, its alerts usually carry a WRONG
predicted class: this is what an analyst actually receives for a novel attack, and
what the H3 study needs to test.

Usage:
    PYTHONPATH=ml/src python ml/scripts/export_escalated_alerts.py --data real.csv --output alerts.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ids_ml.conformal_gate import ConformalGate, calibrate_threshold  # noqa: E402
from ids_ml.conformal_stage1 import calibrate_stage1_threshold  # noqa: E402
from ids_ml.evaluation import _gate_scores, make_stage1  # noqa: E402
from ids_ml.explainability import ShapExplainer, explanation_to_alert_field  # noqa: E402
from ids_ml.features import CANONICAL_FEATURE_COLUMNS as FEATS  # noqa: E402
from ids_ml.ood_gates import KNNGate  # noqa: E402
from ids_ml.pipeline import TwoStageDetector, build_alert  # noqa: E402
from ids_ml.split import random_split  # noqa: E402
from ids_ml.stage2_xgboost import AttackClassifier, Stage2Config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--per-family", type=int, default=20)
    ap.add_argument("--miss-budget", type=float, default=0.05)
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--max-train", type=int, default=80000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    rng = np.random.default_rng(args.seed)
    out_rows = []
    for fam in sorted(set(df["attack_category"]) - {"BENIGN"}):
        known, unk = df[df["attack_category"] != fam], df[df["attack_category"] == fam]
        tr, va, te = random_split(known, train_frac=0.7, val_frac=0.15, random_state=args.seed)
        tr = tr.sample(min(len(tr), args.max_train), random_state=args.seed)
        s1 = make_stage1("hybrid", tr, FEATS, seed=args.seed)
        va_att = va[va["attack_category"] != "BENIGN"]
        calibrate_stage1_threshold(s1.anomaly_score(va_att[FEATS].to_numpy()), args.miss_budget).apply(s1)
        X_tr, y_tr = tr[FEATS].to_numpy(), tr["attack_category"].tolist()
        s2 = AttackClassifier(Stage2Config(n_estimators=200, random_state=args.seed)).fit(X_tr, y_tr)
        gate = KNNGate(s2, random_state=args.seed).fit(X_tr, y_tr)
        Xva = va[FEATS].to_numpy()
        thr = calibrate_threshold(_gate_scores(gate, Xva[s1.flag(Xva)]), budget=args.budget).threshold
        det = TwoStageDetector(s1, s2, gate=gate, escalation_gate=ConformalGate(thr, args.budget, 0), escalation_trigger_name="knn")
        explainer = ShapExplainer(s2, FEATS)
        pool = pd.concat([te[te["attack_category"] != "BENIGN"], unk])
        pool = pool.sample(min(len(pool), 40000), random_state=args.seed)
        X = pool[FEATS].to_numpy()
        scored = det.score(X)
        esc_idx = [i for i, r in enumerate(scored) if r.escalated]
        esc = pool.iloc[esc_idx]
        for tf in sorted(set(esc["attack_category"])):
            pick = [esc_idx[k] for k in np.where(esc["attack_category"].to_numpy() == tf)[0]]
            pick = list(rng.permutation(pick)[: args.per_family])
            if not pick:
                continue
            contribs = explainer.explain_batch(X[pick], [scored[i].stage2_predicted_class for i in pick], top_k=5)
            for i, c in zip(pick, contribs):
                event = {"flow_id": f"real-{fam}-{i}", "src_ip": "", "dst_ip": "", "src_port": 0, "dst_port": 0, "protocol": 6}
                alert = build_alert(event, scored[i], explanation_to_alert_field(c))
                alert.update(true_family=str(pool["attack_category"].iloc[i]), heldout_family=fam)
                out_rows.append(alert)
        print(f"held out {fam}: escalated {len(esc_idx)} of {len(pool)}; exported by true family "
              f"{esc['attack_category'].value_counts().to_dict()}", flush=True)

    alerts = pd.DataFrame(out_rows)
    # keep at most --per-family alerts per true family across folds, preferring alerts from the
    # fold where that family was withheld (the novel-attack case)
    alerts["novel"] = alerts["true_family"] == alerts["heldout_family"]
    alerts = alerts.sort_values("novel", ascending=False).groupby("true_family", group_keys=False).head(args.per_family)
    with open(args.output, "w") as f:
        for rec in alerts.drop(columns=["novel"]).to_dict(orient="records"):
            f.write(json.dumps(rec, default=float) + "\n")
    print(alerts.groupby(["true_family", "novel"]).size().to_string())
    print("wrong predicted class:", round(float((alerts["stage2_predicted_class"] != alerts["true_family"]).mean()), 3))
    print("wrote", args.output, len(alerts))


if __name__ == "__main__":
    main()
