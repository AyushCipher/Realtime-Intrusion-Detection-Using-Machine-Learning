"""Choose a Stage 1 pre-filter by leave-one-family-out (LOFO) validation.

Stage 1 must pass most benign traffic and keep attacks it has never seen, so
candidates are scored on exactly that: for each attack family F, train on all
other families plus benign, then measure attack-vs-benign AUROC on the held-out
family F (with a disjoint benign sample). No candidate is chosen on the
chronological test split. A random-split AUROC (known families) is reported
alongside as the easy case.

Candidates (all cheap enough for per-flow use):
  iforest_benign     Isolation Forest on benign flows only (the original idea)
  xgb_d3             40-tree depth-3 XGBoost, benign vs attack
  xgb_d6             100-tree depth-6 XGBoost
  logreg             logistic regression on signed-log1p, standardised features
  xgb_d3_plus_if     xgb_d3 on features + an Isolation Forest benign-density score
  knn_benign         distance to the k-th nearest benign reference flow (signed-log1p, standardised)
  hyb_sup_if         deployable hybrid: mean percentile of xgb_d3 and iforest_benign (what stage1_hybrid.py ships)
  hyb_sup_knn        mean percentile of xgb_d3 and knn_benign
  hyb3               mean percentile of xgb_d3, iforest_benign and knn_benign
  hyb3_d6            hyb3 with the depth-6 supervised model
  hyb_s6k_w50/70/85  depth-6 supervised percentile and kNN percentile, supervised weight 0.5/0.7/0.85
  max_s6k            max of the depth-6 supervised and kNN percentiles
  efc_benign         reimplemented single-class Energy-based Flow Classifier fitted on benign flows
  max3_s6k_efc       max of the depth-6 supervised, kNN and EFC percentiles
Percentiles are taken against a held-out BENIGN reference sample (20% of training
benign flows, excluded from every model's fit), so every combination is a per-flow
scorer, unlike ranking scores within the test batch.

Writes results/paper/<name>/stage1_selection.csv and a manifest.

Usage:
    PYTHONPATH=ml/src python ml/scripts/stage1_selection.py --data real.csv --name cicids2018_dedup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ids_ml.efc import EFCConfig, EnergyFlowClassifier  # noqa: E402
from ids_ml.features import CANONICAL_FEATURE_COLUMNS as FEATS  # noqa: E402
from ids_ml.manifest import write_manifest  # noqa: E402

CANDIDATES = ["iforest_benign", "xgb_d3", "xgb_d6", "logreg", "xgb_d3_plus_if", "knn_benign", "hyb_sup_if", "hyb_sup_knn", "hyb3", "hyb3_d6", "hyb_s6k_w50", "hyb_s6k_w70", "hyb_s6k_w85", "max_s6k", "efc_benign", "max3_s6k_efc"]


def slog(X: np.ndarray) -> np.ndarray:
    return np.sign(X) * np.log1p(np.abs(X))


def iforest(Xb: np.ndarray, seed: int):
    sc = StandardScaler().fit(slog(Xb))
    m = IsolationForest(n_estimators=100, random_state=seed, n_jobs=4).fit(sc.transform(slog(Xb)))
    return lambda X: -m.score_samples(sc.transform(slog(X)))


def xgb(depth: int, n: int, seed: int):
    return XGBClassifier(n_estimators=n, max_depth=depth, learning_rate=0.2, n_jobs=4, verbosity=0, random_state=seed)


def knn_scorer(Xb: np.ndarray, seed: int, k: int = 10, max_ref: int = 20000):
    rng = np.random.default_rng(seed)
    ref = Xb[rng.permutation(len(Xb))[:max_ref]]
    sc = StandardScaler().fit(slog(ref))
    nn = NearestNeighbors(n_neighbors=k, n_jobs=4).fit(sc.transform(slog(ref)))
    return lambda X: nn.kneighbors(sc.transform(slog(X)))[0][:, -1]


def ecdf(ref_sorted: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.searchsorted(ref_sorted, v, side="right") / len(ref_sorted)


def score_all(Xtr, ytr, Xte, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ben = rng.permutation(np.where(ytr == 0)[0])
    n_ref = max(int(len(ben) * 0.2), 20)
    ref_idx, fit_ben = ben[:n_ref], ben[n_ref:]
    fit_idx = np.concatenate([fit_ben, np.where(ytr == 1)[0]])
    Xref = Xtr[ref_idx]
    out = {}
    ifs = iforest(Xtr[fit_ben], seed)
    knn = knn_scorer(Xtr[fit_ben], seed)
    m3 = xgb(3, 40, seed).fit(Xtr[fit_idx], ytr[fit_idx])
    m6 = xgb(6, 100, seed).fit(Xtr[fit_idx], ytr[fit_idx])
    raw = {"if": ifs, "knn": knn, "s3": lambda X: m3.predict_proba(X)[:, 1], "s6": lambda X: m6.predict_proba(X)[:, 1]}
    refs = {k: np.sort(f(Xref)) for k, f in raw.items()}
    pct = {k: ecdf(refs[k], raw[k](Xte)) for k in raw}
    out["iforest_benign"] = pct["if"]
    out["knn_benign"] = pct["knn"]
    out["xgb_d3"] = pct["s3"]
    out["xgb_d6"] = pct["s6"]
    sc = StandardScaler().fit(slog(Xtr[fit_idx]))
    out["logreg"] = LogisticRegression(max_iter=1000, n_jobs=4).fit(sc.transform(slog(Xtr[fit_idx])), ytr[fit_idx]).predict_proba(sc.transform(slog(Xte)))[:, 1]
    aug = lambda X, s_: np.column_stack([X, s_])  # noqa: E731
    out["xgb_d3_plus_if"] = xgb(3, 40, seed).fit(aug(Xtr[fit_idx], ifs(Xtr[fit_idx])), ytr[fit_idx]).predict_proba(aug(Xte, ifs(Xte)))[:, 1]
    out["hyb_sup_if"] = (pct["s3"] + pct["if"]) / 2
    out["hyb_sup_knn"] = (pct["s3"] + pct["knn"]) / 2
    out["hyb3"] = (pct["s3"] + pct["if"] + pct["knn"]) / 3
    out["hyb3_d6"] = (pct["s6"] + pct["if"] + pct["knn"]) / 3
    for w in (50, 70, 85):  # depth-6 supervised score blended with kNN density at supervised weight w%
        out[f"hyb_s6k_w{w}"] = (w / 100) * pct["s6"] + (1 - w / 100) * pct["knn"]
    out["max_s6k"] = np.maximum(pct["s6"], pct["knn"])  # flag if EITHER view finds the flow unusual
    # Reimplemented single-class Energy-based Flow Classifier (efc.py) fitted on benign flows: a
    # published open-set NIDS method run on the same features and splits.
    efc = EnergyFlowClassifier(EFCConfig(random_state=seed)).fit(Xtr[fit_ben])
    out["efc_benign"] = efc.energies(Xte).ravel()
    # third view for the max fusion: EFC energy as a percentile against the same benign reference
    pct_efc = ecdf(np.sort(efc.energies(Xref).ravel()), out["efc_benign"])
    out["max3_s6k_efc"] = np.maximum.reduce([pct["s6"], pct["knn"], pct_efc])
    return out


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="dataset")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max-train", type=int, default=120_000)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    fams = sorted(set(df["attack_category"]) - {"BENIGN"})
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        ben_idx = rng.permutation(np.where(df["attack_category"] == "BENIGN")[0])
        half = len(ben_idx) // 2
        ben_train_pool, ben_test_pool = ben_idx[:half], ben_idx[half:]
        for fam in fams:
            fam_idx = np.where(df["attack_category"] == fam)[0]
            other_idx = np.where(~df["attack_category"].isin(["BENIGN", fam]))[0]
            tr_idx = np.concatenate([ben_train_pool, other_idx])
            if len(tr_idx) > args.max_train:
                tr_idx = rng.choice(tr_idx, args.max_train, replace=False)
            te_idx = np.concatenate([fam_idx, ben_test_pool[: max(len(fam_idx) * 3, 2000)]])
            Xtr, ytr = df.iloc[tr_idx][FEATS].to_numpy(), (df.iloc[tr_idx]["attack_category"] != "BENIGN").astype(int).to_numpy()
            Xte, yte = df.iloc[te_idx][FEATS].to_numpy(), (df.iloc[te_idx]["attack_category"] != "BENIGN").astype(int).to_numpy()
            for cand, s in score_all(Xtr, ytr, Xte, seed).items():
                rows.append({"protocol": "LOFO_unseen_family", "heldout": fam, "seed": seed, "candidate": cand, "auroc": float(roc_auc_score(yte, s))})
            print(f"seed {seed} family {fam} done", flush=True)
        # known families: random split
        perm = rng.permutation(len(df))
        cut = int(0.7 * len(df))
        tr, te = df.iloc[perm[:cut]], df.iloc[perm[cut:]]
        tr = tr.sample(min(len(tr), args.max_train), random_state=seed)
        Xtr, ytr = tr[FEATS].to_numpy(), (tr["attack_category"] != "BENIGN").astype(int).to_numpy()
        Xte, yte = te[FEATS].to_numpy(), (te["attack_category"] != "BENIGN").astype(int).to_numpy()
        for cand, s in score_all(Xtr, ytr, Xte, seed).items():
            rows.append({"protocol": "random_known_families", "heldout": "-", "seed": seed, "candidate": cand, "auroc": float(roc_auc_score(yte, s))})

    res = pd.DataFrame(rows)
    out = root / "results" / "paper" / args.name
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "stage1_selection_raw.csv", index=False)
    piv = res.groupby(["protocol", "candidate", "heldout"])["auroc"].mean().round(3).unstack("heldout")
    summ = res.groupby(["protocol", "candidate"])["auroc"].agg(["mean", "std"]).round(3).reset_index()
    summ.to_csv(out / "stage1_selection.csv", index=False)
    write_manifest(out / "manifest_stage1_selection.json", inputs=[args.data], outputs=[out / "stage1_selection.csv"], seeds=range(args.seeds), repo_dir=root)
    print(piv.to_string()); print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
