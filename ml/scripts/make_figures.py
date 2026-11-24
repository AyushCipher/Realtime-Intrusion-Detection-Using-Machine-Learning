"""Paper figures from the result CSVs (run after the experiments that write them).

  fig1_recall_vs_budget.png   withheld-family flows escalated end to end vs escalation budget,
                              per gate, CSE-CIC-IDS2018 / CIC-UNSW-NB15 / LYCOS-IDS2017
                              (results/paper/<name>/cascade_openset_budgets.csv)
  fig2_miss_over_time.png     Stage 1 miss rate per window of attacks over the time-ordered
                              CSE-CIC-IDS2018 stream (results/paper/cicids2018_dedup_5s/
                              stage1_stream_hybrid_timeseries.csv)
  fig3_stage1_tradeoff.png    Stage 1 candidates: AUROC on unseen families vs known families
                              (results/paper/<name>/stage1_selection_raw.csv)

Colours: the validated categorical palette (dataviz reference palette, slots 1-6, checked
with validate_palette.js; slots 3-5 are below 3:1 against the surface, so every series also
has its own marker and a direct label).

Usage:
    python ml/scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
RES = ROOT / "results" / "paper"
OUT = RES / "figures"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
MARKERS = ["o", "s", "^", "D", "v", "P"]

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE, "figure.facecolor": "white",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


def fig1() -> None:
    gates = [("knn", "kNN"), ("openset", "OpenMax"), ("softmax", "Softmax"), ("energy", "Energy"), ("efc", "EFC (reimpl.)"), ("mahalanobis", "Mahalanobis")]
    sets = [("cicids2018_dedup", "CSE-CIC-IDS2018"), ("unsw", "CIC-UNSW-NB15"), ("lycos2017", "LYCOS-IDS2017")]
    avail = [(n, t) for n, t in sets if (RES / n / "cascade_openset_budgets.csv").exists()]
    fig, axes = plt.subplots(1, len(avail), figsize=(3.0 * len(avail), 2.8), sharey=True, squeeze=False)
    for ax, (name, title) in zip(axes[0], avail):
        d = pd.read_csv(RES / name / "cascade_openset_budgets.csv")
        for k, (g, label) in enumerate(gates):
            s = d[d["gate"] == g].sort_values("budget")
            if s.empty:
                continue
            ax.plot(s["budget"], s["f_escalated_mean"], color=SLOTS[k], lw=2, marker=MARKERS[k], ms=4.5, label=label)
            ax.fill_between(s["budget"], s["f_escalated_mean"] - s["f_escalated_std"].fillna(0),
                            s["f_escalated_mean"] + s["f_escalated_std"].fillna(0), color=SLOTS[k], alpha=0.12, lw=0)
        ax.set_title(title, color=INK, fontsize=9)
        ax.set_xlabel("escalation budget")
        ax.set_xscale("log")
        ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2])
        ax.set_xticklabels(["1%", "2%", "5%", "10%", "20%"])
        ax.set_ylim(0, 1)
    axes[0][0].set_ylabel("withheld-family flows escalated")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "fig1_recall_vs_budget.png", dpi=220)
    plt.close(fig)


def fig2() -> None:
    p = RES / "cicids2018_dedup_5s" / "stage1_stream_hybrid_timeseries.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d["budget"] == 0.05]
    methods = [("pooled", "static (pooled)"), ("per_family", "per-family, 30% cost cap"), ("adaptive_d0", "adaptive, prompt feedback"), ("adaptive_d1000", "adaptive, 1,000-flow delay")]
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    for k, (m, label) in enumerate(methods):
        s = d[d["method"] == m].sort_values("attack_index")
        ax.plot(s["attack_index"], s["window_miss_rate"], color=SLOTS[k], lw=1.6, marker=MARKERS[k], ms=3, markevery=3, label=label)
    ax.axhline(0.05, color=INK, lw=1, ls="--")
    ax.text(ax.get_xlim()[1], 0.05, " 5% budget", va="center", ha="left", color=INK, fontsize=7.5)
    fam = d[d["method"] == "pooled"].sort_values("attack_index")
    change = fam["family"].ne(fam["family"].shift())
    for _, r in fam[change].iterrows():
        ax.axvline(r["attack_index"], color=MUTED, lw=0.6, ls=":")
        ax.text(r["attack_index"], 1.02, f" {r['family']}", transform=ax.get_xaxis_transform(), fontsize=7, color=MUTED, ha="left")
    ax.set_xlabel("attacks seen in the evaluation stream")
    ax.set_ylabel("miss rate (window of 500 attacks)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_miss_over_time.png", dpi=220)
    plt.close(fig)


# Per-panel label offsets (points) where default placement collides.
OFFSETS = {
    ("unsw", "max_s6k"): (-6, 6), ("unsw", "xgb_d6"): (6, -10), ("unsw", "hyb_s6k_w50"): (6, -12),
    ("unsw", "knn_benign"): (6, 6), ("unsw", "iforest_benign"): (-6, -12),
}


def fig3() -> None:
    sets = [("cicids2018_dedup_5s", "CSE-CIC-IDS2018"), ("unsw", "CIC-UNSW-NB15")]
    cands = [("iforest_benign", "Isolation Forest"), ("xgb_d6", "XGBoost d6"), ("knn_benign", "kNN"),
             ("hyb_s6k_w50", "weighted hybrid"), ("max_s6k", "max-fusion hybrid"), ("efc_benign", "EFC (reimpl.)")]
    avail = [(n, t) for n, t in sets if (RES / n / "stage1_selection_raw.csv").exists()]
    if not avail:
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(3.3 * len(avail), 3.0), squeeze=False)
    for ax, (name, title) in zip(axes[0], avail):
        d = pd.read_csv(RES / name / "stage1_selection_raw.csv")
        per_seed = d.groupby(["protocol", "candidate", "seed"])["auroc"].mean().reset_index()
        g = per_seed.groupby(["protocol", "candidate"])["auroc"].agg(["mean", "std"]).reset_index()
        for k, (c, label) in enumerate(cands):
            u = g[(g["protocol"] == "LOFO_unseen_family") & (g["candidate"] == c)]
            kn = g[(g["protocol"] == "random_known_families") & (g["candidate"] == c)]
            if u.empty or kn.empty:
                continue
            x, y = float(u["mean"].iloc[0]), float(kn["mean"].iloc[0])
            ax.errorbar(x, y, xerr=float(u["std"].iloc[0]), yerr=float(kn["std"].iloc[0]), fmt=MARKERS[k], ms=6,
                        color=SLOTS[k], elinewidth=1, capsize=2)
            off = OFFSETS.get((name, c), (5, -9 if c in ("xgb_d6", "hyb_s6k_w50") else 4))
            ax.annotate(label, (x, y), textcoords="offset points", xytext=off, fontsize=7, color=INK,
                        ha="right" if off[0] < 0 else "left")
        ax.set_title(title, color=INK, fontsize=9)
        ax.set_xlabel("AUROC, family withheld from training")
        ax.set_ylabel("AUROC, known families")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_stage1_tradeoff.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig1()
    fig2()
    fig3()
    print("figures in", OUT, sorted(p.name for p in OUT.glob("*.png")))
