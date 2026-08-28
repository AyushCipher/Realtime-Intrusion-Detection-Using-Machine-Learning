# ML Detection Layer (`ids_ml`)

The ML detection layer for a real-time network intrusion detection system.
It consumes flow-feature events from the ingestion module's Kafka topic,
scores them with a two-stage detector (Isolation Forest pre-filter +
XGBoost attack-type classifier), attaches a SHAP explanation, and publishes
alerts to a downstream topic for the dashboard/API module. This package
does not implement ingestion, a dashboard, or an API -- see
[Out of scope](#out-of-scope).

`ids_ml` has no code dependency on the ingestion module (`ids_ingestion`):
the only coupling between the two is the documented Kafka topic contract in
`schema.py`, duplicated from the ingestion module's schema so each module
stays independently buildable and testable (mirrored by `ids_ingestion`'s
own `consumer_contract.py` stub on its side).

## Architecture

```
Kafka: network.flow.features        Two-stage detector          Kafka: network.ids.alerts
(from ids_ingestion)          --->   Stage 1: IsolationForest    --->  (for dashboard/API)
                                      (cheap anomaly pre-filter)
                                     Stage 2: XGBoost                  + SHAP (TreeSHAP)
                                      (attack-type classifier,           explanation
                                       runs only on stage-1-flagged
                                       flows)
```

- `schema.py` -- the input (flow-feature) and output (alert) topic contracts.
- `features.py` -- the canonical numeric feature vector shared by training and live inference.
- `data.py` -- CICIDS2017/CIC-IDS2018 CSV loading and column mapping.
- `split.py` -- time-based train/val/test splitting (and the random-split comparison used to quantify leakage), plus leave-one-family-out zero-day splitting (see [Open-set escalation](#open-set-escalation-experimental) below).
- `stage1_iforest.py` / `stage2_xgboost.py` -- the two detection stages.
- `openset_head.py` / `softmax_gate.py` / `conformal_gate.py` / `adaptive_conformal_gate.py` -- open-set escalation: the OpenMax recalibration head, its closed-set softmax baseline, static budget-based threshold calibration, and its online-adaptive (drift-robust) counterpart.
- `pipeline.py` -- the deployed stage1 -> stage2 cascade, the optional three-way open-set router, and alert-building/severity logic.
- `evaluation.py` -- per-category metrics, split-leakage comparison, concept drift, low-and-slow robustness probe.
- `explainability.py` -- TreeSHAP per-prediction feature contributions.
- `sequence_model.py` -- optional per-source-IP sequence model for multi-stage attacks.
- `flow_consumer.py` / `alert_producer.py` / `scoring_service.py` -- the live Kafka-facing pipeline.
- `train.py` / `serve.py` -- CLIs.

## Installation

```
pip install -r requirements-ml.txt
```

## Kafka contracts

### Input: `network.flow.features` (published by the ingestion module)

Full field list in `schema.FLOW_EVENT_FIELDS`; see the ingestion module's
own README for the authoritative description. This module validates every
consumed event against that schema (`schema.validate_flow_event`) and maps
it onto `features.CANONICAL_FEATURE_COLUMNS` -- 34 numeric fields (duration,
packet/byte counts, packet-length and inter-arrival-time statistics, TCP
flag counts) -- before scoring.

### Output: `network.ids.alerts` (published by this module)

| Field | Type | Notes |
|---|---|---|
| `alert_id` | string | UUID, generated per alert |
| `flow_id`, `src_ip`, `src_port`, `dst_ip`, `dst_port`, `protocol`, `flow_start_time` | -- | Carried through from the source flow event |
| `scored_at` | number | Unix epoch seconds |
| `stage1_anomaly_score` | number | Higher = more anomalous |
| `stage1_flagged` | boolean | Whether stage 1 flagged the flow |
| `stage2_predicted_class` | string | `BENIGN` if stage 1 didn't flag it (stage 2 never ran), else the predicted attack category |
| `stage2_confidence` | number | Max class probability from stage 2; `1.0` when stage 2 didn't run |
| `stage2_class_probabilities` | object | Full per-class probability map; `{}` when stage 2 didn't run |
| `severity` | string | `info`/`low`/`medium`/`high`/`critical`; see `pipeline.severity_for` |
| `explanation` | array | Top-k `{feature, value, shap_value}` TreeSHAP contributions |
| `model_version`, `schema_version` | -- | |

By default, only alerts where `stage2_predicted_class != "BENIGN"` are
published -- scoring every benign flow would flood the topic with nothing
for the dashboard to act on. Pass `--alert-on-stage1-flag-only` to
`serve.py` to also publish stage-1 false positives (stage 1 flagged, stage 2
resolved it back to BENIGN), which is useful for monitoring the pre-filter's
own false-positive rate from the dashboard side, at the cost of a noisier
topic.

## Dataset

Train on **CICIDS2017** or **CSE-CIC-IDS2018** (both from the Canadian
Institute for Cybersecurity, https://www.unb.ca/cic/datasets/), not
NSL-KDD/KDD99 -- both older sets are stale and have documented label-leakage
issues. Download the per-day labeled flow CSVs and point `train.py` at them:

```
python -m ids_ml.train --data path/to/Monday.csv path/to/Tuesday.csv ... --model-dir models
```

`data.py` expects the standard CICFlowMeter columns (`Flow Duration`, `Total
Fwd Packets`, `SYN Flag Count`, `Timestamp`, `Label`, ...) and tolerates the
dataset's well-known inconsistent leading-whitespace column names.

**This repository does not ship the real dataset** (multi-gigabyte,
separately licensed). It ships a small *synthetic* fixture
(`tests/fixtures/synthetic_cicids_sample.csv`, built by
`tests/generate_ml_fixtures.py`) shaped like a CICIDS2017 CSV, used only to
exercise this module's code paths in tests and in the example run below.
**Every number in this README is from that synthetic fixture, not real
CICIDS2017/2018 data** -- see [Known limitations](#known-limitations) for
exactly what that does and doesn't tell you.

## Example training run (synthetic fixture -- see caveat above)

```
python -m ids_ml.train --data tests/fixtures/synthetic_cicids_sample.csv \
    --model-dir models --stage1-contamination 0.35
```

Stage 2 (XGBoost) alone, evaluated on a time-based test split:

| category | precision | recall | f1 | support |
|---|---|---|---|---|
| BENIGN | 1.00 | 1.00 | 1.00 | 116 |
| DoS/DDoS | 1.00 | 1.00 | 1.00 | 36 |
| PortScan | 1.00 | 1.00 | 1.00 | 19 |
| Brute Force | 1.00 | 1.00 | 1.00 | 6 |
| Botnet | 1.00 | 1.00 | 1.00 | 4 |
| Infiltration | 1.00 | 1.00 | 1.00 | 2 |
| Web Attack | 1.00 | 1.00 | 1.00 | 2 |

The full two-stage cascade (stage 2 only runs on what stage 1 flags), same
split:

| category | precision | recall | f1 | support |
|---|---|---|---|---|
| BENIGN | 0.97 | 1.00 | 0.98 | 116 |
| DoS/DDoS | 1.00 | 0.94 | 0.97 | 36 |
| PortScan | 1.00 | 1.00 | 1.00 | 19 |
| Brute Force | 1.00 | 1.00 | 1.00 | 6 |
| Botnet | 1.00 | 1.00 | 1.00 | 4 |
| **Infiltration** | 1.00 | **0.50** | 0.67 | 2 |
| **Web Attack** | 1.00 | **0.50** | 0.67 | 2 |

This is the honest headline result even on an easy synthetic set: **stage 2
is a near-perfect classifier once it sees a flow, but the cascade's overall
recall on the rarest categories is capped by stage 1's recall.** Infiltration
and Web Attack -- this dataset's closest analogues to NSL-KDD's hard,
low-support R2L/U2R classes (CICIDS2017/2018 don't use that terminology; see
`data.ATTACK_CATEGORY_MAP`) -- are exactly where that cost shows up, same as
it would on real data. DoS is easy; rare, subtle categories are hard, and
the per-category breakdown is what makes that visible instead of averaging
it away into one accuracy number.

**Stage 1's flag threshold matters a lot and needs tuning per deployment.**
The result above uses `--stage1-contamination 0.35`, matched to this small
synthetic set's ~35% attack prevalence. With the default `contamination="auto"`,
stage 1 flagged only 14.5% of test-set attacks and the cascade's recall on
every rare category collapsed to 0%, even though stage 2 alone was still
perfect on those same flows. **Real traffic's attack prevalence is nowhere
near 35%** (usually well under 1%), so this specific contamination value is
not a deployment recommendation -- it illustrates that the threshold has to
be chosen for the target traffic mix, not left at a library default.

### Split-leakage check

```
Split-leakage check: time-based macro-F1=1.0000, random-split macro-F1=1.0000, inflation=0.0000
```

On this synthetic fixture, `evaluation.leakage_comparison_report` shows no
inflation from a random split -- but that's a property of the fixture, not
evidence that random splits are safe. Real CICIDS2017 attacks occur in tight
bursts of near-duplicate flows; a random split scatters those duplicates
across train and test, which is exactly the leakage mechanism the CRITICAL
requirement in this module's spec is about. This synthetic fixture jitters
every row independently (see `tests/generate_ml_fixtures.py`) rather than
generating correlated bursts, so it doesn't reproduce that effect. The
mechanism (`split.time_based_split` vs `split.random_split`, compared by
`evaluation.leakage_comparison_report`) is implemented and tested for
correctness; **the deployed model's reported metrics must come from the
time-based split, on the real dataset, every time** -- never from
`random_split`, which exists in this codebase only to run this comparison.

### Concept drift

`evaluation.concept_drift_report` trains on the chronological first 80% of
one time window and compares per-category F1 on (1) that window's own
held-out tail ("in-window") vs (2) an entirely later window. On the
synthetic fixture (day 1 vs. day 2, where day 2's generator deliberately
shifts each attack category's feature distribution -- see
`DRIFT_FACTOR` in `tests/generate_ml_fixtures.py`):

| category | f1 in-window | f1 later window | degradation |
|---|---|---|---|
| Infiltration | 1.00 | 0.67 | 0.33 |
| Botnet | 1.00 | 0.96 | 0.04 |
| PortScan | 1.00 | 0.99 | 0.01 |
| Brute Force | 1.00 | 1.00 | 0.00 |
| DoS/DDoS | 1.00 | 1.00 | 0.00 |
| BENIGN | 0.99 | 0.99 | ~0.00 |
| Web Attack | 0.70 | 0.98 | -0.28 |
| Heartbleed | 0.00 | 0.67 | -0.67 |

Two things worth being honest about here: the caricatured synthetic
categories are separable enough that most of them barely degrade at all
(this understates real drift), and the negative "degradation" rows
(Web Attack, Heartbleed) are noise from tiny support (2 and ~1-2 samples
respectively) -- not evidence the model got better with age. The mechanism
is real and tested; the specific numbers are a fixture artifact.

### Adversarial robustness: does low-and-slow evade detection?

`evaluation.adversarial_robustness_report` takes real attack flows,
produces a "low-and-slow" variant (`evaluation.simulate_low_and_slow`:
identical packet/byte totals, spread over an N x longer duration --
rate-dependent features scaled accordingly, everything else unchanged), and
compares stage 1's flag rate before and after.

On this synthetic fixture, at 8x slowdown, **the low-and-slow variant was
flagged *more* often, not less** -- e.g. with `contamination=0.35`, the flag
rate went from 94.2% to 100.0%. With the low-`contamination="auto"` setting
from the section above, it went from 14.5% to 100.0%. In both cases,
stretching duration and shrinking rate features pushed those values further
outside the range Isolation Forest saw during training (this fixture's
BENIGN flows aren't themselves slow), which made the perturbed flows *more*
anomalous by the model's own scale, not less.

This complicates the simple "attackers slow down to blend in" intuition,
but it is not a robustness guarantee: it means low-and-slow evasion's
success is a function of what the benign traffic's own duration/rate
distribution looks like and where the flag threshold sits, not a fixed
property of the technique. On real traffic, where benign duration/rate
distributions are wide and genuinely slow attacks may well overlap with
them, the outcome could easily go the other way. Treat this probe as a tool
to run against the real, deployed feature distribution -- not as evidence
either way about production robustness. This is a heuristic check (no
gradient-based or query-based adversarial search), not a formal
adversarial-ML evaluation.

## Explainability (SHAP / TreeSHAP)

`explainability.ShapExplainer` returns the top-k features (by absolute
SHAP value) driving each stage-2 prediction, attached to every published
alert as the `explanation` field.

It uses XGBoost's own `pred_contribs=True` prediction mode rather than the
`shap` package's `TreeExplainer`. Both compute the same TreeSHAP algorithm
(Lundberg & Lee, 2017) for tree ensembles, but at the versions available
while building this module (xgboost 3.x, which serializes a per-class
`base_score` for multiclass models), `shap`'s external model parser fails:

```
ValueError: could not convert string to float: '[2.9169626E0,-4.0982914E-1,...]'
```

This is a known version-compatibility gap in `shap`'s XGBoost model loader,
not a bug in this module's models. XGBoost's native path produces identical
Shapley values without depending on an external parser matching XGBoost's
exact model serialization format, so it's used here directly (see
`explainability.py`'s module docstring for the full explanation).

## Open-set escalation (experimental)

The deployed cascade is closed-set: stage 2 always picks the best-fitting
known class, even for traffic that doesn't resemble anything it trained
on. `pipeline.TwoStageDetector` optionally takes a `gate` that recalibrates
stage-1-flagged rows into a three-way decision instead:

- **known-benign** -- stage 2 (recalibrated) says BENIGN.
- **known-attack** -- stage 2 (recalibrated) confidently predicts a known family.
- **escalated** -- the gate's `unknown_mass` exceeds a calibrated threshold; this
  is the fraction of flagged traffic a downstream tier (not yet built --
  see below) would receive.

Two interchangeable gates, both producing the same
`(predicted_class, known_class_probabilities, unknown_mass)` shape:

- `softmax_gate.SoftmaxGate` -- the baseline. `unknown_mass = 1 -
  stage2_confidence`; this is the pipeline's original behavior, unchanged,
  just exposed behind the gate interface so it's a literal point of
  comparison rather than assumed better-or-worse.
- `openset_head.OpenMaxHead` -- OpenMax (Bendale & Boult, 2016) adapted to
  XGBoost's per-class margins instead of a deep network's logits: fits a
  Weibull tail per class over correctly-classified training examples'
  distance from that class's mean margin vector, then shrinks the
  top-ranked classes' scores by how far outside that tail a query lands,
  redistributing the shrunk mass into `unknown_mass`. See the module
  docstring for the full algorithm and exactly what this adaptation does
  and doesn't inherit from the original paper.

The escalation threshold itself isn't a fixed number: `conformal_gate.
calibrate_threshold` sets it from a calibration set of known traffic via a
split-conformal quantile, against an explicit operator budget (e.g.
"escalate at most 10% of known flows") -- see that module's docstring for
the guarantee this borrows from conformal prediction.

### H1 evaluation: does OpenMax actually beat the softmax baseline?

`evaluation.run_openset_trial` runs one leave-one-family-out fold at one
seed for both gates head-to-head: stage 2 is fit on 75% of the known-family
training rows, the escalation threshold is calibrated (`conformal_gate.
calibrate_threshold`, budget=0.1) on a genuinely separate held-out 25% of
*known* rows (never the same rows stage 2 trained on -- see the function's
docstring for why that distinction matters), and both `unknown_recall`
(escalation rate on the held-out family) and `unknown_auroc` (unknown_mass
as a novelty score, independent of any threshold) are measured on the test
split. `evaluation.openset_vs_softmax_report` repeats this across all 7
non-benign families x 5 seeds; `openset_vs_softmax_significance` runs a
paired Wilcoxon signed-rank test on top of that table.

Result on this synthetic fixture, budget=0.1, seeds 0-4, mean over 5 seeds
per family:

| family | openmax recall | softmax recall | openmax AUROC | softmax AUROC |
|---|---|---|---|---|
| Botnet | 0.993 | 0.900 | 0.995 | 0.972 |
| Brute Force | 1.000 | 1.000 | 0.926 | 0.997 |
| DoS/DDoS | 1.000 | 0.200 | 0.983 | 0.873 |
| Heartbleed | 1.000 | 1.000 | 0.975 | 0.975 |
| Infiltration | 0.080 | 0.000 | 0.465 | 0.493 |
| PortScan | 1.000 | 1.000 | 0.982 | 0.968 |
| Web Attack | 0.800 | 1.000 | 0.934 | 0.994 |

Paired across all 35 (family, seed) pairs: unknown-recall mean 0.839
(OpenMax) vs 0.729 (softmax), Wilcoxon p=0.084 -- suggestive but **not
significant at p<0.05**. Unknown-AUROC is essentially tied (0.894 vs
0.896, p=0.61): the two scores separate known from unknown traffic about
equally well overall, but OpenMax's shape lets more of that separation
land on the right side of a fixed 10%-budget threshold on some families
(most visibly DoS/DDoS) while doing the opposite on at least one (Web
Attack) -- a genuinely mixed result, not a clean win for H1.

**Read this cautiously, for three concrete reasons, not just the usual
"it's synthetic" disclaimer:**

1. **n=35 pairs overstates the independence here.** The leave-one-family-out
   design has exactly 7 independent units (families); the 5 seeds per
   family are correlated resamples (same family, same fixture, only the
   fit/calibration split and XGBoost's own randomness differ), not 5
   independent replications of the underlying comparison. Treat the
   p-value as closer to "n≈7" statistical power than "n=35".
2. **Infiltration is a near-total failure for both gates** (support of 2-5
   correctly-classified training examples per fold -- see `openset_head.
   py`'s `_MIN_EXAMPLES_FOR_TAIL` guard, which is exactly why OpenMax has
   no tail to work with here) -- consistent with this dataset's established
   pattern (see the per-category tables earlier in this README) that rare,
   low-support categories are where every method in this repo struggles,
   open-set escalation included.
3. **Caricatured, well-separated synthetic categories** are not a
   meaningful testbed for an open-set claim either way -- CICIDS2017/2018's
   real families sit far closer together in feature space than this
   fixture's deliberately spread-apart ones do, in both directions: a real
   held-out family might be *easier* to catch as unknown (genuinely novel
   behavior, not just a different label on similar traffic) or *harder*
   (overlapping enough with known families that OpenMax's distance signal
   doesn't separate it at all).

### H1 on real data: CSE-CIC-IDS2018

The synthetic result above was rerun against real network traffic: 8 days
of CSE-CIC-IDS2018 (AWS Open Data release, https://registry.opendata.aws/cse-cic-ids2018/
-- publicly downloadable, no registration wall), covering Brute Force, DoS,
DDoS, Web Attack, Infiltration, and Botnet. `ids_ml.data.load_and_map_2018`
handles this release's different column naming (CICFlowMeter-V3) and two
verified data-quality issues (leaked-header rows, and timestamps that are
day-first -- getting that wrong silently corrupts every time-based split;
see that function's docstring and `scripts/build_cicids2018_subsample.py`,
which builds a capped, documented 333,648-row subsample from the full
~7.5M rows for tractable per-trial refitting). To reproduce: download the
8 days listed in that script's docstring from the AWS bucket, then

```
python -m scripts.build_cicids2018_subsample --input-dir <downloaded CSVs> --output real_cicids2018_subsample.csv
python -m scripts.run_real_data_experiments --data real_cicids2018_subsample.csv
```

Result, same protocol as above (budget=0.1, 5 seeds), mean over 5 seeds per family:

| family | openmax recall | softmax recall | openmax AUROC | softmax AUROC |
|---|---|---|---|---|
| Botnet | 0.037 | 0.000 | 0.838 | 0.821 |
| Brute Force | 0.087 | 0.061 | 0.628 | 0.618 |
| DoS/DDoS | 0.465 | 0.310 | 0.437 | 0.361 |
| Infiltration | 0.070 | 0.074 | 0.510 | 0.531 |
| Web Attack | 0.169 | 0.026 | 0.742 | 0.728 |

Paired across all 25 (family, seed) pairs: unknown-recall mean 0.166
(OpenMax) vs 0.094 (softmax), Wilcoxon **p=0.00012**. Unknown-AUROC mean
0.631 vs 0.612, **p=0.032**. Unlike the synthetic result, **this is a
real, significant win for H1** -- OpenMax beats the softmax baseline on 4
of 5 real families (Infiltration is the one exception, essentially tied),
and the significance is far stronger than anything the synthetic fixture
produced (p=0.00012 vs. synthetic's p=0.084).

Two things worth being precise about rather than just reporting the win:

- **Absolute recall is low for both gates** -- even OpenMax's best case
  (DoS/DDoS, 0.465) means more than half of a genuinely novel attack
  family's traffic still slips past a 10%-budget escalation gate
  undetected. This is a much harder, more realistic result than the
  synthetic fixture's near-perfect recalls, and is the honest number to
  report, not the synthetic table.
- **These 25 pairs have the same independence caveat as the synthetic
  run** -- 5 real families, 5 correlated-by-family seeds each, so treat
  the statistical power as closer to n=5 than n=25. The p-value being much
  smaller than the synthetic run's despite the same nominal pair count is
  itself informative (a larger, more consistent effect size across real
  families), not just a statistical-power artifact.

**What's still open:**

- **No downstream tier consumes `escalated` traffic yet.** The router
  produces the label; nothing currently does LLM/RAG reasoning or any
  other second-tier analysis on it.
- **No escalation-rate-vs-unknown-recall tradeoff curve or latency/
  throughput harness yet.**
- **PortScan and Heartbleed weren't in the 8 downloaded CSE-CIC-IDS2018
  days** (they're on other days of that release, not fetched here) --
  the real-data LOFO above covers 5 of the 7 families the synthetic run
  covered, not all of them.
- **Cross-dataset generalization** (e.g. train on CICIDS2018, test the
  LOFO holdout against UNSW-NB15 or CIC-IoT2023) is untested.

## Adaptive conformal calibration under drift (H2)

`conformal_gate.calibrate_threshold` sets the escalation threshold once,
from one held-out batch of known traffic, via a split-conformal quantile.
That guarantee implicitly assumes the deployment distribution stays
exchangeable with the calibration batch -- an assumption this project's
own `evaluation.concept_drift_report` already demonstrates is false here
(day-2 traffic's feature distributions are deliberately shifted from
day-1's in the synthetic fixture, and real network traffic drifts for the
same reason real training data does). Under that shift, a static
threshold's true escalation rate can silently drift away from the
requested budget, with nothing in the system signaling that it happened.

`adaptive_conformal_gate.AdaptiveConformalGate` replaces the one-shot
calibration with Adaptive Conformal Inference (Gibbs & Candès, NeurIPS
2021): the significance level `alpha_t` updates after every scored
known-traffic flow based on whether that flow was escalated, continuously
pulling the realized escalation rate back toward the target budget instead
of trusting a single calibration pass to hold. See the module docstring
for the exact update rule and, importantly, **what was checked against
the literature before building this**: CALIBURN (arXiv 2605.24696) uses
only static Conformal Risk Control and names this exact adaptation as
unimplemented future work; FIRCE/FADES (arXiv 2605.01962; MDPI Electronics
15(10):2114) use periodic recalibration triggered by a drift-detection
signal, a related but distinct mechanism from continuous online updating.
None of the three combine this with open-set/zero-day detection.

### H2 result: does it actually hold the budget better?

`evaluation.static_vs_adaptive_conformal_drift_report` streams both gates
-- a statically-calibrated `ConformalGate` and a fresh
`AdaptiveConformalGate`, both starting from the *same* initial calibration
batch -- over known traffic spanning day 1 (post-calibration, "pre_drift")
and day 2 ("post_drift"), and measures each one's realized escalation rate
against the target budget. `evaluation.static_vs_adaptive_conformal_report`
repeats this across 10 random fit/calibration splits (seeds); `_significance`
runs a paired Wilcoxon test on the absolute error from budget.

Result on this synthetic fixture, budget=0.1, 10 seeds:

| segment | static error (mean) | adaptive error (mean) | Wilcoxon p |
|---|---|---|---|
| pre_drift | 0.0503 | 0.0027 | **0.00195** |
| post_drift | 0.0310 | 0.0016 | **0.00391** |
| pooled | 0.0407 | 0.0022 | **0.0001** |

The adaptive gate's error is roughly **18-19x smaller** than the static
gate's, and the difference is significant well past p<0.01 in both
segments individually -- a much stronger result than H1's. Worth being
precise about *why*, rather than just calling it a drift-adaptation win:
the static gate is already noticeably off-target even in `pre_drift`
(same distribution as its own calibration batch), because this fixture's
tiny calibration set (order of 100 rows) gives a high-variance one-shot
quantile estimate. The adaptive gate's advantage is real in both segments
because it effectively recalibrates continuously against every flow it
sees, not because it uniquely "detects drift" -- `post_drift`'s static
error being *smaller* than `pre_drift`'s in this run (0.031 vs 0.050) is
itself a reminder that small-sample calibration noise, not drift alone,
is doing a lot of the work here. The honest H2 claim from this experiment
is: **continuous online calibration is far more robust than a one-shot
static calibration, both to ordinary calibration-sample noise and to
concept drift on top of it** -- not narrowly "it detects drift."

### H2 on real data: CSE-CIC-IDS2018

Rerun on the same real subsample H1 uses above, with "drift" now meaning
what it actually should: an early-days window (Feb 14/15/16/21) calibrating
against a later-days window (Feb 22/23, Mar 1/2), 212,720 and 120,928 rows
respectively, 10 seeds:

| segment | static error (mean) | adaptive error (mean) | Wilcoxon p |
|---|---|---|---|
| pre_drift | 0.09932 | 0.00092 | **0.00195** |
| post_drift | 0.09955 | 0.00086 | **0.00195** |
| pooled | 0.09943 | 0.00089 | **1.9e-06** |

This is a starker failure of static calibration than the synthetic run
showed, not a milder one. The static gate's realized escalation rate on
real data collapsed to **~0.05% -- essentially never escalating**, against
a target budget of 10%; the adaptive gate held ~10.08-10.09% throughout,
almost exactly on target, in both segments. The mechanism is the same one
described above (a small one-shot calibration batch on a heavy-tailed
real score distribution can set a threshold high enough that almost
nothing in deployment ever crosses it), just far more pronounced at real
scale than the tiny synthetic fixture's calibration set showed.

**What's still open:**

- **Live deployment needs a ground-truth proxy this experiment doesn't
  need.** The evaluation above calls `AdaptiveConformalGate.update()` only
  on rows *known* (by the labeled dataset) to be in-distribution -- valid
  for testing the calibration mechanism itself, but a live `ScoringService`
  doesn't have ground-truth labels for incoming flows in real time. Wiring
  this into `pipeline.TwoStageDetector`'s live path requires deciding what
  to call `update()` on when the true label isn't available yet (e.g.
  updating on every scored flow under the assumption that the vast
  majority of live traffic is in fact known/benign, similar to how stage
  1's contamination assumption already works) -- not yet done or tested.
- Not yet compared against `openset_head.OpenMaxHead`'s `unknown_mass` as
  the score being calibrated -- this run uses `softmax_gate.SoftmaxGate`
  throughout, to isolate the calibration mechanism from the open-set
  detection question H1 already covers.
- The pre/post drift split here is a single cutoff date on 8 real days,
  not a controlled drift magnitude -- unlike the synthetic fixture's
  `DRIFT_FACTOR`, there's no ground truth for how much the real
  distribution actually shifted, only that it's genuinely different
  traffic, captured on different days.

## Optional: per-source-IP sequence model

`sequence_model.py` targets the multi-stage-attack case (recon -> brute
force -> exfiltration) by windowing each source IP's recent flow history:
`build_sequences` turns every source IP's time-ordered flows into
overlapping windows, labeled by the most recent flow in each window.

The natural model for this is a recurrent one (LSTM/GRU). **PyTorch is not
usable in the environment this module was built in**: `import torch` fails
at DLL load time --

```
OSError: [WinError 4551] An Application Control policy has blocked this file.
Error loading "...\torch\lib\shm.dll" or one of its dependencies.
```

-- which is the host's Application Control (WDAC) security policy blocking
a freshly-downloaded native DLL, not something this module works around.
Rather than drop this optional component, `SequenceAttackModel` implements
the same windowed-history objective with an `MLPClassifier` over flattened
windows: a lightweight, dependency-light stand-in available in this
environment. `build_sequences`'s (windowed samples, next-flow label)
contract is exactly what an LSTM/GRU would consume too, so swapping in a
real recurrent model where PyTorch is actually loadable means swapping
`SequenceAttackModel`'s internals only, not the data pipeline. What a real
recurrent model would add on top: an MLP over a flattened window treats
each window position as an independent input dimension, with no notion of
temporal order beyond concatenation order; an LSTM/GRU would learn temporal
dynamics (e.g. "PortScan followed by Brute Force" as a pattern) directly
instead of relying on the classifier to discover it from flattened,
position-fixed features.

Exercised in `tests/test_ml_sequence_model.py` against
`tests/fixtures/synthetic_sequence_sample.csv`, a fixture built specifically
for this (unlike the main fixture, it gives a handful of "attacker" source
IPs an explicit recon -> brute-force -> DoS/exfiltration campaign, since the
main fixture's near-unique per-row IPs give no per-source-IP history to
window over).

## Serving

```
python -m ids_ml.serve --model-dir models --bootstrap-servers localhost:9092
```

`--use-stub` runs the service against in-memory stubs instead of a real
Kafka broker, for smoke-testing a trained model without standing up Kafka.
The live path (`flow_consumer.py`, `alert_producer.py`) mirrors the
ingestion module's own producer: a bounded queue, retry-with-backoff, a
configurable `drop_oldest`/`block` backpressure policy, and reconnect (tear
down and rebuild the client) on any publish/consume failure.

## Testing

```
pip install -r requirements-ml.txt
PYTHONPATH=src pytest
```

## Known limitations

- **All numbers in this README are from a synthetic fixture, not real
  CICIDS2017/2018 data.** The fixture (`tests/generate_ml_fixtures.py`) is
  shaped like the real dataset (same columns, same whitespace quirks, same
  attack-category vocabulary) but its categories are deliberately far apart
  in feature space and its rows are independently jittered rather than
  temporally correlated. Real performance, real split-leakage inflation,
  and real drift magnitude can only be established by running `train.py`
  against the actual CICIDS2017/CSE-CIC-IDS2018 download.
- **Stage 1's contamination/threshold is a real, per-deployment tuning
  knob**, not a fixed default -- see the example run above, where the
  cascade's rare-category recall went from 0% to 50-100% purely from
  changing `--stage1-contamination`. It must be tuned against the target
  environment's actual (usually <<1%) attack prevalence, not copied from
  this README.
- **PyTorch is unusable in the build environment** (blocked by the host's
  Application Control/WDAC policy at DLL load time), so the "optional"
  LSTM/GRU sequence model is implemented as an MLP-over-flattened-windows
  stand-in instead. See [Optional: per-source-IP sequence
  model](#optional-per-source-ip-sequence-model) for what a real recurrent
  model would add.
- **The `shap` package's TreeExplainer can't load this environment's
  XGBoost models** (a version-incompatibility in `shap`'s model parser, not
  this module). Explanations use XGBoost's own native TreeSHAP
  (`pred_contribs=True`) instead, which computes the same values.
- **CICIDS2017/2018 don't have true NSL-KDD-style R2L/U2R categories.**
  Infiltration and Web Attack are this module's closest analogues (rare,
  subtle, low-support) and are reported as such; don't read "R2L/U2R" claims
  elsewhere as referring to a category this dataset actually has.
- **ece_flag_count/cwr_flag_count**: CICFlowMeter's "CWE Flag Count" column
  (a long-documented typo for CWR) is mapped to `cwr_flag_count`; if a given
  CICIDS2017 redistribution is missing this or the ECE column, those
  features will be absent and `data.map_to_canonical` will raise rather
  than silently zero-filling -- fix the input file/column mapping rather
  than working around it.
- **The low-and-slow robustness probe is a heuristic, not a formal
  adversarial-ML evaluation.** No gradient-based or query-based attack
  search is performed, and its outcome depends on the deployed benign
  traffic's own duration/rate distribution -- see [Adversarial
  robustness](#adversarial-robustness-does-low-and-slow-evade-detection)
  for why the direction of the effect isn't fixed.
- **Backpressure can drop alerts** under sustained broker unavailability,
  same tradeoff as the ingestion module's producer -- see
  `alert_producer.BufferedAlertProducer`'s `drop_oldest`/`block` policies.
- **Open-set escalation's H1 result is significant on real data (CSE-CIC-IDS2018,
  p=0.00012), suggestive-only on synthetic data (p=0.084) -- and even the real
  result has low absolute recall.** See [H1 on real
  data](#h1-on-real-data-cse-cic-ids2018): OpenMax beats the softmax
  baseline's unknown-family recall on 4 of 5 real attack families, but the
  winning recall itself tops out at 0.465 (DoS/DDoS) -- more than half of
  a novel attack family's traffic still gets past a 10%-budget gate
  undetected, on the family OpenMax handles best. Nothing downstream
  consumes the `escalated` decision yet. Don't read the
  `TwoStageDetector(gate=...)` option as "open-set detection is solved
  here" -- it's a real, measured improvement over the closed-set baseline,
  not a solved zero-day detector.
- **Adaptive conformal calibration (H2) is a strong, significant result on
  both synthetic and real data, but isn't wired into the live pipeline.**
  `evaluation.static_vs_adaptive_conformal_report` shows
  `AdaptiveConformalGate` holding its escalation-rate budget far more
  accurately than a static `ConformalGate` -- ~18-19x on the synthetic
  fixture (p<0.004), and on real CSE-CIC-IDS2018 the static gate's
  escalation rate collapsed to ~0.05% (essentially never escalating)
  against a static error ~115x larger than the adaptive gate's
  (p<0.002 in both drift segments, p=1.9e-6 pooled) -- see [Adaptive
  conformal calibration under drift](#adaptive-conformal-calibration-under-drift-h2).
  But the evaluation feeds it ground-truth-known calibration traffic, which
  a live `ScoringService` doesn't have in real time. `TwoStageDetector`
  still only accepts a static `ConformalGate` as `escalation_gate`.

## Out of scope

Ingestion (packet capture, flow extraction, the `network.flow.features`
producer), the dashboard, and the API are not implemented here -- this
module only consumes/publishes via the documented Kafka topic contracts
above.
