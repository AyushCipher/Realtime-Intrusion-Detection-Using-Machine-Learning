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
- `split.py` -- time-based train/val/test splitting (and the random-split comparison used to quantify leakage).
- `stage1_iforest.py` / `stage2_xgboost.py` -- the two detection stages.
- `pipeline.py` -- the deployed stage1 -> stage2 cascade and alert-building/severity logic.
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

## Out of scope

Ingestion (packet capture, flow extraction, the `network.flow.features`
producer), the dashboard, and the API are not implemented here -- this
module only consumes/publishes via the documented Kafka topic contracts
above.
