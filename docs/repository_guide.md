# Repository Guide: Complete Architecture and Code Walkthrough

This document provides a complete file-by-file map of the repository, explaining how every component functions, how modules interact over Apache Kafka, and how to navigate the codebase during faculty review.

---

## 1. High-Level Repository Structure

```
Urop Project/
├── docker-compose.yml          # Container orchestration: Kafka, Ingestion, ML, Tier2, Dashboard-API, Frontend
├── SCHEMA.md                   # Reconciled, verified JSON schemas for Kafka topics
├── Start.txt                   # Execution quick-start notes and research development checklist
│
├── ingestion/                  # MODULE: Packet Capture & Flow Feature Extraction
│   ├── src/ids_ingestion/      # Live packet sniffer, pcap replayer, flow aggregator
│   └── tests/                  # Unit and fixture tests for packet extraction
│
├── ml/                         # MODULE 1 (AYUSH'S CORE): Two-Stage Detector & ML Pipeline
│   ├── src/ids_ml/             # Production ML detection package
│   │   ├── features.py         # Canonical 35-feature schema definitions
│   │   ├── data.py             # CICIDS2017 & CSE-CIC-IDS2018 CSV parsers & normalizers
│   │   ├── split.py            # Chronological time-based splits & LOFO zero-day generators
│   │   ├── stage1_iforest.py   # Stage 1: Isolation Forest unsupervised anomaly pre-filter
│   │   ├── stage2_xgboost.py   # Stage 2: XGBoost multi-class attack classifier
│   │   ├── openset_head.py     # OpenMax Weibull-tail open-set recalibration head
│   │   ├── softmax_gate.py     # Softmax baseline uncertainty gate (1 - max confidence)
│   │   ├── conformal_gate.py   # Static split-conformal threshold calibration
│   │   ├── adaptive_conformal_gate.py # Online-adaptive conformal inference (Gibbs & Candès ACI)
│   │   ├── pipeline.py         # End-to-end TwoStageDetector cascade & severity logic
│   │   ├── explainability.py   # Native TreeSHAP per-flow feature attribution
│   │   ├── sequence_model.py   # Temporal campaign windowing per source IP
│   │   ├── flow_consumer.py    # Kafka consumer reading 'network.flow.features'
│   │   ├── alert_producer.py   # Kafka producer emitting 'network.ids.alerts'
│   │   ├── scoring_service.py  # Live stream event processing worker
│   │   ├── train.py            # CLI entry point for model training & gate calibration
│   │   └── serve.py            # CLI entry point for live Kafka/stub scoring service
│   ├── scripts/
│   │   ├── evaluate_and_compare.py     # Benchmarks selected model against 4 alternatives
│   │   ├── build_cicids2018_subsample.py # Subsampling tool for 7.5M raw 2018 records
│   │   └── run_real_data_experiments.py # Runs H1 & H2 hypothesis tests on real dataset
│   └── tests/                  # 115 passing pytest unit and integration tests
│       └── fixtures/           # synthetic_cicids_sample.csv (1,224 rows, 38 raw columns)
│
├── tier2_reasoner/             # MODULE: LLM Triage & MITRE ATT&CK Knowledge Base
│   ├── src/ids_tier2/          # RAG retriever (TF-IDF cosine similarity) & LLM reasoner
│   └── scripts/run_h3_ablation.py # Live RAG vs ungrounded McNemar significance ablation
│
├── dashboard-api/              # MODULE: FastAPI Backend & SQLite Alert Store
│   ├── src/ids_dashboard/      # REST API, WebSockets broadcaster, alert triage store
│   └── frontend/               # React / Vite / TypeScript analyst dashboard
│
├── integration/
│   └── test_e2e.py             # End-to-end integration test (pcap -> Kafka -> ML -> API)
├── models/                     # Saved model artifacts (.joblib, config files)
├── results/                    # Measured experimental results (CSV, JSON, PNG, TXT)
└── docs/                       # Comprehensive presentation, viva, and technical documentation
```

---

## 2. Component Interactions & Communication

The system follows an asynchronous, event-driven microservices architecture communicating via **Apache Kafka (KRaft mode)**:

```
[ Ingestion Service ]
       │  Publishes extracted flow features
       ▼
  Topic: 'network.flow.features'
       │
       ▼
[ ML Detection Service (ids_ml) ]
  ├── Step 1: Pre-filter with Stage 1 Isolation Forest
  ├── Step 2: (If anomalous) Classify attack with Stage 2 XGBoost
  ├── Step 3: Compute OpenMax unknown mass & conformal escalation check
  └── Step 4: Compute TreeSHAP top-3 feature explanations
       │
       │  Publishes scored alerts
       ▼
  Topic: 'network.ids.alerts'
       │
       ├──────────────────────────────────────────┐
       ▼                                          ▼
[ Dashboard API (FastAPI) ]            [ Tier 2 Reasoner ]
  - Stores in SQLite alert DB            - Filters for `escalated == true`
  - Broadcasts via WebSocket             - RAG query over MITRE ATT&CK
  - Serves REST triage API               - Prompts LLM for reasoning
       │                                          │
       │                                          │ Publishes explanation
       ▼                                          ▼
[ React Frontend UI ]                  Topic: 'network.ids.explanations'
  (Triage, Feed, SHAP, Tier 2)                     │
       ▲                                          │
       └──────────────────────────────────────────┘
```

---

## 3. File-by-File Technical Guide (`ml/src/ids_ml`)

### 3.1 Data Loading & Schema Normalization
- **File:** `features.py`
  - **Purpose:** Defines `CANONICAL_FEATURE_COLUMNS` (35 features). Guarantees that whether features originate from raw CSV files or live streaming JSON packets, the numpy input array has identical ordering and shape `(N, 35)`.
- **File:** `data.py`
  - **Purpose:** Loads CICIDS2017/2018 CSV files.
  - **Key Functions:**
    - `load_and_map(paths)`: Strips inconsistent leading/trailing whitespace from CSV headers, converts microsecond fields (`_us`) to seconds, maps raw names to canonical names via `CICIDS_COLUMN_MAP`.
    - `attack_category(raw_label)`: Maps specific attack tool signatures (e.g., `DoS Hulk`, `FTP-Patator`, `Bot`) into unified families (`DoS/DDoS`, `Brute Force`, `Botnet`).

### 3.2 Splitting & Leakage Prevention
- **File:** `split.py`
  - **Purpose:** Implements rigorous data partitioning to prevent temporal and cross-record data leakage.
  - **Key Functions:**
    - `time_based_split(df, train_frac=0.7, val_frac=0.15)`: Sorts chronologically by timestamp. Train occupies the first 70%, validation the next 15%, and test the final 15%. **No future packets leak into training.**
    - `leave_one_family_out(df, holdout_family)`: Withholds an entire attack category completely from training to simulate zero-day attacks.
    - `random_split(df)`: Implemented strictly as an experimental control to quantify leakage inflation against time-based splits.

### 3.3 Two-Stage Detection Engine
- **File:** `stage1_iforest.py`
  - **Class:** `AnomalyPreFilter`
  - **Algorithm:** Scikit-Learn `IsolationForest` wrapped with `StandardScaler`.
  - **Logic:** Unsupervised anomaly detection. Normalizes features, computes path length scores, and returns an intuitive inverted score where higher = more anomalous (`-model.score_samples()`). Flags flows exceeding `flag_threshold` or default contamination boundary.
- **File:** `stage2_xgboost.py`
  - **Class:** `AttackClassifier`
  - **Algorithm:** XGBoost (`xgb.XGBClassifier`) with multi-class log-loss objective.
  - **Logic:** Fitted on labeled training flows using 300 gradient-boosted trees (max depth 6, learning rate 0.1). Returns predicted attack labels (`predict`) and class probability distributions (`predict_proba`).

### 3.4 Open-Set Recognition & Conformal Calibration
- **File:** `openset_head.py`
  - **Class:** `OpenMaxHead`
  - **Logic:** Extracts XGBoost unnormalized margin vectors (`output_margin=True`). Computes mean activation vectors (MAV) for each known class. Fits Weibull distributions on Euclidean distance tails of correctly classified training instances. During inference, scales margin vectors down and reallocates the reduced mass to an explicit $(K+1)$-th class: `unknown_mass`.
- **File:** `conformal_gate.py`
  - **Class:** `ConformalGate`
  - **Logic:** Computes a distribution-free threshold $\hat{q}$ on validation split unknown mass scores at quantile level $\lceil (n+1)(1 - \text{budget}) \rceil / n$. Guaranteed finite-sample bound on known-traffic escalation rate.
- **File:** `adaptive_conformal_gate.py`
  - **Class:** `AdaptiveConformalGate`
  - **Logic:** Implements Adaptive Conformal Inference (Gibbs & Candès). Dynamically adjusts significance level $\alpha_{t+1} = \alpha_t + \gamma (\text{budget} - \text{escalated}_t)$ using a sliding window to track target escalation budget under concept drift.

### 3.5 Pipeline, Serving, & Explainability
- **File:** `pipeline.py`
  - **Class:** `TwoStageDetector`
  - **Logic:** Chains Stage 1 and Stage 2. If Stage 1 does not flag a flow, it is instantly classified as `BENIGN` (Stage 2 is bypassed). If Stage 1 flags, Stage 2 predicts attack family, and the gate decides whether to mark `escalated: true`.
- **File:** `explainability.py`
  - **Class:** `ShapExplainer`
  - **Logic:** Computes exact TreeSHAP contributions natively using XGBoost's `pred_contribs=True` booster API. Sorts features by absolute Shapley magnitude and attaches the top-$k$ contributors to the alert.
- **File:** `train.py`
  - **Purpose:** CLI for end-to-end training, split-leakage checking, robustness probing, and saving model joblib files.
- **File:** `serve.py`
  - **Purpose:** CLI for launching the live scoring service from Kafka or using in-memory stubs.
