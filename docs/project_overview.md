# Project Overview: Real-Time Network Intrusion Detection System (NIDS)

## 1. Project Title & Executive Summary
- **Project Title:** Real-Time Network Intrusion Detection Using a Two-Stage Machine Learning Cascade and Conformal Open-Set Triage
- **Academic Context:** UROP (Undergraduate Research Opportunities Programme) - Group 1
- **Assigned Module:** **Module 1 (Lead: Ayush)** — Machine Learning Architecture, Training, Testing, Evaluation, Model Selection, and End-to-End Pipeline Integration.

### Executive Summary
Traditional network intrusion detection systems (NIDS) are trained on closed sets of known attack classes. In real-world enterprise deployments, this assumption fails because novel zero-day attacks and mutating threat signatures emerge continuously. When presented with an unseen attack family, standard closed-set classifiers silently misclassify the malicious traffic into whichever known category has the highest superficial similarity.

This project implements an end-to-end, real-time intrusion detection pipeline that solves this critical vulnerability through a **Two-Stage Machine Learning Cascade**:
1. **Stage 1 (Unsupervised Anomaly Pre-Filter):** An **Isolation Forest** pre-filter fitted on statistical flow features to filter out overwhelming benign traffic (~99%+ of normal traffic) at minimal computational cost.
2. **Stage 2 (Supervised Multi-Class Classifier):** An **XGBoost (Extreme Gradient Boosting)** classifier that executes only on flows flagged by Stage 1, accurately categorizing attacks into specific families (DoS/DDoS, PortScan, Brute Force, Botnet, Web Attack, Infiltration).
3. **Open-Set Recalibration Gate (OpenMax):** Re-calibrates XGBoost margin vectors via extreme value theory (Weibull distributions) to compute an explicit *unknown mass* score for novel/zero-day attacks.
4. **Conformal Escalation Thresholding:** Governs escalation to human analysts or downstream Tier-2 reasoners using distribution-free split-conformal prediction bounded by an explicit operational budget (e.g., escalate at most 10% of traffic).
5. **Explainability (TreeSHAP):** Computes exact Shapley feature attributions per flagged flow natively using TreeSHAP to provide immediate triage rationale for security analysts.

---

## 2. Problem Statement & Why It Matters
- **The Threat Landscape:** Modern networks face volumetric attacks (DDoS/DoS) designed to exhaust bandwidth, reconnaissance sweeps (Port Scanning) mapped before compromise, credential attacks (Brute Force SSH/FTP), and stealthy application-layer penetrations (Web Attacks, Infiltration).
- **The Core Flaw in Prior Work:** Traditional models (like Naive Bayes, Random Forests, or standard Deep Neural Networks) assume that test traffic belongs exclusively to the training distribution ($P_{test} = P_{train}$). When exposed to novel attacks, they exhibit overconfident false predictions.
- **The Operational Challenge:** High-speed network backbones process thousands to millions of packets per second. Running complex multi-class models or deep neural networks on 100% of packets causes buffer bloat, packet drops, and unacceptable latency.

### Project Objectives
1. **High Detection Recall:** Detect network attacks with >98% accuracy on known categories.
2. **Computational Efficiency:** Pre-filter benign traffic so heavy classification runs on <5% of flows, keeping median single-flow latency under 20 ms.
3. **Open-Set Zero-Day Awareness:** Quantify prediction uncertainty using OpenMax recalibration and route ambiguous or out-of-distribution flows for Tier-2 expert analysis.
4. **Guaranteed Analyst Budget:** Enforce rigorous conformal prediction bounds so security operations centers (SOC) are not overwhelmed with false escalations.
5. **Local Explainability:** Deliver per-prediction TreeSHAP feature explanations to explain *why* an alert was generated.

---

## 3. ML Task Formulation
- **Task Type:** Hybrid Unsupervised Anomaly Detection (Stage 1) + Supervised Multi-Class Tabular Classification (Stage 2) + Statistical Open-Set Escalation (Conformal Gate).
- **Input:** 35 canonical network flow features extracted from bidirectional packet streams over sliding time windows (duration, packet/byte counts, packet length statistics, inter-arrival time statistics, and TCP flag counts).
- **Target Classes:**
  - `BENIGN`: Normal non-malicious network traffic.
  - `DoS/DDoS`: Denial-of-Service floods (Hulk, GoldenEye, Slowloris, Slowhttptest, LOIC/HOIC).
  - `PortScan`: Reconnaissance probing across destination ports.
  - `Brute Force`: Automated dictionary/credential attacks (FTP-Patator, SSH-Patator).
  - `Botnet`: Command-and-Control (C2) communication channels (ARES bot).
  - `Web Attack`: Application attacks (SQL Injection, Cross-Site Scripting, Brute Force).
  - `Infiltration`: Internal network pivoting and exploitation.
  - `Heartbleed`: TLS heartbeat buffer over-read vulnerability exploitation.
- **Output:** Structured alert event containing `stage1_anomaly_score`, `stage2_predicted_class`, `stage2_confidence`, `unknown_mass`, `escalated` (boolean), `severity` (`info`/`low`/`medium`/`high`/`critical`), and top-3 TreeSHAP feature contributions.

---

## 4. Scope and Limitations
| Aspect | In Scope (Implemented & Verified) | Out of Scope / Future Work |
|---|---|---|
| **Data Processing** | Bidirectional flow statistical feature extraction, handling whitespace quirks, canonical normalization | Deep packet inspection (payload inspection/payload decryption) |
| **Model** | Isolation Forest pre-filter + XGBoost multi-class + OpenMax/Softmax Conformal Gate | Real-time retraining on live network backbones |
| **Explainability** | Native TreeSHAP feature attributions per alert | External perturbation-based explainers (LIME) |
| **Evaluation** | Time-based chronological split, leak-checking random splits, concept drift simulation, low-and-slow evasion probes | Adversarial gradient attacks on physical packet engines |
| **System** | Modular microservices connected via Apache Kafka, Docker Compose, FastAPI, React Dashboard | Hardware FPGA/ASIC acceleration |
| **Tier-2 Escalation**| RAG-grounded MITRE ATT&CK technique mapping via LLM | Unlimited free-tier API quotas for real-time Tier-2 throughput |
