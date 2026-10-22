# Dataset and Features: Inputs, Representations, and Outputs

## 1. Dataset Analysis

### 1.1 Dataset Provenance & Context
- **Primary Research Benchmark:** **CICIDS2017** and **CSE-CIC-IDS2018** created by the Canadian Institute for Cybersecurity (CIC) and Communications Security Establishment (CSE), University of New Brunswick (UNB).
  - *Source:* [UNB Datasets](https://www.unb.ca/cic/datasets/ids-2017.html) & [AWS Open Data CSE-CIC-IDS2018](https://registry.opendata.aws/cse-cic-ids2018/).
  - *Capture Environment:* Emulated enterprise network consisting of attacking and victim networks running over 5 to 10 days, capturing complete packet payloads (`.pcap`) and extracting bidirectional flow records via **CICFlowMeter**.
- **Shipped Repository Fixture:**
  - File: [`ml/tests/fixtures/synthetic_cicids_sample.csv`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/ml/tests/fixtures/synthetic_cicids_sample.csv).
  - Built by: [`ml/tests/generate_ml_fixtures.py`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/ml/tests/generate_ml_fixtures.py).
  - Purpose: Full-pipeline development, test coverage, and repeatable demonstration without downloading 15+ GB of raw AWS CSV files.
  - Characteristics: Mirrors the exact column names, leading whitespace quirks, microsecond scales, and feature distributions of real CICIDS datasets across two synthetic days with simulated concept drift.

### 1.2 Dataset Dimensions & Class Distribution
- **Total Rows:** 1,224 rows
- **Total Columns:** 38 raw CSV columns (35 feature columns + `Timestamp`, `Source IP`, `Label`)
- **Missing Values & Cleanliness:** 0 null values; extreme infinite rates from zero-duration flows are sanitized and clipped.
- **Class Breakdown:**

| Raw Label | Mapped Family | Count | Percentage | Class Characteristics |
|---|---|---|---|---|
| `BENIGN` | `BENIGN` | 800 | 65.36% | High duration, balanced forward/backward packets, standard TCP flags (ACK, PSH, FIN). |
| `DoS Hulk` | `DoS/DDoS` | 200 | 16.34% | High volume of forward packets, short duration, high SYN flag count, low inter-arrival times. |
| `PortScan` | `PortScan` | 100 | 8.17% | Single-packet forward probes (fwd_pkts=1), minimal bytes, RST/SYN flags set. |
| `FTP-Patator` | `Brute Force` | 60 | 4.90% | Repeated rapid connection attempts, moderate duration, moderate packet counts. |
| `Bot` | `Botnet` | 30 | 2.45% | Long-duration periodic beaconing, low byte rate, high inter-arrival times. |
| `Web Attack - Brute Force` | `Web Attack` | 20 | 1.63% | Asymmetric packet sizes, larger backward response payloads, application-layer HTTP POST. |
| `Infiltration` | `Infiltration` | 10 | 0.82% | Stealthy low-volume traffic disguised among normal user flows; hardest category. |
| `Heartbleed` | `Heartbleed` | 4 | 0.33% | Asymmetric buffer over-read (4 forward packets request, 200 backward response packets, 16 KB response). |
| **Total** | | **1,224** | **100.0%** | **Severe class imbalance** (BENIGN 65% vs Infiltration <1%). |

---

## 2. Input to the Model

### 2.1 The 35 Canonical Feature Columns
Training and inference share an identical feature space defined in [`CANONICAL_FEATURE_COLUMNS`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/ml/src/ids_ml/features.py#L22-L58):

1. **Temporal Features:**
   - `flow_duration` (seconds, converted from raw microseconds).
   - `flow_iat_mean`, `flow_iat_std`, `flow_iat_min`, `flow_iat_max` (Flow Inter-Arrival Times in seconds).
   - `fwd_iat_mean`, `fwd_iat_std`, `fwd_iat_min`, `fwd_iat_max` (Forward Inter-Arrival Times in seconds).
   - `bwd_iat_mean`, `bwd_iat_std`, `bwd_iat_min`, `bwd_iat_max` (Backward Inter-Arrival Times in seconds).
2. **Volumetric & Rate Features:**
   - `total_fwd_packets`, `total_bwd_packets` (Counts of forward and backward packets).
   - `total_fwd_bytes`, `total_bwd_bytes` (Total volume transmitted).
   - `flow_bytes_per_sec`, `flow_packets_per_sec` (Bandwidth and packet throughput).
3. **Packet Length Statistics:**
   - `fwd_packet_length_min`, `fwd_packet_length_max`, `fwd_packet_length_mean`, `fwd_packet_length_std`.
   - `bwd_packet_length_min`, `bwd_packet_length_max`, `bwd_packet_length_mean`, `bwd_packet_length_std`.
4. **TCP Control Flags (Binary / Counts):**
   - `syn_flag_count`, `ack_flag_count`, `fin_flag_count`, `rst_flag_count`, `psh_flag_count`, `urg_flag_count`, `ece_flag_count`, `cwr_flag_count` (Note: `cwr_flag_count` maps to CICFlowMeter's historical typo `CWE Flag Count`).

### 2.2 Raw Record vs Model Input Vector
Here is an actual raw record from the dataset compared with the numerical array consumed by the model:

```json
// Raw CSV Record (Excerpt)
{
  " Flow Duration": 20450,
  " Total Fwd Packets": 1,
  " Total Backward Packets": 1,
  "Total Length of Fwd Packets": 40,
  " Total Length of Bwd Packets": 0,
  "SYN Flag Count": 1,
  "RST Flag Count": 1,
  " Timestamp": "2026-10-15 09:12:04",
  " Label": "PortScan"
}
```

**Preprocessing Transformations Applied:**
1. Column whitespace stripping: `" Flow Duration"` $\to$ `"Flow Duration"`.
2. Microsecond to second conversion: $20,450\,\mu s / 10^6 = 0.02045\,s$.
3. Feature alignment: Extracted in exact order of `CANONICAL_FEATURE_COLUMNS`.
4. Final Input Shape: 2D NumPy array of shape `(1, 35)` of dtype `float64`.

---

## 3. Expected Output & Alert Schema

When the pipeline scores an input vector, it produces a structured alert event adhering to [`ALERT_EVENT_FIELDS`](file:///c:/Users/ayush/OneDrive/Desktop/Urop%20Project/ml/src/ids_ml/schema.py#L40-L75):

```json
{
  "alert_id": "c8a4b64e-2f91-4cf5-9988-8123456789ab",
  "flow_id": "10.1.42.15:53210-192.168.1.1:80/TCP",
  "src_ip": "10.1.42.15",
  "src_port": 53210,
  "dst_ip": "192.168.1.1",
  "dst_port": 80,
  "protocol": "TCP",
  "flow_start_time": 1729000000.0,
  "scored_at": 1729000000.0185,
  "stage1_anomaly_score": 0.4820,
  "stage1_flagged": true,
  "stage2_predicted_class": "PortScan",
  "stage2_confidence": 0.9833,
  "stage2_class_probabilities": {
    "BENIGN": 0.0012,
    "Botnet": 0.0021,
    "Brute Force": 0.0015,
    "DoS/DDoS": 0.0043,
    "Infiltration": 0.0001,
    "PortScan": 0.9833,
    "Web Attack": 0.0075
  },
  "unknown_mass": 0.0076,
  "escalated": false,
  "escalation_trigger": "openset",
  "severity": "medium",
  "explanation": [
    {"feature": "total_fwd_packets", "value": 1.0, "shap_value": 4.54},
    {"feature": "flow_duration", "value": 0.02, "shap_value": 0.00},
    {"feature": "fwd_iat_std", "value": 0.00, "shap_value": 0.00}
  ],
  "model_version": "v1.0.0",
  "schema_version": "1.0.0"
}
```

### 3.1 Interpreting Correct vs. Incorrect vs. Escalated Predictions
1. **Normal Benign Traffic (Filtered):**
   - Stage 1 score: $0.395 < 0.450 \implies \text{flagged} = \text{False}$.
   - Result: `BENIGN`, Stage 2 does not run, saving compute.
2. **Confidently Classified Known Attack:**
   - Stage 1 score: $0.495 \ge 0.450 \implies \text{flagged} = \text{True}$.
   - Stage 2 output: `DoS/DDoS`, confidence = 0.9887, `unknown_mass` = 0.0074.
   - Result: `escalated = False`. Alert published with top TreeSHAP features (`total_fwd_packets=249`, `flow_packets_per_sec=3433`).
3. **Escalated Suspicious / Zero-Day Attack:**
   - Example (Row 10 of `sample_predictions.csv`):
   - Stage 1 flags flow (score = 0.4952).
   - Stage 2 predicts `Brute Force` but with low confidence = 0.2459.
   - OpenMax head detects that feature margins deviate from known class centroids, producing **`unknown_mass` = 0.6965**.
   - Because $0.6965 > \hat{q} = 0.1357$ (Conformal threshold), **`escalated = True`**.
   - Result: Alert is escalated to Tier 2 Reasoner for MITRE ATT&CK RAG analysis!

---

## 4. Distinction Between Dataset Replay and Live Traffic
- **Held-Out Test Set Replay:** Flow vectors extracted from historical recordings where ground truth is known. Used for quantitative metrics (precision, recall, F1, confusion matrix).
- **Live Replay in Demo (`docker compose up`):** `ingestion` service streams packets from `ingestion/tests/fixtures/sample_tcp.pcap` into Kafka. The ML service consumes, scores, and emits live JSON alerts.
- **Honest Caveat:** Live packet sniffing requires root/administrator privileges and Npcap drivers (`libpcap`). In standard Docker containers, raw promiscuous interface access is disabled by default; hence the live demo uses pcap stream replay to guarantee 100% reproducibility.
