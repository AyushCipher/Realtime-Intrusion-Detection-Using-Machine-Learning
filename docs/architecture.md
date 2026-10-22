# System Architecture & Component Interaction

## 1. End-to-End System Architecture Diagram

```mermaid
flowchart TD
    subgraph INGESTION ["INGESTION MODULE (ingestion/)"]
        Pcap["PCAP Replay / Live Network Interface<br/>(sample_tcp.pcap)"] --> Sniff["Packet Capture Engine<br/>(Scapy / libpcap)"]
        Sniff --> WinFlow["Sliding Window Flow Aggregator<br/>(Bidirectional statistical accumulation)"]
        WinFlow --> PubFlow["Kafka Flow Producer<br/>(ids_ingestion.producer)"]
    end

    PubFlow -->|"Kafka Topic: network.flow.features<br/>(35 statistical features JSON)"| KafkaBus[("Apache Kafka Message Broker<br/>(KRaft Mode, Port 9092)")]

    subgraph ML_DETECTION ["MODULE 1: ML DETECTION ENGINE (ml/ - Lead: Ayush)"]
        KafkaBus --> Consumer["Flow Consumer & Schema Validator<br/>(ids_ml.flow_consumer)"]
        Consumer --> CanonMap["Canonical Vector Normalizer<br/>(features.py: Shape (1, 35))"]
        
        CanonMap --> Stage1["STAGE 1: Unsupervised Anomaly Pre-Filter<br/>(StandardScaler + Isolation Forest)"]
        
        Stage1 -->|"Score < Threshold<br/>(95%+ Benign Traffic)"| FastBenign["Direct Pass-Through<br/>(predicted_class='BENIGN')"]
        
        Stage1 -->|"Score >= Threshold<br/>(Suspicious / Anomaly)"| Stage2["STAGE 2: Supervised Multi-Class Classifier<br/>(XGBoost: 300 Boosted Decision Trees)"]
        
        Stage2 --> OpenMax["Open-Set Recalibration Head<br/>(Weibull Tail Distance from Class MAVs)"]
        
        OpenMax -->|"Computes unknown_mass"| ConfGate{"Conformal Escalation Gate<br/>(Quantile Threshold q_hat = 0.1357)"}
        
        ConfGate -->|"unknown_mass >= q_hat<br/>(Zero-Day / Ambiguous)"| EscTrue["Mark escalated = true<br/>(Unknown / Novel Family)"]
        ConfGate -->|"unknown_mass < q_hat<br/>(Confident Known Attack)"| EscFalse["Mark escalated = false<br/>(Known Category Label)"]
        
        Stage2 --> TreeSHAP["Native TreeSHAP Engine<br/>(XGBoost pred_contribs=True)"]
        TreeSHAP --> Attrib["Top-3 Local Feature Attributions"]
        
        FastBenign & EscTrue & EscFalse --> AlertBuild["Alert Event Constructor<br/>(schema.py: ALERT_EVENT_FIELDS)"]
        Attrib --> AlertBuild
        AlertBuild --> PubAlert["Kafka Alert Producer<br/>(ids_ml.alert_producer)"]
    end

    PubAlert -->|"Kafka Topic: network.ids.alerts<br/>(Structured Alert JSON)"| KafkaBus

    subgraph TIER2_REASONING ["TIER-2 REASONER (tier2_reasoner/)"]
        KafkaBus --> Tier2Filter{"Filter: escalated == true ?"}
        Tier2Filter -->|"Yes"| RAG["TF-IDF Cosine Retrieval<br/>(MITRE ATT&CK Knowledge Base)"]
        RAG --> PromptEngine["Context-Grounded Prompt Engine"]
        PromptEngine --> LLM["Hosted LLM Reasoner<br/>(Anthropic Claude / Gemini API)"]
        LLM --> Tier2Pub["Publish Explanation Event"]
        Tier2Filter -->|"No"| DropTier2["Ignore (Closed-Set Confirmed)"]
    end

    Tier2Pub -->|"Kafka Topic: network.ids.explanations"| KafkaBus

    subgraph DASHBOARD_BACKEND ["DASHBOARD & TRIAGE API (dashboard-api/)"]
        KafkaBus --> DashAlertCons["Alert Consumer"]
        KafkaBus --> DashExpCons["Explanation Consumer"]
        DashAlertCons & DashExpCons --> SQLite[("SQLite Alert Store<br/>(alerts & explanations tables)")]
        SQLite --> FastAPI["FastAPI REST & WebSocket Server<br/>(Port 8000)"]
    end

    subgraph FRONTEND ["ANALYST DASHBOARD (frontend/)"]
        FastAPI -->|"WebSocket Live Stream & REST Queries"| ReactApp["React 18 / Vite / Tailwind UI<br/>(Port 5173)"]
        ReactApp --> LiveFeed["Live Triage Feed"]
        ReactApp --> ShapView["TreeSHAP Feature Attribution Bar Charts"]
        ReactApp --> Tier2View["Tier-2 LLM MITRE Reasoning Panel"]
    end
```

---

## 2. Two-Minute Oral Architecture Pitch (Script for Sir)

> "Good morning, Sir. I am Ayush, representing Group 1, and I am responsible for **Module 1: The Machine Learning Detection Architecture and Complete Workflow**.
>
> "Our project addresses a fundamental vulnerability in classical network intrusion detection: classical models are closed-set, meaning when an unseen or zero-day attack hits the network, the model blindly forces it into a wrong category with false confidence.
>
> "To solve this in real time, we architected an asynchronous, event-driven pipeline running across four modular services connected via Apache Kafka:
>
> 1. **Data Ingestion:** Captures raw network traffic, aggregates packets into bidirectional flows, and computes 35 canonical statistical features (duration, packet rates, IAT statistics, and TCP flags).
> 2. **Stage 1 (Anomaly Pre-Filter):** Uses an unsupervised Isolation Forest. Because 99% of live network traffic is benign, Stage 1 drops normal traffic immediately, bypassing heavy computation and preserving microsecond latency.
> 3. **Stage 2 (Attack Classifier):** For flows flagged by Stage 1, a 300-tree XGBoost classifier predicts the specific attack family (such as DoS, PortScan, Brute Force, or Botnet).
> 4. **Open-Set Recalibration & Conformal Gate:** Instead of trusting closed-set softmax probabilities, we extract raw XGBoost margins and recalibrate them through an OpenMax Weibull-tail head. If the computed `unknown_mass` exceeds our mathematically proven Conformal Prediction threshold, the alert is automatically flagged as `escalated = true`.
> 5. **Local Explainability:** Every alert carries native TreeSHAP feature attributions, explaining the exact packet metrics driving the detection.
> 6. **Tier-2 LLM Reasoner & Triage UI:** Escalated zero-day alerts are consumed by our Tier-2 reasoner, which performs RAG retrieval against MITRE ATT&CK techniques and provides actionable natural-language guidance to the SOC analyst on our React dashboard.
>
> "Every component is containerized in Docker Compose, and all test benchmarks are empirically verified with zero data leakage. I am ready to demonstrate the code, the saved artifacts, and the experimental results."

---

## 3. Current Implementation vs. Intended Future Architecture

| Dimension | Current Verified Implementation | Intended Future Production Architecture |
|---|---|---|
| **Packet Capture** | Replaying verified PCAP files (`sample_tcp.pcap`) into Kafka | Live promiscuous packet capture using kernel-bypass (DPDK / eBPF / AF_XDP) |
| **Kafka Clustering** | Single-broker KRaft Kafka container | Multi-broker distributed Kafka cluster with partition replication factor $\ge 3$ |
| **Tier-1 Batching** | Single-flow sequential scoring (53.8 flows/s) | Micro-batched vector scoring (achieving >7,000 flows/s as benchmarked) |
| **Escalation Reasoner** | Hosted LLM API (Anthropic Claude / Gemini) with TF-IDF RAG | Local quantized LLM (e.g., Llama-3-8B-Instruct via vLLM) with vector DB (Qdrant) |
| **Online Adaptation** | Opt-in Adaptive Conformal Inference (ACI) with triage audit | Closed-loop automated feedback updating $\alpha_t$ directly from verified analyst triage |
| **Storage Engine** | SQLite database with auto-migrating schema | Distributed PostgreSQL / TimescaleDB with partitioned time-series tables |
