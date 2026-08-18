# IDS Ingestion & Streaming Layer

Packet/flow ingestion and streaming module for a real-time network intrusion
detection system. This module captures traffic (live or replayed from pcap),
groups packets into bidirectional flows, extracts flow-level features over a
sliding window, and publishes those features as events to Kafka for
downstream consumption (e.g. by an ML scoring module).

This module is self-contained: it does not implement detection models, a
dashboard, or an API. The Kafka topic contract is documented so downstream
consumers can be built and tested independently.

## Status

Under active development. See sections below as they are added.
