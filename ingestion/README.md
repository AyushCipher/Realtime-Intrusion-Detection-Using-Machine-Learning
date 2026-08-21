# IDS Ingestion & Streaming Layer

Packet/flow ingestion and streaming module for a real-time network intrusion
detection system. This module captures traffic (live or replayed from pcap),
groups packets into bidirectional flows, extracts flow-level features over a
sliding window, and publishes those features as events to Kafka for
downstream consumption (e.g. by an ML scoring module).

This module is self-contained: it does not implement detection models, a
dashboard, or an API. Kafka is only stubbed on the consumer side (see
[Consumer contract](#consumer-contract-for-downstream-modules)) so this
module is independently testable without a live broker or any downstream
service.

## Architecture

```
PacketSource            FlowFeatureExtractor         BufferedProducer
(LiveCapture /   --->   (sliding-window flow    --->  (backpressure +   --->  Kafka topic
 PcapReplay)             tracking + feature            retry/reconnect)       network.flow.features
                         extraction)
```

- `packet_source.py` -- normalizes Scapy packets into a `PacketMeta` dataclass.
- `capture.py` -- live capture via Scapy's `AsyncSniffer`.
- `replay.py` -- pcap replay that sleeps between packets to reproduce the
  original capture's timing (see [Pcap replay timing](#pcap-replay-timing)).
- `flow.py` / `features.py` -- direction-independent flow keys and
  sliding-window feature extraction, computed from raw packets (no
  pre-extracted CICIDS2017 feature columns are loaded).
- `schema.py` -- the event schema contract published to Kafka.
- `producer.py` -- `KafkaFlowProducer` (real) / `StubFlowProducer`
  (in-memory), wrapped by `BufferedProducer` for backpressure and
  reconnection.
- `consumer_contract.py` -- the read-side counterpart: a real Kafka consumer
  reference implementation plus an in-memory stub for integration tests.
- `pipeline.py` / `config.py` / `__main__.py` -- wiring and a CLI entrypoint.

## Installation

```
pip install -r requirements.txt
```

Live capture additionally requires OS-level packet capture support (Npcap on
Windows, libpcap on Linux/macOS) and elevated privileges.

## Usage

Replay a pcap file at realistic speed, publishing to a real Kafka cluster:

```
python -m ids_ingestion --pcap capture.pcap --bootstrap-servers localhost:9092
```

Replay instantly (for local testing) against an in-memory stub instead of
Kafka:

```
python -m ids_ingestion --pcap capture.pcap --replay-speed 0 --use-stub-producer
```

Live capture on an interface:

```
python -m ids_ingestion --interface eth0 --bootstrap-servers localhost:9092
```

Run `python -m ids_ingestion --help` for the full option list (window
timeouts, drop policy, BPF filter, etc.).

## Consumer contract (for downstream modules)

Flow-feature events are published as JSON to the Kafka topic
`network.flow.features` (`schema.DEFAULT_TOPIC`), keyed by `flow_id` so all
records for one flow land on the same partition. The full field list and
types are defined in `schema.FLOW_EVENT_FIELDS` and enforced at publish/consume
time by `schema.validate_event()`.

| Field | Type | Notes |
|---|---|---|
| `flow_id` | string | Direction-independent flow key |
| `src_ip`, `dst_ip` | string | Forward direction = the flow's first observed packet |
| `src_port`, `dst_port` | integer | |
| `protocol` | integer | IANA transport protocol number (6=TCP, 17=UDP) |
| `flow_start_time`, `flow_end_time` | number | Unix epoch seconds |
| `flow_duration` | number | Seconds |
| `close_reason` | string | `fin`, `rst`, `idle_timeout`, `active_timeout`, or `flush` |
| `total_fwd_packets`, `total_bwd_packets` | integer | |
| `total_fwd_bytes`, `total_bwd_bytes` | integer | |
| `fwd_packet_length_{min,max,mean,std}` | number | |
| `bwd_packet_length_{min,max,mean,std}` | number | |
| `flow_bytes_per_sec`, `flow_packets_per_sec` | number | |
| `flow_iat_{mean,std,min,max}` | number | Inter-arrival time across both directions |
| `fwd_iat_{mean,std,min,max}` | number | Inter-arrival time, forward direction only |
| `bwd_iat_{mean,std,min,max}` | number | Inter-arrival time, backward direction only |
| `{syn,ack,fin,rst,psh,urg,ece,cwr}_flag_count` | integer | TCP flag counts observed in the flow (0 for UDP) |
| `schema_version` | integer | Bump on any breaking change to this table |

To consume: use `consumer_contract.KafkaFlowEventConsumer` directly, or treat
it as a reference implementation and write your own against the schema
above. `consumer_contract.StubFlowEventConsumer` lets the ML module (or any
other consumer) develop and test against this module's output without a
live Kafka broker -- feed it a list of dicts (e.g.
`StubFlowProducer.published` from an end-to-end test run) and it validates
and yields them exactly like the real consumer would.

## Backpressure and reconnection

`BufferedProducer` sits between the pipeline and the Kafka transport:

- A bounded queue decouples packet processing from Kafka's throughput. If
  the queue fills (broker slow/unreachable), the default `drop_oldest`
  policy discards the oldest buffered event to keep memory bounded and
  publish latency low; `block` policy is available when data loss is worse
  than upstream backpressure.
- Publish failures are retried with exponential backoff (capped) up to
  `max_retries`, then the event is dropped and logged.
- `KafkaFlowProducer` tears down its client on any publish failure, so the
  next attempt reconnects from scratch rather than reusing a connection to a
  broker that may no longer be reachable.

## Pcap replay timing

`PcapReplay` sleeps between packets based on their original capture
timestamps (scaled by `--replay-speed`), rather than parsing the file as
fast as possible. This matters because the sliding-window feature extractor
is timestamp-driven: replaying instantly would still compute correct feature
*values*, but would not exercise idle/active timeout behavior the way a real
capture does. Use `--replay-speed 0` when you want instant playback anyway
(e.g. in tests).

## Testing

```
pip install -r requirements.txt
PYTHONPATH=src pytest
```

`tests/fixtures/sample_tcp.pcap` is a small, hand-specified pcap (one TCP
flow with a full handshake/data/FIN teardown, one UDP flow) generated by
`tests/generate_fixtures.py`. Feature-extraction tests assert exact expected
values (packet/byte counts, TCP flag counts, length/inter-arrival
statistics) computed independently from that same per-packet spec, so they
catch real regressions rather than just re-checking the implementation
against itself.

## Known limitations

- **Encrypted traffic is a blind spot.** Features are derived from packet
  headers and sizes only; payload content is never inspected. TLS/QUIC
  traffic yields flow-level features (timing, sizes, flag patterns) but no
  visibility into what's inside the encrypted payload, which limits
  detection of attacks that only manifest in application-layer content.
- **Window size is a real trade-off.** `active_timeout` bounds how long a
  single flow window can grow before it is force-closed and a new window
  starts; `idle_timeout` bounds how long a flow is kept open with no
  traffic. Short timeouts produce more, noisier feature windows and can
  split one logical connection's behavior across multiple events; long
  timeouts delay detection latency and hold more per-flow state in memory.
  The defaults (120s active / 60s idle) are a starting point, not a tuned
  value for any particular deployment or attack class.
- **Fragmented/out-of-order packets are not reassembled.** Each packet is
  processed independently as it's seen; IP fragmentation and TCP
  out-of-order delivery are not explicitly handled, so features on flows
  with heavy fragmentation or reordering may be slightly skewed versus a
  fully reassembling capture engine.
- **Flow direction is "whoever we saw first."** The forward/backward split
  is based on which endpoint sent the first packet this process observed
  for that flow, not necessarily the true connection initiator (e.g. if
  capture starts mid-connection).
- **Backpressure can drop data.** Under sustained broker unavailability or
  overload, the default `drop_oldest` policy silently discards flow events
  rather than blocking capture indefinitely. `dropped`/`retries` counters
  are exposed via `BufferedProducer.stats()` but are not currently exported
  to any metrics system -- that's left to the operational layer.
- **Live capture needs OS-level privileges and drivers** (Npcap/libpcap),
  which are outside this module's control and not covered by its test
  suite; only `PcapReplay` and the flow/feature/producer logic are exercised
  by the unit tests in `tests/`.

## Out of scope

ML models/scoring, the dashboard, and the API are not implemented here. The
Kafka topic and consumer contract above are stubbed only far enough to make
this module independently testable.
