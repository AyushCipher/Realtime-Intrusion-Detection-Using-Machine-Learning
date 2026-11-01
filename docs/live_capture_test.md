# Live capture test

Pcap replay is what the demo stack and the benchmarks use. This documents the
run that exercised `LiveCapture` (Scapy `AsyncSniffer`) on a real network
interface, end to end.

## What was run

```
docker compose -f docker-compose.yml -f docker-compose.live.yml up -d --build
python integration/test_live_capture.py
```

- `ingestion` sniffs the container's `eth0` (`INGESTION_INTERFACE=eth0`, filter
  `(tcp or udp) and not port 9092` so it does not capture its own Kafka output,
  idle timeout 5 s, active timeout 15 s).
- `traffic-gen` (`integration/live_traffic_gen.py`) shares ingestion's network
  namespace and generates real TCP traffic against our own `dashboard-api`
  container: 10 normal GETs, 300 rapid GETs, and connect() attempts across 110
  ports, repeated for 3 rounds. Only a service we own is targeted.
- The chain then ran as normal: Kafka -> ml scoring -> Kafka -> dashboard-api.
- `integration/test_live_capture.py` passes only if alerts exist whose endpoints
  are not in the demo pcap (10.0.0.x), which is only possible if packets were
  sniffed live.

## Result

Passed. The final snapshot had 186 alerts from live-captured flows, all
escalated by the open-set gate. In an earlier snapshot of 149 alerts, the
probe flows (ports other than 8000) were all labelled `PortScan`, and the HTTP
burst flows on port 8000 were labelled `PortScan` (70), `Heartbleed` (24) and
`DoS/DDoS` (1).

## What it found

1. **Bug fixed: silent capture failure.** The ingestion image had no libpcap, so
   Scapy could not compile the BPF filter. The sniffer thread died, but the
   process logged "Live capture started" and then produced nothing. Fixes: the
   Dockerfile now installs `libpcap0.8`, and `LiveCapture` now raises if the
   sniffer thread dies (`ingestion/tests/test_capture.py`).
2. **Classification quality on live traffic is poor, as expected.** The bundled
   model is trained on the synthetic fixture, whose attack share is about 35%.
   Real HTTP bursts were labelled Heartbleed and PortScan, and every flow was
   escalated. This is a limit of the training data, not of the live path. A
   real deployment needs retraining on real traffic (for example the
   CSE-CIC-IDS2018 subsample) and a recalibrated threshold.
3. **Not tested:** capture directly on a Windows host (needs Npcap and
   Administrator rights, neither present here), long-duration runs, and
   high-rate traffic.
