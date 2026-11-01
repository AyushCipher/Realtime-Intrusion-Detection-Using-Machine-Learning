#!/bin/sh
# Integration-only glue around ids_ingestion's own CLI (unmodified).
#
# Default: loop a pcap replay so the demo stack keeps producing traffic
# instead of a single one-shot burst.
# Live mode: set INGESTION_INTERFACE (e.g. eth0) to sniff that interface once
# with ids_ingestion's LiveCapture instead -- see docker-compose.live.yml.
set -eu

BOOTSTRAP="${INGESTION_BOOTSTRAP_SERVERS:-kafka:9092}"
PCAP="${INGESTION_PCAP:-/app/sample.pcap}"
SPEED="${INGESTION_REPLAY_SPEED:-1}"
LOOP_DELAY="${INGESTION_LOOP_DELAY_SECONDS:-15}"

if [ -n "${INGESTION_INTERFACE:-}" ]; then
  FILTER="${INGESTION_BPF_FILTER:-tcp or udp}"
  IDLE="${INGESTION_IDLE_TIMEOUT:-60}"
  ACTIVE="${INGESTION_ACTIVE_TIMEOUT:-120}"
  echo "ingestion: LIVE capture on $INGESTION_INTERFACE (filter='$FILTER', idle=${IDLE}s, active=${ACTIVE}s) -> $BOOTSTRAP"
  exec python -m ids_ingestion --interface "$INGESTION_INTERFACE" --bpf-filter "$FILTER" \
    --idle-timeout "$IDLE" --active-timeout "$ACTIVE" --bootstrap-servers "$BOOTSTRAP"
fi

echo "ingestion: replaying $PCAP to $BOOTSTRAP (speed=$SPEED), looping every ${LOOP_DELAY}s"

while true; do
  python -m ids_ingestion --pcap "$PCAP" --replay-speed "$SPEED" --bootstrap-servers "$BOOTSTRAP" \
    || echo "ingestion: replay pass failed, retrying after ${LOOP_DELAY}s"
  sleep "$LOOP_DELAY"
done
