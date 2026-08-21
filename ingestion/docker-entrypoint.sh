#!/bin/sh
# Integration-only glue: loops ids_ingestion's own CLI (unmodified) so the
# demo stack keeps producing traffic instead of a single one-shot burst.
set -eu

BOOTSTRAP="${INGESTION_BOOTSTRAP_SERVERS:-kafka:9092}"
PCAP="${INGESTION_PCAP:-/app/sample.pcap}"
SPEED="${INGESTION_REPLAY_SPEED:-1}"
LOOP_DELAY="${INGESTION_LOOP_DELAY_SECONDS:-15}"

echo "ingestion: replaying $PCAP to $BOOTSTRAP (speed=$SPEED), looping every ${LOOP_DELAY}s"

while true; do
  python -m ids_ingestion --pcap "$PCAP" --replay-speed "$SPEED" --bootstrap-servers "$BOOTSTRAP" \
    || echo "ingestion: replay pass failed, retrying after ${LOOP_DELAY}s"
  sleep "$LOOP_DELAY"
done
