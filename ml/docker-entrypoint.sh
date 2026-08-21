#!/bin/sh
# Integration-only glue: passes the compose-provided broker address into
# ids_ml's own serve.py CLI (unmodified).
set -eu

BOOTSTRAP="${ML_BOOTSTRAP_SERVERS:-kafka:9092}"

echo "ml: serving two-stage detector against $BOOTSTRAP"

exec python -m ids_ml.serve --model-dir /app/models --bootstrap-servers "$BOOTSTRAP" --alert-on-stage1-flag-only
