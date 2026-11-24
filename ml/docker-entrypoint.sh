#!/bin/sh
# Integration-only glue: passes the compose-provided broker address into
# ids_ml's own serve.py CLI, and optionally trains a site model first.
#
# Default: serve the model trained at image build time on the bundled
# synthetic fixture (a demo, not a real-traffic model).
#
# ML_TRAIN_DATA="/data/a.csv /data/b.csv": train the deployment configuration
# on real, canonical-form CSVs mounted into the container (the real datasets
# are not shipped in the image) -- hybrid Stage 1, the ML_GATE escalation
# gate with an adaptive escalation threshold, both stages calibrated jointly
# so the end-to-end known-attack miss rate is at most ML_MISS_BUDGET, and a
# random split across the datasets. The trained models are cached in
# ML_SITE_MODEL_DIR (a volume), so restarts do not retrain; set ML_RETRAIN=1
# to force it. See docker-compose.realdata.yml.
set -eu

BOOTSTRAP="${ML_BOOTSTRAP_SERVERS:-kafka:9092}"
MODEL_DIR=/app/models

if [ -n "${ML_TRAIN_DATA:-}" ]; then
  MODEL_DIR="${ML_SITE_MODEL_DIR:-/app/site_models}"
  if [ ! -f "$MODEL_DIR/stage2_xgboost.joblib" ] || [ "${ML_RETRAIN:-0}" = "1" ]; then
    echo "ml: training site model from $ML_TRAIN_DATA into $MODEL_DIR (gate=${ML_GATE:-knn})"
    # shellcheck disable=SC2086 -- ML_TRAIN_DATA is a deliberately word-split file list
    python -m ids_ml.train --data $ML_TRAIN_DATA --model-dir "$MODEL_DIR" --split random \
      --stage1-kind hybrid --joint-miss-budget "${ML_MISS_BUDGET:-0.05}" \
      --gate "${ML_GATE:-knn}" --adaptive-escalation --escalation-budget "${ML_ESCALATION_BUDGET:-0.1}" \
      --skip-leakage-check --skip-robustness-probe
  else
    echo "ml: using cached site model in $MODEL_DIR"
  fi
fi

echo "ml: serving two-stage detector from $MODEL_DIR against $BOOTSTRAP"

exec python -m ids_ml.serve --model-dir "$MODEL_DIR" --bootstrap-servers "$BOOTSTRAP" --alert-on-stage1-flag-only
