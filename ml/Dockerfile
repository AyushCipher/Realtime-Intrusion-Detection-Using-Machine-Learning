FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY tests/fixtures/synthetic_cicids_sample.csv /app/training_data.csv
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

ENV PYTHONPATH=/app/src

# Trains at build time on the synthetic fixture bundled with the ml module's
# own test suite (see ml/README.md: this repo does not ship real
# CICIDS2017/2018 data). This is a demo-quality model, not a production one
# -- see the top-level README's known-limitations section.
#
# The deployment configuration found best on real traffic
# (docs/real_data_findings.md): hybrid Stage 1 with conformal recall
# calibration, the kNN escalation gate (best on unseen attack families;
# results/paper/fused_gate/), an adaptive escalation threshold, and joint
# calibration of both stages for a 5% end-to-end miss budget, so the demo
# produces escalated alerts for tier2_reasoner. --escalation-budget 0.1
# matches the budget used throughout this project's evaluation. To train it
# on real data instead, see docker-compose.realdata.yml.
RUN python -m ids_ml.train --data /app/training_data.csv --model-dir /app/models --split random \
    --stage1-kind hybrid --skip-leakage-check --skip-robustness-probe \
    --gate knn --adaptive-escalation --escalation-budget 0.1 --joint-miss-budget 0.05

ENTRYPOINT ["/app/docker-entrypoint.sh"]
