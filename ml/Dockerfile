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
# --gate openset fits and calibrates the OpenMax escalation gate (see
# ml/README.md's "Open-set escalation" section) so this demo actually
# produces escalated: true alerts for tier2_reasoner to consume, instead
# of running the closed-set-only cascade. --escalation-budget 0.1 matches
# the budget used throughout this project's H1/H2 evaluation.
RUN python -m ids_ml.train --data /app/training_data.csv --model-dir /app/models \
    --stage1-contamination 0.35 --skip-leakage-check --skip-robustness-probe \
    --gate openset --escalation-budget 0.1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
