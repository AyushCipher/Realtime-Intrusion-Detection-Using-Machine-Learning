#!/bin/sh
# Integration-only glue: passes the compose-provided broker address into
# ids_tier2's own serve.py CLI, and picks a real LLM client automatically
# if a key is available, falling back to the deterministic stub otherwise
# -- so `docker compose up` works out of the box with no API key, and
# picking up a real one (via a local, gitignored .env file -- see
# tier2_reasoner/README.md) needs no other change.
set -eu

BOOTSTRAP="${TIER2_BOOTSTRAP_SERVERS:-kafka:9092}"

if [ -n "${GEMINI_API_KEY:-}" ]; then
  LLM_FLAG="--llm gemini"
elif [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  LLM_FLAG="--llm anthropic"
else
  LLM_FLAG="--llm stub"
fi

echo "tier2-reasoner: serving against $BOOTSTRAP ($LLM_FLAG)"

exec python -m ids_tier2.serve --bootstrap-servers "$BOOTSTRAP" $LLM_FLAG
