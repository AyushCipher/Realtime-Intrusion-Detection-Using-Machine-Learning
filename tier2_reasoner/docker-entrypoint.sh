#!/bin/sh
# Integration-only glue: passes the compose-provided broker address into
# ids_tier2's own serve.py CLI and picks the LLM client.
#
# TIER2_LLM=ollama|gemini|anthropic|stub selects one explicitly. Without it, a
# real client is picked if a key is available (GEMINI_API_KEY, then
# ANTHROPIC_API_KEY), falling back to the deterministic stub -- so
# `docker compose up` works out of the box with no key.
#
# ollama runs a local open-weights model on the HOST (no key, no quota; the
# hosted free tiers hit 503s and daily quotas in live testing). The container
# reaches it at OLLAMA_HOST_URL, by default Docker Desktop's
# host.docker.internal.
#
# TIER2_RETRIEVAL_MODE (default auto) keys ATT&CK retrieval on the flow's
# behaviour for escalated alerts, whose predicted class is usually wrong.
set -eu

BOOTSTRAP="${TIER2_BOOTSTRAP_SERVERS:-kafka:9092}"
RETRIEVAL_MODE="${TIER2_RETRIEVAL_MODE:-auto}"

LLM="${TIER2_LLM:-}"
if [ -z "$LLM" ]; then
  if [ -n "${GEMINI_API_KEY:-}" ]; then
    LLM="gemini"
  elif [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    LLM="anthropic"
  else
    LLM="stub"
  fi
fi

EXTRA=""
if [ "$LLM" = "ollama" ]; then
  EXTRA="--ollama-host ${OLLAMA_HOST_URL:-http://host.docker.internal:11434} --ollama-model ${OLLAMA_MODEL:-llama3.2:1b} --min-interval-s 0"
fi

echo "tier2-reasoner: serving against $BOOTSTRAP (--llm $LLM, --retrieval-mode $RETRIEVAL_MODE)"

# shellcheck disable=SC2086 -- EXTRA is a deliberately word-split flag list
exec python -m ids_tier2.serve --bootstrap-servers "$BOOTSTRAP" --llm "$LLM" --retrieval-mode "$RETRIEVAL_MODE" $EXTRA
