#!/usr/bin/env bash
# Run e2e against a live ihd-faiss-backend on shdnetwork.
# Usage:
#   ./run.sh                      # fast suite (no OpenAI calls — skips search tests)
#   RUN_EXPENSIVE_E2E=1 ./run.sh  # include search tests (real OpenAI embedding)
#   ./run.sh -k health            # pytest filter
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BASE_URL="${E2E_BASE_URL:-http://ihd-faiss-backend}"

ENV_ARGS=(-e "E2E_BASE_URL=$BASE_URL")
if [[ -n "${RUN_EXPENSIVE_E2E:-}" ]]; then
    ENV_ARGS+=(-e "RUN_EXPENSIVE_E2E=$RUN_EXPENSIVE_E2E")
fi
for var in API_TOKEN E2E_ROUTE_HEALTH E2E_ROUTE_PROJECTS E2E_ROUTE_SEARCH; do
    if [[ -n "${!var:-}" ]]; then
        ENV_ARGS+=(-e "$var=${!var}")
    fi
done

exec docker run --rm \
    --network shdnetwork \
    "${ENV_ARGS[@]}" \
    -v "$ROOT:/e2e" \
    -w /e2e \
    python:3.11-slim \
    bash -c "pip install -q -r requirements.txt && pytest -v --tb=short $*"
