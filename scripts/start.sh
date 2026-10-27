#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/uvicorn backend.app:app --host "${WEAVE_HOST:-0.0.0.0}" --port "${WEAVE_PORT:-8173}"
