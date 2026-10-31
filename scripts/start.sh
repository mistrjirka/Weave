#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
  echo 'Weave is not installed. Run: bash scripts/install.sh' >&2
  exit 1
fi
exec .venv/bin/python -m backend.installation run "$@"
