#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/visualize_run.sh <run-dir> [extra args ...]" >&2
  exit 2
fi

python3 scripts/visualize_run.py "$@"
