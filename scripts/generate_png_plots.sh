#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/generate_png_plots.sh <run-dir> [extra args ...]" >&2
  exit 2
fi

python3 scripts/generate_png_plots.py "$@"
