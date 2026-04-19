#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "$ROOT_DIR/scripts/benchmark_related_work.py" \
  --target-config "$ROOT_DIR/configs/related_work_targets.example.json" \
  "$@"
