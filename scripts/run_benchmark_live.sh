#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
default_model="$repo_root/models/qwen2.5-3b-instruct-q4_k_m.gguf"
model_path="${ORBIT_MODEL_PATH:-${ORBIT_LIVE_TEST_MODEL:-$default_model}}"

if [[ ! -f "$model_path" ]]; then
  echo "live model not found: $model_path" >&2
  echo "set ORBIT_MODEL_PATH or ORBIT_LIVE_TEST_MODEL to a local GGUF path" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
output_dir="${ORBIT_OUTPUT_DIR:-results/live-benchmark-${timestamp}}"
args=("$@")

for ((index = 0; index < ${#args[@]}; index++)); do
  case "${args[index]}" in
    --output-dir)
      if (( index + 1 < ${#args[@]} )); then
        output_dir="${args[index + 1]}"
      fi
      ;;
    --output-dir=*)
      output_dir="${args[index]#--output-dir=}"
      ;;
  esac
done

python3 scripts/benchmark_policies.py \
  --backend llama_cpp \
  --model "$model_path" \
  --workload-kind "${ORBIT_WORKLOAD_KIND:-mixed_realistic}" \
  --requests "${ORBIT_REQUESTS:-20}" \
  --warmup-requests "${ORBIT_WARMUP_REQUESTS:-8}" \
  --validation-requests "${ORBIT_VALIDATION_REQUESTS:-4}" \
  --routers "${ORBIT_ROUTERS:-2}" \
  --clusters "${ORBIT_CLUSTERS:-2}" \
  --live-arrival-scale "${ORBIT_LIVE_ARRIVAL_SCALE:-0.01}" \
  --calibrate-router \
  --output-dir "$output_dir" \
  "${args[@]}"

python3 scripts/visualize_run.py "$output_dir"
echo "wrote live benchmark artifacts to $output_dir"
