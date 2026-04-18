#!/usr/bin/env bash

set -euo pipefail

timestamp="$(date +%Y%m%d-%H%M%S)"
output_dir="${ORBIT_OUTPUT_DIR:-results/synthetic-benchmark-${timestamp}}"
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
  --backend synthetic \
  --workload-kind "${ORBIT_WORKLOAD_KIND:-mixed_realistic}" \
  --requests "${ORBIT_REQUESTS:-40}" \
  --warmup-requests "${ORBIT_WARMUP_REQUESTS:-10}" \
  --validation-requests "${ORBIT_VALIDATION_REQUESTS:-10}" \
  --routers "${ORBIT_ROUTERS:-2}" \
  --clusters "${ORBIT_CLUSTERS:-3}" \
  --output-dir "$output_dir" \
  "${args[@]}"

python3 scripts/visualize_run.py "$output_dir"
echo "wrote synthetic benchmark artifacts to $output_dir"
