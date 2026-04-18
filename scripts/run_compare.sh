#!/usr/bin/env bash

set -euo pipefail

requests="${ORBIT_REQUESTS:-40}"
routers="${ORBIT_ROUTERS:-2}"
clusters="${ORBIT_CLUSTERS:-3}"
backend="${ORBIT_BACKEND:-synthetic}"

python3 -m orbit \
  --compare \
  --backend "$backend" \
  --requests "$requests" \
  --routers "$routers" \
  --clusters "$clusters" \
  "$@"
