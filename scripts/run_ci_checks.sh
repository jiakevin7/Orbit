#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_ci_checks.sh [--unit-only|--with-live|--live-only]

  --unit-only  Run the default unit suite only.
  --with-live  Run the default unit suite, then the opt-in live llama.cpp integration test.
  --live-only  Run only the opt-in live llama.cpp integration test.

When the live test is selected, set ORBIT_LIVE_TEST_MODEL to a local GGUF path
or place the default test model under models/.
EOF
}

run_unit=1
run_live=0

case "${1:-}" in
  "")
    ;;
  --unit-only)
    ;;
  --with-live)
    run_live=1
    ;;
  --live-only)
    run_unit=0
    run_live=1
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [[ "$run_unit" -eq 1 ]]; then
  python3 -m unittest discover -s tests -v
fi

if [[ "$run_live" -eq 1 ]]; then
  export ORBIT_RUN_LIVE_TESTS=1
  python3 -m unittest tests.test_live_llamacpp_integration -v
fi
