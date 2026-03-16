#!/usr/bin/env bash
set -eo pipefail

#
# Run the Orbit benchmark locally with llama.cpp backends.
# Restarts llama.cpp between each strategy for a truly cold cache.
#
# Usage:
#   ./scripts/run_benchmark_local.sh                     # chat workload, 100 reqs
#   ./scripts/run_benchmark_local.sh chat 50 5.0         # chat, 50 reqs, 5 req/s
#   ./scripts/run_benchmark_local.sh mixed 100 5.0 4     # mixed, 4 replicas
#
# Prerequisites:
#   brew install llama.cpp
#   pip install -e .    (in the project venv)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -x "$PROJECT_ROOT/.venv/bin/python3" ]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python3"
else
    PYTHON="python3"
fi

WORKLOAD="${1:-chat}"
NUM_REQUESTS="${2:-100}"
ARRIVAL_RATE="${3:-10.0}"
NUM_REPLICAS="${4:-4}"

MODEL_DIR="$PROJECT_ROOT/models"
MODEL_PATH="${ORBIT_MODEL:-$MODEL_DIR/qwen2.5-1.5b-instruct-q4_k_m.gguf}"

ROUTER_PORT=8000
MONITOR_PORT=8080
REPLICA_BASE_PORT=8001
LLAMACPP_BASE_PORT=9001

STRATEGIES="orbit round_robin random hash_based least_loaded"

# ── PID tracking ─────────────────────────────────────────────────────────

# Service PIDs (router, monitor, replicas) — live for the whole run
SVC_PIDS=()
# llama.cpp PIDs — killed and restarted between strategies
LLAMA_PIDS=()

cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in ${LLAMA_PIDS[@]+"${LLAMA_PIDS[@]}"}; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in ${SVC_PIDS[@]+"${SVC_PIDS[@]}"}; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "All processes stopped."
}
trap cleanup EXIT INT TERM

# ── Helpers ──────────────────────────────────────────────────────────────

start_llama_servers() {
    LLAMA_PIDS=()
    for i in $(seq 1 "$NUM_REPLICAS"); do
        LLAMA_PORT=$((LLAMACPP_BASE_PORT + i - 1))
        llama-server \
            --model "$MODEL_PATH" \
            --port "$LLAMA_PORT" \
            --n-gpu-layers 99 \
            --ctx-size 4096 \
            --cache-type-k q4_0 \
            --cache-type-v q4_0 \
            --parallel 1 \
            --log-disable \
            2>/dev/null &
        LLAMA_PIDS+=($!)
    done

    for i in $(seq 1 "$NUM_REPLICAS"); do
        LLAMA_PORT=$((LLAMACPP_BASE_PORT + i - 1))
        for attempt in $(seq 1 60); do
            if curl -sf "http://localhost:$LLAMA_PORT/health" >/dev/null 2>&1; then
                break
            fi
            if [ "$attempt" -eq 60 ]; then
                echo "ERROR: llama.cpp replica-$i failed to start on port $LLAMA_PORT"
                exit 1
            fi
            sleep 1
        done
    done

    # Verify each instance can actually serve completions (health != ready)
    for i in $(seq 1 "$NUM_REPLICAS"); do
        LLAMA_PORT=$((LLAMACPP_BASE_PORT + i - 1))
        for attempt in $(seq 1 30); do
            if curl -sf "http://localhost:$LLAMA_PORT/completion" \
                -H 'Content-Type: application/json' \
                -d '{"prompt":"hi","n_predict":1}' >/dev/null 2>&1; then
                break
            fi
            if [ "$attempt" -eq 30 ]; then
                echo "ERROR: llama.cpp replica-$i not serving on port $LLAMA_PORT"
                exit 1
            fi
            sleep 1
        done
    done
}

stop_llama_servers() {
    for pid in ${LLAMA_PIDS[@]+"${LLAMA_PIDS[@]}"}; do
        kill "$pid" 2>/dev/null || true
    done
    # Wait for ports to free up
    for pid in ${LLAMA_PIDS[@]+"${LLAMA_PIDS[@]}"}; do
        wait "$pid" 2>/dev/null || true
    done
    LLAMA_PIDS=()
    sleep 1
}

# ── Check prerequisites ──────────────────────────────────────────────────

if ! command -v llama-server &>/dev/null; then
    echo "ERROR: llama-server not found. Install with: brew install llama.cpp"
    exit 1
fi

if ! "$PYTHON" -c "import orbit" 2>/dev/null; then
    echo "ERROR: orbit package not importable. Run: pip install -e . (in your venv)"
    exit 1
fi

# ── Download model if needed ─────────────────────────────────────────────

if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found at $MODEL_PATH"
    echo "Downloading Qwen2.5-1.5B-Instruct Q4_K_M (~900MB)..."
    mkdir -p "$MODEL_DIR"
    "$PYTHON" -c "
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(
    repo_id='Qwen/Qwen2.5-1.5B-Instruct-GGUF',
    filename='qwen2.5-1.5b-instruct-q4_k_m.gguf',
)
dest = os.path.join('$MODEL_DIR', 'qwen2.5-1.5b-instruct-q4_k_m.gguf')
if not os.path.exists(dest):
    shutil.copy2(path, dest)
print(f'Model saved to {dest}')
" || { echo "ERROR: Failed to download model."; exit 1; }
fi

echo "=== Orbit Local Benchmark (llama.cpp) ==="
echo "Model:       $MODEL_PATH"
echo "Workload:    $WORKLOAD"
echo "Requests:    $NUM_REQUESTS"
echo "Arrival rate: $ARRIVAL_RATE req/s"
echo "Replicas:    $NUM_REPLICAS"
echo "Strategies:  $STRATEGIES"
echo ""

# ── Start llama.cpp (first time) ─────────────────────────────────────────

echo "Starting llama.cpp instances..."
start_llama_servers
echo "  All llama.cpp instances ready."

# ── Start replica FastAPI services (stay up for entire run) ──────────────

echo "Starting replica services..."
REPLICA_URLS=""
for i in $(seq 1 "$NUM_REPLICAS"); do
    REPLICA_PORT=$((REPLICA_BASE_PORT + i - 1))
    LLAMA_PORT=$((LLAMACPP_BASE_PORT + i - 1))

    ORBIT_PORT=$REPLICA_PORT \
    ORBIT_REPLICA_ID="replica-$i" \
    ORBIT_ROUTER_URL="http://localhost:$ROUTER_PORT" \
    ORBIT_BACKEND="llamacpp" \
    ORBIT_LLAMACPP_URL="http://localhost:$LLAMA_PORT" \
    ORBIT_CACHE_CAPACITY=1000 \
    ORBIT_BLOCK_SIZE=16 \
    ORBIT_MAX_CONCURRENT=1 \
    "$PYTHON" -m uvicorn orbit.replica.app:app \
        --host 0.0.0.0 --port "$REPLICA_PORT" \
        --log-level warning \
        2>&1 &
    SVC_PIDS+=($!)

    if [ -n "$REPLICA_URLS" ]; then
        REPLICA_URLS="$REPLICA_URLS,"
    fi
    REPLICA_URLS="${REPLICA_URLS}http://localhost:$REPLICA_PORT"
    echo "  replica-$i on port $REPLICA_PORT -> llama.cpp :$LLAMA_PORT"
done

# ── Start router ─────────────────────────────────────────────────────────

echo "Starting router on port $ROUTER_PORT..."
ORBIT_PORT=$ROUTER_PORT \
ORBIT_REPLICA_URLS="$REPLICA_URLS" \
ORBIT_MONITOR_URL="http://localhost:$MONITOR_PORT" \
"$PYTHON" -m uvicorn orbit.router.app:app \
    --host 0.0.0.0 --port "$ROUTER_PORT" \
    --log-level warning \
    2>&1 &
SVC_PIDS+=($!)

# ── Start monitor ────────────────────────────────────────────────────────

echo "Starting monitor on port $MONITOR_PORT..."
ORBIT_PORT=$MONITOR_PORT \
ORBIT_ROUTER_URL="http://localhost:$ROUTER_PORT" \
ORBIT_REPLICA_URLS="$REPLICA_URLS" \
ORBIT_POLL_INTERVAL=1.0 \
"$PYTHON" -m uvicorn orbit.monitor.app:app \
    --host 0.0.0.0 --port "$MONITOR_PORT" \
    --log-level warning \
    2>&1 &
SVC_PIDS+=($!)

# ── Wait for services ────────────────────────────────────────────────────

echo ""
echo "Waiting for services..."
for i in $(seq 1 "$NUM_REPLICAS"); do
    REPLICA_PORT=$((REPLICA_BASE_PORT + i - 1))
    for attempt in $(seq 1 30); do
        if curl -sf "http://localhost:$REPLICA_PORT/v1/status" >/dev/null 2>&1; then
            break
        fi
        if [ "$attempt" -eq 30 ]; then
            echo "ERROR: replica-$i failed to start"
            exit 1
        fi
        sleep 0.5
    done
done
for attempt in $(seq 1 30); do
    if curl -sf "http://localhost:$ROUTER_PORT/health" >/dev/null 2>&1; then
        break
    fi
    if [ "$attempt" -eq 30 ]; then
        echo "ERROR: Router failed to start"
        exit 1
    fi
    sleep 0.5
done
echo "All services ready."
echo ""

# ── Run each strategy with fresh llama.cpp instances ─────────────────────

for STRATEGY in $STRATEGIES; do
    echo "────────────────────────────────────────"
    echo "Strategy: $STRATEGY"
    echo "  Restarting llama.cpp instances (cold cache)..."
    stop_llama_servers
    start_llama_servers
    echo "  llama.cpp ready. Running benchmark..."

    ORBIT_ROUTER_URL="http://localhost:$ROUTER_PORT" \
    ORBIT_REPLICA_URLS="$REPLICA_URLS" \
    ORBIT_WORKLOAD="$WORKLOAD" \
    ORBIT_STRATEGY="$STRATEGY" \
    ORBIT_NUM_REQUESTS="$NUM_REQUESTS" \
    ORBIT_ARRIVAL_RATE="$ARRIVAL_RATE" \
    "$PYTHON" -m orbit.bench.runner

    echo "  Done."
    echo ""
done

echo "Benchmark complete. Results saved to results/$WORKLOAD/"

# ── Generate report and plots ────────────────────────────────────────────

echo ""
echo "Generating report..."
"$PYTHON" -c "
from orbit.bench.analysis import generate_report, plot_results
print(generate_report('results'))
plot_results('results')
print('Plots saved.')
"
