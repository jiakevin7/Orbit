#!/usr/bin/env bash
set -eo pipefail

#
# Start Orbit locally with llama.cpp backends on macOS (Apple Silicon).
#
# Usage:
#   ./scripts/start_local.sh                    # 4 replicas (default)
#   ./scripts/start_local.sh 2                  # 2 replicas
#   ORBIT_MODEL=/path/to/model.gguf ./scripts/start_local.sh
#
# Prerequisites:
#   brew install llama.cpp
#   pip install -e .    (in the project venv)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Use the project venv's Python if available
if [ -x "$PROJECT_ROOT/.venv/bin/python3" ]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python3"
else
    PYTHON="python3"
fi

NUM_REPLICAS="${1:-4}"
MODEL_DIR="$PROJECT_ROOT/models"
MODEL_PATH="${ORBIT_MODEL:-$MODEL_DIR/qwen2.5-1.5b-instruct-q4_k_m.gguf}"

# Port assignments
ROUTER_PORT=8000
MONITOR_PORT=8080
LLAMACPP_BASE_PORT=9001   # llama.cpp: 9001, 9002, ...
REPLICA_BASE_PORT=8001    # replica FastAPI: 8001, 8002, ...

# PID tracking for cleanup
PIDS=()
cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in "${PIDS[@]+"${PIDS[@]}"}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "All processes stopped."
}
trap cleanup EXIT INT TERM

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

    # Download via the huggingface_hub Python API (works regardless of CLI setup)
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
" || {
        echo "ERROR: Failed to download model."
        echo "  Ensure huggingface-hub is installed: pip install huggingface-hub"
        echo "  Or set ORBIT_MODEL=/path/to/your/model.gguf"
        exit 1
    }

    if [ ! -f "$MODEL_PATH" ]; then
        echo "ERROR: Download succeeded but model not found at expected path."
        echo "  Check $MODEL_DIR for the downloaded file."
        ls -la "$MODEL_DIR"
        exit 1
    fi
fi

echo "=== Orbit Local Startup ==="
echo "Model:    $MODEL_PATH"
echo "Replicas: $NUM_REPLICAS"
echo ""

# ── Start llama.cpp instances ────────────────────────────────────────────

echo "Starting $NUM_REPLICAS llama.cpp instances..."
for i in $(seq 1 "$NUM_REPLICAS"); do
    LLAMA_PORT=$((LLAMACPP_BASE_PORT + i - 1))
    echo "  llama.cpp replica-$i on port $LLAMA_PORT"

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
    PIDS+=($!)
done

# Wait for llama.cpp servers to be ready
echo "Waiting for llama.cpp servers..."
for i in $(seq 1 "$NUM_REPLICAS"); do
    LLAMA_PORT=$((LLAMACPP_BASE_PORT + i - 1))
    for attempt in $(seq 1 60); do
        if curl -sf "http://localhost:$LLAMA_PORT/health" >/dev/null 2>&1; then
            echo "  llama.cpp replica-$i ready"
            break
        fi
        if [ "$attempt" -eq 60 ]; then
            echo "ERROR: llama.cpp replica-$i failed to start on port $LLAMA_PORT"
            exit 1
        fi
        sleep 1
    done
done

# ── Start replica FastAPI services ───────────────────────────────────────

echo "Starting $NUM_REPLICAS replica services..."
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
    PIDS+=($!)

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
    --log-level info \
    2>&1 &
PIDS+=($!)

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
PIDS+=($!)

# ── Wait for router ──────────────────────────────────────────────────────

echo ""
echo "Waiting for router..."
for attempt in $(seq 1 30); do
    if curl -sf "http://localhost:$ROUTER_PORT/health" >/dev/null 2>&1; then
        break
    fi
    if [ "$attempt" -eq 30 ]; then
        echo "WARNING: Router may not be ready yet."
    fi
    sleep 1
done

echo ""
echo "=== Orbit is running ==="
echo "  Router:   http://localhost:$ROUTER_PORT"
echo "  Monitor:  http://localhost:$MONITOR_PORT"
echo "  Replicas: $REPLICA_URLS"
echo ""
echo "Try:"
echo "  curl -X POST http://localhost:$ROUTER_PORT/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":32}'"
echo ""
echo "Press Ctrl+C to stop."
wait
