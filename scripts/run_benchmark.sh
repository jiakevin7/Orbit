#!/usr/bin/env bash
set -euo pipefail

# Resolve project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

WORKLOAD="${1:-chat}"
NUM_REQUESTS="${2:-100}"
ARRIVAL_RATE="${3:-10.0}"

echo "=== Orbit Benchmark ==="
echo "Workload: $WORKLOAD"
echo "Requests: $NUM_REQUESTS"
echo "Arrival rate: $ARRIVAL_RATE req/s"
echo ""

# Use 'docker compose' (v2) if available, fall back to 'docker-compose' (v1)
if docker compose version &>/dev/null; then
    COMPOSE="docker compose"
else
    COMPOSE="docker-compose"
fi

# Ensure services are up
echo "Starting services..."
$COMPOSE up -d orbit-router orbit-replica-1 orbit-replica-2 orbit-replica-3 orbit-replica-4 orbit-monitor

echo "Waiting for services to be ready..."
READY=false
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "Router is ready."
        READY=true
        break
    fi
    echo "  waiting... ($i/30)"
    sleep 1
done

if [ "$READY" = false ]; then
    echo "ERROR: Router did not become ready within 30 seconds."
    echo "Check logs with: $COMPOSE logs orbit-router"
    exit 1
fi

# Ensure results directory exists
mkdir -p results

# Run benchmark
echo ""
echo "Running benchmark..."
$COMPOSE run --rm \
    -e ORBIT_ROUTER_URL=http://orbit-router:8000 \
    -e ORBIT_REPLICA_URLS=http://orbit-replica-1:8001,http://orbit-replica-2:8001,http://orbit-replica-3:8001,http://orbit-replica-4:8001 \
    -e ORBIT_WORKLOAD="$WORKLOAD" \
    -e ORBIT_NUM_REQUESTS="$NUM_REQUESTS" \
    -e ORBIT_ARRIVAL_RATE="$ARRIVAL_RATE" \
    orbit-bench

echo ""
echo "Benchmark complete. Results saved to results/"
