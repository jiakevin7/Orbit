# Orbit

Semantic-aware KV-cache affinity routing layer for LLM serving.

Orbit is a request router that steers inference traffic toward replicas that already hold relevant KV-cache state, reducing redundant prefill computation. It uses a two-stage approach: a fast semantic pre-filter narrows candidates, then exact token-level prefix matching selects the best replica.

## Architecture

```
                    ┌─────────────────┐
   Clients ──────►  │  Orbit Router   │  (cache-affinity + load-aware routing)
                    │  :8000          │
                    └────┬───┬───┬────┘
                         │   │   │
              ┌──────────┘   │   └──────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Replica 1│  │ Replica 2│  │ Replica 3│  ... (N replicas)
        │ :8001    │  │ :8002    │  │ :8003    │
        └──────────┘  └──────────┘  └──────────┘
              ▲              ▲              ▲
              └──────────┬───┴──────────────┘
                         │
                    ┌────┴────┐
                    │ Monitor │  (polls replica load)
                    │ :8080   │
                    └─────────┘
```

All components communicate over HTTP/JSON on a Docker bridge network.

## How It Works

### Two-Stage Routing

**Stage 1 — Semantic Pre-Filter** (fast, approximate)
1. Parse the prompt into segments (system prompt, tools, RAG context, user query)
2. Embed the stable segments (system prompt, tools) with MiniLM
3. Cosine similarity against embeddings of cached prefix segments
4. Return top-k candidate replicas likely to hold relevant KV-cache state

**Stage 2 — Exact Match + Scoring** (correct, decisive)
1. Tokenize the full prompt and compute chained SHA-256 block hashes (16-token blocks)
2. For each candidate replica, look up the prefix trie for the longest exact block-hash match
3. Score each candidate: `score = α · (cached_tokens / total_tokens) - β · normalized_congestion`
4. Select the highest-scoring replica (fallback: least-loaded if no cache overlap)

The semantic layer only narrows the search — it never bypasses hash verification. Only exact token-level prefix matches count.

### Simulated Backend

Since Docker on macOS cannot pass through GPUs, replicas use a simulated LLM backend that models realistic timing:

```
prefill_ms = 0.5ms × (total_prompt_tokens - cached_prefix_tokens)
decode_ms  = 10ms  × output_tokens
queue_wait = FIFO queue with max_concurrent limit (default: 4)
```

The simulated KV cache uses LRU eviction over fixed-size blocks (default: 1000 blocks of 16 tokens).

## Project Structure

```
orbit/
├── common/           # Shared code
│   ├── schemas.py    # Pydantic models (requests, responses, cache updates)
│   ├── hashing.py    # SHA-256 chained block hashing over token sequences
│   └── config.py     # Env-var-driven configuration
├── router/           # Routing service
│   ├── app.py              # FastAPI entrypoint (:8000)
│   ├── routing_engine.py   # Core pipeline: parse → filter → match → score → route
│   ├── prompt_analyzer.py  # Segment, tokenize, and hash chat messages
│   ├── prefix_trie.py      # Radix trie: block hashes → replicas
│   ├── semantic_index.py   # MiniLM embedding search for candidate pre-filtering
│   ├── cache_registry.py   # Bidirectional index (replica ↔ block hashes)
│   ├── scorer.py           # α·prefill_savings - β·congestion scoring
│   └── load_tracker.py     # Per-replica load state
├── replica/          # Simulated LLM replicas
│   ├── app.py              # FastAPI entrypoint (:8001)
│   ├── sim_backend.py      # Timing model + concurrency queue
│   ├── kv_cache.py         # Block-level KV cache with LRU eviction
│   └── backend_interface.py
├── monitor/          # Load collector
│   ├── app.py              # FastAPI entrypoint (:8080)
│   └── collector.py        # Polls replicas, pushes load to router
└── bench/            # Benchmarking
    ├── runner.py           # Async benchmark runner
    ├── workloads.py        # Chat, RAG, agentic, mixed generators
    ├── baselines.py        # Round-robin, random, hash-based, least-loaded
    └── analysis.py         # Metrics (TTFT CDFs, cache hit rates) + plotting
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose

### Local Development

```bash
# Create virtualenv and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

### Docker

```bash
# Start all services (router + 4 replicas + monitor)
docker compose up -d

# Check health
curl http://localhost:8000/health

# Send a request through the router
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is Python?"}
    ],
    "max_tokens": 64
  }'

# Advisory mode (returns routing decision without forwarding)
curl -X POST http://localhost:8000/v1/route \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is Python?"}
    ]
  }'

# Stop services
docker compose down
```

### Benchmarking

```bash
# Run a benchmark (workload, num_requests, arrival_rate)
./scripts/run_benchmark.sh chat 200 20.0

# Available workloads: chat, rag, agentic, mixed
./scripts/run_benchmark.sh rag 200 15.0
./scripts/run_benchmark.sh mixed 300 15.0
```

Results are saved to `results/` as JSON files. The analysis module can generate TTFT CDF plots and cache hit rate comparisons.

## API Endpoints

### Router (`:8000`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Active routing — routes and proxies to best replica |
| POST | `/v1/route` | Advisory — returns `RoutingDecision` without forwarding |
| POST | `/v1/cache/update` | Receive cache insert/evict updates from replicas |
| POST | `/v1/load/update` | Receive load updates from monitor |
| GET | `/v1/status` | Router status (replicas, load) |
| GET | `/health` | Health check |

### Replica (`:8001`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Run simulated inference |
| GET | `/v1/status` | Replica status (load, cache utilization) |
| GET | `/v1/cache/blocks` | List all cached block hashes |

### Monitor (`:8080`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/status` | Latest collected status from all replicas |
| GET | `/health` | Health check |

## Configuration

All configuration is driven by environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ORBIT_ALPHA` | `1.0` | Weight for prefill savings in scoring |
| `ORBIT_BETA` | `0.5` | Weight for congestion penalty in scoring |
| `ORBIT_BLOCK_SIZE` | `16` | Tokens per KV-cache block |
| `ORBIT_CACHE_CAPACITY` | `1000` | Max blocks per replica cache |
| `ORBIT_MAX_CONCURRENT` | `4` | Max concurrent requests per replica |
| `ORBIT_PREFILL_MS` | `0.5` | Simulated prefill time per token (ms) |
| `ORBIT_DECODE_MS` | `10.0` | Simulated decode time per token (ms) |
| `ORBIT_TOKENIZER` | `cl100k_base` | Tokenizer (must match across all components) |
| `ORBIT_STRONG_HIT_THRESHOLD` | `0.9` | Cache overlap fraction to trigger strong-hit override |
| `ORBIT_POLL_INTERVAL` | `1.0` | Monitor polling interval (seconds) |

## Technology Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.11+ | LLM ecosystem standard |
| Framework | FastAPI + uvicorn | Async-native, OpenAI-compatible API |
| Semantic similarity | `all-MiniLM-L6-v2` (384-dim) | Fast on CPU, used only as pre-filter |
| Prefix hashing | SHA-256 chained 16-token blocks | Matches vLLM conventions, collision-resistant |
| Tokenizer | tiktoken `cl100k_base` | Consistent across router + replicas |
| Orchestration | Docker Compose | Single command brings up the full system |
| Testing | pytest + pytest-asyncio | 53 tests covering unit, integration, and e2e |

## Tests

```bash
pytest tests/ -v
```

- `test_hashing.py` — Block hash chaining, prefix stability, determinism
- `test_prefix_trie.py` — Trie insert/lookup/remove, multi-replica tracking
- `test_scorer.py` — Scoring arithmetic, load penalties, strong-hit override
- `test_routing_engine.py` — Cache affinity, fallback behavior, convergence
- `test_sim_backend.py` — KV cache LRU, simulated timing, concurrency limits
- `test_integration.py` — End-to-end replica requests, cache hit verification
