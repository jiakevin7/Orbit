# Orbit Implementation Plan — Docker + TCP

## Context

Orbit is a semantic-aware KV-cache affinity routing layer for LLM serving (CSE 585 project). The original proposal planned RDMA over Thunderbolt 5 on Apple Silicon Macs. This plan adapts to **Docker containers with TCP networking** on a single machine, which means:

- No GPU passthrough in Docker on macOS — we use a **simulated LLM backend** that models prefill/decode timing and KV cache behavior realistically
- Standard Docker bridge networking replaces RDMA
- The simulated backend allows running 4+ replicas without needing massive GPU/RAM resources, while producing credible benchmarks

---

## Architecture Overview

```
                    ┌─────────────────┐
   Clients ──────▶  │  Orbit Router   │  (cache-affinity + load-aware routing)
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

All containers on a single Docker bridge network. REST/HTTP + JSON between all components.

---

## Technology Choices

| Decision | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | LLM ecosystem standard |
| Framework | FastAPI + uvicorn | Async-native, OpenAI-compatible API |
| Protocol | REST/HTTP + JSON | Simple, debuggable, sufficient at this scale |
| Semantic similarity | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) | Fast on CPU, used only as pre-filter |
| Prefix hashing | SHA-256 over chained 16-token blocks | Matches vLLM conventions, collision-resistant |
| Tokenizer | tiktoken `cl100k_base` | Consistent across router + replicas |
| Orchestration | docker-compose | Single `docker-compose.yml` brings up everything |
| Testing | pytest + pytest-asyncio | Standard |

---

## Project Structure

```
Orbit/
├── docker-compose.yml
├── Dockerfile.router
├── Dockerfile.replica
├── Dockerfile.monitor
├── Dockerfile.bench
├── pyproject.toml
├── requirements/
│   ├── base.txt          # fastapi, uvicorn, pydantic, httpx, tiktoken
│   ├── router.txt        # sentence-transformers, numpy
│   ├── replica.txt       # (sim backend deps)
│   └── bench.txt         # matplotlib, pandas
├── orbit/
│   ├── common/
│   │   ├── schemas.py    # Pydantic models (InferenceRequest, ReplicaStatus, RoutingDecision, etc.)
│   │   ├── hashing.py    # SHA-256 block hashing over token ID sequences
│   │   └── config.py     # Env-var-driven configuration
│   ├── router/
│   │   ├── app.py              # FastAPI router service
│   │   ├── routing_engine.py   # Core orchestration: parse → semantic filter → exact match → score → route
│   │   ├── prompt_analyzer.py  # Segment messages into system/tool/rag/user parts
│   │   ├── prefix_trie.py      # Radix trie tracking which block hashes live on which replicas
│   │   ├── semantic_index.py   # Embedding-based similarity search for candidate filtering
│   │   ├── cache_registry.py   # Central metadata: replica→prefixes, prefix→replicas
│   │   ├── scorer.py           # score = α·prefill_savings - β·congestion
│   │   └── load_tracker.py     # Per-replica load state
│   ├── replica/
│   │   ├── app.py              # FastAPI replica service
│   │   ├── backend_interface.py # ABC for LLM backends
│   │   ├── sim_backend.py      # Simulated LLM with timing model + KV cache
│   │   └── kv_cache.py         # Simulated block-level KV cache (LRU, prefix trie)
│   ├── monitor/
│   │   ├── app.py              # Load monitor service
│   │   └── collector.py        # Polls replicas, pushes to router
│   └── bench/
│       ├── workloads.py        # Generators: chat, RAG, agentic, mixed
│       ├── runner.py           # Async benchmark runner with timing
│       ├── baselines.py        # Round-robin, least-loaded, random, hash-based routers
│       └── analysis.py         # Metrics computation + plotting
├── tests/
│   ├── test_hashing.py
│   ├── test_prefix_trie.py
│   ├── test_scorer.py
│   ├── test_routing_engine.py
│   ├── test_sim_backend.py
│   └── test_integration.py
├── configs/
│   ├── default.yaml
│   └── bench_*.yaml            # Per-workload benchmark configs
└── scripts/
    └── run_benchmark.sh
```

---

## Core Algorithm: Two-Stage Routing

### Stage 1 — Semantic Pre-Filter (fast, approximate)
1. Parse prompt into segments (system prompt, tools, RAG context, user query)
2. Embed the stable segments (system prompt, tools) with MiniLM
3. Cosine similarity against embeddings of all cached prefix segments
4. Return top-k candidate replicas likely to hold relevant prefix state

### Stage 2 — Exact Match + Scoring (correct, decisive)
1. Tokenize full prompt → compute chained SHA-256 block hashes
2. For each candidate replica: look up prefix trie for longest exact block-hash match
3. Score each candidate: `score = α · (cached_tokens / total_tokens) - β · normalized_congestion`
4. Select highest-scoring replica (fallback: least-loaded if no cache overlap)

**Correctness guarantee**: Only exact token-level prefix matches count. The semantic layer only narrows the search; it never bypasses hash verification.

---

## Simulated Backend Timing Model

```
prefill_ms = 0.5ms × (total_prompt_tokens - cached_prefix_tokens)
decode_ms  = 10ms × output_tokens
queue_wait = modeled via FIFO queue with max_concurrent limit (default: 4)
```

The simulated KV cache uses:
- Fixed capacity (default: 1000 blocks of 16 tokens each)
- LRU eviction
- Chained block hashing consistent with the router

---

## Docker Architecture

| Container | Port | Role |
|---|---|---|
| `orbit-router` | 8000 | Orbit routing layer |
| `orbit-replica-{1..4}` | 8001-8004 | Simulated LLM replicas |
| `orbit-monitor` | 8080 | Load collector |
| `orbit-bench` | — | Benchmark runner (run-and-exit) |

All on `orbit-net` bridge network. Replicas push cache updates to router after each request. Monitor polls replicas every 1s and pushes load state to router.

---

## Implementation Phases

### Phase 1: Foundation (~3-4 days)
**Goal**: End-to-end skeleton with round-robin routing

1. `pyproject.toml`, `requirements/`, project scaffolding
2. `orbit/common/` — schemas, hashing, config
3. `orbit/replica/kv_cache.py` — simulated KV cache with LRU + prefix lookup
4. `orbit/replica/sim_backend.py` — timing model, queue simulation
5. `orbit/replica/app.py` — FastAPI endpoints
6. `orbit/router/app.py` — minimal round-robin proxy
7. `docker-compose.yml` + Dockerfiles — verify all containers start and round-trip works
8. Unit tests for hashing and kv_cache

**Milestone**: Send requests to router → get simulated responses from replicas

### Phase 2: Intelligent Routing (~5-6 days)
**Goal**: Full cache-aware routing (without semantic layer)

1. `prompt_analyzer.py` — segment messages, tokenize, compute block hashes
2. `prefix_trie.py` — radix trie with per-replica tracking
3. `cache_registry.py` — bidirectional index (replica↔prefix)
4. `scorer.py` — α/β scoring function
5. `routing_engine.py` — full pipeline: parse → hash → trie lookup → score → route
6. Wire replicas to push cache updates to router
7. `load_tracker.py` + monitor service
8. Unit + integration tests (verify repeated system prompts route to same replica, cache hits increase)

**Milestone**: Router makes cache-affinity decisions with load awareness

### Phase 3: Semantic Layer (~3-4 days)
**Goal**: Add embedding-based pre-filter

1. `semantic_index.py` — embed cached prefixes, cosine similarity search
2. Integrate MiniLM model loading (lazy, pre-downloaded in Docker image)
3. Wire into routing_engine as Stage 1 before exact matching
4. Tests: semantically similar but textually different prompts are correctly handled

**Milestone**: Full two-stage routing pipeline operational

### Phase 4: Benchmarking & Evaluation (~4-5 days)
**Goal**: Produce publishable benchmark results

1. Four workload generators (chat, RAG, agentic, mixed)
2. Async benchmark runner with timing instrumentation
3. Three baseline routers (round-robin, least-loaded, hash-based)
4. Analysis + plotting: TTFT CDFs, cache hit rates, prefill savings, goodput
5. Parameter sweeps: replica count (2/4/8), prefix reuse ratio, α/β values, arrival rates

**Milestone**: Quantitative evidence that Orbit improves TTFT and cache hit rate over baselines

---

## Key Design Decisions

1. **Cache update protocol**: Replicas push updates to router (new inserts + evictions) after each request. Router also does periodic full reconciliation pulls. Stale state is safe (affects perf, not correctness).

2. **Tokenizer consistency**: Router and all replicas use identical tokenizer (tiktoken cl100k_base). If hashes don't match, the system fails silently — this is a hard requirement.

3. **Block chaining**: Block N's hash = SHA256(block_{N-1}_hash + token_ids_in_block_N). This ensures a block hash uniquely identifies the entire prefix up to that point.

4. **Advisory vs Active mode**: `POST /v1/chat/completions` = active routing (transparent proxy). `POST /v1/route` = advisory (returns RoutingDecision without forwarding). Same routing engine code.

5. **Scoring fallbacks**: If all replicas have zero cache overlap → least-loaded. If one replica has >90% prefix overlap and load < 2× minimum → always choose it (strong hit override).

---

## Verification Plan

1. **Unit tests**: hashing correctness, trie operations, scorer arithmetic, cache LRU behavior
2. **Integration test**: Start router + 2 replicas, send 10 requests with same system prompt → verify requests converge to one replica, cache hit tokens increase monotonically
3. **Benchmark comparison**: Run chat workload with 90% prefix reuse through Orbit vs round-robin vs least-loaded → compare p50/p95 TTFT and cache hit rate
4. **Correctness check**: Send requests with similar (but not identical) system prompts → verify router does NOT claim cache hits (exact match only)
5. **Load balance check**: Burst 100 requests to one prefix → verify router spills to other replicas when cache-hot replica is overloaded
