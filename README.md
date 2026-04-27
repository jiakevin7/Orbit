# Orbit

Orbit is a cluster-level prefix-aware router for multi-cluster LLM serving. It routes each request to a reachable cluster by estimating where the largest reusable leading prompt prefix is likely cached, then combining that estimate with load, staleness, and network cost.

The repository is centered on the final evaluation setup, so the default benchmark reproduces the research configuration without requiring a separate scenario config file.

## Final Default Benchmark

`scripts/benchmark_policies.py` and `python -m orbit` default to the poster evaluation:

- backend: `llama_cpp`
- model: `models/qwen2.5-3b-instruct-q4_k_m.gguf`
- workload: mixed external chat/RAG/agent traffic
- datasets:
  - `results/external-datasets-20260418/sharegpt_x_chat.json`
  - `results/external-datasets-20260418/ragbench_hotpotqa.json`
  - `results/external-datasets-20260418/toolbench_g123_query.json`
- traffic mix: `43.75%` chat, `31.25%` RAG, `25%` agent, `0%` standalone bursty traffic
- topology: `4` routers, `6` clusters, sparse overlap, `3` reachable clusters per router
- requests per seed: `24` warmup, `24` validation, `96` measured
- seeds: `7 11 17 23 29`
- policies: `summary`, `load_only`, `random`, `round_robin`
- cache token capacity: `4096`
- live arrival scale: `0.01`
- calibration: global warmup fit with held-out p95 guardrail

The benchmark writes raw per-request traces, aggregate summaries, grouped traffic/source summaries, calibration/selection artifacts, and PNG figures.

## Core Mechanics

- Clusters maintain exact local prefix state in a trie.
- Clusters export compact hierarchical summaries with Bloom filters and short-prefix hotsets.
- Routers keep stale-tolerant soft state from cluster summaries and router gossip.
- The Orbit router estimates reusable prefix length from summary matches.
- Routing minimizes a TTFT-heavy cost model:

```text
TTFT = RTT + b0 + bq*q + bp*p + ba*a + bu*u + bm*1[missing_summary]
Latency = TTFT + bd*c
RouteCost = TTFT + w*(Latency - TTFT)
```

Where `q` is visible queue depth, `p` is remaining prefill after estimated reuse, `a` is summary age, `u` is reuse-estimate uncertainty, and `c` is the continuation-token budget.

## Run

Run the default live benchmark:

```bash
bash scripts/run_benchmark_live.sh
```

Or call the Python entrypoint directly:

```bash
python3 scripts/benchmark_policies.py --output-dir results/orbit-default
```

The model file must exist at:

```text
models/qwen2.5-3b-instruct-q4_k_m.gguf
```

`llama-server` must be available on `PATH`. Running these scripts took roughly 1 hour on an M4 MacBook with 24 GBs of RAM.

## Plot

Generate or refresh the curated PNG plots for a run:

```bash
python3 scripts/visualize_run.py results/orbit-default
```

Curated figures include:

- `ttft_cdf.png`
- `latency_cdf.png`
- `ttft_by_policy.png`
- `latency_by_policy.png`
- `reuse_latency_tradeoff.png`
- `latency_by_traffic.png`
- `reuse_by_traffic.png`

## Important Results

The final corrected live benchmark is stored in:

```text
results/mixed-external-renamed-baselines-large-v7-20260420
```

Aggregate results:

- Orbit (`summary`): `ttft_p50_mean=0.561`, `latency_p50_mean=2.509`, `mean_reusable_prefix_mean=100.08`
- Least Loaded (`load_only`): `3.175`, `5.011`, `77.62`
- Random: `3.349`, `5.618`, `75.22`
- Round Robin: `8.832`, `11.868`, `60.57`

The main finding is that approximate hierarchical prefix summaries are sufficient to outperform standard practical baselines on realistic sparse multi-cluster LLM traffic, without requiring exact global KV-cache state.

## Active Layout

- `orbit/router.py`: summary-based reuse estimation and routing cost model
- `orbit/cluster.py`: local cache state, trie-backed ground truth, summary publication
- `orbit/bloom.py`: Bloom filter implementation
- `orbit/hashing.py`: prefix hashing and hotset helpers
- `orbit/workload.py`: final mixed chat/RAG/agent workload generation
- `orbit/llamacpp.py`: live `llama-server` management, tokenization, TTFT measurement
- `orbit/benchmark.py`: default benchmark runner and artifact export
- `orbit/simulation.py`: control-plane simulation and live replay
- `orbit/policies.py`: primary baselines
- `orbit/calibration.py`: global router calibration
- `orbit/png_plots.py`: curated seaborn/matplotlib figures

## Test

Run the unit suite:

```bash
python3 -m unittest discover -s tests -v
```

The live integration test is opt-in:

```bash
ORBIT_RUN_LIVE_TESTS=1 python3 -m unittest tests.test_live_llamacpp_integration -v
```
