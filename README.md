# Orbit

Orbit is a cluster-level, prefix-aware router for multi-cluster LLM serving. It routes each request to a reachable cluster by estimating where the largest reusable leading prompt prefix is likely cached, then combining that prefix-reuse estimate with visible load, summary staleness, and router-to-cluster network cost.

The repository is configured around one reproducible benchmark: mixed external chat/RAG/agent traffic replayed through a live `llama.cpp` backend. The maintained routing policies are `orbit`, `least_loaded`, `random`, and `round_robin`.

## Quick Start

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[plots,dev]"
```

Install `llama.cpp` so `llama-server` is on `PATH`.

```bash
brew install llama.cpp
llama-server --help
```

Download the default GGUF model into the expected path:

```bash
mkdir -p models
curl -L \
  -o models/qwen2.5-3b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
```

Run the default benchmark and generate plots:

```bash
bash scripts/run_benchmark_live.sh --output-dir results/orbit-default
```

The script runs `scripts/benchmark_policies.py` and then `scripts/visualize_run.py`. Outputs are written under the selected `results/` directory.

## Required Inputs

### Model

The default benchmark expects:

```text
models/qwen2.5-3b-instruct-q4_k_m.gguf
```

This is the Qwen2.5 3B Instruct Q4_K_M GGUF model from Hugging Face. The file is about 2.1 GB. You can also use another GGUF model by calling the Python runner directly:

```bash
python3 scripts/benchmark_policies.py \
  --model /absolute/path/to/model.gguf \
  --output-dir results/orbit-custom-model
```

### External Datasets

The default mixed workload expects these local dataset files:

```text
results/external-datasets-20260418/sharegpt_x_chat.json
results/external-datasets-20260418/ragbench_hotpotqa.json
results/external-datasets-20260418/toolbench_g123_query.json
```

These represent the three traffic classes used in the final evaluation:

- `sharegpt_chat`: multi-turn chat traffic
- `rag`: retrieval-augmented generation prompts
- `agent`: tool-use / agentic prompts

If those files are not present, either restore them into the paths above or pass replacement dataset paths to `scripts/benchmark_policies.py` with `--sharegpt-path`, `--rag-path`, and `--agent-path`.

## Default Benchmark Configuration

`scripts/benchmark_policies.py`, `python -m orbit`, and `scripts/run_benchmark_live.sh` default to:

- Backend: `llama_cpp`
- Model: `models/qwen2.5-3b-instruct-q4_k_m.gguf`
- Workload: mixed external chat/RAG/agent traffic
- Traffic mix: `43.75%` chat, `31.25%` RAG, `25%` agent, `0%` standalone bursty traffic
- Topology: `4` routers, `6` clusters, sparse overlap, `3` reachable clusters per router
- Requests per seed: `24` warmup, `120` measured
- Seeds: `7 11 17 23 29`
- Policies: `orbit`, `least_loaded`, `random`, `round_robin`
- Cache token capacity: `4096`
- Live arrival scale: `0.01`

Warmup requests are replayed before each measured policy run so caches and summaries are populated. There is no calibration or validation-selection phase in the current code path.

## Running Benchmarks

Run the full default live benchmark:

```bash
bash scripts/run_benchmark_live.sh --output-dir results/orbit-default
```

Run the Python entrypoint directly:

```bash
python3 scripts/benchmark_policies.py --output-dir results/orbit-default
```

Run via the package entrypoint:

```bash
python3 -m orbit --output-dir results/orbit-default
```

Run a short synthetic smoke test that does not require `llama-server` or a GGUF model:

```bash
python3 scripts/benchmark_policies.py \
  --backend synthetic \
  --requests 16 \
  --warmup-requests 4 \
  --seeds 7 \
  --output-dir results/orbit-smoke
```

The public help intentionally shows only the main reproducibility knobs:

```bash
python3 scripts/benchmark_policies.py --help
```

Most experiment parameters still exist for internal sensitivity checks, but the default configuration is the expected path for reproducing the report figures.

## Generated Artifacts

Each run directory contains:

- `manifest.json`: exact benchmark settings
- `workload.json`: full generated workload after any backend tokenization
- `warmup_workload.json`: cache warmup requests
- `measured_workload.json` and `test_workload.json`: measured requests
- `<policy>_records.json` and/or `<policy>_records.csv`: per-request traces
- `summary.json` and `summary.csv`: per-policy metrics for a single-seed run
- `summary_by_traffic.csv`: metrics grouped by traffic class
- `summary_by_source.csv`: metrics grouped by source dataset/session

For multi-seed runs, the top-level output directory also contains:

- `summary_runs.json` and `summary_runs.csv`
- `summary_aggregate.json` and `summary_aggregate.csv`
- `summary_by_traffic_runs.csv`
- `summary_by_traffic_aggregate.csv`
- `summary_by_source_runs.csv`
- `summary_by_source_aggregate.csv`

## Plotting

Generate or refresh PNG plots for a benchmark directory:

```bash
python3 scripts/visualize_run.py results/orbit-default
```

Plots are written to `plots/` inside the run directory. For multi-seed runs, the visualizer also refreshes each `seed-*` subdirectory.

Curated figures include:

- `orbit_01_ttft_p50.png`
- `orbit_02_reuse_fraction.png`
- `orbit_03_reusable_prefix_by_traffic.png`
- `orbit_04_latency_vs_reuse.png`
- `orbit_combined.png`
- `ttft_cdf.png`
- `latency_cdf.png`
- `ttft_by_policy.png`
- `latency_by_policy.png`
- `reuse_latency_tradeoff.png`
- `latency_by_traffic.png`
- `reuse_by_traffic.png`

To collect plots for a report, copy the generated files from:

```text
results/<run-name>/plots/
results/<run-name>/seed-*/plots/
```

## Testing

Run the unit suite:

```bash
python3 -m unittest discover -s tests -v
```

Or use the project script:

```bash
bash scripts/run_ci_checks.sh --unit-only
```

The live llama.cpp integration test is opt-in because it starts local model servers:

```bash
ORBIT_RUN_LIVE_TESTS=1 \
ORBIT_LIVE_TEST_MODEL=models/qwen2.5-3b-instruct-q4_k_m.gguf \
python3 -m unittest tests.test_live_llamacpp_integration -v
```

## How Orbit Works

Clusters maintain exact local prefix state in a trie. Periodically, each cluster publishes a compact hierarchical summary containing Bloom filters and short-prefix hotsets. Routers receive these summaries directly or through gossip and route only among clusters reachable under the sparse topology.

Orbit estimates reusable prefix length from summary matches, then evaluates a TTFT-heavy route cost:

```text
TTFT = RTT + b0 + bq*q + bp*p + ba*a + bu*u + bm*1[missing_summary]
Latency = TTFT + bd*c
RouteCost = TTFT + w*(Latency - TTFT)
```

Where:

- `RTT` is router-to-cluster network cost.
- `q` is visible queue depth from the latest summary.
- `p` is remaining prefill tokens after estimated prefix reuse.
- `a` is summary age.
- `u` is prefix-reuse uncertainty from summary granularity.
- `c` is continuation-token budget.

The router only takes a prefix-aware route when the summary evidence is fresh and strong enough to beat the least-loaded fallback by a configured margin.

## Project Layout

- `orbit/router.py`: Orbit route scoring, prefix-reuse estimation, and least-loaded fallback
- `orbit/cluster.py`: synthetic cluster cache, trie-backed prefix state, and summary publication
- `orbit/llamacpp.py`: live `llama-server` process management, tokenization, and TTFT measurement
- `orbit/simulation.py`: in-process and live replay simulation loop
- `orbit/policies.py`: four maintained policies: `orbit`, `least_loaded`, `random`, `round_robin`
- `orbit/workload.py`: synthetic and mixed external workload generation
- `orbit/benchmark.py`: benchmark CLI, artifact export, multi-seed aggregation
- `orbit/png_plots.py`: seaborn/matplotlib PNG figure generation
- `orbit/reporting.py`: CSV/JSON serialization and grouped summaries
- `orbit/bloom.py`: Bloom filter implementation for summaries
- `orbit/hashing.py`: prefix hashing and hotset helpers
- `orbit/trie.py`: exact prefix trie used by clusters
- `orbit/models.py`: request, decision, record, and metric dataclasses
- `orbit/process_plane.py`: multiprocessing proxies for control-plane isolation tests
- `scripts/benchmark_policies.py`: thin CLI wrapper for `orbit.benchmark`
- `scripts/run_benchmark_live.sh`: default live benchmark plus plot generation
- `scripts/visualize_run.py`: refresh PNG plots for an existing run
- `scripts/run_ci_checks.sh`: unit and optional live integration test runner

## Reference Result

The latest completed live benchmark in this workspace is:

```text
results/mixed-external-renamed-baselines-large-v7-20260420
```

Headline aggregate metrics from that run:

- Orbit: `ttft_p50_mean=0.561`, `latency_p50_mean=2.509`, `mean_reusable_prefix_mean=100.08`
- Least-Loaded: `3.175`, `5.011`, `77.62`
- Random: `3.349`, `5.618`, `75.22`
- Round-Robin: `8.832`, `11.868`, `60.57`

The main result is that approximate hierarchical prefix summaries can improve both TTFT and prefix reuse over standard practical baselines on sparse multi-cluster LLM traffic, without requiring exact global KV-cache state.
