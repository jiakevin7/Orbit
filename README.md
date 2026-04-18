# Orbit

Orbit is a minimal simulator for cluster-level prefix-aware routing across data centers.

The design follows three components:

- clients send tokenized requests to routers
- routers choose a target cluster using approximate prefix summaries, load, and network cost
- clusters execute requests, maintain exact local cache state, and export compact probabilistic summaries

## Implemented Changes

Since the initial fresh-start design, Orbit now includes:

- backend-aligned live execution through `llama.cpp`, including real TTFT measurement, `/tokenize`-aligned routing tokens, `/slots`-based load snapshots, prompt-progress cache visibility, and concurrent request replay
- realistic workload generation across synthetic, ShareGPT-style chat, RAG, agent/tool, and bursty session traffic, with dataset adapters for BFCL-, tau-bench-, ToolBench-, LMSYS-, and FinanceBench-like shapes
- evaluation tooling beyond raw policy comparison, including warm-up slices, held-out validation, router calibration, grouped traffic/source summaries, multi-seed aggregation, and bootstrap confidence intervals
- robustness testing hooks for stale or dropped control-plane updates, injected cluster outages, and failover accounting
- a run visualizer that turns each benchmark directory into a self-contained HTML report
- packaged bash entrypoints for compare runs, synthetic benchmarks, live benchmarks, visualization, and CI checks

The simulator is intentionally compact. It models:

- exact reusable-prefix ground truth with a trie per cluster
- coarse hierarchical summaries with Bloom filters at fixed prefix depths
- stale metadata via periodic summary publication and router gossip
- routing by predicted latency rather than a raw affinity score
- baseline policies including random, load-only, exact full-prefix, and oracle routing
- optional `llama.cpp`-backed cluster execution so TTFT and total service time can be measured directly
- live prompt token alignment through the `llama.cpp` `/tokenize` endpoint so routed prefixes match the backend tokenizer
- live slot polling through the `llama.cpp` `/slots` endpoint so exported load reflects observed busy slots when available
- real concurrent live replay so queue delay emerges from overlapping `llama.cpp` requests instead of a synthetic worker heap
- warm-up and measured phases so benchmarks can report steady-state behavior separately from cold-start effects
- optional held-out validation slices so base and calibrated routers can be compared before final test evaluation
- optional router calibration from warm-up traces so latency scoring can be fit from observed external workloads
- multi-seed benchmark aggregation for less fragile policy comparisons
- text-driven request generation so routed prefixes correspond to the executed prompt content
- mixed realistic workloads that combine ShareGPT-style chat, RAG, agent/tool, and bursty session traffic
- token-budget cache eviction so long prompts exert more realistic pressure than short prompts
- injected control-plane delay, dropped updates, and cluster outages for robustness testing

## Layout

- `orbit/trie.py`: exact prefix ground truth
- `orbit/bloom.py`: compact approximate membership structure
- `orbit/cluster.py`: cluster execution, cache state, and summary export
- `orbit/benchmark.py`: shared-workload policy runner and artifact export
- `orbit/llamacpp.py`: `llama-server` process management and streamed TTFT measurement
- `orbit/router.py`: router soft-state, reuse estimation, and latency prediction
- `orbit/workload.py`: synthetic and mixed realistic workloads, including ShareGPT, BFCL, tau-bench, and related dataset ingestion
- `orbit/calibration.py`: fit router latency coefficients from observed traces
- `orbit/visualizer.py`: generate self-contained HTML reports from benchmark artifacts
- `orbit/simulation.py`: control-plane events, request execution, and metrics
- `orbit/policies.py`: baseline routing policies
- `DATASETS.md`: recommended public datasets and how they map to Orbit traffic classes
- `ROADMAP.md`: production-realism upgrades in priority order

## Run

Compare all policies on the same workload:

```bash
python -m orbit --compare
```

Or use the bash wrapper:

```bash
bash scripts/run_compare.sh
```

Run a single policy:

```bash
python -m orbit --policy summary --requests 200 --routers 2 --clusters 3
```

Run against real `llama.cpp` cluster processes:

```bash
python -m orbit \
  --backend llama_cpp \
  --model /absolute/path/to/model.gguf \
  --policy summary \
  --requests 20 \
  --live-arrival-scale 0.01 \
  --clusters 2
```

In `llama.cpp` mode, Orbit retokenizes prompts through the backend `/tokenize` API before routing, so the trie, Bloom summaries, and router decisions all operate on the same token sequence the model executes. Cluster execution time, TTFT, prompt-progress cache visibility, and queue delay come from real streamed requests to one `llama-server` process per cluster. The benchmark driver replays arrivals concurrently in wall clock time; by default it applies `--live-arrival-scale 0.01` so realistic overlap appears without forcing multi-minute runs.

Export a full multi-policy benchmark with per-request traces:

```bash
python3 scripts/benchmark_policies.py \
  --backend llama_cpp \
  --model /absolute/path/to/model.gguf \
  --warmup-requests 10 \
  --requests 50 \
  --clusters 2 \
  --live-arrival-scale 0.01 \
  --output-dir results/llama-benchmark
```

Or use the packaged live benchmark wrapper:

```bash
bash scripts/run_benchmark_live.sh
```

Calibrate the router on warm-up traces before evaluating policies:

```bash
python3 scripts/benchmark_policies.py \
  --backend llama_cpp \
  --model /absolute/path/to/model.gguf \
  --workload-kind mixed_realistic \
  --sharegpt-path /absolute/path/to/sharegpt.json \
  --rag-path /absolute/path/to/rag.json \
  --agent-path /absolute/path/to/agent.json \
  --warmup-requests 10 \
  --requests 40 \
  --calibrate-router \
  --calibration-policy summary \
  --output-dir results/calibrated-benchmark
```

If the warm-up slice is too small, Orbit writes `calibration.json` with `applied=false` and keeps the base router coefficients.

Use a held-out validation slice to choose between the base and calibrated router before final evaluation:

```bash
python3 scripts/benchmark_policies.py \
  --backend synthetic \
  --workload-kind mixed_realistic \
  --warmup-requests 10 \
  --validation-requests 10 \
  --requests 60 \
  --calibrate-router \
  --output-dir results/heldout-benchmark
```

When `--validation-requests` is non-zero, Orbit replays both router configs on the validation slice, compares prediction MAE, and writes `selection.json` with the chosen config.

Run the mixed realistic workload with external chat, RAG, agent/tool, and bursty traffic:

```bash
python3 scripts/benchmark_policies.py \
  --backend synthetic \
  --workload-kind mixed_realistic \
  --sharegpt-path /absolute/path/to/sharegpt.json \
  --rag-path /absolute/path/to/rag.json \
  --agent-path /absolute/path/to/agent.json \
  --cache-token-capacity 4096 \
  --traffic-mix-chat 0.35 \
  --traffic-mix-rag 0.25 \
  --traffic-mix-agent 0.20 \
  --traffic-mix-bursty 0.20 \
  --requests 40 \
  --output-dir results/mixed-realistic
```

Or use the packaged synthetic benchmark wrapper:

```bash
bash scripts/run_benchmark_synthetic.sh
```

If `--sharegpt-path` is omitted, Orbit falls back to a small built-in chat corpus so the mixed workload mode remains runnable.
If `--rag-path` or `--agent-path` is omitted, Orbit falls back to built-in retrieval and tool-use prompt catalogs.
The agent loader now accepts ToolBench-style tool corpora as well as BFCL- and tau-bench-like trajectory records through the same `--agent-path` flag.
In mixed realistic mode, Orbit defaults `--cache-token-capacity` to `4096` unless you override it.
See `DATASETS.md` for recommended public sources, currently supported shapes, and which datasets are direct drop-ins versus adapter candidates.

Inject stale metadata and cluster failures to test robustness:

```bash
python3 scripts/benchmark_policies.py \
  --backend synthetic \
  --workload-kind mixed_realistic \
  --requests 40 \
  --warmup-requests 10 \
  --validation-requests 10 \
  --summary-delay 2.0 \
  --gossip-delay 2.0 \
  --summary-drop-probability 0.10 \
  --gossip-drop-probability 0.10 \
  --failed-clusters cluster-0 \
  --failure-start 8.0 \
  --failure-duration 6.0 \
  --retry-penalty 1.0 \
  --output-dir results/fault-injection
```

These knobs let you evaluate stale control-plane state, delayed or dropped summary propagation, and router failover behavior when clusters become unavailable mid-run.

## Visualizer

Generate a self-contained HTML report for any benchmark output directory:

```bash
python3 scripts/visualize_run.py results/live-smoke
```

or with the bash wrapper:

```bash
bash scripts/visualize_run.sh results/live-smoke
```

The visualizer writes `report.html` into the target directory and, when seaborn is installed, also renders PNG plots under `plots/`. Reports include policy summary bars, grouped traffic/source charts, per-request latency traces, scatter plots, reuse histograms, cluster/failover distributions, and embedded PNGs such as the TTFT CDF.

Generate only the seaborn PNG plots:

```bash
python3 scripts/generate_png_plots.py results/live-smoke
```

If seaborn is not installed, install the plotting extras first:

```bash
python3 -m pip install -e .[plots]
```

The benchmark runner writes:

- `manifest.json`: benchmark config and run metadata
- `calibration.json`: fitted router coefficients when `--calibrate-router` is used
- `selection.json`: held-out base-vs-calibrated selection details when `--validation-requests` is used
- `workload.json`: the full shared request stream used for all policies, including the final routed token sequence, scaled arrival times, traffic class, and session metadata
- `warmup_workload.json`: the warm-up subset when `--warmup-requests` is used
- `validation_workload.json`: the held-out validation subset when `--validation-requests` is used
- `measured_workload.json`: the measured subset used for reported metrics
- `<policy>_records.json` and/or `<policy>_records.csv`: per-request TTFT, latency, routing, and reuse outcomes
- `summary.json` and `summary.csv`: policy-level aggregates
- `summary_by_traffic.csv`: per-policy aggregates grouped by traffic class
- `summary_by_source.csv`: per-policy aggregates grouped by dataset or source id
- `report.html`: generated visualization for the run when you render it with the visualizer script
- `plots/*.png`: seaborn-generated PNG plots such as `ttft_cdf.png`, `latency_cdf.png`, and `predicted_vs_actual_latency.png`

Run multiple seeds and emit aggregate summaries:

```bash
python3 scripts/benchmark_policies.py \
  --backend synthetic \
  --requests 40 \
  --warmup-requests 10 \
  --seeds 7 11 17 \
  --output-dir results/multi-seed
```

When multiple seeds are provided, the runner creates one subdirectory per seed and writes root-level `summary_runs.*`, `summary_aggregate.*`, `summary_by_traffic_*`, and `summary_by_source_*` files. Aggregate CSVs include bootstrap confidence intervals for numeric metrics.

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

Run the opt-in live `llama.cpp` integration test:

```bash
ORBIT_RUN_LIVE_TESTS=1 \
python -m unittest tests.test_live_llamacpp_integration -v
```

By default the test looks for [models/qwen2.5-3b-instruct-q4_k_m.gguf](/Users/baseb/Documents/6th%20Year/CSE%20585/Orbit/models/qwen2.5-3b-instruct-q4_k_m.gguf). Override it with `ORBIT_LIVE_TEST_MODEL=/absolute/path/to/model.gguf` if you want to use a different local GGUF. The integration target runs a small real benchmark with held-out validation and an injected cluster outage, then checks the emitted artifacts and failover records.

Run the packaged CI targets locally:

```bash
bash scripts/run_ci_checks.sh --unit-only
bash scripts/run_ci_checks.sh --with-live
```

## Bash Scripts

- [scripts/run_compare.sh](/Users/baseb/Documents/6th%20Year/CSE%20585/Orbit/scripts/run_compare.sh): quick `python -m orbit --compare` wrapper
- [scripts/run_benchmark_synthetic.sh](/Users/baseb/Documents/6th%20Year/CSE%20585/Orbit/scripts/run_benchmark_synthetic.sh): synthetic or mixed-realistic benchmark run plus HTML report generation
- [scripts/run_benchmark_live.sh](/Users/baseb/Documents/6th%20Year/CSE%20585/Orbit/scripts/run_benchmark_live.sh): `llama.cpp` benchmark run plus HTML report generation
- [scripts/visualize_run.sh](/Users/baseb/Documents/6th%20Year/CSE%20585/Orbit/scripts/visualize_run.sh): wrapper for the run visualizer
- [scripts/generate_png_plots.sh](/Users/baseb/Documents/6th%20Year/CSE%20585/Orbit/scripts/generate_png_plots.sh): wrapper for generating seaborn PNG plots only
- [scripts/run_ci_checks.sh](/Users/baseb/Documents/6th%20Year/CSE%20585/Orbit/scripts/run_ci_checks.sh): unit-only or live CI-style verification entrypoint

## CI

Orbit now includes two GitHub Actions workflows:

- [.github/workflows/unit-tests.yml](/Users/baseb/Documents/6th%20Year/CSE%20585/Orbit/.github/workflows/unit-tests.yml): runs the standard unit suite on every push and pull request.
- [.github/workflows/live-llamacpp.yml](/Users/baseb/Documents/6th%20Year/CSE%20585/Orbit/.github/workflows/live-llamacpp.yml): manually runs the real `llama.cpp` integration test on a self-hosted macOS ARM64 runner labeled `orbit-live`.

The live workflow expects:

- `llama-server` installed on the self-hosted runner
- either a workflow input `model_path` or a repository variable `ORBIT_LIVE_TEST_MODEL`
- a runner with labels `self-hosted`, `macOS`, `ARM64`, and `orbit-live`

This split keeps normal PR CI fast while still giving you a real pre-merge path for the backend-aligned live test.
