# Orbit Roadmap

## Current State

The framework now supports:

- text-driven workloads instead of `tok_N` placeholder prompts
- mixed realistic workloads spanning ShareGPT-style chat, RAG, agent/tool, and bursty session traffic
- direct agent-path adapters for ToolBench-, BFCL-, and tau-bench-like records
- token-budget cache eviction so long prompts consume more cache than short prompts
- backend-aligned routing tokens via the live `llama.cpp` `/tokenize` endpoint
- live slot-derived queue-depth snapshots via the `llama.cpp` `/slots` endpoint
- prompt-progress driven cache visibility in live mode, so prefixes become reusable after prefill rather than after full decode
- real concurrent live replay, so measured queue delay comes from overlapping backend requests
- warm-up requests that populate cache state without affecting reported metrics
- held-out validation slices that choose between base and calibrated routers before final test evaluation
- router calibration from observed warm-up traces, so latency scoring no longer depends entirely on hand-set coefficients
- multi-seed benchmark runs with aggregate summaries
- grouped reporting by traffic class and dataset/source, with bootstrap confidence intervals on aggregate metrics
- injected control-plane delay, dropped updates, and cluster outages for robustness testing
- optional multiprocess control-plane execution, with routers and synthetic or live clusters running as separate worker processes
- cluster-specific router calibration with validation-time shadow and canary selection under p95 guardrails
- a standard external benchmark matrix with scenario manifests and root-level matrix summaries
- live `llama.cpp` execution for TTFT and service-time measurement
- an opt-in live `llama.cpp` integration test that checks held-out selection, artifact export, and failover under a real cluster outage
- GitHub Actions coverage for unit tests plus a dedicated self-hosted live `llama.cpp` workflow

## Next Up

1. Make the process-separated control plane harsher and more realistic.
   - Add transport-level jitter, restarts, and partitions between workers rather than only logical delivery delay.
   - Measure router/cluster worker recovery after process death.

2. Make the live workflow a stronger gate.
   - Promote the self-hosted `llama.cpp` workflow to a required pre-merge check where appropriate.
   - Extend it to validate TTFT scale and multi-policy live comparisons end to end.

3. Replace slot polling with richer backend telemetry when available.
   - Prefer task queue depth or scheduler occupancy over busy-slot counts alone.
   - Capture cancellation, timeout, and retry behavior in the live benchmark path.

4. Make router calibration online instead of warm-up only.
   - Refresh per-cluster latency surfaces continuously from recent traces.
   - Add longer-horizon rollback logic based on live canary windows, not only held-out replay.

5. Add stronger fault models and recovery metrics.
   - Inject jittery clocks, router restarts, partial network partitions, and summary corruption.
   - Track recovery time, misroute rate during faults, and post-failure load rebalancing quality.
