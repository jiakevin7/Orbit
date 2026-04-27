import argparse
import random
import statistics
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from .cluster import ClusterConfig
from .defaults import (
    DEFAULT_AGENT_PATH,
    DEFAULT_BACKEND,
    DEFAULT_CACHE_CAPACITY,
    DEFAULT_CACHE_TOKEN_CAPACITY,
    DEFAULT_CLUSTER_COUNT,
    DEFAULT_LLAMA_CTX_SIZE,
    DEFAULT_LLAMA_EXECUTABLE,
    DEFAULT_LLAMA_PARALLEL,
    DEFAULT_LLAMA_PORT_BASE,
    DEFAULT_LLAMA_STARTUP_TIMEOUT,
    DEFAULT_LLAMA_THREADS,
    DEFAULT_LLAMA_TIMEOUT,
    DEFAULT_LIVE_ARRIVAL_SCALE,
    DEFAULT_MODEL_PATH,
    DEFAULT_POLICIES,
    DEFAULT_RAG_PATH,
    DEFAULT_REACHABLE_CLUSTERS_PER_ROUTER,
    DEFAULT_ROUTER_COUNT,
    DEFAULT_SEEDS,
    DEFAULT_SHAREGPT_PATH,
    DEFAULT_TOPOLOGY_MODE,
    DEFAULT_TOTAL_REQUESTS,
    DEFAULT_TRAFFIC_MIX_AGENT,
    DEFAULT_TRAFFIC_MIX_BURSTY,
    DEFAULT_TRAFFIC_MIX_CHAT,
    DEFAULT_TRAFFIC_MIX_RAG,
    DEFAULT_WARMUP_REQUESTS,
    DEFAULT_WORKLOAD_KIND,
)
from .llamacpp import LlamaCppClusterConfig
from .reporting import (
    execution_records_as_dicts,
    metrics_as_dict,
    metrics_rows_by_policy,
    requests_as_dicts,
    summarize_execution_records,
    write_execution_records_csv,
    write_json,
    write_rows_csv,
)
from .simulation import FaultInjectionConfig, Simulation, SimulationConfig
from .router import RouterConfig
from .workload import WorkloadConfig, generate_workload


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run Orbit policy benchmarks and export per-request traces"
    )
    parser.add_argument(
        "--backend", choices=("synthetic", "llama_cpp"), default=DEFAULT_BACKEND
    )
    parser.add_argument(
        "--control-plane-mode",
        choices=("inprocess", "multiprocess"),
        default="inprocess",
        help="execution mode for router and synthetic cluster state",
    )
    parser.add_argument(
        "--control-plane-start-method",
        default="spawn",
        help="multiprocessing start method for multiprocess control-plane mode",
    )
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES))
    parser.add_argument(
        "--requests",
        type=int,
        default=DEFAULT_TOTAL_REQUESTS,
        help="number of requests",
    )
    parser.add_argument(
        "--topology-mode",
        choices=("all_to_all", "sparse_overlap"),
        default=DEFAULT_TOPOLOGY_MODE,
        help="router-to-cluster connectivity model",
    )
    parser.add_argument(
        "--reachable-clusters-per-router",
        type=int,
        default=DEFAULT_REACHABLE_CLUSTERS_PER_ROUTER,
        help="number of clusters each router can reach in sparse_overlap mode",
    )
    parser.add_argument(
        "--continuation-token-cap",
        type=int,
        help="optional cap applied to generated continuation token budgets",
    )
    parser.add_argument(
        "--prompt-prefix-token-cap",
        type=int,
        help="optional cap applied to generated prompt prefixes before backend tokenization",
    )
    parser.add_argument(
        "--cache-capacity",
        type=int,
        default=DEFAULT_CACHE_CAPACITY,
        help="maximum number of cached prefixes retained per cluster",
    )
    parser.add_argument(
        "--cache-token-capacity", type=int, default=DEFAULT_CACHE_TOKEN_CAPACITY
    )
    parser.add_argument(
        "--workload-kind",
        choices=("synthetic", "mixed_realistic"),
        default=DEFAULT_WORKLOAD_KIND,
        help="workload generator to use",
    )
    parser.add_argument(
        "--sharegpt-path",
        default=str(DEFAULT_SHAREGPT_PATH),
        help="path to a ShareGPT-style JSON or JSONL dataset for chat traffic in mixed_realistic mode",
    )
    parser.add_argument(
        "--rag-path",
        default=str(DEFAULT_RAG_PATH),
        help="path to a RAG-style JSON or JSONL dataset for retrieval traffic in mixed_realistic mode",
    )
    parser.add_argument(
        "--agent-path",
        default=str(DEFAULT_AGENT_PATH),
        help="path to an agent/tool-use JSON or JSONL dataset for agent traffic in mixed_realistic mode",
    )
    parser.add_argument(
        "--sharegpt-sample-limit",
        type=int,
        default=2000,
        help="maximum number of conversations to load from the ShareGPT dataset",
    )
    parser.add_argument(
        "--rag-sample-limit",
        type=int,
        default=2000,
        help="maximum number of external RAG examples to load",
    )
    parser.add_argument(
        "--agent-sample-limit",
        type=int,
        default=2000,
        help="maximum number of external agent examples to load",
    )
    parser.add_argument(
        "--traffic-mix-chat",
        type=float,
        default=DEFAULT_TRAFFIC_MIX_CHAT,
        help="weight for ShareGPT-style chat traffic",
    )
    parser.add_argument(
        "--traffic-mix-rag",
        type=float,
        default=DEFAULT_TRAFFIC_MIX_RAG,
        help="weight for RAG traffic",
    )
    parser.add_argument(
        "--traffic-mix-agent",
        type=float,
        default=DEFAULT_TRAFFIC_MIX_AGENT,
        help="weight for agent/tool traffic",
    )
    parser.add_argument(
        "--traffic-mix-bursty",
        type=float,
        default=DEFAULT_TRAFFIC_MIX_BURSTY,
        help="weight for bursty session traffic",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=DEFAULT_WARMUP_REQUESTS,
        help="number of initial requests to use for warm-up and exclude from reported metrics",
    )
    parser.add_argument(
        "--routers", type=int, default=DEFAULT_ROUTER_COUNT, help="number of routers"
    )
    parser.add_argument(
        "--clusters", type=int, default=DEFAULT_CLUSTER_COUNT, help="number of clusters"
    )
    parser.add_argument("--seed", type=int, default=7, help="workload seed")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="explicit list of workload seeds; overrides --seed",
    )
    parser.add_argument(
        "--record-format",
        choices=("json", "csv", "both"),
        default="both",
        help="per-policy record export format",
    )
    parser.add_argument(
        "--output-dir",
        help="directory for output artifacts, defaults to results/benchmark-<timestamp>",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="GGUF model path for the llama.cpp backend",
    )
    parser.add_argument(
        "--llama-executable",
        default=DEFAULT_LLAMA_EXECUTABLE,
        help="path to llama-server",
    )
    parser.add_argument(
        "--llama-port-base",
        type=int,
        default=DEFAULT_LLAMA_PORT_BASE,
        help="first port for cluster servers",
    )
    parser.add_argument(
        "--llama-threads",
        type=int,
        default=DEFAULT_LLAMA_THREADS,
        help="threads passed to llama-server",
    )
    parser.add_argument(
        "--llama-ctx-size",
        type=int,
        default=DEFAULT_LLAMA_CTX_SIZE,
        help="context size passed to llama-server",
    )
    parser.add_argument(
        "--llama-parallel",
        type=int,
        default=DEFAULT_LLAMA_PARALLEL,
        help="server slots per cluster",
    )
    parser.add_argument(
        "--llama-timeout",
        type=float,
        default=DEFAULT_LLAMA_TIMEOUT,
        help="request timeout in seconds",
    )
    parser.add_argument(
        "--llama-startup-timeout",
        type=float,
        default=DEFAULT_LLAMA_STARTUP_TIMEOUT,
        help="server startup timeout in seconds",
    )
    parser.add_argument(
        "--live-arrival-scale",
        type=float,
        help="scale factor applied to workload arrival times and control-plane timers in live llama.cpp mode",
    )
    parser.add_argument(
        "--summary-delay",
        type=float,
        default=0.0,
        help="fault injection delay applied to cluster summary delivery",
    )
    parser.add_argument(
        "--gossip-delay",
        type=float,
        default=0.0,
        help="fault injection delay applied to router gossip delivery",
    )
    parser.add_argument(
        "--summary-drop-probability",
        type=float,
        default=0.0,
        help="probability of dropping a cluster summary update",
    )
    parser.add_argument(
        "--gossip-drop-probability",
        type=float,
        default=0.0,
        help="probability of dropping a router gossip update",
    )
    parser.add_argument(
        "--failed-clusters",
        nargs="*",
        default=[],
        help="cluster ids to mark unavailable during the injected outage window",
    )
    parser.add_argument(
        "--failure-start",
        type=float,
        default=0.0,
        help="start time of the injected cluster outage window",
    )
    parser.add_argument(
        "--failure-duration",
        type=float,
        default=0.0,
        help="duration of the injected cluster outage window",
    )
    parser.add_argument(
        "--retry-penalty",
        type=float,
        default=0.0,
        help="additional delay before rerouting after an injected cluster failure",
    )
    parser.add_argument(
        "--llama-extra-arg",
        action="append",
        default=[],
        help="extra argument forwarded to llama-server",
    )
    _hide_nondefault_options(parser)
    return parser


def _hide_nondefault_options(parser):
    # Keep the public CLI focused on the reproducible default run while leaving
    # internal knobs available for tests and sensitivity checks.
    visible = {"help", "output_dir", "model", "llama_port_base"}
    for action in parser._actions:
        if action.dest not in visible:
            action.help = argparse.SUPPRESS


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate arguments before creating result directories or starting servers.
    if args.backend == "llama_cpp" and (not args.model):
        parser.error("--model is required when --backend llama_cpp is used")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must be non-negative")
    if args.warmup_requests >= args.requests:
        parser.error("--warmup-requests must be smaller than --requests")
    if args.live_arrival_scale is not None and args.live_arrival_scale <= 0:
        parser.error("--live-arrival-scale must be positive")
    if args.continuation_token_cap is not None and args.continuation_token_cap <= 0:
        parser.error("--continuation-token-cap must be positive")
    if (
        args.reachable_clusters_per_router is not None
        and args.reachable_clusters_per_router <= 0
    ):
        parser.error("--reachable-clusters-per-router must be positive")
    for probability in (args.summary_drop_probability, args.gossip_drop_probability):
        if not 0.0 <= probability <= 1.0:
            parser.error("drop probabilities must be between 0 and 1")
    if (
        args.summary_delay < 0
        or args.gossip_delay < 0
        or args.failure_duration < 0
        or (args.retry_penalty < 0)
    ):
        parser.error("fault injection delays and durations must be non-negative")

    # Resolve run-wide settings once so every seed uses the same policy set,
    # output root, and effective cache capacity.
    seeds = tuple(args.seeds) if args.seeds else (args.seed,)
    output_dir = resolve_output_dir(args.output_dir)
    policies = tuple(args.policies)
    cache_token_capacity = resolve_cache_token_capacity(args)

    # The manifest is the reproducibility contract for a results directory.
    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "backend": args.backend,
        "policies": list(policies),
        "request_count": args.requests,
        "warmup_requests": args.warmup_requests,
        "router_count": args.routers,
        "cluster_count": args.clusters,
        "topology_mode": args.topology_mode,
        "reachable_clusters_per_router": resolve_reachable_clusters_per_router(args),
        "continuation_token_cap": args.continuation_token_cap,
        "seeds": list(seeds),
        "record_format": args.record_format,
        "workload_kind": args.workload_kind,
        "sharegpt_path": args.sharegpt_path,
        "rag_path": args.rag_path,
        "agent_path": args.agent_path,
        "cache_capacity": args.cache_capacity,
        "cache_token_capacity": cache_token_capacity,
        "faults": {
            "summary_delay": args.summary_delay,
            "gossip_delay": args.gossip_delay,
            "summary_drop_probability": args.summary_drop_probability,
            "gossip_drop_probability": args.gossip_drop_probability,
            "failed_clusters": list(args.failed_clusters),
            "failure_start": args.failure_start,
            "failure_duration": args.failure_duration,
            "retry_penalty": args.retry_penalty,
        },
        "control_plane_mode": args.control_plane_mode,
        "control_plane_start_method": args.control_plane_start_method,
        "prefix_token_source": "llama_cpp"
        if args.backend == "llama_cpp"
        else "synthetic_lexical",
        "live_arrival_scale": resolve_live_arrival_scale(args),
    }
    write_json(output_dir / "manifest.json", manifest)

    # Accumulate per-seed tables here, then write both per-seed and cross-seed
    # artifacts after the final policy replays finish.
    metrics_by_seed: dict[int, dict[str, object]] = {}
    summary_run_rows: list[dict[str, object]] = []
    traffic_run_rows: list[dict[str, object]] = []
    source_run_rows: list[dict[str, object]] = []
    seed_runs: list[dict[str, object]] = []

    # Materialize all workloads before evaluation. For llama.cpp, this also
    # performs backend tokenization so every policy sees real token ids.
    for seed in seeds:
        config = build_simulation_config(args, seed=seed)
        simulation_allocation_index = 0

        def allocate_config(base_config):
            nonlocal simulation_allocation_index
            allocated = allocate_llama_cpp_ports(
                base_config, simulation_allocation_index
            )
            simulation_allocation_index += 1
            return allocated

        requests = generate_workload(config.workload)
        if args.backend == "llama_cpp":
            prepare_simulation = Simulation(allocate_config(config))
            try:
                requests = prepare_simulation.prepare_requests(requests)
            finally:
                prepare_simulation.close()

        warmup_requests, test_requests = split_workload(requests, args.warmup_requests)
        run_dir = output_dir if len(seeds) == 1 else output_dir / f"seed-{seed}"

        # Persist every split so the exact benchmark input is auditable.
        write_json(run_dir / "workload.json", requests_as_dicts(requests))
        if warmup_requests:
            write_json(
                run_dir / "warmup_workload.json", requests_as_dicts(warmup_requests)
            )
        write_json(run_dir / "measured_workload.json", requests_as_dicts(test_requests))
        write_json(run_dir / "test_workload.json", requests_as_dicts(test_requests))
        seed_runs.append(
            {
                "seed": seed,
                "config": config,
                "allocate_config": allocate_config,
                "run_dir": run_dir,
                "warmup_requests": warmup_requests,
                "test_requests": test_requests,
            }
        )

    # Final evaluation: replay warm-up to rebuild cache state, then measure only
    # the test split for every policy under the fixed router configuration.
    for seed_run in seed_runs:
        seed = seed_run["seed"]
        metrics_by_policy = {}
        for policy_name in policies:
            simulation = Simulation(seed_run["allocate_config"](seed_run["config"]))
            if seed_run["warmup_requests"]:
                simulation.run(
                    policy_name=policy_name,
                    requests=seed_run["warmup_requests"],
                    close_on_finish=False,
                )
            records, metrics = simulation.run(
                policy_name=policy_name,
                requests=seed_run["test_requests"],
                close_on_finish=True,
            )
            metrics_by_policy[policy_name] = metrics
            summary_run_rows.append(
                {"seed": seed, **flatten_metrics_row(metrics_as_dict(metrics))}
            )
            traffic_rows = summarize_execution_records(
                records, policy_name, group_field="traffic_class"
            )
            for row in traffic_rows:
                traffic_run_rows.append({"seed": seed, **row})
            source_rows = summarize_execution_records(
                records, policy_name, group_field="source_id"
            )
            for row in source_rows:
                source_run_rows.append({"seed": seed, **row})
            if args.record_format in ("json", "both"):
                write_json(
                    seed_run["run_dir"] / f"{policy_name}_records.json",
                    execution_records_as_dicts(records),
                )
            if args.record_format in ("csv", "both"):
                write_execution_records_csv(
                    seed_run["run_dir"] / f"{policy_name}_records.csv", records
                )

        # Per-seed summaries are small enough to inspect directly and feed the
        # plotting script without loading every per-request trace.
        write_json(
            seed_run["run_dir"] / "summary.json",
            {
                policy_name: metrics_as_dict(metrics)
                for policy_name, metrics in metrics_by_policy.items()
            },
        )
        write_rows_csv(
            seed_run["run_dir"] / "summary.csv",
            metrics_rows_by_policy(metrics_by_policy),
        )
        write_rows_csv(
            seed_run["run_dir"] / "summary_by_traffic.csv",
            [row for row in traffic_run_rows if row["seed"] == seed],
        )
        write_rows_csv(
            seed_run["run_dir"] / "summary_by_source.csv",
            [row for row in source_run_rows if row["seed"] == seed],
        )
        metrics_by_seed[seed] = {
            "simulation_config": asdict(seed_run["config"]),
            "metrics": {
                policy_name: metrics_as_dict(metrics)
                for policy_name, metrics in metrics_by_policy.items()
            },
        }

    # Cross-seed aggregation includes bootstrap CIs for the paper figures.
    if len(seeds) > 1:
        write_json(output_dir / "summary_runs.json", metrics_by_seed)
        write_rows_csv(output_dir / "summary_runs.csv", summary_run_rows)
        aggregate_rows = aggregate_summary_rows(
            summary_run_rows, group_keys=("policy",)
        )
        write_json(output_dir / "summary_aggregate.json", aggregate_rows)
        write_rows_csv(output_dir / "summary_aggregate.csv", aggregate_rows)
        write_rows_csv(output_dir / "summary_by_traffic_runs.csv", traffic_run_rows)
        write_rows_csv(
            output_dir / "summary_by_traffic_aggregate.csv",
            aggregate_summary_rows(
                traffic_run_rows, group_keys=("policy", "traffic_class")
            ),
        )
        write_rows_csv(output_dir / "summary_by_source_runs.csv", source_run_rows)
        write_rows_csv(
            output_dir / "summary_by_source_aggregate.csv",
            aggregate_summary_rows(source_run_rows, group_keys=("policy", "source_id")),
        )

    # Keep CLI output concise: artifact path plus the headline latency metrics.
    print(f"wrote benchmark artifacts to {output_dir}")
    if len(seeds) == 1:
        metrics_payload = metrics_by_seed[seeds[0]]["metrics"]
        for policy_name in policies:
            metrics = metrics_payload[policy_name]
            print(
                f"{policy_name}: ttft_p50={metrics['ttft_p50']:.3f} ttft_p95={metrics['ttft_p95']:.3f} latency_p50={metrics['latency_p50']:.3f} latency_p95={metrics['latency_p95']:.3f}"
            )
    else:
        for row in aggregate_summary_rows(summary_run_rows):
            print(
                f"{row['policy']}: ttft_p50_mean={row['ttft_p50_mean']:.3f} ttft_p95_mean={row['ttft_p95_mean']:.3f} latency_p50_mean={row['latency_p50_mean']:.3f} latency_p95_mean={row['latency_p95_mean']:.3f} runs={row['runs']}"
            )
    return 0


def build_simulation_config(args, seed=None):
    router_ids = tuple((f"router-{index}" for index in range(args.routers)))
    cluster_ids = tuple((f"cluster-{index}" for index in range(args.clusters)))
    cache_token_capacity = resolve_cache_token_capacity(args)
    return SimulationConfig(
        backend=args.backend,
        control_plane_mode=args.control_plane_mode,
        control_plane_start_method=args.control_plane_start_method,
        router_ids=router_ids,
        cluster_ids=cluster_ids,
        topology_mode=args.topology_mode,
        reachable_clusters_per_router=resolve_reachable_clusters_per_router(args),
        cluster_config=ClusterConfig(
            cache_capacity=args.cache_capacity,
            cache_capacity_tokens=cache_token_capacity,
        ),
        llama_cpp=LlamaCppClusterConfig(
            model_path=args.model,
            executable=args.llama_executable,
            port_base=args.llama_port_base,
            threads=args.llama_threads,
            ctx_size=args.llama_ctx_size,
            parallel=args.llama_parallel,
            request_timeout=args.llama_timeout,
            startup_timeout=args.llama_startup_timeout,
            prompt_token_cap=resolve_prompt_prefix_token_cap(args),
            extra_args=tuple(args.llama_extra_arg),
        )
        if args.backend == "llama_cpp"
        else None,
        router_config=RouterConfig(),
        live_arrival_scale=resolve_live_arrival_scale(args),
        faults=FaultInjectionConfig(
            summary_delay=args.summary_delay,
            gossip_delay=args.gossip_delay,
            summary_drop_probability=args.summary_drop_probability,
            gossip_drop_probability=args.gossip_drop_probability,
            failed_cluster_ids=tuple(args.failed_clusters),
            failure_start=args.failure_start,
            failure_duration=args.failure_duration,
            retry_penalty=args.retry_penalty,
        ),
        workload=WorkloadConfig(
            num_requests=args.requests,
            router_ids=router_ids,
            continuation_token_range=_continuation_token_range(
                args.continuation_token_cap
            ),
            workload_kind=args.workload_kind,
            sharegpt_path=args.sharegpt_path,
            sharegpt_sample_limit=args.sharegpt_sample_limit,
            rag_path=args.rag_path,
            rag_sample_limit=args.rag_sample_limit,
            agent_path=args.agent_path,
            agent_sample_limit=args.agent_sample_limit,
            traffic_mix_chat=args.traffic_mix_chat,
            traffic_mix_rag=args.traffic_mix_rag,
            traffic_mix_agent=args.traffic_mix_agent,
            traffic_mix_bursty=args.traffic_mix_bursty,
            dataset_continuation_floor=_continuation_floor(args.continuation_token_cap),
            dataset_continuation_cap=_continuation_cap(args.continuation_token_cap),
            prompt_prefix_token_cap=resolve_prompt_prefix_token_cap(args),
            seed=args.seed if seed is None else seed,
        ),
    )


def resolve_live_arrival_scale(args):
    if args.live_arrival_scale is not None:
        return args.live_arrival_scale
    if args.backend == "llama_cpp":
        return DEFAULT_LIVE_ARRIVAL_SCALE
    return 1.0


def resolve_cache_token_capacity(args):
    if args.cache_token_capacity is not None:
        return args.cache_token_capacity
    if args.workload_kind == "mixed_realistic":
        return DEFAULT_CACHE_TOKEN_CAPACITY
    return None


def resolve_prompt_prefix_token_cap(args):
    explicit_cap = getattr(args, "prompt_prefix_token_cap", None)
    if explicit_cap is not None:
        return explicit_cap
    if args.backend != "llama_cpp":
        return WorkloadConfig.prompt_prefix_token_cap
    continuation_cap = _continuation_cap(args.continuation_token_cap)
    reserved_tokens = max(512, continuation_cap + 128)
    return max(256, args.llama_ctx_size - reserved_tokens)


def resolve_reachable_clusters_per_router(args):
    if args.topology_mode != "sparse_overlap":
        return None
    if args.reachable_clusters_per_router is not None:
        return min(args.reachable_clusters_per_router, args.clusters)
    return min(args.clusters, max(2, -(-args.clusters // max(args.routers, 1)) + 1))


def _continuation_cap(cap):
    return cap if cap is not None else 96


def _continuation_floor(cap):
    return min(8, _continuation_cap(cap))


def _continuation_token_range(cap):
    upper = min(24, _continuation_cap(cap))
    lower = min(8, upper)
    return (lower, upper)


def allocate_llama_cpp_ports(config, allocation_index):
    if config.backend != "llama_cpp" or config.llama_cpp is None:
        return config
    stride = max(len(config.cluster_ids) + 2, 16)
    return replace(
        config,
        llama_cpp=replace(
            config.llama_cpp,
            port_base=config.llama_cpp.port_base + allocation_index * stride,
        ),
    )


def resolve_output_dir(output_dir):
    if output_dir:
        return Path(output_dir).resolve()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return (Path.cwd() / "results" / f"benchmark-{timestamp}").resolve()


def split_workload(requests, warmup_requests):
    requests = list(requests)
    warmup = requests[:warmup_requests]
    test = requests[warmup_requests:]
    return (warmup, test)


def aggregate_summary_rows(rows, group_keys=("policy",)):
    if not rows:
        return []
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        group_value = tuple((row[key] for key in group_keys))
        grouped.setdefault(group_value, []).append(row)
    numeric_fields = [
        key
        for key in rows[0].keys()
        if key not in set(group_keys) | {"seed"}
        and all((isinstance(row.get(key), (int, float)) for row in rows))
    ]
    aggregates: list[dict[str, object]] = []
    for group_value in sorted(grouped):
        policy_rows = grouped[group_value]
        aggregate = {key: value for key, value in zip(group_keys, group_value)}
        aggregate["runs"] = len(policy_rows)
        for field in numeric_fields:
            values = [float(row[field]) for row in policy_rows]
            aggregate[f"{field}_mean"] = statistics.fmean(values)
            ci_low, ci_high = bootstrap_mean_confidence_interval(values)
            aggregate[f"{field}_ci_low"] = ci_low
            aggregate[f"{field}_ci_high"] = ci_high
        aggregates.append(aggregate)
    return aggregates


def flatten_metrics_row(row):
    row = dict(row)
    cluster_counts = row.pop("cluster_request_counts", {})
    if isinstance(cluster_counts, dict):
        for cluster_id, count in cluster_counts.items():
            row[f"cluster_requests_{cluster_id}"] = count
    return row


def bootstrap_mean_confidence_interval(
    values, confidence=0.95, bootstrap_samples=1000, seed=17
):
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(bootstrap_samples):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(statistics.fmean(sample))
    means.sort()
    lower_index = int((1.0 - confidence) / 2.0 * (bootstrap_samples - 1))
    upper_index = int((1.0 - (1.0 - confidence) / 2.0) * (bootstrap_samples - 1))
    return (means[lower_index], means[upper_index])


if __name__ == "__main__":
    raise SystemExit(main())
