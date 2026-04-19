from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .calibration import fit_router_config
from .cluster import ClusterConfig
from .llamacpp import LlamaCppClusterConfig
from .policies import POLICIES
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


DEFAULT_POLICIES = ("summary", "random", "load_only", "exact_prefix", "oracle")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Orbit policy benchmarks and export per-request traces",
    )
    parser.add_argument("--backend", choices=("synthetic", "llama_cpp"), default="synthetic")
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
    parser.add_argument("--requests", type=int, default=200, help="number of requests")
    parser.add_argument(
        "--topology-mode",
        choices=("all_to_all", "sparse_overlap"),
        default="all_to_all",
        help="router-to-cluster connectivity model",
    )
    parser.add_argument(
        "--reachable-clusters-per-router",
        type=int,
        help="number of clusters each router can reach in sparse_overlap mode",
    )
    parser.add_argument(
        "--continuation-token-cap",
        type=int,
        help="optional cap applied to generated continuation token budgets",
    )
    parser.add_argument(
        "--cache-capacity",
        type=int,
        default=256,
        help="maximum number of cached prefixes retained per cluster",
    )
    parser.add_argument(
        "--cache-token-capacity",
        type=int,
        help="maximum total cached prefix tokens retained per cluster",
    )
    parser.add_argument(
        "--workload-kind",
        choices=("synthetic", "mixed_realistic"),
        default="synthetic",
        help="workload generator to use",
    )
    parser.add_argument(
        "--sharegpt-path",
        help="path to a ShareGPT-style JSON or JSONL dataset for chat traffic in mixed_realistic mode",
    )
    parser.add_argument(
        "--rag-path",
        help="path to a RAG-style JSON or JSONL dataset for retrieval traffic in mixed_realistic mode",
    )
    parser.add_argument(
        "--agent-path",
        help="path to an agent/tool-use JSON or JSONL dataset for agent traffic in mixed_realistic mode (for example ToolBench, BFCL, or tau-bench-style traces)",
    )
    parser.add_argument(
        "--sharegpt-sample-limit",
        type=int,
        default=2000,
        help="maximum number of conversations to load from the ShareGPT dataset",
    )
    parser.add_argument("--rag-sample-limit", type=int, default=2000, help="maximum number of external RAG examples to load")
    parser.add_argument("--agent-sample-limit", type=int, default=2000, help="maximum number of external agent examples to load")
    parser.add_argument("--traffic-mix-chat", type=float, default=0.35, help="weight for ShareGPT-style chat traffic")
    parser.add_argument("--traffic-mix-rag", type=float, default=0.25, help="weight for RAG traffic")
    parser.add_argument("--traffic-mix-agent", type=float, default=0.20, help="weight for agent/tool traffic")
    parser.add_argument("--traffic-mix-bursty", type=float, default=0.20, help="weight for bursty session traffic")
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help="number of initial requests to use for warm-up and exclude from reported metrics",
    )
    parser.add_argument(
        "--validation-requests",
        type=int,
        default=0,
        help="number of held-out requests used to select between base and calibrated router configs before final test evaluation",
    )
    parser.add_argument(
        "--calibrate-router",
        action="store_true",
        help="fit router latency coefficients from the warm-up requests before evaluating policies",
    )
    parser.add_argument(
        "--calibration-policy",
        choices=tuple(POLICIES),
        default="summary",
        help="policy used to collect warm-up traces for router calibration",
    )
    parser.add_argument("--routers", type=int, default=2, help="number of routers")
    parser.add_argument("--clusters", type=int, default=3, help="number of clusters")
    parser.add_argument("--seed", type=int, default=7, help="workload seed")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
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
    parser.add_argument("--model", help="GGUF model path for the llama.cpp backend")
    parser.add_argument("--llama-executable", default="llama-server", help="path to llama-server")
    parser.add_argument("--llama-port-base", type=int, default=8081, help="first port for cluster servers")
    parser.add_argument("--llama-threads", type=int, default=4, help="threads passed to llama-server")
    parser.add_argument("--llama-ctx-size", type=int, default=4096, help="context size passed to llama-server")
    parser.add_argument("--llama-parallel", type=int, default=1, help="server slots per cluster")
    parser.add_argument("--llama-timeout", type=float, default=120.0, help="request timeout in seconds")
    parser.add_argument("--llama-startup-timeout", type=float, default=120.0, help="server startup timeout in seconds")
    parser.add_argument(
        "--live-arrival-scale",
        type=float,
        help="scale factor applied to workload arrival times and control-plane timers in live llama.cpp mode",
    )
    parser.add_argument("--summary-delay", type=float, default=0.0, help="fault injection delay applied to cluster summary delivery")
    parser.add_argument("--gossip-delay", type=float, default=0.0, help="fault injection delay applied to router gossip delivery")
    parser.add_argument("--summary-drop-probability", type=float, default=0.0, help="probability of dropping a cluster summary update")
    parser.add_argument("--gossip-drop-probability", type=float, default=0.0, help="probability of dropping a router gossip update")
    parser.add_argument("--failed-clusters", nargs="*", default=[], help="cluster ids to mark unavailable during the injected outage window")
    parser.add_argument("--failure-start", type=float, default=0.0, help="start time of the injected cluster outage window")
    parser.add_argument("--failure-duration", type=float, default=0.0, help="duration of the injected cluster outage window")
    parser.add_argument("--retry-penalty", type=float, default=0.0, help="additional delay before rerouting after an injected cluster failure")
    parser.add_argument(
        "--validation-p95-regression-tolerance",
        type=float,
        default=0.05,
        help="maximum relative p95 TTFT/latency regression allowed during validation canary selection",
    )
    parser.add_argument("--llama-extra-arg", action="append", default=[], help="extra argument forwarded to llama-server")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.backend == "llama_cpp" and not args.model:
        parser.error("--model is required when --backend llama_cpp is used")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must be non-negative")
    if args.validation_requests < 0:
        parser.error("--validation-requests must be non-negative")
    if args.warmup_requests + args.validation_requests >= args.requests:
        parser.error("--warmup-requests plus --validation-requests must be smaller than --requests")
    if args.calibrate_router and args.warmup_requests <= 0:
        parser.error("--calibrate-router requires --warmup-requests to be greater than zero")
    if args.live_arrival_scale is not None and args.live_arrival_scale <= 0:
        parser.error("--live-arrival-scale must be positive")
    if args.continuation_token_cap is not None and args.continuation_token_cap <= 0:
        parser.error("--continuation-token-cap must be positive")
    if args.reachable_clusters_per_router is not None and args.reachable_clusters_per_router <= 0:
        parser.error("--reachable-clusters-per-router must be positive")
    if args.validation_p95_regression_tolerance < 0:
        parser.error("--validation-p95-regression-tolerance must be non-negative")
    for probability in (args.summary_drop_probability, args.gossip_drop_probability):
        if not 0.0 <= probability <= 1.0:
            parser.error("drop probabilities must be between 0 and 1")
    if args.summary_delay < 0 or args.gossip_delay < 0 or args.failure_duration < 0 or args.retry_penalty < 0:
        parser.error("fault injection delays and durations must be non-negative")

    seeds = tuple(args.seeds) if args.seeds else (args.seed,)
    output_dir = resolve_output_dir(args.output_dir)
    policies = tuple(args.policies)
    cache_token_capacity = resolve_cache_token_capacity(args)

    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "backend": args.backend,
        "policies": list(policies),
        "request_count": args.requests,
        "warmup_requests": args.warmup_requests,
        "validation_requests": args.validation_requests,
        "router_count": args.routers,
        "cluster_count": args.clusters,
        "topology_mode": args.topology_mode,
        "reachable_clusters_per_router": resolve_reachable_clusters_per_router(args),
        "continuation_token_cap": args.continuation_token_cap,
        "calibrate_router": args.calibrate_router,
        "calibration_policy": args.calibration_policy,
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
        "calibration_scope": "per_cluster_shadow_canary" if args.calibrate_router else "disabled",
        "validation_p95_regression_tolerance": args.validation_p95_regression_tolerance,
        "prefix_token_source": "llama_cpp" if args.backend == "llama_cpp" else "synthetic_lexical",
        "live_arrival_scale": resolve_live_arrival_scale(args),
    }
    write_json(output_dir / "manifest.json", manifest)

    metrics_by_seed: dict[int, dict[str, object]] = {}
    calibration_by_seed: dict[int, dict[str, object]] = {}
    selection_by_seed: dict[int, dict[str, object]] = {}
    summary_run_rows: list[dict[str, object]] = []
    traffic_run_rows: list[dict[str, object]] = []
    source_run_rows: list[dict[str, object]] = []

    for seed in seeds:
        config = build_simulation_config(args, seed=seed)
        requests = generate_workload(config.workload)
        if args.backend == "llama_cpp":
            prepare_simulation = Simulation(config)
            try:
                requests = prepare_simulation.prepare_requests(requests)
            finally:
                prepare_simulation.close()
        warmup_requests, validation_requests, test_requests = split_workload(
            requests,
            args.warmup_requests,
            args.validation_requests,
        )
        run_dir = output_dir if len(seeds) == 1 else output_dir / f"seed-{seed}"

        write_json(run_dir / "workload.json", requests_as_dicts(requests))
        if warmup_requests:
            write_json(run_dir / "warmup_workload.json", requests_as_dicts(warmup_requests))
        if validation_requests:
            write_json(run_dir / "validation_workload.json", requests_as_dicts(validation_requests))
        write_json(run_dir / "measured_workload.json", requests_as_dicts(test_requests))
        write_json(run_dir / "test_workload.json", requests_as_dicts(test_requests))

        calibrated_config = config
        calibration_payload: dict[str, object] | None = None
        if args.calibrate_router:
            calibration_simulation = Simulation(config)
            try:
                calibration_records, _ = calibration_simulation.run(
                    policy_name=args.calibration_policy,
                    requests=warmup_requests,
                    close_on_finish=True,
                )
            finally:
                calibration_simulation.close()
            calibrated_router_config, calibration = fit_router_config(
                calibration_records,
                config.router_config,
                source_policy=args.calibration_policy,
                cluster_specific=True,
            )
            calibrated_config = replace(config, router_config=calibrated_router_config)
            calibration_payload = asdict(calibration)
            calibration_by_seed[seed] = calibration_payload
            write_json(run_dir / "calibration.json", calibration_payload)

        selected_config = calibrated_config
        selection_payload: dict[str, object] | None = None
        if validation_requests:
            selected_config, selection_payload = select_config_by_validation(
                base_config=config,
                calibrated_config=calibrated_config,
                calibration_policy=args.calibration_policy,
                warmup_requests=warmup_requests,
                validation_requests=validation_requests,
                p95_regression_tolerance=args.validation_p95_regression_tolerance,
            )
            selection_by_seed[seed] = selection_payload
            write_json(run_dir / "selection.json", selection_payload)

        metrics_by_policy = {}
        for policy_name in policies:
            simulation = Simulation(selected_config)
            if warmup_requests:
                simulation.run(
                    policy_name=policy_name,
                    requests=warmup_requests,
                    close_on_finish=False,
                )
            records, metrics = simulation.run(
                policy_name=policy_name,
                requests=test_requests,
                close_on_finish=True,
            )
            metrics_by_policy[policy_name] = metrics
            summary_run_rows.append(
                {"seed": seed, **flatten_metrics_row(metrics_as_dict(metrics))}
            )
            traffic_rows = summarize_execution_records(records, policy_name, group_field="traffic_class")
            for row in traffic_rows:
                traffic_run_rows.append({"seed": seed, **row})
            source_rows = summarize_execution_records(records, policy_name, group_field="source_id")
            for row in source_rows:
                source_run_rows.append({"seed": seed, **row})

            if args.record_format in ("json", "both"):
                write_json(
                    run_dir / f"{policy_name}_records.json",
                    execution_records_as_dicts(records),
                )
            if args.record_format in ("csv", "both"):
                write_execution_records_csv(run_dir / f"{policy_name}_records.csv", records)

        write_json(
            run_dir / "summary.json",
            {policy_name: metrics_as_dict(metrics) for policy_name, metrics in metrics_by_policy.items()},
        )
        write_rows_csv(run_dir / "summary.csv", metrics_rows_by_policy(metrics_by_policy))
        write_rows_csv(
            run_dir / "summary_by_traffic.csv",
            [
                row
                for row in traffic_run_rows
                if row["seed"] == seed
            ],
        )
        write_rows_csv(
            run_dir / "summary_by_source.csv",
            [
                row
                for row in source_run_rows
                if row["seed"] == seed
            ],
        )
        metrics_by_seed[seed] = {
            "simulation_config": asdict(selected_config),
            "metrics": {policy_name: metrics_as_dict(metrics) for policy_name, metrics in metrics_by_policy.items()},
        }

    if len(seeds) > 1:
        write_json(output_dir / "summary_runs.json", metrics_by_seed)
        write_rows_csv(output_dir / "summary_runs.csv", summary_run_rows)
        aggregate_rows = aggregate_summary_rows(summary_run_rows, group_keys=("policy",))
        write_json(output_dir / "summary_aggregate.json", aggregate_rows)
        write_rows_csv(output_dir / "summary_aggregate.csv", aggregate_rows)
        write_rows_csv(output_dir / "summary_by_traffic_runs.csv", traffic_run_rows)
        write_rows_csv(
            output_dir / "summary_by_traffic_aggregate.csv",
            aggregate_summary_rows(traffic_run_rows, group_keys=("policy", "traffic_class")),
        )
        write_rows_csv(output_dir / "summary_by_source_runs.csv", source_run_rows)
        write_rows_csv(
            output_dir / "summary_by_source_aggregate.csv",
            aggregate_summary_rows(source_run_rows, group_keys=("policy", "source_id")),
        )
        if calibration_by_seed:
            write_json(output_dir / "calibration_runs.json", calibration_by_seed)
        if selection_by_seed:
            write_json(output_dir / "selection_runs.json", selection_by_seed)

    print(f"wrote benchmark artifacts to {output_dir}")
    if len(seeds) == 1:
        metrics_payload = metrics_by_seed[seeds[0]]["metrics"]
        for policy_name in policies:
            metrics = metrics_payload[policy_name]
            print(
                f"{policy_name}: "
                f"ttft_p50={metrics['ttft_p50']:.3f} "
                f"ttft_p95={metrics['ttft_p95']:.3f} "
                f"latency_p50={metrics['latency_p50']:.3f} "
                f"latency_p95={metrics['latency_p95']:.3f}"
            )
    else:
        for row in aggregate_summary_rows(summary_run_rows):
            print(
                f"{row['policy']}: "
                f"ttft_p50_mean={row['ttft_p50_mean']:.3f} "
                f"ttft_p95_mean={row['ttft_p95_mean']:.3f} "
                f"latency_p50_mean={row['latency_p50_mean']:.3f} "
                f"latency_p95_mean={row['latency_p95_mean']:.3f} "
                f"runs={row['runs']}"
            )
    return 0


def build_simulation_config(args: argparse.Namespace, seed: int | None = None) -> SimulationConfig:
    router_ids = tuple(f"router-{index}" for index in range(args.routers))
    cluster_ids = tuple(f"cluster-{index}" for index in range(args.clusters))
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
        llama_cpp=(
            LlamaCppClusterConfig(
                model_path=args.model,
                executable=args.llama_executable,
                port_base=args.llama_port_base,
                threads=args.llama_threads,
                ctx_size=args.llama_ctx_size,
                parallel=args.llama_parallel,
                request_timeout=args.llama_timeout,
                startup_timeout=args.llama_startup_timeout,
                extra_args=tuple(args.llama_extra_arg),
            )
            if args.backend == "llama_cpp"
            else None
        ),
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
            continuation_token_range=_continuation_token_range(args.continuation_token_cap),
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
            seed=args.seed if seed is None else seed,
        ),
    )


def resolve_live_arrival_scale(args: argparse.Namespace) -> float:
    if args.live_arrival_scale is not None:
        return args.live_arrival_scale
    if args.backend == "llama_cpp":
        return 0.01
    return 1.0


def resolve_cache_token_capacity(args: argparse.Namespace) -> int | None:
    if args.cache_token_capacity is not None:
        return args.cache_token_capacity
    if args.workload_kind == "mixed_realistic":
        return 4096
    return None


def resolve_reachable_clusters_per_router(args: argparse.Namespace) -> int | None:
    if args.topology_mode != "sparse_overlap":
        return None
    if args.reachable_clusters_per_router is not None:
        return min(args.reachable_clusters_per_router, args.clusters)
    return min(args.clusters, max(2, -(-args.clusters // max(args.routers, 1)) + 1))


def _continuation_cap(cap: int | None) -> int:
    return cap if cap is not None else 96


def _continuation_floor(cap: int | None) -> int:
    return min(8, _continuation_cap(cap))


def _continuation_token_range(cap: int | None) -> tuple[int, int]:
    upper = min(24, _continuation_cap(cap))
    lower = min(8, upper)
    return (lower, upper)


def resolve_output_dir(output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).resolve()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return (Path.cwd() / "results" / f"benchmark-{timestamp}").resolve()


def split_workload(
    requests: Sequence,
    warmup_requests: int,
    validation_requests: int = 0,
) -> tuple[list, list, list]:
    requests = list(requests)
    warmup = requests[:warmup_requests]
    validation_start = warmup_requests
    validation_end = validation_start + validation_requests
    validation = requests[validation_start:validation_end]
    test = requests[validation_end:]
    return warmup, validation, test


def aggregate_summary_rows(
    rows: Sequence[dict[str, object]],
    group_keys: Sequence[str] = ("policy",),
) -> list[dict[str, object]]:
    if not rows:
        return []

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        group_value = tuple(row[key] for key in group_keys)
        grouped.setdefault(group_value, []).append(row)

    numeric_fields = [
        key
        for key in rows[0].keys()
        if key not in set(group_keys) | {"seed"}
        and all(isinstance(row.get(key), (int, float)) for row in rows)
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


def flatten_metrics_row(row: dict[str, object]) -> dict[str, object]:
    row = dict(row)
    cluster_counts = row.pop("cluster_request_counts", {})
    if isinstance(cluster_counts, dict):
        for cluster_id, count in cluster_counts.items():
            row[f"cluster_requests_{cluster_id}"] = count
    return row


def bootstrap_mean_confidence_interval(
    values: Sequence[float],
    confidence: float = 0.95,
    bootstrap_samples: int = 1000,
    seed: int = 17,
) -> tuple[float, float]:
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
    lower_index = int(((1.0 - confidence) / 2.0) * (bootstrap_samples - 1))
    upper_index = int((1.0 - (1.0 - confidence) / 2.0) * (bootstrap_samples - 1))
    return means[lower_index], means[upper_index]


def replay_policy(
    config: SimulationConfig,
    policy_name: str,
    warmup_requests: Sequence,
    eval_requests: Sequence,
) -> list:
    simulation = Simulation(config)
    try:
        if warmup_requests:
            simulation.run(policy_name=policy_name, requests=warmup_requests, close_on_finish=False)
        records, _ = simulation.run(policy_name=policy_name, requests=eval_requests, close_on_finish=True)
        return records
    finally:
        simulation.close()


def select_config_by_validation(
    base_config: SimulationConfig,
    calibrated_config: SimulationConfig,
    calibration_policy: str,
    warmup_requests: Sequence,
    validation_requests: Sequence,
    p95_regression_tolerance: float,
) -> tuple[SimulationConfig, dict[str, object]]:
    base_records = replay_policy(base_config, calibration_policy, warmup_requests, validation_requests)
    base_validation_error = prediction_error_summary(base_records)
    base_validation_metrics = validation_metrics_summary(base_records)

    payload: dict[str, object] = {
        "selected_config": "base",
        "selection_metric": "validation_prediction_mae_with_p95_guardrail",
        "p95_regression_tolerance": p95_regression_tolerance,
        "base_validation_error": base_validation_error,
        "base_validation_metrics": base_validation_metrics,
    }

    cluster_overrides = dict(calibrated_config.router_config.cluster_overrides)
    if not cluster_overrides:
        calibrated_records = replay_policy(calibrated_config, calibration_policy, warmup_requests, validation_requests)
        calibrated_validation_error = prediction_error_summary(calibrated_records)
        calibrated_validation_metrics = validation_metrics_summary(calibrated_records)
        accepted, reasons = validation_candidate_accepted(
            base_validation_error,
            base_validation_metrics,
            calibrated_validation_error,
            calibrated_validation_metrics,
            p95_regression_tolerance,
        )
        payload["calibrated_validation_error"] = calibrated_validation_error
        payload["calibrated_validation_metrics"] = calibrated_validation_metrics
        payload["calibrated_rejection_reasons"] = reasons
        if accepted:
            payload["selected_config"] = "calibrated"
            return calibrated_config, payload
        return base_config, payload

    cluster_shadow_results: dict[str, object] = {}
    accepted_cluster_overrides: dict[str, dict[str, float]] = {}
    for cluster_id, override in sorted(cluster_overrides.items()):
        cluster_candidate = replace(
            base_config,
            router_config=replace(base_config.router_config, cluster_overrides={cluster_id: override}),
        )
        cluster_records = replay_policy(cluster_candidate, calibration_policy, warmup_requests, validation_requests)
        cluster_error = prediction_error_summary(cluster_records)
        cluster_metrics = validation_metrics_summary(cluster_records)
        accepted, reasons = validation_candidate_accepted(
            base_validation_error,
            base_validation_metrics,
            cluster_error,
            cluster_metrics,
            p95_regression_tolerance,
        )
        cluster_shadow_results[cluster_id] = {
            "accepted": accepted,
            "validation_error": cluster_error,
            "validation_metrics": cluster_metrics,
            "rejection_reasons": reasons,
        }
        if accepted:
            accepted_cluster_overrides[cluster_id] = override

    payload["cluster_shadow_results"] = cluster_shadow_results
    payload["accepted_clusters"] = sorted(accepted_cluster_overrides)
    payload["rejected_clusters"] = sorted(cluster_id for cluster_id in cluster_overrides if cluster_id not in accepted_cluster_overrides)

    if not accepted_cluster_overrides:
        payload["canary_validation_error"] = base_validation_error
        payload["canary_validation_metrics"] = base_validation_metrics
        payload["canary_rejection_reasons"] = ["no_cluster_overrides_passed"]
        return base_config, payload

    canary_config = replace(
        base_config,
        router_config=replace(base_config.router_config, cluster_overrides=accepted_cluster_overrides),
    )
    canary_records = replay_policy(canary_config, calibration_policy, warmup_requests, validation_requests)
    canary_validation_error = prediction_error_summary(canary_records)
    canary_validation_metrics = validation_metrics_summary(canary_records)
    accepted, reasons = validation_candidate_accepted(
        base_validation_error,
        base_validation_metrics,
        canary_validation_error,
        canary_validation_metrics,
        p95_regression_tolerance,
    )
    payload["canary_validation_error"] = canary_validation_error
    payload["canary_validation_metrics"] = canary_validation_metrics
    payload["canary_rejection_reasons"] = reasons
    if accepted:
        payload["selected_config"] = "canary"
        return canary_config, payload
    return base_config, payload


def prediction_error_summary(records: Sequence) -> dict[str, float]:
    if not records:
        return {"mae": 0.0, "rmse": 0.0}
    errors = [record.predicted_latency - record.actual_latency for record in records]
    mae = statistics.fmean(abs(error) for error in errors)
    rmse = (statistics.fmean(error * error for error in errors)) ** 0.5
    return {"mae": mae, "rmse": rmse}


def validation_metrics_summary(records: Sequence) -> dict[str, object]:
    if not records:
        return {
            "ttft_p50": 0.0,
            "ttft_p95": 0.0,
            "latency_p50": 0.0,
            "latency_p95": 0.0,
            "per_cluster": {},
        }
    ttfts = sorted(record.actual_ttft for record in records)
    latencies = sorted(record.actual_latency for record in records)
    per_cluster: dict[str, dict[str, float]] = {}
    for cluster_id in sorted({record.cluster_id for record in records}):
        cluster_records = [record for record in records if record.cluster_id == cluster_id]
        per_cluster[cluster_id] = {
            "request_count": float(len(cluster_records)),
            "ttft_p95": _percentile([record.actual_ttft for record in cluster_records], 0.95),
            "latency_p95": _percentile([record.actual_latency for record in cluster_records], 0.95),
        }
    return {
        "ttft_p50": _percentile(ttfts, 0.50),
        "ttft_p95": _percentile(ttfts, 0.95),
        "latency_p50": _percentile(latencies, 0.50),
        "latency_p95": _percentile(latencies, 0.95),
        "per_cluster": per_cluster,
    }


def validation_candidate_accepted(
    base_error: dict[str, float],
    base_metrics: dict[str, object],
    candidate_error: dict[str, float],
    candidate_metrics: dict[str, object],
    p95_regression_tolerance: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if candidate_error["mae"] >= base_error["mae"]:
        reasons.append("prediction_mae_not_improved")
    if _regressed(float(base_metrics["latency_p95"]), float(candidate_metrics["latency_p95"]), p95_regression_tolerance):
        reasons.append("latency_p95_regressed")
    if _regressed(float(base_metrics["ttft_p95"]), float(candidate_metrics["ttft_p95"]), p95_regression_tolerance):
        reasons.append("ttft_p95_regressed")

    base_clusters = base_metrics.get("per_cluster", {})
    candidate_clusters = candidate_metrics.get("per_cluster", {})
    if isinstance(base_clusters, dict) and isinstance(candidate_clusters, dict):
        for cluster_id in sorted(set(base_clusters) & set(candidate_clusters)):
            base_cluster_metrics = base_clusters[cluster_id]
            candidate_cluster_metrics = candidate_clusters[cluster_id]
            if _regressed(
                float(base_cluster_metrics["latency_p95"]),
                float(candidate_cluster_metrics["latency_p95"]),
                p95_regression_tolerance,
            ):
                reasons.append(f"cluster_{cluster_id}_latency_p95_regressed")
    return len(reasons) == 0, reasons


def _regressed(base_value: float, candidate_value: float, tolerance: float) -> bool:
    if base_value <= 0.0:
        return candidate_value > base_value
    return candidate_value > base_value * (1.0 + tolerance)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


if __name__ == "__main__":
    raise SystemExit(main())
