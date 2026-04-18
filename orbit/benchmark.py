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
from .simulation import FaultInjectionConfig
from .router import RouterConfig
from .simulation import Simulation, SimulationConfig
from .workload import WorkloadConfig, generate_workload


DEFAULT_POLICIES = ("summary", "random", "load_only", "exact_prefix", "oracle")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Orbit policy benchmarks and export per-request traces",
    )
    parser.add_argument("--backend", choices=("synthetic", "llama_cpp"), default="synthetic")
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES))
    parser.add_argument("--requests", type=int, default=200, help="number of requests")
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
        choices=DEFAULT_POLICIES,
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
            )
            calibrated_config = replace(config, router_config=calibrated_router_config)
            calibration_payload = asdict(calibration)
            calibration_by_seed[seed] = calibration_payload
            write_json(run_dir / "calibration.json", calibration_payload)

        selected_config = calibrated_config
        selection_payload: dict[str, object] | None = None
        if validation_requests:
            base_validation_records = replay_policy(config, args.calibration_policy, warmup_requests, validation_requests)
            calibrated_validation_records = replay_policy(calibrated_config, args.calibration_policy, warmup_requests, validation_requests)
            base_validation_error = prediction_error_summary(base_validation_records)
            calibrated_validation_error = prediction_error_summary(calibrated_validation_records)
            use_calibrated = calibrated_validation_error["mae"] < base_validation_error["mae"]
            selected_config = calibrated_config if use_calibrated else config
            selection_payload = {
                "selected_config": "calibrated" if use_calibrated else "base",
                "selection_metric": "validation_prediction_mae",
                "base_validation_error": base_validation_error,
                "calibrated_validation_error": calibrated_validation_error,
            }
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
        router_ids=router_ids,
        cluster_ids=cluster_ids,
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


def prediction_error_summary(records: Sequence) -> dict[str, float]:
    if not records:
        return {"mae": 0.0, "rmse": 0.0}
    errors = [record.predicted_latency - record.actual_latency for record in records]
    mae = statistics.fmean(abs(error) for error in errors)
    rmse = (statistics.fmean(error * error for error in errors)) ** 0.5
    return {"mae": mae, "rmse": rmse}


if __name__ == "__main__":
    raise SystemExit(main())
