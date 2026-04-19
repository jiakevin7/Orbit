from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit.benchmark import build_simulation_config, resolve_output_dir, split_workload
from orbit.reporting import (
    execution_records_as_dicts,
    metrics_as_dict,
    requests_as_dicts,
    summarize_execution_records,
    write_execution_records_csv,
    write_json,
    write_rows_csv,
)
from orbit.related_work import (
    cap_request_continuations,
    external_metrics_as_dict,
    external_records_as_dicts,
    load_related_work_targets,
    run_external_target,
    scale_request_arrivals,
    summarize_external_records,
)
from orbit.simulation import Simulation
from orbit.workload import WorkloadConfig, generate_workload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Orbit directly against deployed related-work systems",
    )
    parser.add_argument("--target-config", help="JSON config describing external systems to benchmark")
    parser.add_argument("--systems", nargs="+", help="subset of target names from --target-config")
    parser.add_argument("--skip-orbit", action="store_true", help="skip local Orbit baselines")
    parser.add_argument("--orbit-policies", nargs="+", default=["summary"], help="Orbit policies to benchmark locally")
    parser.add_argument("--backend", choices=("synthetic", "llama_cpp"), default="llama_cpp")
    parser.add_argument("--control-plane-mode", choices=("inprocess", "multiprocess"), default="inprocess")
    parser.add_argument("--control-plane-start-method", default="spawn")
    parser.add_argument("--requests", type=int, default=64, help="number of measured requests")
    parser.add_argument("--warmup-requests", type=int, default=16, help="number of warmup requests")
    parser.add_argument("--routers", type=int, default=2)
    parser.add_argument("--clusters", type=int, default=2)
    parser.add_argument("--cache-capacity", type=int, default=256)
    parser.add_argument("--cache-token-capacity", type=int)
    parser.add_argument("--workload-kind", choices=("synthetic", "mixed_realistic"), default="mixed_realistic")
    parser.add_argument("--sharegpt-path")
    parser.add_argument("--rag-path")
    parser.add_argument("--agent-path")
    parser.add_argument("--sharegpt-sample-limit", type=int, default=2000)
    parser.add_argument("--rag-sample-limit", type=int, default=2000)
    parser.add_argument("--agent-sample-limit", type=int, default=2000)
    parser.add_argument("--traffic-mix-chat", type=float, default=0.35)
    parser.add_argument("--traffic-mix-rag", type=float, default=0.25)
    parser.add_argument("--traffic-mix-agent", type=float, default=0.20)
    parser.add_argument("--traffic-mix-bursty", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--arrival-scale",
        type=float,
        default=0.01,
        help="scale factor applied to workload arrivals for both Orbit live mode and external endpoints",
    )
    parser.add_argument(
        "--continuation-token-cap",
        type=int,
        help="optional shared cap applied to request continuation tokens for both Orbit and external systems",
    )
    parser.add_argument("--external-max-workers", type=int, default=16)
    parser.add_argument("--output-dir")
    parser.add_argument("--model", help="GGUF model path for Orbit local llama.cpp runs")
    parser.add_argument("--llama-executable", default="llama-server")
    parser.add_argument("--llama-port-base", type=int, default=19410)
    parser.add_argument("--llama-threads", type=int, default=4)
    parser.add_argument("--llama-ctx-size", type=int, default=4096)
    parser.add_argument("--llama-parallel", type=int, default=1)
    parser.add_argument("--llama-timeout", type=float, default=120.0)
    parser.add_argument("--llama-startup-timeout", type=float, default=120.0)
    parser.add_argument("--summary-delay", type=float, default=0.0)
    parser.add_argument("--gossip-delay", type=float, default=0.0)
    parser.add_argument("--summary-drop-probability", type=float, default=0.0)
    parser.add_argument("--gossip-drop-probability", type=float, default=0.0)
    parser.add_argument("--failed-clusters", nargs="*", default=[])
    parser.add_argument("--failure-start", type=float, default=0.0)
    parser.add_argument("--failure-duration", type=float, default=0.0)
    parser.add_argument("--retry-penalty", type=float, default=0.0)
    parser.add_argument("--llama-extra-arg", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.skip_orbit and not args.target_config:
        parser.error("at least one of Orbit or --target-config must be enabled")
    if not args.skip_orbit and args.backend == "llama_cpp" and not args.model:
        parser.error("--model is required for local Orbit llama.cpp comparisons")
    if args.arrival_scale <= 0:
        parser.error("--arrival-scale must be positive")

    total_requests = args.requests + args.warmup_requests
    workload_config = WorkloadConfig(
        num_requests=total_requests,
        router_ids=tuple(f"router-{index}" for index in range(args.routers)),
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
        seed=args.seed,
    )
    logical_requests = cap_request_continuations(
        generate_workload(workload_config),
        args.continuation_token_cap,
    )
    scheduled_requests = scale_request_arrivals(logical_requests, args.arrival_scale)
    logical_warmup, _, logical_measured = split_workload(logical_requests, args.warmup_requests, 0)
    scheduled_warmup, _, scheduled_measured = split_workload(scheduled_requests, args.warmup_requests, 0)

    output_dir = resolve_output_dir(args.output_dir or str(REPO_ROOT / "results" / f"related-work-{timestamp()}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "workload_kind": args.workload_kind,
        "seed": args.seed,
        "measured_requests": args.requests,
        "warmup_requests": args.warmup_requests,
        "arrival_scale": args.arrival_scale,
        "continuation_token_cap": args.continuation_token_cap,
        "orbit_enabled": not args.skip_orbit,
        "orbit_policies": list(args.orbit_policies),
        "target_config": args.target_config,
        "systems": list(args.systems or []),
        "backend": args.backend,
        "control_plane_mode": args.control_plane_mode,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "logical_workload.json", requests_as_dicts(logical_requests))
    write_json(output_dir / "scheduled_workload.json", requests_as_dicts(scheduled_requests))
    write_json(output_dir / "warmup_workload.json", requests_as_dicts(scheduled_warmup))
    write_json(output_dir / "measured_workload.json", requests_as_dicts(scheduled_measured))
    if args.target_config:
        shutil.copyfile(args.target_config, output_dir / "target_config.json")

    comparison_rows: list[dict[str, object]] = []

    if not args.skip_orbit:
        orbit_dir = output_dir / "orbit"
        orbit_rows = run_orbit_policies(args, logical_warmup, logical_measured, orbit_dir)
        comparison_rows.extend(orbit_rows)

    if args.target_config:
        target_dir = output_dir / "external"
        targets = list(load_related_work_targets(args.target_config))
        if args.systems:
            requested = set(args.systems)
            targets = [target for target in targets if target.name in requested]
        if not targets:
            parser.error("no external targets selected from --target-config")

        for target in targets:
            system_dir = target_dir / target.name
            records, metrics = run_external_target(
                target,
                warmup_requests=scheduled_warmup,
                measured_requests=scheduled_measured,
                max_workers=args.external_max_workers,
            )
            write_json(system_dir / "records.json", external_records_as_dicts(records))
            write_rows_csv(system_dir / "records.csv", external_records_as_dicts(records))
            write_json(system_dir / "summary.json", external_metrics_as_dict(metrics))
            write_rows_csv(system_dir / "summary.csv", [external_metrics_as_dict(metrics)])
            write_rows_csv(
                system_dir / "summary_by_traffic.csv",
                summarize_external_records(records, group_field="traffic_class"),
            )
            write_rows_csv(
                system_dir / "summary_by_source.csv",
                summarize_external_records(records, group_field="source_id"),
            )
            comparison_rows.append(_comparison_row_for_external(metrics))

    write_json(output_dir / "comparison_summary.json", comparison_rows)
    write_rows_csv(output_dir / "comparison_summary.csv", comparison_rows)
    print(f"wrote related-work benchmark artifacts to {output_dir}")
    for row in comparison_rows:
        print(
            f"{row['system']}: "
            f"ttft_p50={row['ttft_p50']:.3f} "
            f"ttft_p95={row['ttft_p95']:.3f} "
            f"latency_p50={row['latency_p50']:.3f} "
            f"latency_p95={row['latency_p95']:.3f}"
        )
    return 0


def run_orbit_policies(
    args: argparse.Namespace,
    warmup_requests,
    measured_requests,
    output_dir: Path,
) -> list[dict[str, object]]:
    orbit_args = argparse.Namespace(
        backend=args.backend,
        control_plane_mode=args.control_plane_mode,
        control_plane_start_method=args.control_plane_start_method,
        routers=args.routers,
        clusters=args.clusters,
        cache_capacity=args.cache_capacity,
        cache_token_capacity=args.cache_token_capacity,
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
        requests=len(warmup_requests) + len(measured_requests),
        seed=args.seed,
        model=args.model,
        llama_executable=args.llama_executable,
        llama_port_base=args.llama_port_base,
        llama_threads=args.llama_threads,
        llama_ctx_size=args.llama_ctx_size,
        llama_parallel=args.llama_parallel,
        llama_timeout=args.llama_timeout,
        llama_startup_timeout=args.llama_startup_timeout,
        live_arrival_scale=args.arrival_scale,
        summary_delay=args.summary_delay,
        gossip_delay=args.gossip_delay,
        summary_drop_probability=args.summary_drop_probability,
        gossip_drop_probability=args.gossip_drop_probability,
        failed_clusters=args.failed_clusters,
        failure_start=args.failure_start,
        failure_duration=args.failure_duration,
        retry_penalty=args.retry_penalty,
        llama_extra_arg=args.llama_extra_arg,
    )
    config = build_simulation_config(orbit_args, seed=args.seed)
    comparison_rows: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []

    for policy_name in args.orbit_policies:
        simulation = Simulation(config)
        if warmup_requests:
            simulation.run(policy_name=policy_name, requests=warmup_requests, close_on_finish=False)
        records, metrics = simulation.run(
            policy_name=policy_name,
            requests=measured_requests,
            close_on_finish=True,
        )
        write_json(output_dir / f"{policy_name}_records.json", execution_records_as_dicts(records))
        write_execution_records_csv(output_dir / f"{policy_name}_records.csv", records)
        write_rows_csv(
            output_dir / f"{policy_name}_summary_by_traffic.csv",
            summarize_execution_records(records, policy_name, group_field="traffic_class"),
        )
        write_rows_csv(
            output_dir / f"{policy_name}_summary_by_source.csv",
            summarize_execution_records(records, policy_name, group_field="source_id"),
        )
        metric_row = metrics_as_dict(metrics)
        metric_row["system"] = f"orbit:{policy_name}"
        metrics_rows.append(metric_row)
        comparison_rows.append(_comparison_row_for_orbit(policy_name, metric_row))

    write_json(output_dir / "summary.json", metrics_rows)
    write_rows_csv(output_dir / "summary.csv", metrics_rows)
    return comparison_rows


def _comparison_row_for_orbit(policy_name: str, metric_row: dict[str, object]) -> dict[str, object]:
    return {
        "system": f"orbit:{policy_name}",
        "family": "orbit",
        "mode": "orbit_policy",
        "request_count": metric_row["request_count"],
        "success_count": metric_row["request_count"],
        "failure_count": 0,
        "failure_rate": 0.0,
        "ttft_p50": metric_row["ttft_p50"],
        "ttft_p95": metric_row["ttft_p95"],
        "latency_p50": metric_row["latency_p50"],
        "latency_p95": metric_row["latency_p95"],
        "throughput_rps": None,
        "mean_reusable_prefix": metric_row["mean_reusable_prefix"],
        "mean_reuse_fraction": metric_row["mean_reuse_fraction"],
        "control_plane_bytes": metric_row["control_plane_bytes"],
        "summary_memory_bytes": metric_row["summary_memory_bytes"],
    }


def _comparison_row_for_external(metric_row) -> dict[str, object]:
    row = external_metrics_as_dict(metric_row)
    return {
        "system": row["system"],
        "family": row["family"],
        "mode": "external_endpoint",
        "request_count": row["request_count"],
        "success_count": row["success_count"],
        "failure_count": row["failure_count"],
        "failure_rate": row["failure_rate"],
        "ttft_p50": row["ttft_p50"],
        "ttft_p95": row["ttft_p95"],
        "latency_p50": row["latency_p50"],
        "latency_p95": row["latency_p95"],
        "throughput_rps": row["throughput_rps"],
        "mean_reusable_prefix": None,
        "mean_reuse_fraction": None,
        "control_plane_bytes": None,
        "summary_memory_bytes": None,
    }


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())
