from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit.benchmark import (
    DEFAULT_POLICIES,
    main as benchmark_main,
    resolve_output_dir,
    resolve_reachable_clusters_per_router,
)
from orbit.matrix import collect_matrix_summary_rows, load_external_benchmark_matrix, matrix_manifest
from orbit.reporting import write_json, write_rows_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standard external Orbit benchmark matrix")
    parser.add_argument("--backend", choices=("synthetic", "llama_cpp"), default="synthetic")
    parser.add_argument("--control-plane-mode", choices=("inprocess", "multiprocess"))
    parser.add_argument("--control-plane-start-method", default="spawn")
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES))
    parser.add_argument("--routers", type=int, default=4)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--topology-mode", choices=("all_to_all", "sparse_overlap"), default="sparse_overlap")
    parser.add_argument("--reachable-clusters-per-router", type=int, default=3)
    parser.add_argument(
        "--scenario-names",
        nargs="+",
        default=["mixed_external"],
        help="subset of named scenarios to run; defaults to the mixed external workload",
    )
    parser.add_argument("--measured-requests", type=int, default=96)
    parser.add_argument("--warmup-requests", type=int, default=24)
    parser.add_argument("--validation-requests", type=int, default=24)
    parser.add_argument("--continuation-token-cap", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 17, 23, 29])
    parser.add_argument("--cache-capacity", type=int, default=256)
    parser.add_argument("--cache-token-capacity", type=int, default=4096)
    parser.add_argument("--sharegpt-path")
    parser.add_argument("--rag-path")
    parser.add_argument("--agent-path")
    parser.add_argument("--sharegpt-sample-limit", type=int, default=2000)
    parser.add_argument("--rag-sample-limit", type=int, default=2000)
    parser.add_argument("--agent-sample-limit", type=int, default=2000)
    parser.add_argument("--output-dir")
    parser.add_argument("--model")
    parser.add_argument("--llama-executable", default="llama-server")
    parser.add_argument("--llama-port-base", type=int, default=19010)
    parser.add_argument("--llama-threads", type=int, default=4)
    parser.add_argument("--llama-ctx-size", type=int, default=4096)
    parser.add_argument("--llama-parallel", type=int, default=1)
    parser.add_argument("--llama-timeout", type=float, default=120.0)
    parser.add_argument("--llama-startup-timeout", type=float, default=120.0)
    parser.add_argument("--live-arrival-scale", type=float)
    parser.add_argument("--validation-p95-regression-tolerance", type=float, default=0.05)
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

    if args.backend == "llama_cpp" and not args.model:
        parser.error("--model is required when --backend llama_cpp is used")
    if args.reachable_clusters_per_router is not None and args.reachable_clusters_per_router <= 0:
        parser.error("--reachable-clusters-per-router must be positive")
    control_plane_mode = args.control_plane_mode
    if control_plane_mode is None:
        control_plane_mode = "multiprocess" if args.backend == "synthetic" else "inprocess"
    matrix_name, description, scenarios = load_external_benchmark_matrix()
    selected_names = set(args.scenario_names)
    scenarios = tuple(scenario for scenario in scenarios if scenario.name in selected_names)
    missing_scenarios = sorted(selected_names - {scenario.name for scenario in scenarios})
    if missing_scenarios:
        parser.error(f"unknown scenario names: {', '.join(missing_scenarios)}")
    total_requests = args.measured_requests + args.warmup_requests + args.validation_requests
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    output_dir = resolve_output_dir(args.output_dir or str(REPO_ROOT / "results" / f"matrix-{timestamp}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        output_dir / "matrix_manifest.json",
        matrix_manifest(
            matrix_name,
            description,
            scenarios,
            backend=args.backend,
            control_plane_mode=control_plane_mode,
            router_count=args.routers,
            cluster_count=args.clusters,
            topology_mode=args.topology_mode,
            reachable_clusters_per_router=resolve_reachable_clusters_per_router(args),
            seeds=args.seeds,
            measured_requests=args.measured_requests,
            warmup_requests=args.warmup_requests,
            validation_requests=args.validation_requests,
            sharegpt_path=args.sharegpt_path,
            rag_path=args.rag_path,
            agent_path=args.agent_path,
        ),
    )

    for scenario_index, scenario in enumerate(scenarios):
        scenario_dir = output_dir / scenario.name
        scenario_args = [
            "--backend",
            args.backend,
            "--control-plane-mode",
            control_plane_mode,
            "--control-plane-start-method",
            args.control_plane_start_method,
            "--workload-kind",
            scenario.workload_kind,
            "--requests",
            str(total_requests),
            "--warmup-requests",
            str(args.warmup_requests),
            "--validation-requests",
            str(args.validation_requests),
            "--calibrate-router",
            "--routers",
            str(args.routers),
            "--clusters",
            str(args.clusters),
            "--topology-mode",
            args.topology_mode,
            "--cache-capacity",
            str(args.cache_capacity),
            "--cache-token-capacity",
            str(args.cache_token_capacity),
            "--traffic-mix-chat",
            str(scenario.traffic_mix_chat),
            "--traffic-mix-rag",
            str(scenario.traffic_mix_rag),
            "--traffic-mix-agent",
            str(scenario.traffic_mix_agent),
            "--traffic-mix-bursty",
            str(scenario.traffic_mix_bursty),
            "--sharegpt-sample-limit",
            str(args.sharegpt_sample_limit),
            "--rag-sample-limit",
            str(args.rag_sample_limit),
            "--agent-sample-limit",
            str(args.agent_sample_limit),
            "--validation-p95-regression-tolerance",
            str(args.validation_p95_regression_tolerance),
            "--summary-delay",
            str(args.summary_delay),
            "--gossip-delay",
            str(args.gossip_delay),
            "--summary-drop-probability",
            str(args.summary_drop_probability),
            "--gossip-drop-probability",
            str(args.gossip_drop_probability),
            "--failure-start",
            str(args.failure_start),
            "--failure-duration",
            str(args.failure_duration),
            "--retry-penalty",
            str(args.retry_penalty),
            "--output-dir",
            str(scenario_dir),
            "--record-format",
            "both",
            "--policies",
            *args.policies,
            "--seeds",
            *[str(seed) for seed in args.seeds],
        ]
        if args.reachable_clusters_per_router is not None:
            scenario_args.extend(
                ["--reachable-clusters-per-router", str(args.reachable_clusters_per_router)]
            )
        if args.continuation_token_cap is not None:
            scenario_args.extend(["--continuation-token-cap", str(args.continuation_token_cap)])
        if args.sharegpt_path:
            scenario_args.extend(["--sharegpt-path", args.sharegpt_path])
        if args.rag_path:
            scenario_args.extend(["--rag-path", args.rag_path])
        if args.agent_path:
            scenario_args.extend(["--agent-path", args.agent_path])
        if args.failed_clusters:
            scenario_args.extend(["--failed-clusters", *args.failed_clusters])
        if args.backend == "llama_cpp":
            scenario_args.extend(
                [
                    "--model",
                    args.model,
                    "--llama-executable",
                    args.llama_executable,
                    "--llama-port-base",
                    str(args.llama_port_base + scenario_index * max(args.clusters, 1) * 10),
                    "--llama-threads",
                    str(args.llama_threads),
                    "--llama-ctx-size",
                    str(args.llama_ctx_size),
                    "--llama-parallel",
                    str(args.llama_parallel),
                    "--llama-timeout",
                    str(args.llama_timeout),
                    "--llama-startup-timeout",
                    str(args.llama_startup_timeout),
                ]
            )
            if args.live_arrival_scale is not None:
                scenario_args.extend(["--live-arrival-scale", str(args.live_arrival_scale)])
            for extra_arg in args.llama_extra_arg:
                scenario_args.extend(["--llama-extra-arg", extra_arg])

        print(f"running scenario {scenario.name} -> {scenario_dir}")
        benchmark_main(scenario_args)

    matrix_rows = collect_matrix_summary_rows(output_dir)
    write_json(output_dir / "matrix_summary.json", matrix_rows)
    write_rows_csv(output_dir / "matrix_summary.csv", matrix_rows)
    print(f"wrote benchmark matrix artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
