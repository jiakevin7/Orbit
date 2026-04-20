from __future__ import annotations

import argparse
import itertools
from datetime import datetime
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit.benchmark import aggregate_summary_rows, allocate_llama_cpp_ports, flatten_metrics_row, split_workload
from orbit.cluster import ClusterConfig
from orbit.llamacpp import LlamaCppClusterConfig
from orbit.matrix import load_external_benchmark_matrix
from orbit.reporting import metrics_as_dict, write_json, write_rows_csv
from orbit.router import RouterConfig
from orbit.simulation import Simulation, SimulationConfig
from orbit.workload import WorkloadConfig, generate_workload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep Orbit explicit recent-assignment pressure penalties on the exact large mixed_external setup",
    )
    parser.add_argument(
        "--model",
        default=str(REPO_ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"),
        help="GGUF model path for llama.cpp",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--llama-executable", default="llama-server")
    parser.add_argument("--llama-port-base", type=int, default=28010)
    parser.add_argument("--llama-threads", type=int, default=4)
    parser.add_argument("--llama-ctx-size", type=int, default=8192)
    parser.add_argument("--llama-parallel", type=int, default=1)
    parser.add_argument("--llama-timeout", type=float, default=300.0)
    parser.add_argument("--llama-startup-timeout", type=float, default=300.0)
    parser.add_argument("--routers", type=int, default=4)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--reachable-clusters-per-router", type=int, default=3)
    parser.add_argument("--measured-requests", type=int, default=96)
    parser.add_argument("--warmup-requests", type=int, default=24)
    parser.add_argument("--validation-requests", type=int, default=24)
    parser.add_argument("--continuation-token-cap", type=int, default=4)
    parser.add_argument("--cache-capacity", type=int, default=256)
    parser.add_argument("--cache-token-capacity", type=int, default=4096)
    parser.add_argument("--live-arrival-scale", type=float, default=0.01)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 17, 23, 29])
    parser.add_argument(
        "--window-values",
        nargs="+",
        type=float,
        default=[0.25],
        help="recent-assignment windows to sweep in simulated-time seconds",
    )
    parser.add_argument(
        "--pressure-values",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1.0, 2.0],
        help="explicit recent-assignment pressure penalties to sweep",
    )
    parser.add_argument("--sharegpt-path", default=str(REPO_ROOT / "results" / "external-datasets-20260418" / "sharegpt_x_chat.json"))
    parser.add_argument("--rag-path", default=str(REPO_ROOT / "results" / "external-datasets-20260418" / "ragbench_hotpotqa.json"))
    parser.add_argument("--agent-path", default=str(REPO_ROOT / "results" / "external-datasets-20260418" / "toolbench_g123_query.json"))
    return parser


def build_output_dir(path: str | None) -> Path:
    if path:
        return Path(path).resolve()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return (REPO_ROOT / "results" / f"orbit-pressure-term-sweep-{timestamp}").resolve()


def candidate_name(window: float, pressure: float) -> str:
    if pressure <= 0 or window <= 0:
        return "baseline"
    return f"win{window:.2f}_press{pressure:.2f}".replace(".", "p")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = build_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, scenarios = load_external_benchmark_matrix()
    scenario = next(s for s in scenarios if s.name == "mixed_external")
    router_ids = tuple(f"router-{index}" for index in range(args.routers))
    cluster_ids = tuple(f"cluster-{index}" for index in range(args.clusters))
    total_requests = args.measured_requests + args.warmup_requests + args.validation_requests

    candidate_grid = [(0.0, 0.0)] + list(itertools.product(args.window_values, args.pressure_values))

    write_json(
        output_dir / "manifest.json",
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "backend": "llama_cpp",
            "policy": "orbit",
            "scenario": "mixed_external",
            "routers": len(router_ids),
            "clusters": len(cluster_ids),
            "reachable_clusters_per_router": args.reachable_clusters_per_router,
            "measured_requests": args.measured_requests,
            "warmup_requests": args.warmup_requests,
            "validation_requests": args.validation_requests,
            "continuation_token_cap": args.continuation_token_cap,
            "seeds": list(args.seeds),
            "model": args.model,
            "window_values": list(args.window_values),
            "pressure_values": list(args.pressure_values),
            "candidate_count": len(candidate_grid),
        },
    )

    workloads: dict[int, tuple[list, list]] = {}
    for seed_index, seed in enumerate(args.seeds):
        base_config = SimulationConfig(
            backend="llama_cpp",
            control_plane_mode="inprocess",
            router_ids=router_ids,
            cluster_ids=cluster_ids,
            topology_mode="sparse_overlap",
            reachable_clusters_per_router=args.reachable_clusters_per_router,
            cluster_config=ClusterConfig(
                cache_capacity=args.cache_capacity,
                cache_capacity_tokens=args.cache_token_capacity,
            ),
            llama_cpp=LlamaCppClusterConfig(
                model_path=args.model,
                executable=args.llama_executable,
                port_base=args.llama_port_base + seed_index * 1000,
                threads=args.llama_threads,
                ctx_size=args.llama_ctx_size,
                parallel=args.llama_parallel,
                request_timeout=args.llama_timeout,
                startup_timeout=args.llama_startup_timeout,
            ),
            router_config=RouterConfig(),
            live_arrival_scale=args.live_arrival_scale,
            workload=WorkloadConfig(
                num_requests=total_requests,
                router_ids=router_ids,
                continuation_token_range=(args.continuation_token_cap, args.continuation_token_cap),
                workload_kind=scenario.workload_kind,
                sharegpt_path=args.sharegpt_path,
                rag_path=args.rag_path,
                agent_path=args.agent_path,
                traffic_mix_chat=scenario.traffic_mix_chat,
                traffic_mix_rag=scenario.traffic_mix_rag,
                traffic_mix_agent=scenario.traffic_mix_agent,
                traffic_mix_bursty=scenario.traffic_mix_bursty,
                dataset_continuation_floor=args.continuation_token_cap,
                dataset_continuation_cap=args.continuation_token_cap,
                seed=seed,
            ),
        )
        requests = generate_workload(base_config.workload)
        prep = Simulation(allocate_llama_cpp_ports(base_config, 0))
        try:
            requests = prep.prepare_requests(requests)
        finally:
            prep.close()
        warmup_requests, _, measured_requests = split_workload(
            requests,
            args.warmup_requests,
            args.validation_requests,
        )
        workloads[seed] = (warmup_requests, measured_requests)

    run_rows: list[dict[str, object]] = []
    for candidate_index, (window, pressure) in enumerate(candidate_grid):
        name = candidate_name(window, pressure)
        print(f"candidate {name} ({candidate_index + 1}/{len(candidate_grid)})", flush=True)
        candidate_dir = output_dir / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        config_payload = {
            "candidate": name,
            "recent_assignment_window": window,
            "recent_assignment_weight": 0.0,
            "recent_assignment_penalty": pressure,
            "ttft_recent_assignment_penalty": pressure,
        }
        write_json(candidate_dir / "config.json", config_payload)

        candidate_rows: list[dict[str, object]] = []
        for seed_index, seed in enumerate(args.seeds):
            warmup_requests, measured_requests = workloads[seed]
            router_config = RouterConfig(
                recent_assignment_window=window,
                recent_assignment_weight=0.0,
                recent_assignment_penalty=pressure,
                ttft_recent_assignment_penalty=pressure,
            )
            config = SimulationConfig(
                backend="llama_cpp",
                control_plane_mode="inprocess",
                router_ids=router_ids,
                cluster_ids=cluster_ids,
                topology_mode="sparse_overlap",
                reachable_clusters_per_router=args.reachable_clusters_per_router,
                cluster_config=ClusterConfig(
                    cache_capacity=args.cache_capacity,
                    cache_capacity_tokens=args.cache_token_capacity,
                ),
                llama_cpp=LlamaCppClusterConfig(
                    model_path=args.model,
                    executable=args.llama_executable,
                    port_base=args.llama_port_base + candidate_index * 200 + seed_index * 40,
                    threads=args.llama_threads,
                    ctx_size=args.llama_ctx_size,
                    parallel=args.llama_parallel,
                    request_timeout=args.llama_timeout,
                    startup_timeout=args.llama_startup_timeout,
                ),
                router_config=router_config,
                live_arrival_scale=args.live_arrival_scale,
            )
            simulation = Simulation(allocate_llama_cpp_ports(config, 0))
            try:
                if warmup_requests:
                    simulation.run("orbit", warmup_requests, close_on_finish=False)
                _, metrics = simulation.run("orbit", measured_requests)
            finally:
                simulation.close()

            metrics_payload = metrics_as_dict(metrics)
            write_json(candidate_dir / f"seed-{seed}.json", metrics_payload)
            row = {
                "policy": "orbit",
                "candidate": name,
                "seed": seed,
                "recent_assignment_window": window,
                "recent_assignment_penalty": pressure,
                **flatten_metrics_row(metrics_payload),
            }
            candidate_rows.append(row)
            run_rows.append(row)

        aggregate_rows = aggregate_summary_rows(candidate_rows, group_keys=("candidate",))
        write_json(candidate_dir / "aggregate.json", aggregate_rows[0])

    write_rows_csv(output_dir / "runs.csv", run_rows)
    aggregate_rows = aggregate_summary_rows(run_rows, group_keys=("candidate",))
    aggregate_rows.sort(key=lambda row: (row["ttft_p50_mean"], row["latency_p50_mean"], row["ttft_p95_mean"]))
    write_rows_csv(output_dir / "aggregate.csv", aggregate_rows)
    write_json(output_dir / "aggregate.json", aggregate_rows)
    if aggregate_rows:
        write_json(output_dir / "best.json", aggregate_rows[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
