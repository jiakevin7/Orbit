from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit.benchmark import aggregate_summary_rows, allocate_llama_cpp_ports, flatten_metrics_row, split_workload
from orbit.cluster import ClusterConfig
from orbit.llamacpp import LlamaCppClusterConfig
from orbit.reporting import metrics_as_dict, write_json, write_rows_csv
from orbit.router import RouterConfig
from orbit.simulation import Simulation, SimulationConfig
from orbit.workload import WorkloadConfig, generate_workload


DEFAULT_CANDIDATES = (
    {
        "name": "default",
        "low_overlap_fraction": 0.10,
        "min_summary_overlap_tokens": 4,
        "max_summary_overlap_tokens": 32,
        "summary_advantage_margin": 1.0,
        "summary_advantage_uncertainty_scale": 0.25,
    },
    {
        "name": "gate16",
        "low_overlap_fraction": 0.05,
        "min_summary_overlap_tokens": 4,
        "max_summary_overlap_tokens": 16,
        "summary_advantage_margin": 1.0,
        "summary_advantage_uncertainty_scale": 0.25,
    },
    {
        "name": "gate8",
        "low_overlap_fraction": 0.02,
        "min_summary_overlap_tokens": 4,
        "max_summary_overlap_tokens": 8,
        "summary_advantage_margin": 1.0,
        "summary_advantage_uncertainty_scale": 0.25,
    },
    {
        "name": "gate16_soft",
        "low_overlap_fraction": 0.02,
        "min_summary_overlap_tokens": 4,
        "max_summary_overlap_tokens": 16,
        "summary_advantage_margin": 0.5,
        "summary_advantage_uncertainty_scale": 0.25,
    },
    {
        "name": "gate16_hard",
        "low_overlap_fraction": 0.02,
        "min_summary_overlap_tokens": 4,
        "max_summary_overlap_tokens": 16,
        "summary_advantage_margin": 2.0,
        "summary_advantage_uncertainty_scale": 0.25,
    },
    {
        "name": "gate16_nomargin",
        "low_overlap_fraction": 0.02,
        "min_summary_overlap_tokens": 4,
        "max_summary_overlap_tokens": 16,
        "summary_advantage_margin": 0.25,
        "summary_advantage_uncertainty_scale": 0.0,
    },
    {
        "name": "gate8_nomargin",
        "low_overlap_fraction": 0.02,
        "min_summary_overlap_tokens": 4,
        "max_summary_overlap_tokens": 8,
        "summary_advantage_margin": 0.25,
        "summary_advantage_uncertainty_scale": 0.0,
    },
    {
        "name": "gate24_soft",
        "low_overlap_fraction": 0.05,
        "min_summary_overlap_tokens": 4,
        "max_summary_overlap_tokens": 24,
        "summary_advantage_margin": 0.5,
        "summary_advantage_uncertainty_scale": 0.25,
    },
    {
        "name": "min0_gate16",
        "low_overlap_fraction": 0.02,
        "min_summary_overlap_tokens": 0,
        "max_summary_overlap_tokens": 16,
        "summary_advantage_margin": 1.0,
        "summary_advantage_uncertainty_scale": 0.25,
    },
    {
        "name": "min8_gate16",
        "low_overlap_fraction": 0.02,
        "min_summary_overlap_tokens": 8,
        "max_summary_overlap_tokens": 16,
        "summary_advantage_margin": 1.0,
        "summary_advantage_uncertainty_scale": 0.25,
    },
    {
        "name": "gate32_hard",
        "low_overlap_fraction": 0.05,
        "min_summary_overlap_tokens": 4,
        "max_summary_overlap_tokens": 32,
        "summary_advantage_margin": 2.0,
        "summary_advantage_uncertainty_scale": 0.5,
    },
    {
        "name": "fallback_hard",
        "low_overlap_fraction": 0.02,
        "min_summary_overlap_tokens": 4,
        "max_summary_overlap_tokens": 16,
        "summary_advantage_margin": 3.0,
        "summary_advantage_uncertainty_scale": 1.0,
    },
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep Orbit summary-threshold and fallback knobs on the large mixed_external setup",
    )
    parser.add_argument(
        "--model",
        default=str(REPO_ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"),
        help="GGUF model path for llama.cpp",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--llama-executable", default="llama-server")
    parser.add_argument("--llama-port-base", type=int, default=26010)
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
    parser.add_argument("--sharegpt-path", default=str(REPO_ROOT / "results" / "external-datasets-20260418" / "sharegpt_x_chat.json"))
    parser.add_argument("--rag-path", default=str(REPO_ROOT / "results" / "external-datasets-20260418" / "ragbench_hotpotqa.json"))
    parser.add_argument("--agent-path", default=str(REPO_ROOT / "results" / "external-datasets-20260418" / "toolbench_g123_query.json"))
    return parser


def build_output_dir(path: str | None) -> Path:
    if path:
        return Path(path).resolve()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return (REPO_ROOT / "results" / f"orbit-summary-gate-sweep-{timestamp}").resolve()


def build_base_config(args: argparse.Namespace, *, seed: int, total_requests: int, router_ids: tuple[str, ...], cluster_ids: tuple[str, ...], port_base: int) -> SimulationConfig:
    return SimulationConfig(
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
            port_base=port_base,
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
            workload_kind="mixed_realistic",
            sharegpt_path=args.sharegpt_path,
            rag_path=args.rag_path,
            agent_path=args.agent_path,
            traffic_mix_chat=0.4375,
            traffic_mix_rag=0.3125,
            traffic_mix_agent=0.25,
            traffic_mix_bursty=0.0,
            dataset_continuation_floor=args.continuation_token_cap,
            dataset_continuation_cap=args.continuation_token_cap,
            seed=seed,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = build_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    router_ids = tuple(f"router-{index}" for index in range(args.routers))
    cluster_ids = tuple(f"cluster-{index}" for index in range(args.clusters))
    total_requests = args.measured_requests + args.warmup_requests + args.validation_requests

    write_json(
        output_dir / "manifest.json",
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "backend": "llama_cpp",
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
            "candidate_count": len(DEFAULT_CANDIDATES),
            "candidates": DEFAULT_CANDIDATES,
        },
    )

    workloads: dict[int, tuple[list, list]] = {}
    for seed_index, seed in enumerate(args.seeds):
        base_config = build_base_config(
            args,
            seed=seed,
            total_requests=total_requests,
            router_ids=router_ids,
            cluster_ids=cluster_ids,
            port_base=args.llama_port_base + seed_index * 1000,
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
    for candidate_index, candidate in enumerate(DEFAULT_CANDIDATES):
        name = str(candidate["name"])
        print(f"candidate {name} ({candidate_index + 1}/{len(DEFAULT_CANDIDATES)})", flush=True)
        candidate_dir = output_dir / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        write_json(candidate_dir / "config.json", candidate)
        candidate_rows: list[dict[str, object]] = []

        for seed_index, seed in enumerate(args.seeds):
            warmup_requests, measured_requests = workloads[seed]
            router_config = replace(
                RouterConfig(),
                low_overlap_fraction=float(candidate["low_overlap_fraction"]),
                min_summary_overlap_tokens=int(candidate["min_summary_overlap_tokens"]),
                max_summary_overlap_tokens=int(candidate["max_summary_overlap_tokens"]),
                summary_advantage_margin=float(candidate["summary_advantage_margin"]),
                summary_advantage_uncertainty_scale=float(candidate["summary_advantage_uncertainty_scale"]),
            )
            config = build_base_config(
                args,
                seed=seed,
                total_requests=total_requests,
                router_ids=router_ids,
                cluster_ids=cluster_ids,
                port_base=args.llama_port_base + candidate_index * 200 + seed_index * 40,
            )
            config = replace(config, router_config=router_config)

            allocation_index = [0]

            def alloc(cfg: SimulationConfig) -> SimulationConfig:
                allocated = allocate_llama_cpp_ports(cfg, allocation_index[0])
                allocation_index[0] += 1
                return allocated

            simulation = Simulation(alloc(config))
            try:
                if warmup_requests:
                    simulation.run(policy_name="orbit", requests=warmup_requests, close_on_finish=False)
                _, metrics = simulation.run(
                    policy_name="orbit",
                    requests=measured_requests,
                    close_on_finish=True,
                )
            finally:
                simulation.close()

            row = {"candidate": name, "seed": seed, **flatten_metrics_row(metrics_as_dict(metrics))}
            run_rows.append(row)
            candidate_rows.append(row)
            write_json(candidate_dir / f"seed-{seed}-summary.json", row)
            print(
                f"  seed {seed}: ttft_p50={row['ttft_p50']:.3f} "
                f"latency_p50={row['latency_p50']:.3f} "
                f"ttft_p95={row['ttft_p95']:.3f} "
                f"latency_p95={row['latency_p95']:.3f}",
                flush=True,
            )

        write_rows_csv(candidate_dir / "runs.csv", candidate_rows)
        write_rows_csv(
            candidate_dir / "aggregate.csv",
            aggregate_summary_rows(candidate_rows, group_keys=("candidate",)),
        )

    write_rows_csv(output_dir / "runs.csv", run_rows)
    aggregate_rows = aggregate_summary_rows(run_rows, group_keys=("candidate",))
    write_rows_csv(output_dir / "aggregate.csv", aggregate_rows)
    write_json(output_dir / "aggregate.json", aggregate_rows)
    best = min(
        aggregate_rows,
        key=lambda row: (
            float(row["ttft_p50_mean"]),
            float(row["ttft_p95_mean"]),
            float(row["latency_p50_mean"]),
        ),
    )
    write_json(output_dir / "best.json", best)
    print(
        f"best={best['candidate']} "
        f"ttft_p50={best['ttft_p50_mean']:.3f} "
        f"ttft_p95={best['ttft_p95_mean']:.3f} "
        f"latency_p50={best['latency_p50_mean']:.3f}",
        flush=True,
    )
    print(output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
