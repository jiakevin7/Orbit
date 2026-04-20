from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace

from .benchmark import (
    resolve_prompt_prefix_token_cap,
    resolve_reachable_clusters_per_router,
    select_config_by_validation,
    split_workload,
)
from .calibration import fit_router_config
from .cluster import ClusterConfig
from .llamacpp import LlamaCppClusterConfig
from .policies import POLICIES
from .router import RouterConfig
from .simulation import FaultInjectionConfig, Simulation, SimulationConfig, metrics_as_dict
from .workload import WorkloadConfig, generate_workload


def main() -> None:
    parser = argparse.ArgumentParser(description="Orbit cluster-level routing simulator")
    parser.add_argument("--backend", choices=("synthetic", "llama_cpp"), default="synthetic")
    parser.add_argument("--control-plane-mode", choices=("inprocess", "multiprocess"), default="inprocess")
    parser.add_argument("--control-plane-start-method", default="spawn")
    parser.add_argument("--policy", default="summary", help="policy to run")
    parser.add_argument("--compare", action="store_true", help="run all built-in policies")
    parser.add_argument("--requests", type=int, default=200, help="number of requests")
    parser.add_argument("--topology-mode", choices=("all_to_all", "sparse_overlap"), default="all_to_all")
    parser.add_argument("--reachable-clusters-per-router", type=int)
    parser.add_argument("--cache-capacity", type=int, default=256)
    parser.add_argument("--cache-token-capacity", type=int)
    parser.add_argument("--prompt-prefix-token-cap", type=int)
    parser.add_argument("--workload-kind", choices=("synthetic", "mixed_realistic"), default="synthetic")
    parser.add_argument("--sharegpt-path", help="path to a ShareGPT-style JSON or JSONL dataset")
    parser.add_argument("--rag-path", help="path to a RAG-style JSON or JSONL dataset")
    parser.add_argument("--agent-path", help="path to an agent/tool-use JSON or JSONL dataset such as ToolBench, BFCL, or tau-bench traces")
    parser.add_argument("--sharegpt-sample-limit", type=int, default=2000)
    parser.add_argument("--rag-sample-limit", type=int, default=2000)
    parser.add_argument("--agent-sample-limit", type=int, default=2000)
    parser.add_argument("--traffic-mix-chat", type=float, default=0.4375)
    parser.add_argument("--traffic-mix-rag", type=float, default=0.3125)
    parser.add_argument("--traffic-mix-agent", type=float, default=0.25)
    parser.add_argument("--traffic-mix-bursty", type=float, default=0.0)
    parser.add_argument("--warmup-requests", type=int, default=0, help="number of warm-up requests excluded from reported metrics")
    parser.add_argument(
        "--validation-requests",
        type=int,
        default=0,
        help="number of held-out requests used to choose between base and calibrated router configs before final evaluation",
    )
    parser.add_argument("--calibrate-router", action="store_true", help="fit router latency coefficients from the warm-up requests")
    parser.add_argument("--calibration-policy", choices=tuple(POLICIES), default="summary")
    parser.add_argument("--routers", type=int, default=2, help="number of routers")
    parser.add_argument("--clusters", type=int, default=3, help="number of clusters")
    parser.add_argument("--seed", type=int, default=7, help="workload seed")
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
    parser.add_argument("--validation-p95-regression-tolerance", type=float, default=0.05)
    parser.add_argument("--llama-extra-arg", action="append", default=[], help="extra argument forwarded to llama-server")
    args = parser.parse_args()

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
    if args.reachable_clusters_per_router is not None and args.reachable_clusters_per_router <= 0:
        parser.error("--reachable-clusters-per-router must be positive")
    if args.validation_p95_regression_tolerance < 0:
        parser.error("--validation-p95-regression-tolerance must be non-negative")
    for probability in (args.summary_drop_probability, args.gossip_drop_probability):
        if not 0.0 <= probability <= 1.0:
            parser.error("drop probabilities must be between 0 and 1")
    if args.summary_delay < 0 or args.gossip_delay < 0 or args.failure_duration < 0 or args.retry_penalty < 0:
        parser.error("fault injection delays and durations must be non-negative")

    router_ids = tuple(f"router-{index}" for index in range(args.routers))
    cluster_ids = tuple(f"cluster-{index}" for index in range(args.clusters))
    cache_token_capacity = args.cache_token_capacity
    if cache_token_capacity is None and args.workload_kind == "mixed_realistic":
        cache_token_capacity = 4096

    config = SimulationConfig(
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
                prompt_token_cap=resolve_prompt_prefix_token_cap(args),
                extra_args=tuple(args.llama_extra_arg),
            )
            if args.backend == "llama_cpp"
            else None
        ),
        router_config=RouterConfig(),
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
        live_arrival_scale=(
            args.live_arrival_scale
            if args.live_arrival_scale is not None
            else (0.01 if args.backend == "llama_cpp" else 1.0)
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
            prompt_prefix_token_cap=resolve_prompt_prefix_token_cap(args),
            seed=args.seed,
        ),
    )

    workload = generate_workload(config.workload)
    if args.backend == "llama_cpp":
        prepare_simulation = Simulation(config)
        try:
            workload = prepare_simulation.prepare_requests(workload)
        finally:
            prepare_simulation.close()

    warmup_requests, validation_requests, measured_requests = split_workload(
        workload,
        args.warmup_requests,
        args.validation_requests,
    )
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

    policy_names = tuple(POLICIES) if args.compare else (args.policy,)
    results: dict[str, dict[str, object]] = {}
    for policy_name in policy_names:
        simulation = Simulation(selected_config)
        if warmup_requests:
            simulation.run(policy_name=policy_name, requests=warmup_requests, close_on_finish=False)
        _, metrics = simulation.run(policy_name=policy_name, requests=measured_requests, close_on_finish=True)
        results[policy_name] = metrics_as_dict(metrics)

    payload: dict[str, object] = {"metrics": results}
    if calibration_payload is not None:
        payload["calibration"] = calibration_payload
    if selection_payload is not None:
        payload["selection"] = selection_payload
    if args.compare:
        print(json.dumps(payload, indent=2))
        return

    print(
        json.dumps(
            {
                "metrics": results[args.policy],
                "calibration": calibration_payload,
                "selection": selection_payload,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
