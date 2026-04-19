from __future__ import annotations

import random
from typing import Callable, Dict, Mapping

from .cluster import Cluster
from .models import Request, RouteDecision
from .router import Router


PolicyFn = Callable[[Router, Request, Mapping[str, Cluster], float, random.Random], RouteDecision]


def _load_fallback_route(
    router: Router,
    request: Request,
    clusters: Mapping[str, Cluster],
    now: float,
    policy_name: str,
) -> RouteDecision:
    base = router.load_only_route(request, clusters.keys(), now)
    return RouteDecision(
        policy=policy_name,
        cluster_id=base.cluster_id,
        estimated_reusable_tokens=0,
        predicted_latency=base.predicted_latency,
        used_fallback=True,
        details=base.details,
    )


def summary_policy(
    router: Router,
    request: Request,
    clusters: Mapping[str, Cluster],
    now: float,
    rng: random.Random,
) -> RouteDecision:
    del rng
    return router.route(request, clusters.keys(), now)


def random_policy(
    router: Router,
    request: Request,
    clusters: Mapping[str, Cluster],
    now: float,
    rng: random.Random,
) -> RouteDecision:
    cluster_id = rng.choice(sorted(clusters))
    predicted_latency = (
        router.network_cost(cluster_id)
        + router.config.fixed_overhead
        + request.input_length * router.config.prefill_cost_per_token
        + request.continuation_tokens * router.config.decode_cost_per_token
    )
    return RouteDecision(
        policy="random",
        cluster_id=cluster_id,
        estimated_reusable_tokens=0,
        predicted_latency=predicted_latency,
        details={
            "network_cost": router.network_cost(cluster_id),
            "queue_delay": 0.0,
            "raw_queue_depth": 0,
            "estimated_remaining_prefill_tokens": request.input_length,
            "stale_penalty": 0.0,
            "metadata_age": 0.0,
            "uncertainty_gap": 0,
            "uncertainty_penalty": 0.0,
            "missing_summary": 0.0,
            "missing_summary_penalty": 0.0,
        },
    )


def load_only_policy(
    router: Router,
    request: Request,
    clusters: Mapping[str, Cluster],
    now: float,
    rng: random.Random,
) -> RouteDecision:
    del rng
    return router.load_only_route(request, clusters.keys(), now)


def exact_prefix_policy(
    router: Router,
    request: Request,
    clusters: Mapping[str, Cluster],
    now: float,
    rng: random.Random,
) -> RouteDecision:
    del rng
    best_cluster = None
    best_latency = float("inf")
    best_reuse = 0
    best_details: dict[str, float] = {}
    request_length = request.input_length
    for cluster_id, cluster in clusters.items():
        exact_match = cluster.exact_prefix_match(request.prefix_tokens, now)
        estimated_reuse = request_length if exact_match else 0
        raw_queue_depth = cluster.queue_depth(now)
        predicted, details = router.predict_latency(
            cluster_id=cluster_id,
            request=request,
            estimated_reusable_tokens=estimated_reuse,
            raw_queue_depth=raw_queue_depth,
        )
        if predicted < best_latency:
            best_latency = predicted
            best_cluster = cluster_id
            best_reuse = estimated_reuse
            best_details = details
    return RouteDecision(
        policy="exact_prefix",
        cluster_id=best_cluster or next(iter(clusters)),
        estimated_reusable_tokens=best_reuse,
        predicted_latency=best_latency,
        details=best_details,
    )


def oracle_policy(
    router: Router,
    request: Request,
    clusters: Mapping[str, Cluster],
    now: float,
    rng: random.Random,
) -> RouteDecision:
    del rng
    best_cluster = None
    best_latency = float("inf")
    best_reuse = 0
    best_details: dict[str, float] = {}
    request_length = request.input_length
    for cluster_id, cluster in clusters.items():
        estimated_reuse = cluster.true_reusable_prefix(request.prefix_tokens, now)
        raw_queue_depth = cluster.queue_depth(now)
        predicted, details = router.predict_latency(
            cluster_id=cluster_id,
            request=request,
            estimated_reusable_tokens=estimated_reuse,
            raw_queue_depth=raw_queue_depth,
        )
        if predicted < best_latency:
            best_latency = predicted
            best_cluster = cluster_id
            best_reuse = estimated_reuse
            best_details = details
    return RouteDecision(
        policy="oracle",
        cluster_id=best_cluster or next(iter(clusters)),
        estimated_reusable_tokens=best_reuse,
        predicted_latency=best_latency,
        details=best_details,
    )


def vllm_prefix_mock_policy(
    router: Router,
    request: Request,
    clusters: Mapping[str, Cluster],
    now: float,
    rng: random.Random,
) -> RouteDecision:
    del rng
    request_length = request.input_length
    exact_candidates: list[RouteDecision] = []

    for cluster_id, cluster in clusters.items():
        exact_match = cluster.exact_prefix_match(request.prefix_tokens, now)
        if not exact_match:
            continue
        raw_queue_depth = cluster.queue_depth(now)
        predicted, details = router.predict_latency(
            cluster_id=cluster_id,
            request=request,
            estimated_reusable_tokens=request_length,
            raw_queue_depth=raw_queue_depth,
        )
        exact_candidates.append(
            RouteDecision(
                policy="vllm_prefix_mock",
                cluster_id=cluster_id,
                estimated_reusable_tokens=request_length,
                predicted_latency=predicted,
                details=details,
            )
        )

    if exact_candidates:
        return min(exact_candidates, key=lambda candidate: candidate.predicted_latency)
    return _load_fallback_route(router, request, clusters, now, "vllm_prefix_mock")


def vllm_kv_mock_policy(
    router: Router,
    request: Request,
    clusters: Mapping[str, Cluster],
    now: float,
    rng: random.Random,
) -> RouteDecision:
    del rng
    best_reuse = -1
    best_candidates: list[RouteDecision] = []

    for cluster_id, cluster in clusters.items():
        exact_reuse = cluster.true_reusable_prefix(request.prefix_tokens, now)
        raw_queue_depth = cluster.queue_depth(now)
        predicted, details = router.predict_latency(
            cluster_id=cluster_id,
            request=request,
            estimated_reusable_tokens=exact_reuse,
            raw_queue_depth=raw_queue_depth,
        )
        candidate = RouteDecision(
            policy="vllm_kv_mock",
            cluster_id=cluster_id,
            estimated_reusable_tokens=exact_reuse,
            predicted_latency=predicted,
            used_fallback=exact_reuse == 0,
            details=details,
        )
        if exact_reuse > best_reuse:
            best_reuse = exact_reuse
            best_candidates = [candidate]
        elif exact_reuse == best_reuse:
            best_candidates.append(candidate)

    if not best_candidates:
        return _load_fallback_route(router, request, clusters, now, "vllm_kv_mock")
    return min(best_candidates, key=lambda candidate: candidate.predicted_latency)


POLICIES: Dict[str, PolicyFn] = {
    "summary": summary_policy,
    "random": random_policy,
    "load_only": load_only_policy,
    "exact_prefix": exact_prefix_policy,
    "oracle": oracle_policy,
    "vllm_prefix_mock": vllm_prefix_mock_policy,
    "vllm_kv_mock": vllm_kv_mock_policy,
}
