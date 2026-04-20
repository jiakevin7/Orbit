from __future__ import annotations

import random
from typing import Callable, Dict, Mapping

from .cluster import Cluster
from .hashing import hash_prefix
from .models import Request, RouteDecision
from .router import Router


PolicyFn = Callable[[Router, Request, Mapping[str, Cluster], float, random.Random], RouteDecision]
_VLLM_PREFIX_CACHE_BLOCK_SIZE = 16


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


def _full_prefix_key(tokens: tuple[int, ...]) -> int:
    return hash_prefix(tokens, len(tokens))


def _block_aligned_reuse(reusable_tokens: int, request_length: int) -> int:
    if reusable_tokens <= 0 or request_length <= 0:
        return 0
    aligned = (reusable_tokens // _VLLM_PREFIX_CACHE_BLOCK_SIZE) * _VLLM_PREFIX_CACHE_BLOCK_SIZE
    return max(0, min(request_length, aligned))


def summary_policy(
    router: Router,
    request: Request,
    clusters: Mapping[str, Cluster],
    now: float,
    rng: random.Random,
) -> RouteDecision:
    del rng
    return router.route(request, clusters.keys(), now)


def orbit_policy(
    router: Router,
    request: Request,
    clusters: Mapping[str, Cluster],
    now: float,
    rng: random.Random,
) -> RouteDecision:
    decision = summary_policy(router, request, clusters, now, rng)
    return RouteDecision(
        policy="orbit",
        cluster_id=decision.cluster_id,
        estimated_reusable_tokens=decision.estimated_reusable_tokens,
        predicted_latency=decision.predicted_latency,
        used_fallback=decision.used_fallback,
        details=decision.details,
    )


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


def least_loaded_policy(
    router: Router,
    request: Request,
    clusters: Mapping[str, Cluster],
    now: float,
    rng: random.Random,
) -> RouteDecision:
    decision = load_only_policy(router, request, clusters, now, rng)
    return RouteDecision(
        policy="least_loaded",
        cluster_id=decision.cluster_id,
        estimated_reusable_tokens=decision.estimated_reusable_tokens,
        predicted_latency=decision.predicted_latency,
        used_fallback=decision.used_fallback,
        details=decision.details,
    )


def round_robin_policy(
    router: Router,
    request: Request,
    clusters: Mapping[str, Cluster],
    now: float,
    rng: random.Random,
) -> RouteDecision:
    del rng
    cluster_ids = tuple(sorted(clusters))
    if not cluster_ids:
        raise ValueError(f"router {router.router_id} has no reachable clusters")
    index = router.round_robin_cursor % len(cluster_ids)
    router.round_robin_cursor += 1
    cluster_id = cluster_ids[index]

    raw_queue_depth = 0
    metadata_age = 0.0
    view = getattr(router, "views", {}).get(cluster_id)
    if view is not None:
        raw_queue_depth = view.summary.queue_depth
        metadata_age = max(0.0, now - view.summary.created_at)
    predicted_latency, details = router.predict_latency(
        cluster_id=cluster_id,
        request=request,
        estimated_reusable_tokens=0,
        raw_queue_depth=raw_queue_depth,
        metadata_age=metadata_age,
        uncertainty_gap=0,
        missing_summary=view is None,
    )
    return RouteDecision(
        policy="round_robin",
        cluster_id=cluster_id,
        estimated_reusable_tokens=0,
        predicted_latency=predicted_latency,
        details=details,
    )


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
    prefix_key = _full_prefix_key(request.prefix_tokens)
    sticky_cluster_id = router.prefix_affinity.get(prefix_key)
    if sticky_cluster_id in clusters:
        sticky_cluster = clusters[sticky_cluster_id]
        raw_queue_depth = sticky_cluster.queue_depth(now)
        predicted, details = router.predict_latency(
            cluster_id=sticky_cluster_id,
            request=request,
            estimated_reusable_tokens=request.input_length,
            raw_queue_depth=raw_queue_depth,
        )
        return RouteDecision(
            policy="vllm_prefix_mock",
            cluster_id=sticky_cluster_id,
            estimated_reusable_tokens=request.input_length,
            predicted_latency=predicted,
            details=details,
        )

    fallback = _load_fallback_route(router, request, clusters, now, "vllm_prefix_mock")
    router.prefix_affinity[prefix_key] = fallback.cluster_id
    return fallback


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
        exact_reuse = _block_aligned_reuse(
            cluster.true_reusable_prefix(request.prefix_tokens, now),
            request.input_length,
        )
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
    "orbit": orbit_policy,
    "random": random_policy,
    "load_only": load_only_policy,
    "least_loaded": least_loaded_policy,
    "round_robin": round_robin_policy,
    "exact_prefix": exact_prefix_policy,
    "oracle": oracle_policy,
    "vllm_prefix_mock": vllm_prefix_mock_policy,
    "vllm_kv_mock": vllm_kv_mock_policy,
}
