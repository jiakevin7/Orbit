from __future__ import annotations

import random
from typing import Callable, Dict, Mapping

from .cluster import Cluster
from .models import Request, RouteDecision
from .router import Router


PolicyFn = Callable[[Router, Request, Mapping[str, Cluster], float, random.Random], RouteDecision]


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
    best_cluster = None
    best_cost = float("inf")
    best_details: dict[str, float] = {}
    for cluster_id in clusters:
        view = router.views.get(cluster_id)
        queue_delay = 0.0
        stale_penalty = 0.0
        raw_queue_depth = 0
        if view is not None:
            raw_queue_depth = view.summary.queue_depth
            queue_delay = raw_queue_depth * router.config.queue_depth_penalty
            stale_penalty = max(0.0, now - view.summary.created_at) * router.config.stale_penalty_per_second
        cost = (
            router.network_cost(cluster_id)
            + router.config.fixed_overhead
            + queue_delay
            + router.config.prefill_cost_per_token * request.input_length
            + request.continuation_tokens * router.config.decode_cost_per_token
            + stale_penalty
        )
        if cost < best_cost:
            best_cost = cost
            best_cluster = cluster_id
            best_details = {
                "network_cost": router.network_cost(cluster_id),
                "queue_delay": queue_delay,
                "raw_queue_depth": raw_queue_depth,
                "estimated_remaining_prefill_tokens": request.input_length,
                "stale_penalty": stale_penalty,
                "metadata_age": max(0.0, now - view.summary.created_at) if view is not None else 0.0,
                "uncertainty_gap": 0,
                "uncertainty_penalty": 0.0,
                "missing_summary": 0.0 if view is not None else 1.0,
                "missing_summary_penalty": 0.0,
            }
    return RouteDecision(
        policy="load_only",
        cluster_id=best_cluster or next(iter(clusters)),
        estimated_reusable_tokens=0,
        predicted_latency=best_cost,
        details=best_details,
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
        remaining_prefill = request_length - estimated_reuse
        raw_queue_depth = cluster.queue_depth(now)
        predicted = (
            router.network_cost(cluster_id)
            + router.config.fixed_overhead
            + raw_queue_depth * router.config.queue_depth_penalty
            + router.config.prefill_cost_per_token * remaining_prefill
            + request.continuation_tokens * router.config.decode_cost_per_token
        )
        if predicted < best_latency:
            best_latency = predicted
            best_cluster = cluster_id
            best_reuse = estimated_reuse
            best_details = {
                "network_cost": router.network_cost(cluster_id),
                "queue_delay": raw_queue_depth * router.config.queue_depth_penalty,
                "raw_queue_depth": raw_queue_depth,
                "estimated_remaining_prefill_tokens": remaining_prefill,
                "stale_penalty": 0.0,
                "metadata_age": 0.0,
                "uncertainty_gap": 0,
                "uncertainty_penalty": 0.0,
                "missing_summary": 0.0,
                "missing_summary_penalty": 0.0,
            }
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
        remaining_prefill = request_length - estimated_reuse
        raw_queue_depth = cluster.queue_depth(now)
        predicted = (
            router.network_cost(cluster_id)
            + router.config.fixed_overhead
            + raw_queue_depth * router.config.queue_depth_penalty
            + router.config.prefill_cost_per_token * remaining_prefill
            + request.continuation_tokens * router.config.decode_cost_per_token
        )
        if predicted < best_latency:
            best_latency = predicted
            best_cluster = cluster_id
            best_reuse = estimated_reuse
            best_details = {
                "network_cost": router.network_cost(cluster_id),
                "queue_delay": raw_queue_depth * router.config.queue_depth_penalty,
                "raw_queue_depth": raw_queue_depth,
                "estimated_remaining_prefill_tokens": remaining_prefill,
                "stale_penalty": 0.0,
                "metadata_age": 0.0,
                "uncertainty_gap": 0,
                "uncertainty_penalty": 0.0,
                "missing_summary": 0.0,
                "missing_summary_penalty": 0.0,
            }
    return RouteDecision(
        policy="oracle",
        cluster_id=best_cluster or next(iter(clusters)),
        estimated_reusable_tokens=best_reuse,
        predicted_latency=best_latency,
        details=best_details,
    )


POLICIES: Dict[str, PolicyFn] = {
    "summary": summary_policy,
    "random": random_policy,
    "load_only": load_only_policy,
    "exact_prefix": exact_prefix_policy,
    "oracle": oracle_policy,
}
