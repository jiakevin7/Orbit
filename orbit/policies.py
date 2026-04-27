from .models import RouteDecision


# Primary evaluation baselines share the same router/cluster interface so they
# differ only in routing strategy, not in backend execution or workload.
def orbit_policy(router, request, clusters, now, rng):
    return router.route(request, clusters.keys(), now)


def random_policy(router, request, clusters, now, rng):
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


def least_loaded_policy(router, request, clusters, now, rng):
    decision = router.least_loaded_route(request, clusters.keys(), now)
    return RouteDecision(
        policy="least_loaded",
        cluster_id=decision.cluster_id,
        estimated_reusable_tokens=decision.estimated_reusable_tokens,
        predicted_latency=decision.predicted_latency,
        used_fallback=decision.used_fallback,
        details=decision.details,
    )


def round_robin_policy(router, request, clusters, now, rng):
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


POLICIES = {
    "orbit": orbit_policy,
    "least_loaded": least_loaded_policy,
    "random": random_policy,
    "round_robin": round_robin_policy,
}
