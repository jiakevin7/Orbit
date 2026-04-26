from .models import RouteDecision

def summary_policy(router, request, clusters, now, rng):
    return router.route(request, clusters.keys(), now)

def orbit_policy(router, request, clusters, now, rng):
    decision = summary_policy(router, request, clusters, now, rng)
    return RouteDecision(policy='orbit', cluster_id=decision.cluster_id, estimated_reusable_tokens=decision.estimated_reusable_tokens, predicted_latency=decision.predicted_latency, used_fallback=decision.used_fallback, details=decision.details)

def random_policy(router, request, clusters, now, rng):
    cluster_id = rng.choice(sorted(clusters))
    predicted_latency = router.network_cost(cluster_id) + router.config.fixed_overhead + request.input_length * router.config.prefill_cost_per_token + request.continuation_tokens * router.config.decode_cost_per_token
    return RouteDecision(policy='random', cluster_id=cluster_id, estimated_reusable_tokens=0, predicted_latency=predicted_latency, details={'network_cost': router.network_cost(cluster_id), 'queue_delay': 0.0, 'raw_queue_depth': 0, 'estimated_remaining_prefill_tokens': request.input_length, 'stale_penalty': 0.0, 'metadata_age': 0.0, 'uncertainty_gap': 0, 'uncertainty_penalty': 0.0, 'missing_summary': 0.0, 'missing_summary_penalty': 0.0})

def load_only_policy(router, request, clusters, now, rng):
    return router.load_only_route(request, clusters.keys(), now)

def least_loaded_policy(router, request, clusters, now, rng):
    decision = load_only_policy(router, request, clusters, now, rng)
    return RouteDecision(policy='least_loaded', cluster_id=decision.cluster_id, estimated_reusable_tokens=decision.estimated_reusable_tokens, predicted_latency=decision.predicted_latency, used_fallback=decision.used_fallback, details=decision.details)

def round_robin_policy(router, request, clusters, now, rng):
    cluster_ids = tuple(sorted(clusters))
    if not cluster_ids:
        raise ValueError(f'router {router.router_id} has no reachable clusters')
    index = router.round_robin_cursor % len(cluster_ids)
    router.round_robin_cursor += 1
    cluster_id = cluster_ids[index]
    raw_queue_depth = 0
    metadata_age = 0.0
    view = getattr(router, 'views', {}).get(cluster_id)
    if view is not None:
        raw_queue_depth = view.summary.queue_depth
        metadata_age = max(0.0, now - view.summary.created_at)
    predicted_latency, details = router.predict_latency(cluster_id=cluster_id, request=request, estimated_reusable_tokens=0, raw_queue_depth=raw_queue_depth, metadata_age=metadata_age, uncertainty_gap=0, missing_summary=view is None)
    return RouteDecision(policy='round_robin', cluster_id=cluster_id, estimated_reusable_tokens=0, predicted_latency=predicted_latency, details=details)

def exact_prefix_policy(router, request, clusters, now, rng):
    best_cluster = None
    best_latency = float('inf')
    best_reuse = 0
    best_details: dict[str, float] = {}
    request_length = request.input_length
    for cluster_id, cluster in clusters.items():
        exact_match = cluster.exact_prefix_match(request.prefix_tokens, now)
        estimated_reuse = request_length if exact_match else 0
        raw_queue_depth = cluster.queue_depth(now)
        predicted, details = router.predict_latency(cluster_id=cluster_id, request=request, estimated_reusable_tokens=estimated_reuse, raw_queue_depth=raw_queue_depth)
        if predicted < best_latency:
            best_latency = predicted
            best_cluster = cluster_id
            best_reuse = estimated_reuse
            best_details = details
    return RouteDecision(policy='exact_prefix', cluster_id=best_cluster or next(iter(clusters)), estimated_reusable_tokens=best_reuse, predicted_latency=best_latency, details=best_details)

def oracle_policy(router, request, clusters, now, rng):
    best_cluster = None
    best_latency = float('inf')
    best_reuse = 0
    best_details: dict[str, float] = {}
    request_length = request.input_length
    for cluster_id, cluster in clusters.items():
        estimated_reuse = cluster.true_reusable_prefix(request.prefix_tokens, now)
        raw_queue_depth = cluster.queue_depth(now)
        predicted, details = router.predict_latency(cluster_id=cluster_id, request=request, estimated_reusable_tokens=estimated_reuse, raw_queue_depth=raw_queue_depth)
        if predicted < best_latency:
            best_latency = predicted
            best_cluster = cluster_id
            best_reuse = estimated_reuse
            best_details = details
    return RouteDecision(policy='oracle', cluster_id=best_cluster or next(iter(clusters)), estimated_reusable_tokens=best_reuse, predicted_latency=best_latency, details=best_details)
POLICIES = {'summary': summary_policy, 'orbit': orbit_policy, 'random': random_policy, 'load_only': load_only_policy, 'least_loaded': least_loaded_policy, 'round_robin': round_robin_policy, 'exact_prefix': exact_prefix_policy, 'oracle': oracle_policy}
