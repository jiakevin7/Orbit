from dataclasses import dataclass
from .hashing import prefix_hashes
from .models import ClusterSummary, RouteDecision

# Orbit uses one global calibrated cost model. Per-cluster differences enter only
# through router-to-cluster network cost and the summary state seen by the router.
_COST_FIELDS = (
    "fixed_overhead",
    "prefill_cost_per_token",
    "decode_cost_per_token",
    "queue_depth_penalty",
    "stale_penalty_per_second",
    "uncertainty_penalty_per_token",
    "missing_summary_penalty",
)
_REUSE_FIELDS = (
    "reuse_intercept",
    "reuse_estimate_scale",
    "reuse_match_level_bonus",
    "reuse_hotset_match_bonus",
)


@dataclass(frozen=True)
class RouterConfig:
    summary_depths: tuple[int, ...] = (
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        20,
        24,
        28,
        32,
        40,
        48,
        64,
        96,
        128,
        192,
        256,
        384,
        512,
    )
    fixed_overhead: float = 0.0
    prefill_cost_per_token: float = 1.0
    decode_cost_per_token: float = 2.0
    queue_depth_penalty: float = 4.0
    stale_penalty_per_second: float = 0.25
    uncertainty_penalty_per_token: float = 0.25
    routing_latency_weight: float = 0.25
    low_overlap_fraction: float = 0.1
    min_summary_overlap_tokens: int = 4
    max_summary_overlap_tokens: int = 32
    max_summary_age: float = 30.0
    missing_summary_penalty: float = 64.0
    summary_advantage_margin: float = 1.0
    summary_advantage_uncertainty_scale: float = 0.25
    reuse_intercept: float = 0.0
    reuse_estimate_scale: float = 1.0
    reuse_match_level_bonus: float = 0.0
    reuse_hotset_match_bonus: float = 0.0


@dataclass
class RouterView:
    summary: ClusterSummary
    received_at: float
    source: str


@dataclass(frozen=True)
class ReuseEstimate:
    calibrated_tokens: int
    raw_tokens: int
    matched_levels: int
    hotset_matched_levels: int
    uncertainty_gap: int


def build_prediction_details(
    *,
    config,
    network_cost,
    request,
    estimated_reusable_tokens,
    raw_queue_depth,
    metadata_age=0.0,
    uncertainty_gap=0,
    missing_summary=False,
    extra_uncertainty_penalty=0.0,
):
    # Keep every cost component in the decision trace so plots and calibration
    # can explain why a route was selected.
    coefficients = {
        field_name: float(getattr(config, field_name)) for field_name in _COST_FIELDS
    }
    remaining_prefill = max(0, request.input_length - estimated_reusable_tokens)
    queue_delay = raw_queue_depth * coefficients["queue_depth_penalty"]
    stale_penalty = max(0.0, metadata_age) * coefficients["stale_penalty_per_second"]
    uncertainty_penalty = (
        uncertainty_gap * coefficients["uncertainty_penalty_per_token"]
        + extra_uncertainty_penalty
    )
    missing_summary_penalty = (
        coefficients["missing_summary_penalty"] if missing_summary else 0.0
    )
    predicted_ttft = (
        network_cost
        + coefficients["fixed_overhead"]
        + queue_delay
        + remaining_prefill * coefficients["prefill_cost_per_token"]
        + stale_penalty
        + uncertainty_penalty
        + missing_summary_penalty
    )
    decode_penalty = request.continuation_tokens * coefficients["decode_cost_per_token"]
    predicted_latency = predicted_ttft + decode_penalty
    predicted_route_cost = (
        predicted_ttft + decode_penalty * config.routing_latency_weight
    )
    return {
        "network_cost": network_cost,
        "queue_delay": queue_delay,
        "decode_penalty": decode_penalty,
        "predicted_latency": predicted_latency,
        "predicted_ttft": predicted_ttft,
        "predicted_route_cost": predicted_route_cost,
        "raw_queue_depth": raw_queue_depth,
        "estimated_remaining_prefill_tokens": remaining_prefill,
        "stale_penalty": stale_penalty,
        "metadata_age": max(0.0, metadata_age),
        "uncertainty_gap": uncertainty_gap,
        "uncertainty_penalty": uncertainty_penalty,
        "missing_summary": 1.0 if missing_summary else 0.0,
        "missing_summary_penalty": missing_summary_penalty,
    }


class Router:
    def __init__(self, router_id, network_costs, config=None):
        self.router_id = router_id
        self.network_costs = dict(network_costs)
        self.config = config or RouterConfig()
        self.views = {}
        self.round_robin_cursor = 0

    def network_cost(self, cluster_id):
        return self.network_costs.get(cluster_id, float("inf"))

    def reachable_cluster_ids(self):
        return tuple(self.network_costs)

    def receive_summary(self, summary, received_at, source):
        # Summaries can arrive from the home cluster or router gossip; keep the
        # newest version and break ties by the summary creation timestamp.
        current = self.views.get(summary.cluster_id)
        if current is None:
            self.views[summary.cluster_id] = RouterView(summary, received_at, source)
            return
        if summary.version > current.summary.version:
            self.views[summary.cluster_id] = RouterView(summary, received_at, source)
            return
        if (
            summary.version == current.summary.version
            and summary.created_at >= current.summary.created_at
        ):
            self.views[summary.cluster_id] = RouterView(summary, received_at, source)

    def export_summaries(self):
        return {cluster_id: view.summary for cluster_id, view in self.views.items()}

    def summary_memory_bytes(self):
        return sum((view.summary.byte_size for view in self.views.values()))

    def estimate_reusable_prefix(self, tokens, summary):
        # Probe hierarchical Bloom filters and hotsets from shallow to deep.
        # The first missing level bounds the reusable prefix estimate.
        hashes = prefix_hashes(tokens, summary.depths)
        deepest_match = 0
        matched_levels = 0
        hotset_matched_levels = 0
        next_depth: int | None = None
        for depth in sorted(summary.depths):
            if depth > len(tokens):
                break
            prefix_hash = hashes[depth]
            hotset_match = prefix_hash in summary.hot_prefix_hashes.get(depth, ())
            if hotset_match or summary.filters[depth].contains(prefix_hash):
                deepest_match = depth
                matched_levels += 1
                if hotset_match:
                    hotset_matched_levels += 1
            else:
                next_depth = depth
                break
        raw_estimate = self._interpolate_reuse_estimate(
            deepest_match=deepest_match,
            next_depth=next_depth,
            request_length=len(tokens),
        )
        calibrated_estimate = self.calibrate_reuse(
            request_length=len(tokens),
            raw_estimated_reusable_tokens=raw_estimate,
            matched_levels=matched_levels,
            hotset_matched_levels=hotset_matched_levels,
        )
        uncertainty_gap = self._uncertainty_gap(
            request_length=len(tokens),
            estimated_reuse=raw_estimate,
            depths=summary.depths,
        )
        return ReuseEstimate(
            calibrated_tokens=calibrated_estimate,
            raw_tokens=raw_estimate,
            matched_levels=matched_levels,
            hotset_matched_levels=hotset_matched_levels,
            uncertainty_gap=uncertainty_gap,
        )

    def calibrate_reuse(
        self,
        request_length,
        raw_estimated_reusable_tokens,
        matched_levels,
        hotset_matched_levels,
    ):
        if (
            raw_estimated_reusable_tokens <= 0
            and matched_levels <= 0
            and (hotset_matched_levels <= 0)
        ):
            return 0
        coefficients = {
            field_name: float(getattr(self.config, field_name))
            for field_name in _REUSE_FIELDS
        }
        predicted = (
            coefficients["reuse_intercept"]
            + raw_estimated_reusable_tokens * coefficients["reuse_estimate_scale"]
            + matched_levels * coefficients["reuse_match_level_bonus"]
            + hotset_matched_levels * coefficients["reuse_hotset_match_bonus"]
        )
        return max(0, min(request_length, int(round(predicted))))

    def least_loaded_route(self, request, cluster_ids, now):
        cluster_ids = tuple(cluster_ids)
        if not cluster_ids:
            raise ValueError(f"router {self.router_id} has no reachable clusters")

        # Baseline route score ignores prefix locality but still uses visible
        # queue state, summary age, and network cost.
        best_cluster = None
        best_cost = float("inf")
        best_details: dict[str, float] = {}

        for cluster_id in cluster_ids:
            view = self.views.get(cluster_id)
            raw_queue_depth = 0
            metadata_age = 0.0

            if view is not None:
                raw_queue_depth = view.summary.queue_depth
                metadata_age = max(0.0, now - view.summary.created_at)

            predicted_latency, details = self.predict_latency(
                cluster_id=cluster_id,
                request=request,
                estimated_reusable_tokens=0,
                raw_queue_depth=raw_queue_depth,
                metadata_age=metadata_age,
                uncertainty_gap=0,
                missing_summary=view is None,
            )
            route_score = self._route_score_from_details(details)
            if route_score < best_cost:
                best_cost = route_score
                best_cluster = cluster_id
                best_details = details
        return RouteDecision(
            policy="least_loaded",
            cluster_id=best_cluster or next(iter(cluster_ids)),
            estimated_reusable_tokens=0,
            predicted_latency=float(best_details.get("predicted_latency", best_cost)),
            details=best_details,
        )

    def predict_latency(
        self,
        cluster_id,
        request,
        estimated_reusable_tokens,
        raw_queue_depth,
        metadata_age=0.0,
        uncertainty_gap=0,
        missing_summary=False,
        extra_uncertainty_penalty=0.0,
    ):
        details = build_prediction_details(
            config=self.config,
            network_cost=self.network_cost(cluster_id),
            request=request,
            estimated_reusable_tokens=estimated_reusable_tokens,
            raw_queue_depth=raw_queue_depth,
            metadata_age=metadata_age,
            uncertainty_gap=uncertainty_gap,
            missing_summary=missing_summary,
            extra_uncertainty_penalty=extra_uncertainty_penalty,
        )
        return (float(details["predicted_latency"]), details)

    def route(self, request, cluster_ids, now):
        # Orbit first computes the least-loaded fallback, then only takes a
        # prefix-aware route when the summary signal is fresh and large enough.
        cluster_ids = tuple(cluster_ids)
        if not cluster_ids:
            raise ValueError(f"router {self.router_id} has no reachable clusters")

        request_length = request.input_length
        least_loaded_decision = self.least_loaded_route(request, cluster_ids, now)
        candidates: list[RouteDecision] = []
        has_fresh_summary = False

        # Build one candidate per reachable cluster using only the summaries
        # currently visible to this router.
        for cluster_id in cluster_ids:
            summary_view = self.views.get(cluster_id)
            if summary_view is None:
                uncertainty_gap = request_length
                predicted_latency, details = self.predict_latency(
                    cluster_id=cluster_id,
                    request=request,
                    estimated_reusable_tokens=0,
                    raw_queue_depth=0,
                    metadata_age=0.0,
                    uncertainty_gap=uncertainty_gap,
                    missing_summary=True,
                )
                candidates.append(
                    RouteDecision(
                        policy="orbit",
                        cluster_id=cluster_id,
                        estimated_reusable_tokens=0,
                        predicted_latency=predicted_latency,
                        used_fallback=True,
                        details=details,
                    )
                )
                continue

            summary = summary_view.summary
            metadata_age = max(0.0, now - summary.created_at)
            if metadata_age <= self.config.max_summary_age:
                has_fresh_summary = True

            reuse_estimate = self.estimate_reusable_prefix(
                request.prefix_tokens, summary
            )
            estimated_reuse = reuse_estimate.calibrated_tokens
            raw_queue_depth = summary.queue_depth
            uncertainty_gap = reuse_estimate.uncertainty_gap
            extra_uncertainty_penalty = 0.0

            if reuse_estimate.matched_levels == 0:
                extra_uncertainty_penalty = self.config.queue_depth_penalty * 0.5

            predicted_latency, details = self.predict_latency(
                cluster_id=cluster_id,
                request=request,
                estimated_reusable_tokens=estimated_reuse,
                raw_queue_depth=raw_queue_depth,
                metadata_age=metadata_age,
                uncertainty_gap=uncertainty_gap,
                missing_summary=False,
                extra_uncertainty_penalty=extra_uncertainty_penalty,
            )
            details.update(
                {
                    "raw_estimated_reusable_tokens": reuse_estimate.raw_tokens,
                    "summary_matched_levels": reuse_estimate.matched_levels,
                    "hotset_matched_levels": reuse_estimate.hotset_matched_levels,
                }
            )
            candidates.append(
                RouteDecision(
                    policy="orbit",
                    cluster_id=cluster_id,
                    estimated_reusable_tokens=estimated_reuse,
                    predicted_latency=predicted_latency,
                    used_fallback=False,
                    details=details,
                )
            )

        # Guardrail 1: if prefix evidence is stale or too weak, do not let
        # approximate summaries override the least-loaded decision.
        best_overlap = max(
            (candidate.estimated_reusable_tokens for candidate in candidates)
        )
        if not has_fresh_summary or best_overlap < self._summary_overlap_threshold(
            request_length
        ):
            return self._summary_fallback(
                least_loaded_decision, "insufficient_summary_overlap"
            )

        # Guardrail 2: require a margin over least-loaded to avoid chasing noisy
        # approximate matches that do not materially improve route cost.
        best_summary = min(candidates, key=self._route_sort_key)
        required_margin = self._summary_advantage_margin(best_summary)
        if self._route_score(best_summary) + required_margin >= self._route_score(
            least_loaded_decision
        ):
            return self._summary_fallback(
                least_loaded_decision, "summary_advantage_too_small"
            )
        return best_summary

    def _summary_fallback(self, least_loaded_decision, reason):
        details = dict(least_loaded_decision.details)
        details["fallback_reason"] = reason
        return RouteDecision(
            policy="orbit",
            cluster_id=least_loaded_decision.cluster_id,
            estimated_reusable_tokens=0,
            predicted_latency=least_loaded_decision.predicted_latency,
            used_fallback=True,
            details=details,
        )

    def _uncertainty_gap(self, request_length, estimated_reuse, depths):
        next_depth = request_length
        for depth in sorted(depths):
            if estimated_reuse < depth <= request_length:
                next_depth = depth
                break
        return max(0, next_depth - estimated_reuse)

    def _summary_overlap_threshold(self, request_length):
        fractional_threshold = request_length * self.config.low_overlap_fraction
        threshold = max(
            float(max(0, self.config.min_summary_overlap_tokens)),
            max(0.0, fractional_threshold),
        )
        max_threshold = float(max(0, self.config.max_summary_overlap_tokens))
        if max_threshold > 0:
            threshold = min(threshold, max_threshold)
        return threshold

    def _interpolate_reuse_estimate(self, *, deepest_match, next_depth, request_length):
        if deepest_match <= 0:
            return 0
        if next_depth is None or next_depth <= deepest_match:
            return min(deepest_match, request_length)
        gap = next_depth - deepest_match
        return min(request_length, deepest_match + max(1, gap // 2))

    def _route_score(self, decision):
        return self._route_score_from_details(
            decision.details, default=decision.predicted_latency
        )

    def _route_score_from_details(self, details, default=None):
        if "predicted_route_cost" in details:
            return float(details["predicted_route_cost"])
        if default is not None:
            return float(default)
        return float("inf")

    def _route_sort_key(self, decision):
        return (
            self._route_score(decision),
            float(decision.details.get("predicted_ttft", decision.predicted_latency)),
        )

    def _summary_advantage_margin(self, decision):
        uncertainty = float(decision.details.get("uncertainty_penalty", 0.0))
        return (
            self.config.summary_advantage_margin
            + uncertainty * self.config.summary_advantage_uncertainty_scale
        )
