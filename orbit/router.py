from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Sequence

from .hashing import prefix_hashes
from .models import ClusterSummary, Request, RouteDecision


_COST_FIELDS = (
    "fixed_overhead",
    "prefill_cost_per_token",
    "decode_cost_per_token",
    "queue_depth_penalty",
    "stale_penalty_per_second",
    "uncertainty_penalty_per_token",
    "missing_summary_penalty",
)


@dataclass(frozen=True)
class RouterConfig:
    summary_depths: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512)
    fixed_overhead: float = 0.0
    prefill_cost_per_token: float = 1.0
    decode_cost_per_token: float = 2.0
    queue_depth_penalty: float = 4.0
    stale_penalty_per_second: float = 0.25
    uncertainty_penalty_per_token: float = 0.25
    low_overlap_fraction: float = 0.1
    min_summary_overlap_tokens: int = 8
    max_summary_age: float = 30.0
    missing_summary_penalty: float = 64.0
    cluster_overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class RouterView:
    summary: ClusterSummary
    received_at: float
    source: str


class Router:
    def __init__(
        self,
        router_id: str,
        network_costs: Dict[str, float],
        config: RouterConfig | None = None,
    ) -> None:
        self.router_id = router_id
        self.network_costs = dict(network_costs)
        self.config = config or RouterConfig()
        self.views: Dict[str, RouterView] = {}

    def network_cost(self, cluster_id: str) -> float:
        return self.network_costs.get(cluster_id, float("inf"))

    def reachable_cluster_ids(self) -> tuple[str, ...]:
        return tuple(self.network_costs)

    def receive_summary(self, summary: ClusterSummary, received_at: float, source: str) -> None:
        current = self.views.get(summary.cluster_id)
        if current is None:
            self.views[summary.cluster_id] = RouterView(summary, received_at, source)
            return

        if summary.version > current.summary.version:
            self.views[summary.cluster_id] = RouterView(summary, received_at, source)
            return

        if summary.version == current.summary.version and summary.created_at >= current.summary.created_at:
            self.views[summary.cluster_id] = RouterView(summary, received_at, source)

    def export_summaries(self) -> Dict[str, ClusterSummary]:
        return {cluster_id: view.summary for cluster_id, view in self.views.items()}

    def summary_memory_bytes(self) -> int:
        return sum(view.summary.byte_size for view in self.views.values())

    def estimate_reusable_prefix(
        self,
        tokens: Sequence[int],
        summary: ClusterSummary,
    ) -> tuple[int, int]:
        hashes = prefix_hashes(tokens, summary.depths)
        deepest_match = 0
        matched_levels = 0
        for depth in sorted(summary.depths):
            if depth > len(tokens):
                break
            if summary.filters[depth].contains(hashes[depth]):
                deepest_match = depth
                matched_levels += 1
            else:
                break
        return deepest_match, matched_levels

    def load_only_route(self, request: Request, cluster_ids: Iterable[str], now: float) -> RouteDecision:
        cluster_ids = tuple(cluster_ids)
        if not cluster_ids:
            raise ValueError(f"router {self.router_id} has no reachable clusters")
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
            if predicted_latency < best_cost:
                best_cost = predicted_latency
                best_cluster = cluster_id
                best_details = details
        return RouteDecision(
            policy="load_only",
            cluster_id=best_cluster or next(iter(cluster_ids)),
            estimated_reusable_tokens=0,
            predicted_latency=best_cost,
            details=best_details,
        )

    def predict_latency(
        self,
        cluster_id: str,
        request: Request,
        estimated_reusable_tokens: int,
        raw_queue_depth: int,
        metadata_age: float = 0.0,
        uncertainty_gap: int = 0,
        missing_summary: bool = False,
        extra_uncertainty_penalty: float = 0.0,
    ) -> tuple[float, dict[str, float]]:
        coefficients = self._coefficients_for_cluster(cluster_id)
        remaining_prefill = max(0, request.input_length - estimated_reusable_tokens)
        queue_delay = raw_queue_depth * coefficients["queue_depth_penalty"]
        stale_penalty = max(0.0, metadata_age) * coefficients["stale_penalty_per_second"]
        uncertainty_penalty = (
            uncertainty_gap * coefficients["uncertainty_penalty_per_token"]
            + extra_uncertainty_penalty
        )
        missing_summary_penalty = coefficients["missing_summary_penalty"] if missing_summary else 0.0
        predicted_latency = (
            self.network_cost(cluster_id)
            + coefficients["fixed_overhead"]
            + queue_delay
            + remaining_prefill * coefficients["prefill_cost_per_token"]
            + request.continuation_tokens * coefficients["decode_cost_per_token"]
            + stale_penalty
            + uncertainty_penalty
            + missing_summary_penalty
        )
        return predicted_latency, {
            "network_cost": self.network_cost(cluster_id),
            "queue_delay": queue_delay,
            "raw_queue_depth": raw_queue_depth,
            "estimated_remaining_prefill_tokens": remaining_prefill,
            "stale_penalty": stale_penalty,
            "metadata_age": max(0.0, metadata_age),
            "uncertainty_gap": uncertainty_gap,
            "uncertainty_penalty": uncertainty_penalty,
            "missing_summary": 1.0 if missing_summary else 0.0,
            "missing_summary_penalty": missing_summary_penalty,
        }

    def route(self, request: Request, cluster_ids: Iterable[str], now: float) -> RouteDecision:
        cluster_ids = tuple(cluster_ids)
        if not cluster_ids:
            raise ValueError(f"router {self.router_id} has no reachable clusters")
        request_length = request.input_length
        candidates: list[RouteDecision] = []
        has_fresh_summary = False

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
                        policy="summary",
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
            estimated_reuse, matched_levels = self.estimate_reusable_prefix(
                request.prefix_tokens,
                summary,
            )
            estimated_remaining_prefill = max(0, request_length - estimated_reuse)
            raw_queue_depth = summary.queue_depth
            uncertainty_gap = self._uncertainty_gap(request_length, estimated_reuse, summary.depths)
            extra_uncertainty_penalty = 0.0
            if matched_levels == 0:
                extra_uncertainty_penalty = self._coefficients_for_cluster(cluster_id)["queue_depth_penalty"] * 0.5
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
            candidates.append(
                RouteDecision(
                    policy="summary",
                    cluster_id=cluster_id,
                    estimated_reusable_tokens=estimated_reuse,
                    predicted_latency=predicted_latency,
                    used_fallback=False,
                    details=details,
                )
            )

        best_overlap = max(candidate.estimated_reusable_tokens for candidate in candidates)
        if not has_fresh_summary or best_overlap < self._summary_overlap_threshold(request_length):
            return self._load_fallback(candidates, request.continuation_tokens)

        return min(candidates, key=lambda candidate: candidate.predicted_latency)

    def _load_fallback(self, candidates: Sequence[RouteDecision], continuation_tokens: int) -> RouteDecision:
        scored: list[RouteDecision] = []
        for candidate in candidates:
            load_cost = (
                candidate.details.get("network_cost", 0.0)
                + self.config.fixed_overhead
                + candidate.details.get("queue_delay", 0.0)
                + candidate.details.get("estimated_remaining_prefill_tokens", 0.0) * self.config.prefill_cost_per_token
                + continuation_tokens * self.config.decode_cost_per_token
                + candidate.details.get("stale_penalty", 0.0)
                + candidate.details.get("uncertainty_penalty", 0.0)
                + candidate.details.get("missing_summary_penalty", 0.0)
            )
            scored.append(
                RouteDecision(
                    policy=candidate.policy,
                    cluster_id=candidate.cluster_id,
                    estimated_reusable_tokens=candidate.estimated_reusable_tokens,
                    predicted_latency=load_cost,
                    used_fallback=True,
                    details=dict(candidate.details),
                )
            )
        return min(scored, key=lambda candidate: candidate.predicted_latency)

    def _uncertainty_gap(
        self,
        request_length: int,
        estimated_reuse: int,
        depths: Sequence[int],
    ) -> int:
        next_depth = request_length
        for depth in sorted(depths):
            if estimated_reuse < depth <= request_length:
                next_depth = depth
                break
        return max(0, next_depth - estimated_reuse)

    def _summary_overlap_threshold(self, request_length: int) -> float:
        fractional_threshold = request_length * self.config.low_overlap_fraction
        if fractional_threshold <= 0:
            return 0.0
        return min(
            fractional_threshold,
            float(max(0, self.config.min_summary_overlap_tokens)),
        )

    def _coefficients_for_cluster(self, cluster_id: str) -> dict[str, float]:
        overrides = self.config.cluster_overrides.get(cluster_id, {})
        return {
            field_name: float(overrides.get(field_name, getattr(self.config, field_name)))
            for field_name in _COST_FIELDS
        }
