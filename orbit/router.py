from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

from .hashing import prefix_hashes
from .models import ClusterSummary, Request, RouteDecision


@dataclass(frozen=True)
class RouterConfig:
    summary_depths: tuple[int, ...] = (64, 128, 256, 512)
    fixed_overhead: float = 0.0
    prefill_cost_per_token: float = 1.0
    decode_cost_per_token: float = 2.0
    queue_depth_penalty: float = 4.0
    stale_penalty_per_second: float = 0.25
    uncertainty_penalty_per_token: float = 0.25
    low_overlap_fraction: float = 0.1
    max_summary_age: float = 30.0
    missing_summary_penalty: float = 64.0


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
        return self.network_costs.get(cluster_id, 0.0)

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

    def route(self, request: Request, cluster_ids: Iterable[str], now: float) -> RouteDecision:
        request_length = request.input_length
        candidates: list[RouteDecision] = []
        has_fresh_summary = False

        for cluster_id in cluster_ids:
            summary_view = self.views.get(cluster_id)
            network = self.network_cost(cluster_id)
            if summary_view is None:
                estimated_remaining_prefill = request_length
                uncertainty_gap = request_length
                raw_queue_depth = 0
                predicted_latency = (
                    network
                    + self.config.fixed_overhead
                    + estimated_remaining_prefill * self.config.prefill_cost_per_token
                    + request.continuation_tokens * self.config.decode_cost_per_token
                    + uncertainty_gap * self.config.uncertainty_penalty_per_token
                    + self.config.missing_summary_penalty
                )
                candidates.append(
                    RouteDecision(
                        policy="summary",
                        cluster_id=cluster_id,
                        estimated_reusable_tokens=0,
                        predicted_latency=predicted_latency,
                        used_fallback=True,
                        details={
                            "network_cost": network,
                            "queue_delay": 0.0,
                            "raw_queue_depth": raw_queue_depth,
                            "estimated_remaining_prefill_tokens": estimated_remaining_prefill,
                            "stale_penalty": 0.0,
                            "metadata_age": 0.0,
                            "uncertainty_gap": uncertainty_gap,
                            "uncertainty_penalty": uncertainty_gap * self.config.uncertainty_penalty_per_token,
                            "missing_summary": 1.0,
                            "missing_summary_penalty": self.config.missing_summary_penalty,
                        },
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
            queue_delay = raw_queue_depth * self.config.queue_depth_penalty
            stale_penalty = metadata_age * self.config.stale_penalty_per_second
            uncertainty_gap = self._uncertainty_gap(request_length, estimated_reuse, summary.depths)
            uncertainty_penalty = uncertainty_gap * self.config.uncertainty_penalty_per_token
            if matched_levels == 0:
                uncertainty_penalty += self.config.queue_depth_penalty * 0.5

            predicted_latency = (
                network
                + self.config.fixed_overhead
                + queue_delay
                + self.config.prefill_cost_per_token * estimated_remaining_prefill
                + request.continuation_tokens * self.config.decode_cost_per_token
                + stale_penalty
                + uncertainty_penalty
            )
            candidates.append(
                RouteDecision(
                    policy="summary",
                    cluster_id=cluster_id,
                    estimated_reusable_tokens=estimated_reuse,
                    predicted_latency=predicted_latency,
                        used_fallback=False,
                        details={
                            "network_cost": network,
                            "queue_delay": queue_delay,
                            "raw_queue_depth": raw_queue_depth,
                            "estimated_remaining_prefill_tokens": estimated_remaining_prefill,
                            "stale_penalty": stale_penalty,
                            "metadata_age": metadata_age,
                            "uncertainty_gap": uncertainty_gap,
                            "uncertainty_penalty": uncertainty_penalty,
                            "missing_summary": 0.0,
                            "missing_summary_penalty": 0.0,
                        },
                    )
                )

        best_overlap = max(candidate.estimated_reusable_tokens for candidate in candidates)
        if not has_fresh_summary or best_overlap < request_length * self.config.low_overlap_fraction:
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
