from __future__ import annotations

import unittest

from orbit.cluster import Cluster, ClusterConfig
from orbit.models import Request
from orbit.router import Router, RouterConfig


class RouterTests(unittest.TestCase):
    def test_monotonic_summary_matching_estimates_deepest_valid_depth(self) -> None:
        cluster = Cluster(
            "cluster-a",
            ClusterConfig(
                cache_capacity=8,
                summary_depths=(2, 4, 6),
                bloom_bits=2048,
                bloom_hashes=4,
            ),
        )
        request = Request(
            request_id="seed",
            arrival_time=0.0,
            router_id="router-a",
            prefix_tokens=(1, 2, 3, 4, 5, 6),
            continuation_tokens=1,
        )
        execution = cluster.execute(request)
        cluster.advance_time(execution.finished_at)
        summary = cluster.publish_summary(execution.finished_at)

        router = Router(
            "router-a",
            {"cluster-a": 5.0},
            RouterConfig(summary_depths=(2, 4, 6)),
        )
        router.receive_summary(summary, summary.created_at, source="cluster-a")

        estimated_reuse, matched_levels = router.estimate_reusable_prefix((1, 2, 3, 4, 9, 9), summary)
        self.assertEqual(estimated_reuse, 4)
        self.assertEqual(matched_levels, 2)

    def test_router_prefers_cluster_with_better_predicted_reuse(self) -> None:
        fast_cluster = Cluster(
            "cluster-fast",
            ClusterConfig(summary_depths=(2, 4), decode_cost_per_token=1.0),
        )
        slow_cluster = Cluster(
            "cluster-slow",
            ClusterConfig(summary_depths=(2, 4), decode_cost_per_token=1.0),
        )

        seed_request = Request(
            request_id="cached",
            arrival_time=0.0,
            router_id="router-a",
            prefix_tokens=(5, 6, 7, 8),
            continuation_tokens=1,
        )
        execution = fast_cluster.execute(seed_request)
        fast_cluster.advance_time(execution.finished_at)

        router = Router(
            "router-a",
            {"cluster-fast": 10.0, "cluster-slow": 10.0},
            RouterConfig(summary_depths=(2, 4), low_overlap_fraction=0.0),
        )
        router.receive_summary(fast_cluster.publish_summary(execution.finished_at), execution.finished_at, "cluster-fast")
        router.receive_summary(slow_cluster.publish_summary(execution.finished_at), execution.finished_at, "cluster-slow")

        decision = router.route(
            Request(
                request_id="new",
                arrival_time=execution.finished_at + 1.0,
                router_id="router-a",
                prefix_tokens=(5, 6, 7, 8),
                continuation_tokens=1,
            ),
            ["cluster-fast", "cluster-slow"],
            execution.finished_at + 1.0,
        )

        self.assertEqual(decision.cluster_id, "cluster-fast")
        self.assertEqual(decision.estimated_reusable_tokens, 4)


if __name__ == "__main__":
    unittest.main()

