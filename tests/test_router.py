import unittest
from orbit.cluster import Cluster, ClusterConfig
from orbit.models import Request
from orbit.router import Router, RouterConfig


class RouterTests(unittest.TestCase):
    def test_monotonic_summary_matching_estimates_deepest_valid_depth(self):
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
            "router-a", {"cluster-a": 5.0}, RouterConfig(summary_depths=(2, 4, 6))
        )
        router.receive_summary(summary, summary.created_at, source="cluster-a")
        reuse_estimate = router.estimate_reusable_prefix((1, 2, 3, 4, 9, 9), summary)
        self.assertEqual(reuse_estimate.raw_tokens, 5)
        self.assertEqual(reuse_estimate.estimated_tokens, 5)
        self.assertEqual(reuse_estimate.matched_levels, 2)

    def test_router_prefers_cluster_with_better_predicted_reuse(self):
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
            RouterConfig(
                summary_depths=(2, 4),
                low_overlap_fraction=0.0,
                summary_advantage_margin=0.0,
            ),
        )
        router.receive_summary(
            fast_cluster.publish_summary(execution.finished_at),
            execution.finished_at,
            "cluster-fast",
        )
        router.receive_summary(
            slow_cluster.publish_summary(execution.finished_at),
            execution.finished_at,
            "cluster-slow",
        )
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

    def test_router_uses_absolute_overlap_threshold_for_long_prompts(self):
        cached_cluster = Cluster(
            "cluster-cached", ClusterConfig(summary_depths=(8, 16))
        )
        empty_cluster = Cluster("cluster-empty", ClusterConfig(summary_depths=(8, 16)))
        seed_tokens = tuple(range(100))
        seed_request = Request(
            request_id="cached",
            arrival_time=0.0,
            router_id="router-a",
            prefix_tokens=seed_tokens,
            continuation_tokens=1,
        )
        execution = cached_cluster.execute(seed_request)
        cached_cluster.advance_time(execution.finished_at)
        router = Router(
            "router-a",
            {"cluster-cached": 10.0, "cluster-empty": 10.0},
            RouterConfig(summary_depths=(8, 16)),
        )
        router.receive_summary(
            cached_cluster.publish_summary(execution.finished_at),
            execution.finished_at,
            "cluster-cached",
        )
        router.receive_summary(
            empty_cluster.publish_summary(execution.finished_at),
            execution.finished_at,
            "cluster-empty",
        )
        request_tokens = tuple(list(range(8)) + list(range(1000, 1092)))
        decision = router.route(
            Request(
                request_id="new",
                arrival_time=execution.finished_at + 1.0,
                router_id="router-a",
                prefix_tokens=request_tokens,
                continuation_tokens=1,
            ),
            ["cluster-cached", "cluster-empty"],
            execution.finished_at + 1.0,
        )
        self.assertEqual(decision.cluster_id, "cluster-cached")
        self.assertGreaterEqual(decision.estimated_reusable_tokens, 8)


if __name__ == "__main__":
    unittest.main()
