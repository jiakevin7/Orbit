from __future__ import annotations

import random
import unittest

from orbit.cluster import Cluster, ClusterConfig
from orbit.models import Request
from orbit.policies import (
    exact_prefix_policy,
    oracle_policy,
    vllm_kv_mock_policy,
    vllm_prefix_mock_policy,
)
from orbit.router import Router, RouterConfig


class MockPolicyTests(unittest.TestCase):
    def test_vllm_prefix_mock_prefers_exact_match_before_load(self) -> None:
        matching_cluster = Cluster(
            "cluster-match",
            ClusterConfig(
                cache_capacity=8,
                concurrency=1,
                prefill_cost_per_token=1.0,
                decode_cost_per_token=1.0,
            ),
        )
        empty_cluster = Cluster(
            "cluster-empty",
            ClusterConfig(
                cache_capacity=8,
                concurrency=1,
                prefill_cost_per_token=1.0,
                decode_cost_per_token=1.0,
            ),
        )
        router = Router(
            "router-0",
            {"cluster-match": 0.0, "cluster-empty": 0.0},
            RouterConfig(
                fixed_overhead=0.0,
                prefill_cost_per_token=1.0,
                decode_cost_per_token=1.0,
                queue_depth_penalty=100.0,
                low_overlap_fraction=0.0,
            ),
        )

        seed = Request(
            request_id="seed",
            arrival_time=0.0,
            router_id="router-0",
            prefix_tokens=(1, 2, 3, 4),
            continuation_tokens=1,
        )
        seed_execution = matching_cluster.execute(seed)
        matching_cluster.advance_time(seed_execution.finished_at)

        blocker = Request(
            request_id="blocker",
            arrival_time=seed_execution.finished_at + 1.0,
            router_id="router-0",
            prefix_tokens=(9, 9, 9, 9),
            continuation_tokens=20,
        )
        blocker_execution = matching_cluster.execute(blocker)

        request_obj = Request(
            request_id="new",
            arrival_time=blocker.arrival_time + 1.0,
            router_id="router-0",
            prefix_tokens=(1, 2, 3, 4),
            continuation_tokens=1,
        )

        decision = vllm_prefix_mock_policy(
            router,
            request_obj,
            {"cluster-match": matching_cluster, "cluster-empty": empty_cluster},
            request_obj.arrival_time,
            random.Random(0),
        )
        exact_decision = exact_prefix_policy(
            router,
            request_obj,
            {"cluster-match": matching_cluster, "cluster-empty": empty_cluster},
            request_obj.arrival_time,
            random.Random(0),
        )

        self.assertGreater(blocker_execution.finished_at, request_obj.arrival_time)
        self.assertEqual(decision.cluster_id, "cluster-match")
        self.assertEqual(decision.estimated_reusable_tokens, 4)
        self.assertEqual(exact_decision.cluster_id, "cluster-empty")

    def test_vllm_kv_mock_prefers_more_reuse_before_load(self) -> None:
        deep_cluster = Cluster(
            "cluster-deep",
            ClusterConfig(
                cache_capacity=8,
                concurrency=1,
                prefill_cost_per_token=1.0,
                decode_cost_per_token=1.0,
            ),
        )
        shallow_cluster = Cluster(
            "cluster-shallow",
            ClusterConfig(
                cache_capacity=8,
                concurrency=1,
                prefill_cost_per_token=1.0,
                decode_cost_per_token=1.0,
            ),
        )
        router = Router(
            "router-0",
            {"cluster-deep": 0.0, "cluster-shallow": 0.0},
            RouterConfig(
                fixed_overhead=0.0,
                prefill_cost_per_token=1.0,
                decode_cost_per_token=1.0,
                queue_depth_penalty=100.0,
                low_overlap_fraction=0.0,
            ),
        )

        deep_seed = Request(
            request_id="deep-seed",
            arrival_time=0.0,
            router_id="router-0",
            prefix_tokens=(1, 2, 3, 4, 5, 6),
            continuation_tokens=1,
        )
        deep_execution = deep_cluster.execute(deep_seed)
        deep_cluster.advance_time(deep_execution.finished_at)

        shallow_seed = Request(
            request_id="shallow-seed",
            arrival_time=0.0,
            router_id="router-0",
            prefix_tokens=(1, 2, 3),
            continuation_tokens=1,
        )
        shallow_execution = shallow_cluster.execute(shallow_seed)
        shallow_cluster.advance_time(shallow_execution.finished_at)

        blocker = Request(
            request_id="deep-blocker",
            arrival_time=deep_execution.finished_at + 1.0,
            router_id="router-0",
            prefix_tokens=(7, 7, 7, 7),
            continuation_tokens=20,
        )
        blocker_execution = deep_cluster.execute(blocker)

        request_obj = Request(
            request_id="new",
            arrival_time=blocker.arrival_time + 1.0,
            router_id="router-0",
            prefix_tokens=(1, 2, 3, 4, 5, 9),
            continuation_tokens=1,
        )

        decision = vllm_kv_mock_policy(
            router,
            request_obj,
            {"cluster-deep": deep_cluster, "cluster-shallow": shallow_cluster},
            request_obj.arrival_time,
            random.Random(0),
        )
        oracle_decision = oracle_policy(
            router,
            request_obj,
            {"cluster-deep": deep_cluster, "cluster-shallow": shallow_cluster},
            request_obj.arrival_time,
            random.Random(0),
        )

        self.assertGreater(blocker_execution.finished_at, request_obj.arrival_time)
        self.assertEqual(decision.cluster_id, "cluster-deep")
        self.assertEqual(decision.estimated_reusable_tokens, 5)
        self.assertEqual(oracle_decision.cluster_id, "cluster-shallow")


if __name__ == "__main__":
    unittest.main()
