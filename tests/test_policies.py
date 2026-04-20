from __future__ import annotations

import random
import unittest

from orbit.cluster import Cluster, ClusterConfig
from orbit.models import Request
from orbit.policies import (
    exact_prefix_policy,
    oracle_policy,
    round_robin_policy,
    vllm_kv_mock_policy,
    vllm_prefix_mock_policy,
)
from orbit.router import Router, RouterConfig


class MockPolicyTests(unittest.TestCase):
    def test_round_robin_policy_cycles_clusters(self) -> None:
        cluster_a = Cluster("cluster-a", ClusterConfig())
        cluster_b = Cluster("cluster-b", ClusterConfig())
        router = Router("router-0", {"cluster-a": 0.0, "cluster-b": 0.0}, RouterConfig())
        request_obj = Request(
            request_id="req",
            arrival_time=0.0,
            router_id="router-0",
            prefix_tokens=(1, 2, 3),
            continuation_tokens=1,
        )

        first = round_robin_policy(
            router,
            request_obj,
            {"cluster-a": cluster_a, "cluster-b": cluster_b},
            request_obj.arrival_time,
            random.Random(0),
        )
        second = round_robin_policy(
            router,
            request_obj,
            {"cluster-a": cluster_a, "cluster-b": cluster_b},
            request_obj.arrival_time,
            random.Random(0),
        )
        third = round_robin_policy(
            router,
            request_obj,
            {"cluster-a": cluster_a, "cluster-b": cluster_b},
            request_obj.arrival_time,
            random.Random(0),
        )

        self.assertEqual(first.cluster_id, "cluster-a")
        self.assertEqual(second.cluster_id, "cluster-b")
        self.assertEqual(third.cluster_id, "cluster-a")

    def test_vllm_prefix_mock_sticks_to_first_chosen_cluster(self) -> None:
        sticky_cluster = Cluster(
            "cluster-sticky",
            ClusterConfig(
                cache_capacity=8,
                concurrency=1,
                prefill_cost_per_token=1.0,
                decode_cost_per_token=1.0,
            ),
        )
        other_cluster = Cluster(
            "cluster-other",
            ClusterConfig(
                cache_capacity=8,
                concurrency=1,
                prefill_cost_per_token=1.0,
                decode_cost_per_token=1.0,
            ),
        )
        router = Router(
            "router-0",
            {"cluster-sticky": 0.0, "cluster-other": 0.0},
            RouterConfig(
                fixed_overhead=0.0,
                prefill_cost_per_token=1.0,
                decode_cost_per_token=1.0,
                queue_depth_penalty=100.0,
                low_overlap_fraction=0.0,
            ),
        )

        shared_prefix = tuple(range(1, 17))
        other_seed = Request(
            request_id="other-seed",
            arrival_time=0.0,
            router_id="router-0",
            prefix_tokens=shared_prefix,
            continuation_tokens=1,
        )
        other_seed_execution = other_cluster.execute(other_seed)
        other_cluster.advance_time(other_seed_execution.finished_at)

        other_blocker = Request(
            request_id="other-blocker",
            arrival_time=other_seed_execution.finished_at + 1.0,
            router_id="router-0",
            prefix_tokens=(9, 9, 9, 9),
            continuation_tokens=5,
        )
        other_blocker_execution = other_cluster.execute(other_blocker)

        first_request = Request(
            request_id="first",
            arrival_time=other_blocker.arrival_time + 1.0,
            router_id="router-0",
            prefix_tokens=shared_prefix,
            continuation_tokens=1,
        )

        first_decision = vllm_prefix_mock_policy(
            router,
            first_request,
            {"cluster-sticky": sticky_cluster, "cluster-other": other_cluster},
            first_request.arrival_time,
            random.Random(0),
        )

        self.assertGreater(other_blocker_execution.finished_at, first_request.arrival_time)
        self.assertEqual(first_decision.cluster_id, "cluster-sticky")
        self.assertEqual(first_decision.estimated_reusable_tokens, 0)
        self.assertTrue(first_decision.used_fallback)

        first_execution = sticky_cluster.execute(first_request)
        sticky_cluster.advance_time(first_execution.finished_at)
        other_cluster.advance_time(other_blocker_execution.finished_at)

        sticky_blocker = Request(
            request_id="sticky-blocker",
            arrival_time=first_execution.finished_at + 1.0,
            router_id="router-0",
            prefix_tokens=(7, 7, 7, 7),
            continuation_tokens=20,
        )
        sticky_blocker_execution = sticky_cluster.execute(sticky_blocker)

        second_request = Request(
            request_id="second",
            arrival_time=sticky_blocker.arrival_time + 1.0,
            router_id="router-0",
            prefix_tokens=shared_prefix,
            continuation_tokens=1,
        )

        decision = vllm_prefix_mock_policy(
            router,
            second_request,
            {"cluster-sticky": sticky_cluster, "cluster-other": other_cluster},
            second_request.arrival_time,
            random.Random(0),
        )
        exact_decision = exact_prefix_policy(
            router,
            second_request,
            {"cluster-sticky": sticky_cluster, "cluster-other": other_cluster},
            second_request.arrival_time,
            random.Random(0),
        )

        self.assertGreater(sticky_blocker_execution.finished_at, second_request.arrival_time)
        self.assertEqual(decision.cluster_id, "cluster-sticky")
        self.assertEqual(decision.estimated_reusable_tokens, len(shared_prefix))
        self.assertEqual(exact_decision.cluster_id, "cluster-other")

    def test_vllm_kv_mock_prefers_more_block_aligned_reuse_before_load(self) -> None:
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
            prefix_tokens=tuple(range(1, 34)),
            continuation_tokens=1,
        )
        deep_execution = deep_cluster.execute(deep_seed)
        deep_cluster.advance_time(deep_execution.finished_at)

        shallow_seed = Request(
            request_id="shallow-seed",
            arrival_time=0.0,
            router_id="router-0",
            prefix_tokens=tuple(range(1, 18)),
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
            prefix_tokens=tuple(range(1, 34)) + (99, 100),
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
        self.assertEqual(decision.estimated_reusable_tokens, 32)
        self.assertEqual(oracle_decision.cluster_id, "cluster-shallow")


if __name__ == "__main__":
    unittest.main()
