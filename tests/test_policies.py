import random
import unittest
from orbit.cluster import Cluster, ClusterConfig
from orbit.models import Request
from orbit.policies import round_robin_policy
from orbit.router import Router, RouterConfig


class MockPolicyTests(unittest.TestCase):
    def test_round_robin_policy_cycles_clusters(self):
        cluster_a = Cluster("cluster-a", ClusterConfig())
        cluster_b = Cluster("cluster-b", ClusterConfig())
        router = Router(
            "router-0", {"cluster-a": 0.0, "cluster-b": 0.0}, RouterConfig()
        )
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


if __name__ == "__main__":
    unittest.main()
