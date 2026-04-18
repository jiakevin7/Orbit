from __future__ import annotations

import unittest

from orbit.cluster import Cluster, ClusterConfig


class ClusterTests(unittest.TestCase):
    def test_token_budget_eviction_removes_oldest_prefixes(self) -> None:
        cluster = Cluster(
            cluster_id="cluster-a",
            config=ClusterConfig(
                cache_capacity=8,
                cache_capacity_tokens=5,
            ),
        )

        cluster._insert_into_cache("req-0", (1, 2, 3))
        self.assertEqual(cluster.true_reusable_prefix((1, 2, 3), now=0.0), 3)

        cluster._insert_into_cache("req-1", (4, 5, 6))

        self.assertEqual(cluster.true_reusable_prefix((1, 2, 3), now=0.0), 0)
        self.assertEqual(cluster.true_reusable_prefix((4, 5, 6), now=0.0), 3)


if __name__ == "__main__":
    unittest.main()
