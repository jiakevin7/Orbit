from __future__ import annotations

import argparse
import unittest

from orbit.benchmark import (
    aggregate_summary_rows,
    flatten_metrics_row,
    resolve_cache_token_capacity,
    split_workload,
)
from orbit.models import Request


class BenchmarkTests(unittest.TestCase):
    def test_split_workload_separates_warmup_validation_and_test_requests(self) -> None:
        requests = [
            Request(
                request_id=f"req-{index}",
                arrival_time=float(index),
                router_id="router-0",
                prefix_tokens=(index,),
            )
            for index in range(5)
        ]

        warmup, validation, measured = split_workload(requests, 2, 1)

        self.assertEqual([request.request_id for request in warmup], ["req-0", "req-1"])
        self.assertEqual([request.request_id for request in validation], ["req-2"])
        self.assertEqual([request.request_id for request in measured], ["req-3", "req-4"])

    def test_flatten_and_aggregate_metrics_rows(self) -> None:
        rows = [
            flatten_metrics_row(
                {
                    "policy": "summary",
                    "request_count": 10,
                    "mean_reusable_prefix": 20.0,
                    "mean_reuse_fraction": 0.25,
                    "ttft_p50": 0.4,
                    "ttft_p95": 0.8,
                    "latency_p50": 0.5,
                    "latency_p95": 0.9,
                    "control_plane_bytes": 100,
                    "summary_memory_bytes": 200,
                    "load_stddev": 1.0,
                    "cluster_request_counts": {"cluster-0": 6, "cluster-1": 4},
                }
            ),
            flatten_metrics_row(
                {
                    "policy": "summary",
                    "request_count": 10,
                    "mean_reusable_prefix": 30.0,
                    "mean_reuse_fraction": 0.35,
                    "ttft_p50": 0.6,
                    "ttft_p95": 1.0,
                    "latency_p50": 0.7,
                    "latency_p95": 1.1,
                    "control_plane_bytes": 300,
                    "summary_memory_bytes": 400,
                    "load_stddev": 3.0,
                    "cluster_request_counts": {"cluster-0": 2, "cluster-1": 8},
                }
            ),
        ]

        aggregate_rows = aggregate_summary_rows(rows)

        self.assertEqual(len(aggregate_rows), 1)
        self.assertEqual(aggregate_rows[0]["policy"], "summary")
        self.assertEqual(aggregate_rows[0]["runs"], 2)
        self.assertEqual(aggregate_rows[0]["ttft_p50_mean"], 0.5)
        self.assertIn("ttft_p50_ci_low", aggregate_rows[0])
        self.assertIn("ttft_p50_ci_high", aggregate_rows[0])
        self.assertEqual(aggregate_rows[0]["cluster_requests_cluster-0_mean"], 4.0)

    def test_resolve_cache_token_capacity_uses_effective_mixed_realistic_default(self) -> None:
        args = argparse.Namespace(cache_token_capacity=None, workload_kind="mixed_realistic")
        self.assertEqual(resolve_cache_token_capacity(args), 4096)

        args = argparse.Namespace(cache_token_capacity=8192, workload_kind="mixed_realistic")
        self.assertEqual(resolve_cache_token_capacity(args), 8192)

        args = argparse.Namespace(cache_token_capacity=None, workload_kind="synthetic")
        self.assertIsNone(resolve_cache_token_capacity(args))


if __name__ == "__main__":
    unittest.main()
