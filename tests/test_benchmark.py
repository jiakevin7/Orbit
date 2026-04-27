import argparse
import unittest
from orbit.benchmark import (
    allocate_llama_cpp_ports,
    aggregate_summary_rows,
    build_simulation_config,
    flatten_metrics_row,
    resolve_prompt_prefix_token_cap,
    resolve_reachable_clusters_per_router,
    resolve_cache_token_capacity,
    split_workload,
)
from orbit.models import Request


class BenchmarkTests(unittest.TestCase):
    def test_split_workload_separates_warmup_and_test_requests(self):
        requests = [
            Request(
                request_id=f"req-{index}",
                arrival_time=float(index),
                router_id="router-0",
                prefix_tokens=(index,),
            )
            for index in range(5)
        ]
        warmup, measured = split_workload(requests, 2)
        self.assertEqual([request.request_id for request in warmup], ["req-0", "req-1"])
        self.assertEqual(
            [request.request_id for request in measured],
            ["req-2", "req-3", "req-4"],
        )

    def test_flatten_and_aggregate_metrics_rows(self):
        rows = [
            flatten_metrics_row(
                {
                    "policy": "orbit",
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
                    "policy": "orbit",
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
        self.assertEqual(aggregate_rows[0]["policy"], "orbit")
        self.assertEqual(aggregate_rows[0]["runs"], 2)
        self.assertEqual(aggregate_rows[0]["ttft_p50_mean"], 0.5)
        self.assertIn("ttft_p50_ci_low", aggregate_rows[0])
        self.assertIn("ttft_p50_ci_high", aggregate_rows[0])
        self.assertEqual(aggregate_rows[0]["cluster_requests_cluster-0_mean"], 4.0)

    def test_resolve_cache_token_capacity_uses_effective_mixed_realistic_default(self):
        args = argparse.Namespace(
            cache_token_capacity=None, workload_kind="mixed_realistic"
        )
        self.assertEqual(resolve_cache_token_capacity(args), 4096)
        args = argparse.Namespace(
            cache_token_capacity=8192, workload_kind="mixed_realistic"
        )
        self.assertEqual(resolve_cache_token_capacity(args), 8192)
        args = argparse.Namespace(cache_token_capacity=None, workload_kind="synthetic")
        self.assertIsNone(resolve_cache_token_capacity(args))

    def test_resolve_reachable_clusters_per_router_uses_sparse_default(self):
        args = argparse.Namespace(
            topology_mode="sparse_overlap",
            reachable_clusters_per_router=None,
            routers=4,
            clusters=6,
        )
        self.assertEqual(resolve_reachable_clusters_per_router(args), 3)
        args = argparse.Namespace(
            topology_mode="all_to_all",
            reachable_clusters_per_router=2,
            routers=4,
            clusters=6,
        )
        self.assertIsNone(resolve_reachable_clusters_per_router(args))

    def test_resolve_prompt_prefix_token_cap_reserves_llama_context_headroom(self):
        args = argparse.Namespace(
            backend="llama_cpp", llama_ctx_size=4096, continuation_token_cap=96
        )
        self.assertEqual(resolve_prompt_prefix_token_cap(args), 3584)
        args = argparse.Namespace(
            backend="synthetic", llama_ctx_size=4096, continuation_token_cap=96
        )
        self.assertEqual(resolve_prompt_prefix_token_cap(args), 4096)

    def test_build_simulation_config_records_sparse_topology(self):
        args = argparse.Namespace(
            backend="synthetic",
            control_plane_mode="inprocess",
            control_plane_start_method="spawn",
            routers=4,
            clusters=6,
            topology_mode="sparse_overlap",
            reachable_clusters_per_router=3,
            cache_capacity=256,
            cache_token_capacity=None,
            model=None,
            llama_executable="llama-server",
            llama_port_base=8081,
            llama_threads=4,
            llama_ctx_size=4096,
            llama_parallel=1,
            llama_timeout=120.0,
            llama_startup_timeout=120.0,
            llama_extra_arg=[],
            workload_kind="mixed_realistic",
            requests=32,
            continuation_token_cap=None,
            sharegpt_path=None,
            sharegpt_sample_limit=100,
            rag_path=None,
            rag_sample_limit=100,
            agent_path=None,
            agent_sample_limit=100,
            traffic_mix_chat=0.35,
            traffic_mix_rag=0.25,
            traffic_mix_agent=0.2,
            traffic_mix_bursty=0.2,
            seed=7,
            live_arrival_scale=None,
            summary_delay=0.0,
            gossip_delay=0.0,
            summary_drop_probability=0.0,
            gossip_drop_probability=0.0,
            failed_clusters=[],
            failure_start=0.0,
            failure_duration=0.0,
            retry_penalty=0.0,
        )
        config = build_simulation_config(args)
        self.assertEqual(config.topology_mode, "sparse_overlap")
        self.assertEqual(config.reachable_clusters_per_router, 3)

    def test_allocate_llama_cpp_ports_offsets_port_base_per_simulation(self):
        args = argparse.Namespace(
            backend="llama_cpp",
            control_plane_mode="inprocess",
            control_plane_start_method="spawn",
            routers=2,
            clusters=4,
            topology_mode="all_to_all",
            reachable_clusters_per_router=None,
            cache_capacity=256,
            cache_token_capacity=None,
            model="model.gguf",
            llama_executable="llama-server",
            llama_port_base=18000,
            llama_threads=4,
            llama_ctx_size=4096,
            llama_parallel=1,
            llama_timeout=120.0,
            llama_startup_timeout=120.0,
            llama_extra_arg=[],
            workload_kind="synthetic",
            requests=8,
            continuation_token_cap=None,
            sharegpt_path=None,
            sharegpt_sample_limit=100,
            rag_path=None,
            rag_sample_limit=100,
            agent_path=None,
            agent_sample_limit=100,
            traffic_mix_chat=0.35,
            traffic_mix_rag=0.25,
            traffic_mix_agent=0.2,
            traffic_mix_bursty=0.2,
            seed=7,
            live_arrival_scale=None,
            summary_delay=0.0,
            gossip_delay=0.0,
            summary_drop_probability=0.0,
            gossip_drop_probability=0.0,
            failed_clusters=[],
            failure_start=0.0,
            failure_duration=0.0,
            retry_penalty=0.0,
        )
        config = build_simulation_config(args)
        first = allocate_llama_cpp_ports(config, 0)
        second = allocate_llama_cpp_ports(config, 1)
        self.assertEqual(first.llama_cpp.port_base, 18000)
        self.assertEqual(second.llama_cpp.port_base, 18016)


if __name__ == "__main__":
    unittest.main()
