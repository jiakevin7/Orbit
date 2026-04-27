import threading
import time
import unittest
from unittest import mock
from orbit.cluster import Cluster, ClusterConfig
from orbit.llamacpp import LlamaCppClusterConfig, LlamaCppResult
from orbit.models import Request
from orbit.router import RouterConfig
from orbit.simulation import (
    FaultInjectionConfig,
    Simulation,
    SimulationConfig,
    run_policies,
)
from orbit.workload import WorkloadConfig, generate_workload


class SimulationTests(unittest.TestCase):
    def test_simulation_runs_and_returns_metrics(self):
        config = SimulationConfig(
            router_ids=("router-a", "router-b"),
            cluster_ids=("cluster-a", "cluster-b"),
            cluster_config=ClusterConfig(
                cache_capacity=32,
                summary_depths=(32, 64, 96),
                summary_interval=2.0,
                decode_cost_per_token=4.0,
            ),
            router_config=RouterConfig(
                summary_depths=(32, 64, 96),
                low_overlap_fraction=0.0,
                queue_depth_penalty=2.0,
            ),
            workload=WorkloadConfig(
                num_requests=30,
                router_ids=("router-a", "router-b"),
                prefix_length_choices=(64, 96),
                overlap_length_choices=(0, 32, 64),
                continuation_token_range=(4, 8),
                mean_interarrival=10.0,
                seed=3,
            ),
            gossip_interval=2.0,
        )
        simulation = Simulation(config)
        records, metrics = simulation.run("orbit")
        self.assertEqual(len(records), 30)
        self.assertEqual(metrics.request_count, 30)
        self.assertGreaterEqual(metrics.control_plane_bytes, 0)
        self.assertGreaterEqual(metrics.summary_memory_bytes, 0)

    def test_simulation_runs_with_multiprocess_control_plane(self):
        config = SimulationConfig(
            control_plane_mode="multiprocess",
            router_ids=("router-a", "router-b"),
            cluster_ids=("cluster-a", "cluster-b"),
            cluster_config=ClusterConfig(
                cache_capacity=16,
                summary_depths=(32, 64),
                summary_interval=2.0,
                decode_cost_per_token=3.0,
            ),
            router_config=RouterConfig(
                summary_depths=(32, 64),
                low_overlap_fraction=0.0,
                queue_depth_penalty=1.0,
            ),
            workload=WorkloadConfig(
                num_requests=12,
                router_ids=("router-a", "router-b"),
                prefix_length_choices=(64,),
                overlap_length_choices=(0, 32, 64),
                continuation_token_range=(4, 8),
                mean_interarrival=10.0,
                seed=5,
            ),
            gossip_interval=2.0,
        )
        simulation = Simulation(config)
        try:
            records, metrics = simulation.run("orbit")
        finally:
            simulation.close()
        self.assertEqual(len(records), 12)
        self.assertEqual(metrics.request_count, 12)
        self.assertGreaterEqual(metrics.control_plane_bytes, 0)

    def test_run_policies_uses_only_supported_evaluation_policies(self):
        config = SimulationConfig(
            router_ids=("router-a", "router-b"),
            cluster_ids=("cluster-a", "cluster-b", "cluster-c"),
            cluster_config=ClusterConfig(
                cache_capacity=64,
                summary_depths=(32, 64, 96),
                summary_interval=2.0,
                decode_cost_per_token=3.0,
            ),
            router_config=RouterConfig(
                summary_depths=(32, 64, 96),
                low_overlap_fraction=0.0,
                queue_depth_penalty=0.0,
            ),
            workload=WorkloadConfig(
                num_requests=60,
                router_ids=("router-a", "router-b"),
                prefix_length_choices=(64, 96),
                overlap_length_choices=(0, 32, 64, 96),
                continuation_token_range=(4, 8),
                mean_interarrival=80.0,
                seed=11,
            ),
            gossip_interval=2.0,
        )
        metrics = run_policies(config)
        self.assertEqual(
            set(metrics),
            {"orbit", "least_loaded", "random", "round_robin"},
        )

    def test_llama_cpp_defaults_use_second_scaled_network_costs(self):
        config = SimulationConfig(
            backend="llama_cpp",
            router_ids=("router-a", "router-b"),
            cluster_ids=("cluster-a", "cluster-b"),
            llama_cpp=LlamaCppClusterConfig(model_path="/tmp/placeholder.gguf"),
        )
        simulation = Simulation(config)
        try:
            self.assertEqual(simulation.network_costs["router-a"]["cluster-a"], 0.005)
            self.assertEqual(simulation.network_costs["router-a"]["cluster-b"], 0.025)
            self.assertEqual(simulation.network_costs["router-b"]["cluster-b"], 0.005)
        finally:
            simulation.close()

    def test_sparse_topology_filters_unreachable_clusters(self):
        config = SimulationConfig(
            router_ids=("router-a", "router-b"),
            cluster_ids=("cluster-a", "cluster-b", "cluster-c"),
            network_costs={
                "router-a": {"cluster-a": 1.0, "cluster-b": 2.0},
                "router-b": {"cluster-b": 1.0, "cluster-c": 2.0},
            },
            cluster_config=ClusterConfig(
                cache_capacity=32, summary_depths=(32, 64), summary_interval=1.0
            ),
            workload=WorkloadConfig(num_requests=0),
        )
        requests = [
            Request(
                request_id=f"req-{index}",
                arrival_time=float(index),
                router_id="router-a",
                prefix_tokens=(index, index + 1, index + 2),
                continuation_tokens=1,
            )
            for index in range(8)
        ]
        simulation = Simulation(config)
        try:
            records, _ = simulation.run("random", requests=requests)
        finally:
            simulation.close()
        self.assertTrue(records)
        self.assertTrue(
            all((record.cluster_id in {"cluster-a", "cluster-b"} for record in records))
        )

    def test_sparse_topology_failover_stays_within_reachable_clusters(self):
        config = SimulationConfig(
            router_ids=("router-a", "router-b"),
            cluster_ids=("cluster-a", "cluster-b", "cluster-c"),
            network_costs={
                "router-a": {"cluster-a": 0.0, "cluster-b": 1.0},
                "router-b": {"cluster-b": 0.0, "cluster-c": 1.0},
            },
            cluster_config=ClusterConfig(
                cache_capacity=32, summary_depths=(64, 128), summary_interval=1.0
            ),
            router_config=RouterConfig(
                summary_depths=(64, 128),
                low_overlap_fraction=0.0,
                queue_depth_penalty=0.0,
            ),
            workload=WorkloadConfig(num_requests=0),
            faults=FaultInjectionConfig(
                failed_cluster_ids=("cluster-a",),
                failure_start=1.5,
                failure_duration=10.0,
                retry_penalty=1.0,
            ),
        )
        requests = [
            Request(
                request_id="req-0",
                arrival_time=0.0,
                router_id="router-a",
                prefix_tokens=tuple(range(128)),
                continuation_tokens=1,
            ),
            Request(
                request_id="req-1",
                arrival_time=2.0,
                router_id="router-a",
                prefix_tokens=tuple(range(128)),
                continuation_tokens=1,
            ),
        ]
        simulation = Simulation(config)
        try:
            records, _ = simulation.run("orbit", requests=requests)
        finally:
            simulation.close()
        self.assertEqual(records[1].initial_cluster_id, "cluster-a")
        self.assertEqual(records[1].cluster_id, "cluster-b")
        self.assertNotEqual(records[1].cluster_id, "cluster-c")

    def test_simulation_can_warm_up_before_measured_run(self):
        config = SimulationConfig(
            router_ids=("router-a",),
            cluster_ids=("cluster-a", "cluster-b"),
            cluster_config=ClusterConfig(
                cache_capacity=64,
                summary_depths=(32, 64, 96),
                summary_interval=2.0,
                decode_cost_per_token=3.0,
            ),
            router_config=RouterConfig(
                summary_depths=(32, 64, 96),
                low_overlap_fraction=0.0,
                queue_depth_penalty=1.5,
            ),
            workload=WorkloadConfig(
                num_requests=8,
                router_ids=("router-a",),
                prefix_length_choices=(64, 96),
                overlap_length_choices=(32, 64, 96),
                continuation_token_range=(4, 8),
                mean_interarrival=8.0,
                seed=17,
            ),
            gossip_interval=2.0,
        )
        requests = generate_workload(config.workload)
        simulation = Simulation(config)
        try:
            simulation.run("orbit", requests=requests[:3], close_on_finish=False)
            measured_records, measured_metrics = simulation.run(
                "orbit", requests=requests[3:]
            )
        finally:
            simulation.close()
        self.assertEqual(len(measured_records), 5)
        self.assertEqual(measured_metrics.request_count, 5)
        self.assertGreaterEqual(measured_metrics.control_plane_bytes, 0)

    def test_llama_cpp_prepare_requests_is_idempotent_for_already_aligned_workloads(
        self,
    ):
        requests = [
            Request(
                request_id="req-0",
                arrival_time=0.0,
                router_id="router-a",
                prefix_tokens=(10, 20, 30),
                prompt_prefix_text="System:\nRespond briefly.\n\nUser:\nHello there.",
                prefix_token_source="llama_cpp",
                arrival_scale_applied=0.01,
            )
        ]
        simulation = Simulation(
            SimulationConfig(
                backend="llama_cpp",
                router_ids=("router-a",),
                cluster_ids=("cluster-a",),
                llama_cpp=LlamaCppClusterConfig(
                    model_path="/tmp/placeholder.gguf", manage_server=False
                ),
                live_arrival_scale=0.01,
            )
        )
        cluster = simulation.clusters["cluster-a"]
        try:
            with mock.patch.object(cluster, "prepare_requests") as prepare_requests:
                prepared = simulation.prepare_requests(requests)
            self.assertEqual(prepared, requests)
            prepare_requests.assert_not_called()
        finally:
            simulation.close()

    def test_llama_cpp_prepare_requests_scales_arrivals_once(self):
        simulation = Simulation(
            SimulationConfig(
                backend="llama_cpp",
                router_ids=("router-a",),
                cluster_ids=("cluster-a",),
                llama_cpp=LlamaCppClusterConfig(
                    model_path="/tmp/placeholder.gguf", manage_server=False
                ),
                live_arrival_scale=0.01,
            )
        )
        requests = [
            Request(
                request_id="req-0",
                arrival_time=20.0,
                router_id="router-a",
                prefix_tokens=(10, 20, 30),
                prompt_prefix_text="System:\nRespond briefly.\n\nUser:\nHello there.",
                prefix_token_source="llama_cpp",
            )
        ]
        try:
            prepared = simulation.prepare_requests(requests)
            prepared_twice = simulation.prepare_requests(prepared)
        finally:
            simulation.close()
        self.assertAlmostEqual(prepared[0].arrival_time, 0.2)
        self.assertEqual(prepared[0].arrival_scale_applied, 0.01)
        self.assertEqual(prepared, prepared_twice)

    def test_synthetic_cluster_exposes_prefix_reuse_after_prefill_not_finish(self):
        cluster = Cluster(
            cluster_id="cluster-a",
            config=ClusterConfig(
                concurrency=1, prefill_cost_per_token=1.0, decode_cost_per_token=10.0
            ),
        )
        first_request = Request(
            request_id="req-0",
            arrival_time=0.0,
            router_id="router-a",
            prefix_tokens=(1, 2, 3, 4),
            continuation_tokens=5,
        )
        second_request = Request(
            request_id="req-1",
            arrival_time=4.5,
            router_id="router-a",
            prefix_tokens=(1, 2, 3, 4),
            continuation_tokens=1,
        )
        first_execution = cluster.execute(first_request)
        self.assertGreater(first_execution.finished_at, 4.5)
        self.assertLessEqual(first_execution.cache_ready_at, 4.5)
        second_execution = cluster.execute(second_request)
        self.assertEqual(second_execution.true_reusable_tokens, 4)

    def test_llama_cpp_run_replays_requests_concurrently(self):
        config = SimulationConfig(
            backend="llama_cpp",
            router_ids=("router-a",),
            cluster_ids=("cluster-a",),
            cluster_config=ClusterConfig(summary_interval=1.0),
            llama_cpp=LlamaCppClusterConfig(
                model_path="/tmp/placeholder.gguf", manage_server=False, parallel=1
            ),
            router_config=RouterConfig(low_overlap_fraction=0.0),
            live_arrival_scale=1.0,
        )
        requests = [
            Request(
                request_id="req-0",
                arrival_time=0.0,
                router_id="router-a",
                prefix_tokens=(1, 2, 3),
                prompt_prefix_text="System:\nRespond briefly.\n\nUser:\nHello there.",
                prefix_token_source="llama_cpp",
            ),
            Request(
                request_id="req-1",
                arrival_time=0.02,
                router_id="router-a",
                prefix_tokens=(1, 2, 3),
                prompt_prefix_text="System:\nRespond briefly.\n\nUser:\nHello there.",
                prefix_token_source="llama_cpp",
            ),
        ]
        simulation = Simulation(config)
        cluster = simulation.clusters["cluster-a"]
        cluster._started = True
        gate = threading.Semaphore(1)

        def fake_complete(prompt, max_tokens, event_callback=None):
            submitted_at = time.perf_counter()
            with gate:
                processing_started = time.perf_counter() - submitted_at
                if event_callback is not None:
                    event_callback(
                        {
                            "prompt_progress": {
                                "total": 3,
                                "processed": 0,
                                "time_ms": 0.0,
                            }
                        },
                        processing_started,
                    )
                time.sleep(0.01)
                cache_ready = time.perf_counter() - submitted_at
                if event_callback is not None:
                    event_callback(
                        {
                            "prompt_progress": {
                                "total": 3,
                                "processed": 3,
                                "time_ms": 10.0,
                            }
                        },
                        cache_ready,
                    )
                time.sleep(0.03)
                total_latency = time.perf_counter() - submitted_at
                return LlamaCppResult(
                    total_latency=total_latency,
                    ttft=min(total_latency, cache_ready + 0.01),
                    prompt_eval_latency=0.01,
                    processing_started_latency=processing_started,
                    cache_ready_latency=cache_ready,
                )

        try:
            with (
                mock.patch.object(cluster, "_ensure_started"),
                mock.patch.object(
                    cluster._client, "complete", side_effect=fake_complete
                ),
                mock.patch.object(
                    cluster._client, "slots", side_effect=RuntimeError("offline")
                ),
            ):
                records, _ = simulation.run("orbit", requests=requests)
        finally:
            simulation.close()
        self.assertEqual(len(records), 2)
        self.assertGreater(records[1].queue_delay, 0.0)
        self.assertEqual(records[1].actual_reusable_tokens, 3)
        self.assertLess(records[0].finished_at, records[1].finished_at)

    def test_llama_cpp_multiprocess_prepare_requests_uses_proxy(self):
        simulation = Simulation(
            SimulationConfig(
                backend="llama_cpp",
                control_plane_mode="multiprocess",
                router_ids=("router-a",),
                cluster_ids=("cluster-a",),
                llama_cpp=LlamaCppClusterConfig(
                    model_path="/tmp/placeholder.gguf", manage_server=False
                ),
                live_arrival_scale=0.01,
            )
        )
        requests = [
            Request(
                request_id="req-0",
                arrival_time=10.0,
                router_id="router-a",
                prefix_tokens=(1, 2, 3),
                prompt_prefix_text="System:\nRespond briefly.\n\nUser:\nHello there.",
            )
        ]
        cluster = simulation.clusters["cluster-a"]
        try:
            prepared_request = Request(
                request_id="req-0",
                arrival_time=10.0,
                router_id="router-a",
                prefix_tokens=(10, 20, 30),
                prompt_prefix_text="System:\nRespond briefly.\n\nUser:\nHello there.",
                prefix_token_source="llama_cpp",
            )
            with mock.patch.object(
                cluster, "prepare_requests", return_value=[prepared_request]
            ) as prepare_mock:
                prepared = simulation.prepare_requests(requests)
            self.assertEqual(prepared[0].prefix_token_source, "llama_cpp")
            prepare_mock.assert_called_once()
        finally:
            simulation.close()

    def test_gossip_delay_keeps_remote_router_view_stale(self):
        requests = [
            Request(
                request_id="req-0",
                arrival_time=0.0,
                router_id="router-a",
                prefix_tokens=tuple(range(128)),
                continuation_tokens=1,
            ),
            Request(
                request_id="req-1",
                arrival_time=2.0,
                router_id="router-b",
                prefix_tokens=tuple(range(128)),
                continuation_tokens=1,
            ),
        ]
        base_config = SimulationConfig(
            router_ids=("router-a", "router-b"),
            cluster_ids=("cluster-a", "cluster-b"),
            cluster_config=ClusterConfig(
                cache_capacity=32,
                summary_depths=(64, 128),
                summary_interval=1.0,
                prefill_cost_per_token=0.001,
            ),
            router_config=RouterConfig(
                summary_depths=(64, 128),
                low_overlap_fraction=0.0,
                queue_depth_penalty=0.0,
            ),
            workload=WorkloadConfig(num_requests=0),
            gossip_interval=1.0,
        )
        delayed_config = SimulationConfig(
            **{
                **base_config.__dict__,
                "faults": FaultInjectionConfig(gossip_delay=100.0),
            }
        )
        healthy_simulation = Simulation(base_config)
        delayed_simulation = Simulation(delayed_config)
        try:
            healthy_records, _ = healthy_simulation.run("orbit", requests=requests)
            delayed_records, _ = delayed_simulation.run("orbit", requests=requests)
        finally:
            healthy_simulation.close()
            delayed_simulation.close()
        self.assertEqual(healthy_records[1].cluster_id, "cluster-a")
        self.assertEqual(delayed_records[1].cluster_id, "cluster-b")

    def test_cluster_outage_records_failover(self):
        config = SimulationConfig(
            router_ids=("router-a",),
            cluster_ids=("cluster-a", "cluster-b"),
            cluster_config=ClusterConfig(
                cache_capacity=32, summary_depths=(64, 128), summary_interval=1.0
            ),
            router_config=RouterConfig(
                summary_depths=(64, 128),
                low_overlap_fraction=0.0,
                queue_depth_penalty=0.0,
            ),
            workload=WorkloadConfig(num_requests=0),
            faults=FaultInjectionConfig(
                failed_cluster_ids=("cluster-a",),
                failure_start=1.5,
                failure_duration=10.0,
                retry_penalty=3.0,
            ),
        )
        requests = [
            Request(
                request_id="req-0",
                arrival_time=0.0,
                router_id="router-a",
                prefix_tokens=tuple(range(128)),
                continuation_tokens=1,
            ),
            Request(
                request_id="req-1",
                arrival_time=2.0,
                router_id="router-a",
                prefix_tokens=tuple(range(128)),
                continuation_tokens=1,
            ),
        ]
        simulation = Simulation(config)
        try:
            records, metrics = simulation.run("orbit", requests=requests)
        finally:
            simulation.close()
        self.assertFalse(records[0].had_failover)
        self.assertTrue(records[1].had_failover)
        self.assertEqual(records[1].initial_cluster_id, "cluster-a")
        self.assertEqual(records[1].cluster_id, "cluster-b")
        self.assertEqual(records[1].failover_delay, 3.0)
        self.assertEqual(records[1].attempt_count, 2)
        self.assertEqual(metrics.failover_count, 1)


if __name__ == "__main__":
    unittest.main()
