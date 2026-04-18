from __future__ import annotations

import random
import statistics
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, Iterable, Sequence

from .cluster import Cluster, ClusterConfig
from .llamacpp import LlamaCppCluster, LlamaCppClusterConfig
from .models import ExecutionRecord, Request, SimulationMetrics
from .policies import POLICIES
from .router import Router, RouterConfig
from .workload import WorkloadConfig, generate_workload


@dataclass(frozen=True)
class FaultInjectionConfig:
    summary_delay: float = 0.0
    gossip_delay: float = 0.0
    summary_drop_probability: float = 0.0
    gossip_drop_probability: float = 0.0
    failed_cluster_ids: tuple[str, ...] = ()
    failure_start: float = 0.0
    failure_duration: float = 0.0
    retry_penalty: float = 0.0


@dataclass(frozen=True)
class SimulationConfig:
    backend: str = "synthetic"
    router_ids: tuple[str, ...] = ("router-a", "router-b")
    cluster_ids: tuple[str, ...] = ("cluster-a", "cluster-b", "cluster-c")
    cluster_config: ClusterConfig = field(default_factory=ClusterConfig)
    llama_cpp: LlamaCppClusterConfig | None = None
    router_config: RouterConfig = field(default_factory=RouterConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    gossip_interval: float = 5.0
    control_plane_header_bytes: int = 32
    network_costs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    home_router_for_cluster: Dict[str, str] = field(default_factory=dict)
    random_seed: int = 13
    live_arrival_scale: float = 1.0
    faults: FaultInjectionConfig = field(default_factory=FaultInjectionConfig)


class Simulation:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self.network_costs = self._build_network_costs()
        self.home_router_for_cluster = self._build_home_router_map()
        self.clusters = self._build_clusters()
        self.routers = {
            router_id: Router(
                router_id,
                network_costs=self.network_costs[router_id],
                config=self.config.router_config,
            )
            for router_id in self.config.router_ids
        }
        self._next_summary_due = {cluster_id: 0.0 for cluster_id in self.config.cluster_ids}
        self._next_gossip_at = 0.0
        self._pending_deliveries: list[tuple[float, int, str, object, str]] = []
        self._delivery_sequence = 0
        self.control_plane_bytes = 0
        self.rng = random.Random(self.config.random_seed)
        self._live_time_origin: float | None = None

    def close(self) -> None:
        for cluster in self.clusters.values():
            close = getattr(cluster, "close", None)
            if callable(close):
                close()
        self._live_time_origin = None

    def prepare_requests(self, requests: Sequence[Request]) -> list[Request]:
        return self._prepare_requests_for_backend(requests)

    def run(
        self,
        policy_name: str = "summary",
        requests: Sequence = (),
        close_on_finish: bool = True,
    ) -> tuple[list[ExecutionRecord], SimulationMetrics]:
        if policy_name not in POLICIES:
            raise ValueError(f"unknown policy: {policy_name}")

        workload = list(requests) if requests else generate_workload(self.config.workload)
        workload = self._prepare_requests_for_backend(workload)
        if self.config.backend == "llama_cpp":
            return self._run_llama_cpp_concurrent(
                policy_name=policy_name,
                workload=workload,
                close_on_finish=close_on_finish,
            )

        records: list[ExecutionRecord] = []
        cluster_counts: Counter[str] = Counter()
        control_plane_bytes_start = self.control_plane_bytes

        try:
            for request in workload:
                self._process_control_plane_until(request.arrival_time)
                self._advance_clusters(request.arrival_time)

                router = self.routers[request.router_id]
                routed_request, decision, initial_cluster_id, had_failover, failover_delay, attempt_count = self._route_request(
                    policy_name=policy_name,
                    router=router,
                    request=request,
                    now=request.arrival_time,
                )

                execution = self.clusters[decision.cluster_id].execute(routed_request)
                network_cost = router.network_cost(decision.cluster_id)
                actual_latency = network_cost + (execution.finished_at - request.arrival_time)
                actual_ttft = network_cost + execution.time_to_first_token
                reuse_fraction = 0.0
                if request.input_length > 0:
                    reuse_fraction = execution.true_reusable_tokens / request.input_length

                records.append(
                    ExecutionRecord(
                        request_id=request.request_id,
                        policy=policy_name,
                        router_id=request.router_id,
                        cluster_id=decision.cluster_id,
                        arrival_time=request.arrival_time,
                        started_at=execution.started_at,
                        finished_at=execution.finished_at,
                        predicted_latency=decision.predicted_latency,
                        actual_latency=actual_latency,
                        actual_ttft=actual_ttft,
                        estimated_reusable_tokens=decision.estimated_reusable_tokens,
                        actual_reusable_tokens=execution.true_reusable_tokens,
                        estimated_remaining_prefill_tokens=max(0, request.input_length - decision.estimated_reusable_tokens),
                        input_length=request.input_length,
                        continuation_tokens=request.continuation_tokens,
                        reuse_fraction=reuse_fraction,
                        network_cost=network_cost,
                        queue_delay=execution.queue_delay,
                        queue_depth_before=execution.queue_depth_before,
                        route_queue_depth=int(decision.details.get("raw_queue_depth", 0.0)),
                        metadata_age=float(decision.details.get("metadata_age", 0.0)),
                        uncertainty_gap=int(decision.details.get("uncertainty_gap", 0.0)),
                        missing_summary=bool(decision.details.get("missing_summary", 0.0)),
                        initial_cluster_id=initial_cluster_id,
                        had_failover=had_failover,
                        failover_delay=failover_delay,
                        attempt_count=attempt_count,
                        service_time=execution.service_time,
                        traffic_class=request.traffic_class,
                        session_id=request.session_id,
                        source_id=request.source_id,
                    )
                )
                cluster_counts[decision.cluster_id] += 1
        finally:
            if close_on_finish:
                self.close()

        metrics = self._compute_metrics(
            policy_name,
            records,
            cluster_counts,
            control_plane_bytes=self.control_plane_bytes - control_plane_bytes_start,
        )
        return records, metrics

    def _build_clusters(self) -> Dict[str, object]:
        if self.config.backend == "synthetic":
            return {
                cluster_id: Cluster(cluster_id, self.config.cluster_config)
                for cluster_id in self.config.cluster_ids
            }
        if self.config.backend == "llama_cpp":
            if self.config.llama_cpp is None:
                raise ValueError("llama_cpp backend requires llama_cpp configuration")
            clusters: Dict[str, object] = {}
            for index, cluster_id in enumerate(self.config.cluster_ids):
                clusters[cluster_id] = LlamaCppCluster(
                    cluster_id=cluster_id,
                    cluster_config=self.config.cluster_config,
                    backend_config=self.config.llama_cpp.for_cluster(index),
                )
            return clusters
        raise ValueError(f"unsupported backend: {self.config.backend}")

    def _prepare_requests_for_backend(self, requests: Sequence[Request]) -> list[Request]:
        prepared = list(requests)
        if self.config.backend != "llama_cpp" or not prepared:
            return prepared

        if not all(request_obj.prefix_token_source == "llama_cpp" for request_obj in prepared):
            tokenizer_cluster = next(iter(self.clusters.values()))
            prepare_requests = getattr(tokenizer_cluster, "prepare_requests", None)
            if not callable(prepare_requests):
                raise RuntimeError("llama_cpp backend requires clusters that can prepare requests")
            prepared = prepare_requests(prepared)

        if self.config.live_arrival_scale <= 0:
            raise ValueError("live_arrival_scale must be positive")

        scaled_requests: list[Request] = []
        for request_obj in prepared:
            if request_obj.arrival_scale_applied == self.config.live_arrival_scale:
                scaled_requests.append(request_obj)
                continue

            logical_arrival = request_obj.arrival_time / request_obj.arrival_scale_applied
            scaled_requests.append(
                replace(
                    request_obj,
                    arrival_time=logical_arrival * self.config.live_arrival_scale,
                    arrival_scale_applied=self.config.live_arrival_scale,
                )
            )
        return scaled_requests

    def _build_network_costs(self) -> Dict[str, Dict[str, float]]:
        if self.config.network_costs:
            return {
                router_id: dict(cluster_costs)
                for router_id, cluster_costs in self.config.network_costs.items()
            }

        local_cost = 5.0
        remote_step = 20.0
        if self.config.backend == "llama_cpp":
            # The synthetic simulator uses abstract latency units. For live llama.cpp
            # measurements, convert the default topology costs to seconds.
            local_cost = 0.005
            remote_step = 0.020

        network_costs: Dict[str, Dict[str, float]] = {}
        for router_index, router_id in enumerate(self.config.router_ids):
            router_costs: Dict[str, float] = {}
            for cluster_index, cluster_id in enumerate(self.config.cluster_ids):
                distance = abs(router_index - cluster_index)
                router_costs[cluster_id] = local_cost + distance * remote_step
            network_costs[router_id] = router_costs
        return network_costs

    def _build_home_router_map(self) -> Dict[str, str]:
        if self.config.home_router_for_cluster:
            return dict(self.config.home_router_for_cluster)

        mapping: Dict[str, str] = {}
        router_count = len(self.config.router_ids)
        for index, cluster_id in enumerate(self.config.cluster_ids):
            mapping[cluster_id] = self.config.router_ids[index % router_count]
        return mapping

    def _process_control_plane_until(self, now: float) -> None:
        while True:
            next_summary_time = min(self._next_summary_due.values())
            next_delivery_time = self._pending_deliveries[0][0] if self._pending_deliveries else float("inf")
            next_event_time = min(next_summary_time, self._next_gossip_at, next_delivery_time)
            if next_event_time > now:
                return

            if next_delivery_time <= min(next_summary_time, self._next_gossip_at):
                self._deliver_pending_summaries(next_delivery_time)
                continue

            if self._next_gossip_at <= next_summary_time:
                self._gossip(self._next_gossip_at)
                self._next_gossip_at += self._gossip_interval()
                continue

            due_cluster_ids = [
                cluster_id
                for cluster_id, event_time in self._next_summary_due.items()
                if event_time == next_summary_time
            ]
            for cluster_id in due_cluster_ids:
                event_time = self._next_summary_due[cluster_id]
                cluster = self.clusters[cluster_id]
                cluster.advance_time(event_time)
                if not self._cluster_is_available(cluster_id, event_time):
                    self._next_summary_due[cluster_id] += self._summary_interval()
                    continue
                summary = cluster.publish_summary(event_time)
                home_router_id = self.home_router_for_cluster[cluster_id]
                if self.rng.random() >= self.config.faults.summary_drop_probability:
                    self._schedule_delivery(
                        deliver_at=event_time + self.config.faults.summary_delay,
                        receiver_id=home_router_id,
                        summary=summary,
                        source=cluster_id,
                    )
                self._next_summary_due[cluster_id] += self._summary_interval()

    def _gossip(self, now: float) -> None:
        for sender_id, sender in self.routers.items():
            payload = sender.export_summaries()
            for receiver_id, receiver in self.routers.items():
                if sender_id == receiver_id:
                    continue
                for summary in payload.values():
                    if self.rng.random() >= self.config.faults.gossip_drop_probability:
                        self._schedule_delivery(
                            deliver_at=now + self.config.faults.gossip_delay,
                            receiver_id=receiver_id,
                            summary=summary,
                            source=f"gossip:{sender_id}",
                        )

    def _advance_clusters(self, now: float) -> None:
        for cluster in self.clusters.values():
            cluster.advance_time(now)

    def _run_llama_cpp_concurrent(
        self,
        policy_name: str,
        workload: Sequence[Request],
        close_on_finish: bool,
    ) -> tuple[list[ExecutionRecord], SimulationMetrics]:
        cluster_counts: Counter[str] = Counter()
        control_plane_bytes_start = self.control_plane_bytes
        pending_by_future: dict[Future[object], tuple[int, Request, object, object]] = {}
        records_by_index: dict[int, ExecutionRecord] = {}

        try:
            if self._live_time_origin is None:
                self._live_time_origin = time.monotonic()

            for index, request_obj in enumerate(workload):
                arrival_time = self._wait_for_live_time(request_obj.arrival_time)
                self._process_control_plane_until(arrival_time)
                self._advance_clusters(arrival_time)

                routed_request = (
                    request_obj
                    if abs(arrival_time - request_obj.arrival_time) < 1e-9
                    else replace(request_obj, arrival_time=arrival_time)
                )
                router = self.routers[routed_request.router_id]
                routed_request, decision, initial_cluster_id, had_failover, failover_delay, attempt_count = self._route_request(
                    policy_name=policy_name,
                    router=router,
                    request=routed_request,
                    now=arrival_time,
                )
                cluster = self.clusters[decision.cluster_id]
                submit = getattr(cluster, "submit", None)
                if not callable(submit):
                    raise RuntimeError("llama_cpp backend requires clusters that support submit()")
                future = submit(routed_request)
                pending_by_future[future] = (
                    index,
                    routed_request,
                    decision,
                    router,
                    initial_cluster_id,
                    had_failover,
                    failover_delay,
                    attempt_count,
                )
                cluster_counts[decision.cluster_id] += 1

            while pending_by_future:
                now = self._current_live_time()
                self._process_control_plane_until(now)
                self._advance_clusters(now)

                completed = [future for future in pending_by_future if future.done()]
                if not completed:
                    timeout = 0.01
                    next_control_plane_event = self._next_control_plane_event_time()
                    if next_control_plane_event is not None and next_control_plane_event > now:
                        timeout = min(timeout, max(0.001, next_control_plane_event - now))
                    wait(
                        tuple(pending_by_future),
                        timeout=timeout,
                        return_when=FIRST_COMPLETED,
                    )
                    continue

                for future in completed:
                    index, request_obj, decision, router, initial_cluster_id, had_failover, failover_delay, attempt_count = pending_by_future.pop(future)
                    execution = future.result()
                    records_by_index[index] = self._build_execution_record(
                        request_obj=request_obj,
                        policy_name=policy_name,
                        decision=decision,
                        execution=execution,
                        router=router,
                        initial_cluster_id=initial_cluster_id,
                        had_failover=had_failover,
                        failover_delay=failover_delay,
                        attempt_count=attempt_count,
                    )
        finally:
            if close_on_finish:
                self.close()

        records = [records_by_index[index] for index in range(len(workload))]
        metrics = self._compute_metrics(
            policy_name,
            records,
            cluster_counts,
            control_plane_bytes=self.control_plane_bytes - control_plane_bytes_start,
        )
        return records, metrics

    def _build_execution_record(
        self,
        request_obj: Request,
        policy_name: str,
        decision: object,
        execution: object,
        router: Router,
        initial_cluster_id: str,
        had_failover: bool,
        failover_delay: float,
        attempt_count: int,
    ) -> ExecutionRecord:
        network_cost = router.network_cost(decision.cluster_id)
        actual_latency = network_cost + (execution.finished_at - request_obj.arrival_time)
        actual_ttft = network_cost + execution.time_to_first_token
        reuse_fraction = 0.0
        if request_obj.input_length > 0:
            reuse_fraction = execution.true_reusable_tokens / request_obj.input_length

        return ExecutionRecord(
            request_id=request_obj.request_id,
            policy=policy_name,
            router_id=request_obj.router_id,
            cluster_id=decision.cluster_id,
            arrival_time=request_obj.arrival_time,
            started_at=execution.started_at,
            finished_at=execution.finished_at,
            predicted_latency=decision.predicted_latency,
            actual_latency=actual_latency,
            actual_ttft=actual_ttft,
            estimated_reusable_tokens=decision.estimated_reusable_tokens,
            actual_reusable_tokens=execution.true_reusable_tokens,
            estimated_remaining_prefill_tokens=max(0, request_obj.input_length - decision.estimated_reusable_tokens),
            input_length=request_obj.input_length,
            continuation_tokens=request_obj.continuation_tokens,
            reuse_fraction=reuse_fraction,
            network_cost=network_cost,
            queue_delay=execution.queue_delay,
            queue_depth_before=execution.queue_depth_before,
            route_queue_depth=int(decision.details.get("raw_queue_depth", 0.0)),
            metadata_age=float(decision.details.get("metadata_age", 0.0)),
            uncertainty_gap=int(decision.details.get("uncertainty_gap", 0.0)),
            missing_summary=bool(decision.details.get("missing_summary", 0.0)),
            initial_cluster_id=initial_cluster_id,
            had_failover=had_failover,
            failover_delay=failover_delay,
            attempt_count=attempt_count,
            service_time=execution.service_time,
            traffic_class=request_obj.traffic_class,
            session_id=request_obj.session_id,
            source_id=request_obj.source_id,
        )

    def _summary_interval(self) -> float:
        if self.config.backend == "llama_cpp":
            return self.config.cluster_config.summary_interval * self.config.live_arrival_scale
        return self.config.cluster_config.summary_interval

    def _gossip_interval(self) -> float:
        if self.config.backend == "llama_cpp":
            return self.config.gossip_interval * self.config.live_arrival_scale
        return self.config.gossip_interval

    def _next_control_plane_event_time(self) -> float | None:
        candidates: list[float] = [self._next_gossip_at]
        if self._next_summary_due:
            candidates.append(min(self._next_summary_due.values()))
        if self._pending_deliveries:
            candidates.append(self._pending_deliveries[0][0])
        return min(candidates) if candidates else None

    def _current_live_time(self) -> float:
        if self._live_time_origin is None:
            return 0.0
        return time.monotonic() - self._live_time_origin

    def _wait_for_live_time(self, target_time: float) -> float:
        while True:
            now = self._current_live_time()
            self._process_control_plane_until(now)
            self._advance_clusters(now)
            if now >= target_time:
                return now

            next_control_plane_event = self._next_control_plane_event_time()
            wake_time = target_time
            if next_control_plane_event is not None:
                wake_time = min(wake_time, next_control_plane_event)

            sleep_for = max(0.0, wake_time - now)
            if sleep_for <= 0:
                return self._current_live_time()
            time.sleep(sleep_for)

    def _compute_metrics(
        self,
        policy_name: str,
        records: Sequence[ExecutionRecord],
        cluster_counts: Counter[str],
        control_plane_bytes: int,
    ) -> SimulationMetrics:
        latencies = [record.actual_latency for record in records]
        ttfts = [record.actual_ttft for record in records]
        reusable_prefixes = [record.actual_reusable_tokens for record in records]
        reuse_fractions = [record.reuse_fraction for record in records]
        count_values = [cluster_counts.get(cluster_id, 0) for cluster_id in self.config.cluster_ids]
        failover_count = sum(1 for record in records if record.had_failover)

        summary_memory_bytes = 0
        for router in self.routers.values():
            summary_memory_bytes += sum(view.summary.byte_size for view in router.views.values())

        return SimulationMetrics(
            policy=policy_name,
            request_count=len(records),
            mean_reusable_prefix=statistics.fmean(reusable_prefixes) if reusable_prefixes else 0.0,
            mean_reuse_fraction=statistics.fmean(reuse_fractions) if reuse_fractions else 0.0,
            ttft_p50=_percentile(ttfts, 0.50),
            ttft_p95=_percentile(ttfts, 0.95),
            latency_p50=_percentile(latencies, 0.50),
            latency_p95=_percentile(latencies, 0.95),
            control_plane_bytes=control_plane_bytes,
            summary_memory_bytes=summary_memory_bytes,
            load_stddev=statistics.pstdev(count_values) if len(count_values) > 1 else 0.0,
            failover_count=failover_count,
            failover_rate=(failover_count / len(records)) if records else 0.0,
            cluster_request_counts={cluster_id: cluster_counts.get(cluster_id, 0) for cluster_id in self.config.cluster_ids},
        )

    def _route_request(
        self,
        policy_name: str,
        router: Router,
        request: Request,
        now: float,
    ) -> tuple[Request, object, str, bool, float, int]:
        decision = POLICIES[policy_name](
            router,
            request,
            self.clusters,
            now,
            self.rng,
        )
        initial_cluster_id = decision.cluster_id
        if self._cluster_is_available(decision.cluster_id, now):
            return request, decision, initial_cluster_id, False, 0.0, 1

        available_clusters = self._available_clusters(now, exclude={decision.cluster_id})
        if not available_clusters:
            raise RuntimeError("no clusters available to fail over after injected outage")

        failover_delay = self.config.faults.retry_penalty
        retry_time = now + failover_delay
        self._process_control_plane_until(retry_time)
        self._advance_clusters(retry_time)
        rerouted_request = request if failover_delay == 0.0 else replace(request, arrival_time=request.arrival_time + failover_delay)
        retry_decision = POLICIES[policy_name](
            router,
            rerouted_request,
            available_clusters,
            retry_time,
            self.rng,
        )
        return rerouted_request, retry_decision, initial_cluster_id, True, failover_delay, 2

    def _cluster_is_available(self, cluster_id: str, now: float) -> bool:
        if cluster_id not in self.config.faults.failed_cluster_ids:
            return True
        if self.config.faults.failure_duration <= 0.0:
            return True
        end = self.config.faults.failure_start + self.config.faults.failure_duration
        return not (self.config.faults.failure_start <= now < end)

    def _available_clusters(self, now: float, exclude: set[str] | None = None) -> Dict[str, object]:
        exclude = exclude or set()
        return {
            cluster_id: cluster
            for cluster_id, cluster in self.clusters.items()
            if cluster_id not in exclude and self._cluster_is_available(cluster_id, now)
        }

    def _schedule_delivery(
        self,
        deliver_at: float,
        receiver_id: str,
        summary: object,
        source: str,
    ) -> None:
        self._delivery_sequence += 1
        heap_entry = (deliver_at, self._delivery_sequence, receiver_id, summary, source)
        self._pending_deliveries.append(heap_entry)
        self._pending_deliveries.sort(key=lambda item: (item[0], item[1]))

    def _deliver_pending_summaries(self, now: float) -> None:
        while self._pending_deliveries and self._pending_deliveries[0][0] <= now:
            _, _, receiver_id, summary, source = self._pending_deliveries.pop(0)
            self.routers[receiver_id].receive_summary(summary, now, source=source)
            self.control_plane_bytes += summary.byte_size + self.config.control_plane_header_bytes


def run_policies(
    config: SimulationConfig | None = None,
    policy_names: Iterable[str] = ("summary", "random", "load_only", "exact_prefix", "oracle"),
) -> Dict[str, SimulationMetrics]:
    config = config or SimulationConfig()
    requests = generate_workload(config.workload)
    results: Dict[str, SimulationMetrics] = {}
    for policy_name in policy_names:
        simulation = Simulation(config)
        _, metrics = simulation.run(policy_name=policy_name, requests=requests)
        results[policy_name] = metrics
    return results


def metrics_as_dict(metrics: SimulationMetrics) -> Dict[str, object]:
    return asdict(metrics)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
