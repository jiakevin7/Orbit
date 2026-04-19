from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from multiprocessing import get_context
from multiprocessing.connection import Connection
from typing import Any, Callable

from .cluster import Cluster, ClusterConfig
from .llamacpp import LlamaCppCluster, LlamaCppClusterConfig
from .models import Request
from .router import Router, RouterConfig


def _worker_loop(connection: Connection, factory: Callable[[], object]) -> None:
    target = factory()
    pending_tasks: dict[int, Future[Any]] = {}
    next_task_id = 0
    try:
        while True:
            try:
                message = connection.recv()
            except EOFError:
                break
            if not isinstance(message, tuple) or not message:
                continue
            method_name = message[0]
            if method_name == "__close__":
                break
            if method_name == "__submit__":
                _, submit_args, submit_kwargs = message
                if not submit_args:
                    connection.send(("error", {"type": "ValueError", "message": "missing submit method"}))
                    continue
                submit_method_name = str(submit_args[0])
                args = tuple(submit_args[1:])
                kwargs = dict(submit_kwargs)
                try:
                    future = getattr(target, submit_method_name)(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    connection.send(("error", {"type": exc.__class__.__name__, "message": str(exc)}))
                    continue
                next_task_id += 1
                pending_tasks[next_task_id] = future
                connection.send(("ok", next_task_id))
                continue
            if method_name == "__collect_completed__":
                completed: list[tuple[int, str, object]] = []
                for task_id, future in list(pending_tasks.items()):
                    if not future.done():
                        continue
                    try:
                        completed.append((task_id, "ok", future.result()))
                    except Exception as exc:  # noqa: BLE001
                        completed.append(
                            (task_id, "error", {"type": exc.__class__.__name__, "message": str(exc)})
                        )
                    del pending_tasks[task_id]
                connection.send(("ok", completed))
                continue
            _, args, kwargs = message
            try:
                result = getattr(target, method_name)(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                connection.send(("error", {"type": exc.__class__.__name__, "message": str(exc)}))
                continue
            connection.send(("ok", result))
    finally:
        close = getattr(target, "close", None)
        if callable(close):
            close()
        connection.close()


def _cluster_worker(connection: Connection, cluster_id: str, config: ClusterConfig) -> None:
    _worker_loop(connection, lambda: Cluster(cluster_id=cluster_id, config=config))


def _router_worker(
    connection: Connection,
    router_id: str,
    network_costs: dict[str, float],
    config: RouterConfig,
) -> None:
    _worker_loop(
        connection,
        lambda: Router(router_id=router_id, network_costs=network_costs, config=config),
    )


def _llama_cluster_worker(
    connection: Connection,
    cluster_id: str,
    cluster_config: ClusterConfig,
    backend_config: LlamaCppClusterConfig,
) -> None:
    _worker_loop(
        connection,
        lambda: LlamaCppCluster(
            cluster_id=cluster_id,
            cluster_config=cluster_config,
            backend_config=backend_config,
        ),
    )


class _ProcessProxy:
    def __init__(self, start_method: str, target: Callable[..., None], *args: object) -> None:
        context = get_context(start_method)
        parent, child = context.Pipe()
        self._connection = parent
        self._connection_lock = threading.Lock()
        self._process = context.Process(target=target, args=(child, *args), daemon=True)
        self._process.start()
        child.close()

    def _call(self, method_name: str, *args: object, **kwargs: object) -> Any:
        with self._connection_lock:
            self._connection.send((method_name, args, kwargs))
            status, payload = self._connection.recv()
        if status == "ok":
            return payload
        raise RuntimeError(f"worker call {method_name} failed: {payload['type']}: {payload['message']}")

    def close(self) -> None:
        if getattr(self, "_connection", None) is None:
            return
        try:
            self._connection.send(("__close__", (), {}))
        except (BrokenPipeError, EOFError, OSError):
            pass
        try:
            self._connection.close()
        finally:
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2.0)
            self._connection = None


class ProcessClusterProxy(_ProcessProxy):
    def __init__(self, cluster_id: str, config: ClusterConfig, start_method: str = "spawn") -> None:
        self.cluster_id = cluster_id
        self.config = config
        super().__init__(start_method, _cluster_worker, cluster_id, config)

    def advance_time(self, now: float) -> None:
        self._call("advance_time", now)

    def queue_depth(self, now: float) -> int:
        return int(self._call("queue_depth", now))

    def exact_prefix_match(self, tokens: tuple[int, ...], now: float) -> bool:
        return bool(self._call("exact_prefix_match", tuple(tokens), now))

    def true_reusable_prefix(self, tokens: tuple[int, ...], now: float) -> int:
        return int(self._call("true_reusable_prefix", tuple(tokens), now))

    def execute(self, request: object) -> object:
        return self._call("execute", request)

    def publish_summary(self, now: float) -> object:
        return self._call("publish_summary", now)


class ProcessLlamaCppClusterProxy(_ProcessProxy):
    def __init__(
        self,
        cluster_id: str,
        cluster_config: ClusterConfig,
        backend_config: LlamaCppClusterConfig,
        start_method: str = "spawn",
    ) -> None:
        self.cluster_id = cluster_id
        self.cluster_config = cluster_config
        self.backend_config = backend_config
        self._task_futures: dict[int, Future[Any]] = {}
        self._task_lock = threading.Lock()
        self._poller_stop = threading.Event()
        self._poller_thread: threading.Thread | None = None
        super().__init__(start_method, _llama_cluster_worker, cluster_id, cluster_config, backend_config)

    def advance_time(self, now: float) -> None:
        self._call("advance_time", now)

    def prepare_requests(self, requests: list[Request]) -> list[Request]:
        return list(self._call("prepare_requests", list(requests)))

    def queue_depth(self, now: float) -> int:
        return int(self._call("queue_depth", now))

    def exact_prefix_match(self, tokens: tuple[int, ...], now: float) -> bool:
        return bool(self._call("exact_prefix_match", tuple(tokens), now))

    def true_reusable_prefix(self, tokens: tuple[int, ...], now: float) -> int:
        return int(self._call("true_reusable_prefix", tuple(tokens), now))

    def execute(self, request: Request) -> object:
        return self._call("execute", request)

    def publish_summary(self, now: float) -> object:
        return self._call("publish_summary", now)

    def submit(self, request: Request) -> Future[Any]:
        task_id = int(self._call("__submit__", "submit", request))
        future: Future[Any] = Future()
        with self._task_lock:
            self._task_futures[task_id] = future
        self._ensure_poller()
        return future

    def close(self) -> None:
        self._poller_stop.set()
        poller = self._poller_thread
        if poller is not None and poller.is_alive():
            poller.join(timeout=0.5)
        with self._task_lock:
            pending = list(self._task_futures.values())
            self._task_futures.clear()
        for future in pending:
            if not future.done():
                future.set_exception(RuntimeError("cluster proxy closed before task completion"))
        super().close()

    def _ensure_poller(self) -> None:
        if self._poller_thread is not None and self._poller_thread.is_alive():
            return
        self._poller_stop.clear()
        self._poller_thread = threading.Thread(target=self._poll_tasks, daemon=True)
        self._poller_thread.start()

    def _poll_tasks(self) -> None:
        while True:
            with self._task_lock:
                has_tasks = bool(self._task_futures)
            if self._poller_stop.is_set() and not has_tasks:
                return
            if not has_tasks:
                time.sleep(0.01)
                continue
            try:
                completed = self._call("__collect_completed__")
            except Exception as exc:  # noqa: BLE001
                with self._task_lock:
                    pending = list(self._task_futures.values())
                    self._task_futures.clear()
                for future in pending:
                    if not future.done():
                        future.set_exception(exc)
                return
            if not completed:
                time.sleep(0.01)
                continue
            with self._task_lock:
                for task_id, status, payload in completed:
                    future = self._task_futures.pop(int(task_id), None)
                    if future is None or future.done():
                        continue
                    if status == "ok":
                        future.set_result(payload)
                    else:
                        future.set_exception(
                            RuntimeError(f"cluster task failed: {payload['type']}: {payload['message']}")
                        )


class ProcessRouterProxy(_ProcessProxy):
    def __init__(
        self,
        router_id: str,
        network_costs: dict[str, float],
        config: RouterConfig,
        start_method: str = "spawn",
    ) -> None:
        self.router_id = router_id
        self.network_costs = dict(network_costs)
        self.config = config
        super().__init__(start_method, _router_worker, router_id, dict(network_costs), config)

    def network_cost(self, cluster_id: str) -> float:
        return self.network_costs.get(cluster_id, float("inf"))

    def receive_summary(self, summary: object, received_at: float, source: str) -> None:
        self._call("receive_summary", summary, received_at, source)

    def export_summaries(self) -> dict[str, object]:
        return dict(self._call("export_summaries"))

    def route(self, request: object, cluster_ids: object, now: float) -> object:
        return self._call("route", request, tuple(cluster_ids), now)

    def load_only_route(self, request: object, cluster_ids: object, now: float) -> object:
        return self._call("load_only_route", request, tuple(cluster_ids), now)

    def summary_memory_bytes(self) -> int:
        return int(self._call("summary_memory_bytes"))

    def predict_latency(
        self,
        cluster_id: str,
        request: Request,
        estimated_reusable_tokens: int,
        raw_queue_depth: int,
        metadata_age: float = 0.0,
        uncertainty_gap: int = 0,
        missing_summary: bool = False,
        extra_uncertainty_penalty: float = 0.0,
    ) -> tuple[float, dict[str, float]]:
        coefficients = self._coefficients_for_cluster(cluster_id)
        remaining_prefill = max(0, request.input_length - estimated_reusable_tokens)
        queue_delay = raw_queue_depth * coefficients["queue_depth_penalty"]
        stale_penalty = max(0.0, metadata_age) * coefficients["stale_penalty_per_second"]
        uncertainty_penalty = (
            uncertainty_gap * coefficients["uncertainty_penalty_per_token"]
            + extra_uncertainty_penalty
        )
        missing_summary_penalty = coefficients["missing_summary_penalty"] if missing_summary else 0.0
        predicted_latency = (
            self.network_cost(cluster_id)
            + coefficients["fixed_overhead"]
            + queue_delay
            + remaining_prefill * coefficients["prefill_cost_per_token"]
            + request.continuation_tokens * coefficients["decode_cost_per_token"]
            + stale_penalty
            + uncertainty_penalty
            + missing_summary_penalty
        )
        return predicted_latency, {
            "network_cost": self.network_cost(cluster_id),
            "queue_delay": queue_delay,
            "raw_queue_depth": raw_queue_depth,
            "estimated_remaining_prefill_tokens": remaining_prefill,
            "stale_penalty": stale_penalty,
            "metadata_age": max(0.0, metadata_age),
            "uncertainty_gap": uncertainty_gap,
            "uncertainty_penalty": uncertainty_penalty,
            "missing_summary": 1.0 if missing_summary else 0.0,
            "missing_summary_penalty": missing_summary_penalty,
        }

    def _coefficients_for_cluster(self, cluster_id: str) -> dict[str, float]:
        overrides = self.config.cluster_overrides.get(cluster_id, {})
        return {
            "fixed_overhead": float(overrides.get("fixed_overhead", self.config.fixed_overhead)),
            "prefill_cost_per_token": float(overrides.get("prefill_cost_per_token", self.config.prefill_cost_per_token)),
            "decode_cost_per_token": float(overrides.get("decode_cost_per_token", self.config.decode_cost_per_token)),
            "queue_depth_penalty": float(overrides.get("queue_depth_penalty", self.config.queue_depth_penalty)),
            "stale_penalty_per_second": float(overrides.get("stale_penalty_per_second", self.config.stale_penalty_per_second)),
            "uncertainty_penalty_per_token": float(overrides.get("uncertainty_penalty_per_token", self.config.uncertainty_penalty_per_token)),
            "missing_summary_penalty": float(overrides.get("missing_summary_penalty", self.config.missing_summary_penalty)),
        }
