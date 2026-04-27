import threading
import time
from concurrent.futures import Future
from multiprocessing import get_context
from .cluster import Cluster
from .llamacpp import LlamaCppCluster
from .router import Router, build_prediction_details


def _worker_loop(connection, factory):
    # Multiprocess mode keeps router/cluster state behind RPC-like proxies so
    # tests can exercise process isolation without changing simulation logic.
    target = factory()
    pending_tasks = {}
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
                    connection.send(
                        (
                            "error",
                            {"type": "ValueError", "message": "missing submit method"},
                        )
                    )
                    continue
                submit_method_name = str(submit_args[0])
                args = tuple(submit_args[1:])
                kwargs = dict(submit_kwargs)
                try:
                    future = getattr(target, submit_method_name)(*args, **kwargs)
                except Exception as exc:
                    connection.send(
                        ("error", {"type": exc.__class__.__name__, "message": str(exc)})
                    )
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
                    except Exception as exc:
                        completed.append(
                            (
                                task_id,
                                "error",
                                {"type": exc.__class__.__name__, "message": str(exc)},
                            )
                        )
                    del pending_tasks[task_id]
                connection.send(("ok", completed))
                continue
            _, args, kwargs = message
            try:
                result = getattr(target, method_name)(*args, **kwargs)
            except Exception as exc:
                connection.send(
                    ("error", {"type": exc.__class__.__name__, "message": str(exc)})
                )
                continue
            connection.send(("ok", result))
    finally:
        close = getattr(target, "close", None)
        if callable(close):
            close()
        connection.close()


def _cluster_worker(connection, cluster_id, config):
    _worker_loop(connection, lambda: Cluster(cluster_id=cluster_id, config=config))


def _router_worker(connection, router_id, network_costs, config):
    _worker_loop(
        connection,
        lambda: Router(router_id=router_id, network_costs=network_costs, config=config),
    )


def _llama_cluster_worker(connection, cluster_id, cluster_config, backend_config):
    _worker_loop(
        connection,
        lambda: LlamaCppCluster(
            cluster_id=cluster_id,
            cluster_config=cluster_config,
            backend_config=backend_config,
        ),
    )


class _ProcessProxy:
    def __init__(self, start_method, target, *args):
        context = get_context(start_method)
        parent, child = context.Pipe()
        self._connection = parent
        self._connection_lock = threading.Lock()
        self._process = context.Process(target=target, args=(child, *args), daemon=True)
        self._process.start()
        child.close()

    def _call(self, method_name, *args, **kwargs):
        with self._connection_lock:
            self._connection.send((method_name, args, kwargs))
            status, payload = self._connection.recv()
        if status == "ok":
            return payload
        raise RuntimeError(
            f"worker call {method_name} failed: {payload['type']}: {payload['message']}"
        )

    def close(self):
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
    def __init__(self, cluster_id, config, start_method="spawn"):
        self.cluster_id = cluster_id
        self.config = config
        super().__init__(start_method, _cluster_worker, cluster_id, config)

    def advance_time(self, now):
        self._call("advance_time", now)

    def queue_depth(self, now):
        return int(self._call("queue_depth", now))

    def true_reusable_prefix(self, tokens, now):
        return int(self._call("true_reusable_prefix", tuple(tokens), now))

    def execute(self, request):
        return self._call("execute", request)

    def publish_summary(self, now):
        return self._call("publish_summary", now)


class ProcessLlamaCppClusterProxy(_ProcessProxy):
    def __init__(
        self, cluster_id, cluster_config, backend_config, start_method="spawn"
    ):
        self.cluster_id = cluster_id
        self.cluster_config = cluster_config
        self.backend_config = backend_config
        self._task_futures = {}
        self._task_lock = threading.Lock()
        self._poller_stop = threading.Event()
        self._poller_thread: threading.Thread | None = None
        super().__init__(
            start_method,
            _llama_cluster_worker,
            cluster_id,
            cluster_config,
            backend_config,
        )

    def advance_time(self, now):
        self._call("advance_time", now)

    def prepare_requests(self, requests):
        return list(self._call("prepare_requests", list(requests)))

    def queue_depth(self, now):
        return int(self._call("queue_depth", now))

    def true_reusable_prefix(self, tokens, now):
        return int(self._call("true_reusable_prefix", tuple(tokens), now))

    def execute(self, request):
        return self._call("execute", request)

    def publish_summary(self, now):
        return self._call("publish_summary", now)

    def submit(self, request):
        # Live llama.cpp requests return futures; a poller transfers completion
        # or errors back from the worker process.
        task_id = int(self._call("__submit__", "submit", request))
        future = Future()
        with self._task_lock:
            self._task_futures[task_id] = future
        self._ensure_poller()
        return future

    def close(self):
        self._poller_stop.set()
        poller = self._poller_thread
        if poller is not None and poller.is_alive():
            poller.join(timeout=0.5)
        with self._task_lock:
            pending = list(self._task_futures.values())
            self._task_futures.clear()
        for future in pending:
            if not future.done():
                future.set_exception(
                    RuntimeError("cluster proxy closed before task completion")
                )
        super().close()

    def _ensure_poller(self):
        if self._poller_thread is not None and self._poller_thread.is_alive():
            return
        self._poller_stop.clear()
        self._poller_thread = threading.Thread(target=self._poll_tasks, daemon=True)
        self._poller_thread.start()

    def _poll_tasks(self):
        while True:
            with self._task_lock:
                has_tasks = bool(self._task_futures)
            if self._poller_stop.is_set() and (not has_tasks):
                return
            if not has_tasks:
                time.sleep(0.01)
                continue
            try:
                completed = self._call("__collect_completed__")
            except Exception as exc:
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
                            RuntimeError(
                                f"cluster task failed: {payload['type']}: {payload['message']}"
                            )
                        )


class ProcessRouterProxy(_ProcessProxy):
    def __init__(self, router_id, network_costs, config, start_method="spawn"):
        self.router_id = router_id
        self.network_costs = dict(network_costs)
        self.config = config
        super().__init__(
            start_method, _router_worker, router_id, dict(network_costs), config
        )

    def network_cost(self, cluster_id):
        return self.network_costs.get(cluster_id, float("inf"))

    def receive_summary(self, summary, received_at, source):
        self._call("receive_summary", summary, received_at, source)

    def export_summaries(self):
        return dict(self._call("export_summaries"))

    def route(self, request, cluster_ids, now):
        return self._call("route", request, tuple(cluster_ids), now)

    def least_loaded_route(self, request, cluster_ids, now):
        return self._call("least_loaded_route", request, tuple(cluster_ids), now)

    def summary_memory_bytes(self):
        return int(self._call("summary_memory_bytes"))

    def predict_latency(
        self,
        cluster_id,
        request,
        estimated_reusable_tokens,
        raw_queue_depth,
        metadata_age=0.0,
        uncertainty_gap=0,
        missing_summary=False,
        extra_uncertainty_penalty=0.0,
    ):
        details = build_prediction_details(
            config=self.config,
            network_cost=self.network_cost(cluster_id),
            request=request,
            estimated_reusable_tokens=estimated_reusable_tokens,
            raw_queue_depth=raw_queue_depth,
            metadata_age=metadata_age,
            uncertainty_gap=uncertainty_gap,
            missing_summary=missing_summary,
            extra_uncertainty_penalty=extra_uncertainty_penalty,
        )
        return (float(details["predicted_latency"]), details)
