import heapq
import json
import subprocess
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, replace
from pathlib import Path
from urllib import request
from .bloom import BloomFilter
from .cluster import ClusterConfig
from .hashing import hash_prefix, hot_prefix_hashes
from .models import ClusterExecution, ClusterSummary, Request
from .trie import PrefixTrie


@dataclass(frozen=True)
class LlamaCppClusterConfig:
    model_path: str | None = None
    executable: str = "llama-server"
    host: str = "127.0.0.1"
    port_base: int = 8081
    threads: int = 4
    ctx_size: int = 4096
    parallel: int = 1
    request_timeout: float = 120.0
    startup_timeout: float = 120.0
    prompt_token_cap: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    manage_server: bool = True
    extra_args: tuple[str, ...] = ()

    def for_cluster(self, index):
        return LlamaCppClusterConfig(
            model_path=self.model_path,
            executable=self.executable,
            host=self.host,
            port_base=self.port_base + index,
            threads=self.threads,
            ctx_size=self.ctx_size,
            parallel=self.parallel,
            request_timeout=self.request_timeout,
            startup_timeout=self.startup_timeout,
            prompt_token_cap=self.prompt_token_cap,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            manage_server=self.manage_server,
            extra_args=self.extra_args,
        )

    @property
    def base_url(self):
        return f"http://{self.host}:{self.port_base}"


@dataclass(frozen=True)
class LlamaCppResult:
    total_latency: float
    ttft: float
    prompt_eval_latency: float
    processing_started_latency: float
    cache_ready_latency: float


@dataclass(frozen=True)
class LlamaCppSlot:
    slot_id: int
    context_size: int
    is_processing: bool
    task_id: int | None = None


class LlamaCppClient:
    def __init__(
        self, base_url, request_timeout=120.0, temperature=0.0, top_p=1.0, seed=0
    ):
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed

    def wait_until_ready(self, timeout):
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._healthcheck()
                return
            except Exception as exc:
                last_error = exc
                time.sleep(0.25)
        raise RuntimeError(
            f"llama.cpp server did not become ready: {last_error}"
        ) from last_error

    def tokenize(self, text):
        payload = self._json_request(
            "/tokenize", payload={"content": text}, method="POST"
        )
        if not isinstance(payload, dict):
            raise RuntimeError(f"unexpected tokenize payload: {payload!r}")
        tokens = payload.get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError(f"tokenize response missing tokens: {payload!r}")
        return tuple((int(token) for token in tokens))

    def slots(self):
        payload = self._json_request("/slots", method="GET")
        if not isinstance(payload, list):
            raise RuntimeError(f"unexpected slots payload: {payload!r}")
        slots: list[LlamaCppSlot] = []
        for slot_payload in payload:
            if not isinstance(slot_payload, dict):
                raise RuntimeError(f"unexpected slot payload: {slot_payload!r}")
            task_id = slot_payload.get("id_task")
            slots.append(
                LlamaCppSlot(
                    slot_id=int(slot_payload.get("id", -1)),
                    context_size=int(slot_payload.get("n_ctx", 0)),
                    is_processing=bool(slot_payload.get("is_processing", False)),
                    task_id=int(task_id) if isinstance(task_id, int) else None,
                )
            )
        return tuple(slots)

    def complete(self, prompt, max_tokens, event_callback=None):
        # Streaming lets the benchmark measure TTFT and prompt-cache readiness
        # separately from full completion latency.
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "stream": True,
            "cache_prompt": True,
            "return_progress": True,
        }
        request_body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}/completion",
            data=request_body,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        started_at = time.perf_counter()
        ttft: float | None = None
        prompt_eval_latency: float | None = None
        processing_started_latency: float | None = None
        cache_ready_latency: float | None = None

        # Read server-sent events until completion. Different llama.cpp builds
        # expose timing fields slightly differently, so each metric has a
        # fallback path below.
        with request.urlopen(http_request, timeout=self.request_timeout) as response:
            for raw_line in response:
                elapsed = time.perf_counter() - started_at
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload_text = line[5:].strip()
                if not payload_text:
                    continue
                if payload_text == "[DONE]":
                    break
                event = json.loads(payload_text)
                if event_callback is not None:
                    event_callback(event, elapsed)

                if ttft is None and self._looks_like_first_token(event):
                    ttft = elapsed

                prompt_progress = self._extract_prompt_progress(event)
                if prompt_progress is not None:
                    if processing_started_latency is None:
                        processing_started_latency = elapsed
                    prompt_eval_latency = self._extract_prompt_eval_latency(
                        event, fallback=prompt_eval_latency
                    )
                    if (
                        cache_ready_latency is None
                        and prompt_progress["total"] > 0
                        and (prompt_progress["processed"] >= prompt_progress["total"])
                    ):
                        cache_ready_latency = elapsed
                prompt_eval_latency = self._extract_prompt_eval_latency(
                    event, fallback=prompt_eval_latency
                )

        finished_at = time.perf_counter()
        total_latency = finished_at - started_at

        # Normalize missing telemetry into conservative, bounded timings so the
        # rest of the benchmark can use one result schema.
        if ttft is None:
            ttft = total_latency
        if processing_started_latency is None:
            processing_started_latency = 0.0
        if prompt_eval_latency is None:
            prompt_eval_latency = max(
                0.0, min(ttft, total_latency) - processing_started_latency
            )
        if cache_ready_latency is None:
            cache_ready_latency = min(
                total_latency,
                processing_started_latency + max(prompt_eval_latency, 0.0),
            )
        return LlamaCppResult(
            total_latency=total_latency,
            ttft=ttft,
            prompt_eval_latency=min(max(prompt_eval_latency, 0.0), total_latency),
            processing_started_latency=min(
                max(processing_started_latency, 0.0), total_latency
            ),
            cache_ready_latency=min(max(cache_ready_latency, 0.0), total_latency),
        )

    def _healthcheck(self):
        health_request = request.Request(f"{self.base_url}/health", method="GET")
        with request.urlopen(
            health_request, timeout=min(self.request_timeout, 5.0)
        ) as response:
            if response.status >= 400:
                raise RuntimeError(f"health check failed with status {response.status}")

    def _json_request(self, path, payload=None, method="GET"):
        request_body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            request_body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        http_request = request.Request(
            f"{self.base_url}/{path.lstrip('/')}",
            data=request_body,
            headers=headers,
            method=method,
        )
        with request.urlopen(http_request, timeout=self.request_timeout) as response:
            if response.status >= 400:
                raise RuntimeError(
                    f"request to {path} failed with status {response.status}"
                )
            body = response.read()
        if not body:
            return None
        return json.loads(body.decode("utf-8"))

    @staticmethod
    def _looks_like_first_token(event):
        if LlamaCppClient._extract_prompt_progress(event) is not None:
            return False
        content = event.get("content")
        if isinstance(content, str) and content:
            return True
        tokens = event.get("tokens")
        if isinstance(tokens, list) and len(tokens) > 0:
            return True
        token = event.get("token")
        if isinstance(token, str) and token:
            return True
        return False

    @staticmethod
    def _extract_prompt_eval_latency(event, fallback=None):
        prompt_progress = LlamaCppClient._extract_prompt_progress(event)
        if prompt_progress is not None and prompt_progress["time_ms"] is not None:
            return float(prompt_progress["time_ms"]) / 1000.0
        timings = event.get("timings")
        if not isinstance(timings, dict):
            return fallback
        prompt_ms = timings.get("prompt_ms")
        if isinstance(prompt_ms, (int, float)):
            return float(prompt_ms) / 1000.0
        return fallback

    @staticmethod
    def _extract_prompt_progress(event):
        payload = event.get("prompt_progress")
        if not isinstance(payload, dict):
            return None
        total = payload.get("total")
        processed = payload.get("processed")
        if not isinstance(total, int) or not isinstance(processed, int):
            return None
        time_ms = payload.get("time_ms")
        normalized_time_ms: float | None = None
        if isinstance(time_ms, (int, float)):
            normalized_time_ms = float(time_ms)
        return {"total": total, "processed": processed, "time_ms": normalized_time_ms}


class ManagedLlamaCppServer:
    def __init__(self, config):
        self.config = config
        self.process: subprocess.Popen[bytes] | None = None

    def start(self):
        # Each cluster gets its own llama-server process and port to approximate
        # independent serving clusters on one local machine.
        if self.process is not None:
            return
        if not self.config.model_path:
            raise ValueError("model_path is required when manage_server is enabled")
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")
        command = [
            self.config.executable,
            "--model",
            str(model_path),
            "--host",
            self.config.host,
            "--port",
            str(self.config.port_base),
            "--threads",
            str(self.config.threads),
            "--ctx-size",
            str(self.config.ctx_size),
            "--parallel",
            str(self.config.parallel),
            "--cache-prompt",
            "--slots",
        ]
        command.extend(self.config.extra_args)
        self.process = subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def stop(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None


class LlamaCppCluster:
    def __init__(self, cluster_id, cluster_config=None, backend_config=None):
        self.cluster_id = cluster_id
        self.cluster_config = cluster_config or ClusterConfig()
        self.backend_config = backend_config or LlamaCppClusterConfig()
        # Mirror the synthetic cluster's cache accounting using real token ids
        # and live completion timing from llama.cpp.
        self.trie = PrefixTrie()
        self._cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self._cached_token_total = 0
        self._pending_insertions: list[tuple[float, int, str, tuple[int, ...]]] = []
        self._active_requests = 0
        self._cache_sequence = 0
        self._summary_version = 0
        self._lock = threading.RLock()
        self._client = LlamaCppClient(
            base_url=self.backend_config.base_url,
            request_timeout=self.backend_config.request_timeout,
            temperature=self.backend_config.temperature,
            top_p=self.backend_config.top_p,
            seed=self.backend_config.seed,
        )
        self._server = (
            ManagedLlamaCppServer(self.backend_config)
            if self.backend_config.manage_server
            else None
        )
        self._started = False

    def advance_time(self, now):
        with self._lock:
            self._advance_time_locked(now)

    def prepare_requests(self, requests):
        if not requests:
            return []
        self._ensure_started()
        prepared: list[Request] = []
        for request_obj in requests:
            prepared.append(self._prepare_request(request_obj))
        return prepared

    def _prepare_request(self, request_obj):
        # Re-tokenize prompts with the serving backend so routing summaries and
        # actual KV reuse operate on the same token boundaries.
        prompt_prefix_text = request_obj.prompt_prefix_text or request_obj.prompt_text
        budget = self._prompt_token_budget(request_obj.continuation_tokens)
        tokens = self._client.tokenize(_prompt_text_from_prefix(prompt_prefix_text))
        if budget is not None and len(tokens) > budget:
            prompt_prefix_text = self._truncate_prompt_prefix_to_budget(
                prompt_prefix_text, token_budget=budget
            )
            tokens = self._client.tokenize(_prompt_text_from_prefix(prompt_prefix_text))
        return replace(
            request_obj,
            prompt_prefix_text=prompt_prefix_text,
            prefix_tokens=tokens,
            prefix_token_source="llama_cpp",
        )

    def _prompt_token_budget(self, continuation_tokens):
        configured_cap = self.backend_config.prompt_token_cap
        hard_cap = max(1, self.backend_config.ctx_size - max(continuation_tokens, 1))
        if configured_cap is None:
            return hard_cap
        return max(1, min(configured_cap, hard_cap))

    def _truncate_prompt_prefix_to_budget(self, prompt_prefix_text, *, token_budget):
        normalized = prompt_prefix_text.rstrip()
        if not normalized:
            return normalized
        low = 0
        high = len(normalized)
        best = ""
        while low <= high:
            mid = (low + high) // 2
            candidate = normalized[:mid].rstrip()
            candidate_tokens = self._client.tokenize(
                _prompt_text_from_prefix(candidate)
            )
            if len(candidate_tokens) <= token_budget:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1
        return best

    def queue_depth(self, now):
        observed_busy_slots = self._observed_busy_slots()
        with self._lock:
            active_requests = self._active_requests
        if observed_busy_slots is not None:
            return max(active_requests, observed_busy_slots)
        return active_requests

    def true_reusable_prefix(self, tokens, now):
        with self._lock:
            self._advance_time_locked(now)
            return self.trie.longest_prefix(tokens)

    def execute(self, request_obj):
        return self._execute_request(request_obj)

    def submit(self, request_obj):
        self._ensure_started()
        future: Future[ClusterExecution] = Future()
        worker = threading.Thread(
            target=self._execute_request_async, args=(request_obj, future), daemon=True
        )
        worker.start()
        return future

    def publish_summary(self, now):
        with self._lock:
            self._advance_time_locked(now)
            filters = {
                depth: BloomFilter(
                    num_bits=self.cluster_config.bloom_bits,
                    num_hashes=self.cluster_config.bloom_hashes,
                )
                for depth in self.cluster_config.summary_depths
            }
            for tokens in self._cache.values():
                for depth in self.cluster_config.summary_depths:
                    if len(tokens) >= depth:
                        filters[depth].add(hash_prefix(tokens, depth))
            hotsets = hot_prefix_hashes(
                self._cache.values(),
                self.cluster_config.hotset_depths,
                self.cluster_config.hotset_capacity_per_depth,
            )
            self._summary_version += 1
            version = self._summary_version
            byte_size = (
                sum((bloom.byte_size for bloom in filters.values()))
                + sum((8 * len(values) for values in hotsets.values()))
                + 64
            )
        return ClusterSummary(
            cluster_id=self.cluster_id,
            version=version,
            created_at=now,
            queue_depth=self.queue_depth(now),
            depths=self.cluster_config.summary_depths,
            filters=filters,
            byte_size=byte_size,
            hot_prefix_hashes=hotsets,
        )

    def close(self):
        if self._server is not None:
            self._server.stop()
        self._started = False

    def _ensure_started(self):
        if self._started:
            return
        if self._server is not None:
            self._server.start()
        self._client.wait_until_ready(self.backend_config.startup_timeout)
        self._started = True

    def _observed_busy_slots(self):
        if not self._started:
            return None
        try:
            slots = self._client.slots()
        except Exception:
            return None
        return sum((1 for slot in slots if slot.is_processing))

    def _advance_time_locked(self, now):
        while self._pending_insertions and self._pending_insertions[0][0] <= now:
            _, _, request_id, tokens = heapq.heappop(self._pending_insertions)
            self._insert_into_cache(request_id, tokens)

    def _execute_request_async(self, request_obj, future):
        try:
            future.set_result(self._execute_request(request_obj))
        except Exception as exc:
            future.set_exception(exc)

    def _execute_request(self, request_obj):
        # Insert cache state as soon as prompt evaluation completes if the
        # stream exposes progress; otherwise defer to the measured fallback.
        self._ensure_started()
        now = request_obj.arrival_time

        # Snapshot exact reuse before submitting and mark this cluster as busy
        # for subsequent queue-depth observations.
        with self._lock:
            self._advance_time_locked(now)
            true_reusable = self.trie.longest_prefix(request_obj.prefix_tokens)
            queue_depth_before = self._active_requests
            self._active_requests += 1

        cache_ready_latency: float | None = None
        cache_inserted = False

        def handle_event(event, elapsed):
            nonlocal cache_ready_latency, cache_inserted
            prompt_progress = self._client._extract_prompt_progress(event)
            if prompt_progress is None:
                return
            if (
                cache_inserted
                or prompt_progress["total"] <= 0
                or prompt_progress["processed"] < prompt_progress["total"]
            ):
                return
            cache_ready_latency = elapsed
            with self._lock:
                self._insert_into_cache(
                    request_obj.request_id, request_obj.prefix_tokens
                )
            cache_inserted = True

        # Completion blocks this worker thread while the simulation can continue
        # routing other live requests through separate cluster threads.
        try:
            completion = self._client.complete(
                prompt=request_obj.prompt_text,
                max_tokens=request_obj.continuation_tokens,
                event_callback=handle_event,
            )
        finally:
            with self._lock:
                self._active_requests = max(0, self._active_requests - 1)

        # If the stream did not expose prompt progress, schedule cache insertion
        # at the best available cache-ready timestamp.
        if not cache_inserted:
            cache_ready_latency = completion.cache_ready_latency
            with self._lock:
                self._cache_sequence += 1
                heapq.heappush(
                    self._pending_insertions,
                    (
                        now + completion.cache_ready_latency,
                        self._cache_sequence,
                        request_obj.request_id,
                        request_obj.prefix_tokens,
                    ),
                )
                self._advance_time_locked(now + completion.total_latency)

        processing_started_latency = completion.processing_started_latency
        cache_ready_at = now + (
            cache_ready_latency
            if cache_ready_latency is not None
            else completion.cache_ready_latency
        )
        started_at = now + processing_started_latency
        finished_at = now + completion.total_latency
        return ClusterExecution(
            cluster_id=self.cluster_id,
            queue_depth_before=queue_depth_before,
            true_reusable_tokens=true_reusable,
            service_time=max(
                0.0, completion.total_latency - processing_started_latency
            ),
            queue_delay=max(0.0, processing_started_latency),
            started_at=started_at,
            finished_at=finished_at,
            time_to_first_token=completion.ttft,
            cache_ready_at=cache_ready_at,
        )

    def _insert_into_cache(self, request_id, tokens):
        if request_id in self._cache:
            old_tokens = self._cache.pop(request_id)
            self._remove_cached_tokens(old_tokens)
        self._cache[request_id] = tuple(tokens)
        self._add_cached_tokens(tokens)
        while self._cache_limit_exceeded():
            evicted_request_id, evicted_tokens = self._cache.popitem(last=False)
            self._remove_cached_tokens(evicted_tokens)
            if evicted_request_id == request_id and (not self._cache_limit_exceeded()):
                break

    def _cache_limit_exceeded(self):
        if len(self._cache) > self.cluster_config.cache_capacity:
            return True
        if (
            self.cluster_config.cache_capacity_tokens is not None
            and self._cached_token_total > self.cluster_config.cache_capacity_tokens
        ):
            return True
        return False

    def _add_cached_tokens(self, tokens):
        self.trie.insert(tokens)
        self._cached_token_total += len(tokens)

    def _remove_cached_tokens(self, tokens):
        self.trie.remove(tokens)
        self._cached_token_total -= len(tokens)


def _prompt_text_from_prefix(prompt_prefix_text):
    normalized = prompt_prefix_text.rstrip()
    if normalized.endswith("Assistant:"):
        return normalized
    return f"{normalized}\nAssistant:"
