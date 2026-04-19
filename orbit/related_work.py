from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib import error, request

from .models import Request


@dataclass(frozen=True)
class ExternalProcessConfig:
    startup_command: tuple[str, ...] = ()
    shutdown_command: tuple[str, ...] = ()
    reset_command: tuple[str, ...] = ()
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    startup_timeout: float = 120.0
    shutdown_timeout: float = 30.0


@dataclass(frozen=True)
class ExternalSystemTarget:
    name: str
    family: str
    base_url: str
    model: str
    description: str = ""
    request_format: str = "chat"
    endpoint_path: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    health_path: str | None = "/health"
    request_timeout: float = 300.0
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = True
    max_tokens_override: int | None = None
    process: ExternalProcessConfig | None = None

    @property
    def resolved_endpoint_path(self) -> str:
        if self.endpoint_path:
            return self.endpoint_path
        if self.request_format == "completion":
            return "/v1/completions"
        return "/v1/chat/completions"

    @property
    def normalized_base_url(self) -> str:
        return self.base_url.rstrip("/")


@dataclass(frozen=True)
class ExternalCompletionResult:
    total_latency: float
    ttft: float
    status_code: int


@dataclass(frozen=True)
class ExternalRequestRecord:
    system: str
    family: str
    request_id: str
    arrival_time: float
    started_at: float
    finished_at: float
    actual_ttft: float
    actual_latency: float
    input_length: int
    continuation_tokens: int
    success: bool
    status_code: int | None
    error: str | None = None
    traffic_class: str = "synthetic"
    session_id: str | None = None
    source_id: str | None = None


@dataclass(frozen=True)
class ExternalRunMetrics:
    system: str
    family: str
    request_count: int
    success_count: int
    failure_count: int
    failure_rate: float
    ttft_p50: float
    ttft_p95: float
    latency_p50: float
    latency_p95: float
    throughput_rps: float


class OpenAICompatibleClient:
    def __init__(self, target: ExternalSystemTarget) -> None:
        self.target = target

    def wait_until_ready(self, timeout: float) -> None:
        if self.target.health_path is None:
            return
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._healthcheck()
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(0.25)
        raise RuntimeError(f"{self.target.name} did not become ready: {last_error}") from last_error

    def complete(self, request_obj: Request) -> ExternalCompletionResult:
        payload = self._build_payload(request_obj)
        request_body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if self.target.stream else "application/json",
            **self.target.headers,
        }
        http_request = request.Request(
            f"{self.target.normalized_base_url}{self.target.resolved_endpoint_path}",
            data=request_body,
            headers=headers,
            method="POST",
        )

        started_at = time.perf_counter()
        ttft: float | None = None
        with request.urlopen(http_request, timeout=self.target.request_timeout) as response:
            status_code = int(response.status)
            if self.target.stream:
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
                    if ttft is None and self._looks_like_first_token(event):
                        ttft = elapsed
            else:
                response.read()
        total_latency = time.perf_counter() - started_at
        if ttft is None:
            ttft = total_latency
        return ExternalCompletionResult(
            total_latency=total_latency,
            ttft=ttft,
            status_code=status_code,
        )

    def _healthcheck(self) -> None:
        health_request = request.Request(
            f"{self.target.normalized_base_url}{self.target.health_path}",
            headers=self.target.headers,
            method="GET",
        )
        with request.urlopen(health_request, timeout=min(self.target.request_timeout, 5.0)) as response:
            if response.status >= 400:
                raise RuntimeError(f"health check failed with status {response.status}")

    def _build_payload(self, request_obj: Request) -> dict[str, object]:
        max_tokens = self.target.max_tokens_override or request_obj.continuation_tokens
        if self.target.request_format == "completion":
            return {
                "model": self.target.model,
                "prompt": request_obj.prompt_text,
                "max_tokens": max_tokens,
                "temperature": self.target.temperature,
                "top_p": self.target.top_p,
                "stream": self.target.stream,
            }
        if self.target.request_format != "chat":
            raise ValueError(f"unsupported request_format: {self.target.request_format}")
        return {
            "model": self.target.model,
            "messages": prompt_text_to_messages(request_obj.prompt_text),
            "max_tokens": max_tokens,
            "temperature": self.target.temperature,
            "top_p": self.target.top_p,
            "stream": self.target.stream,
        }

    @staticmethod
    def _looks_like_first_token(event: Mapping[str, object]) -> bool:
        choices = event.get("choices")
        if not isinstance(choices, list):
            return False
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            text = choice.get("text")
            if isinstance(text, str) and text:
                return True
            delta = choice.get("delta")
            if isinstance(delta, Mapping):
                content = delta.get("content")
                if isinstance(content, str) and content:
                    return True
                if isinstance(content, list) and any(
                    isinstance(part, Mapping) and part.get("type") == "text" and part.get("text")
                    for part in content
                ):
                    return True
            message = choice.get("message")
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str) and content:
                    return True
        return False


class ManagedExternalSystem:
    def __init__(self, target: ExternalSystemTarget) -> None:
        self.target = target
        self.process: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> "ManagedExternalSystem":
        process = self.target.process
        if process is None:
            return self

        if process.reset_command:
            _run_command(
                process.reset_command,
                cwd=process.cwd,
                env=process.env,
                timeout=process.shutdown_timeout,
            )
        if process.startup_command:
            env = os.environ.copy()
            env.update(process.env)
            self.process = subprocess.Popen(
                list(process.startup_command),
                cwd=process.cwd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        process = self.target.process
        if process is None:
            return False

        if process.shutdown_command:
            _run_command(
                process.shutdown_command,
                cwd=process.cwd,
                env=process.env,
                timeout=process.shutdown_timeout,
            )
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=process.shutdown_timeout)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5.0)
            self.process = None
        return False


def load_related_work_targets(path: str | Path) -> tuple[ExternalSystemTarget, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_targets = payload.get("targets", payload)
    if not isinstance(raw_targets, list):
        raise ValueError("related work target config must contain a list of targets")

    targets: list[ExternalSystemTarget] = []
    for raw_target in raw_targets:
        if not isinstance(raw_target, Mapping):
            raise ValueError(f"unexpected target payload: {raw_target!r}")
        raw_process = raw_target.get("process")
        process = None
        if isinstance(raw_process, Mapping):
            process = ExternalProcessConfig(
                startup_command=_tuple_of_strings(raw_process.get("startup_command")),
                shutdown_command=_tuple_of_strings(raw_process.get("shutdown_command")),
                reset_command=_tuple_of_strings(raw_process.get("reset_command")),
                cwd=_string_or_none(raw_process.get("cwd")),
                env=_dict_of_strings(raw_process.get("env")),
                startup_timeout=float(raw_process.get("startup_timeout", 120.0)),
                shutdown_timeout=float(raw_process.get("shutdown_timeout", 30.0)),
            )
        targets.append(
            ExternalSystemTarget(
                name=str(raw_target["name"]),
                family=str(raw_target.get("family", raw_target["name"])),
                base_url=str(raw_target["base_url"]),
                model=str(raw_target["model"]),
                description=str(raw_target.get("description", "")),
                request_format=str(raw_target.get("request_format", "chat")),
                endpoint_path=_string_or_none(raw_target.get("endpoint_path")),
                headers=_dict_of_strings(raw_target.get("headers")),
                health_path=_string_or_none(raw_target.get("health_path", "/health")),
                request_timeout=float(raw_target.get("request_timeout", 300.0)),
                temperature=float(raw_target.get("temperature", 0.0)),
                top_p=float(raw_target.get("top_p", 1.0)),
                stream=bool(raw_target.get("stream", True)),
                max_tokens_override=(
                    int(raw_target["max_tokens_override"])
                    if raw_target.get("max_tokens_override") is not None
                    else None
                ),
                process=process,
            )
        )
    return tuple(targets)


def prompt_text_to_messages(prompt_text: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    current_role: str | None = None
    current_lines: list[str] = []
    role_map = {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "tool": "tool",
    }

    def flush() -> None:
        nonlocal current_role, current_lines
        if current_role is None:
            return
        content = "\n".join(current_lines).strip()
        if current_role == "assistant" and not content:
            current_role = None
            current_lines = []
            return
        if content:
            messages.append({"role": current_role, "content": content})
        current_role = None
        current_lines = []

    for raw_line in prompt_text.splitlines():
        stripped = raw_line.strip()
        normalized_role = role_map.get(stripped[:-1].lower()) if stripped.endswith(":") else None
        if normalized_role is not None:
            flush()
            current_role = normalized_role
            current_lines = []
            continue
        current_lines.append(raw_line)
    flush()

    if messages:
        return messages
    return [{"role": "user", "content": prompt_text.strip()}]


def scale_request_arrivals(requests: Sequence[Request], scale: float) -> list[Request]:
    if scale <= 0:
        raise ValueError("arrival scale must be positive")
    return [
        replace(
            request_obj,
            arrival_time=request_obj.arrival_time * scale,
            arrival_scale_applied=request_obj.arrival_scale_applied * scale,
        )
        for request_obj in requests
    ]


def cap_request_continuations(requests: Sequence[Request], max_tokens: int | None) -> list[Request]:
    if max_tokens is None:
        return list(requests)
    if max_tokens <= 0:
        raise ValueError("continuation token cap must be positive")
    return [
        replace(request_obj, continuation_tokens=min(request_obj.continuation_tokens, max_tokens))
        for request_obj in requests
    ]


def run_external_target(
    target: ExternalSystemTarget,
    *,
    warmup_requests: Sequence[Request],
    measured_requests: Sequence[Request],
    max_workers: int = 16,
) -> tuple[list[ExternalRequestRecord], ExternalRunMetrics]:
    client = OpenAICompatibleClient(target)
    with ManagedExternalSystem(target):
        process = target.process
        if process is not None:
            client.wait_until_ready(timeout=process.startup_timeout)
        if warmup_requests:
            _replay_requests(client, target, warmup_requests, max_workers=max_workers)
        records = _replay_requests(client, target, measured_requests, max_workers=max_workers)
    return records, summarize_external_run(records, target.name, target.family)


def external_records_as_dicts(records: Sequence[ExternalRequestRecord]) -> list[dict[str, object]]:
    return [asdict(record) for record in records]


def external_metrics_as_dict(metrics: ExternalRunMetrics) -> dict[str, object]:
    return asdict(metrics)


def summarize_external_run(
    records: Sequence[ExternalRequestRecord],
    system: str,
    family: str,
) -> ExternalRunMetrics:
    ttfts = [record.actual_ttft for record in records if record.success]
    latencies = [record.actual_latency for record in records if record.success]
    failures = [record for record in records if not record.success]
    success_count = len(records) - len(failures)
    throughput_window = 0.0
    if records:
        throughput_window = max(record.finished_at for record in records) - min(record.arrival_time for record in records)
    throughput_rps = success_count / throughput_window if throughput_window > 0 else 0.0
    return ExternalRunMetrics(
        system=system,
        family=family,
        request_count=len(records),
        success_count=success_count,
        failure_count=len(failures),
        failure_rate=(len(failures) / len(records)) if records else 0.0,
        ttft_p50=_percentile(ttfts, 0.50),
        ttft_p95=_percentile(ttfts, 0.95),
        latency_p50=_percentile(latencies, 0.50),
        latency_p95=_percentile(latencies, 0.95),
        throughput_rps=throughput_rps,
    )


def summarize_external_records(
    records: Sequence[ExternalRequestRecord],
    *,
    group_field: str | None = None,
) -> list[dict[str, object]]:
    grouped: dict[str | None, list[ExternalRequestRecord]] = {}
    for record in records:
        group_value = getattr(record, group_field) if group_field else None
        grouped.setdefault(group_value, []).append(record)

    rows: list[dict[str, object]] = []
    for group_value, group_records in sorted(
        grouped.items(),
        key=lambda item: (str(item[0]) if item[0] is not None else ""),
    ):
        summary = external_metrics_as_dict(
            summarize_external_run(
                group_records,
                system=group_records[0].system if group_records else "",
                family=group_records[0].family if group_records else "",
            )
        )
        if group_field is not None:
            summary[group_field] = group_value
        rows.append(summary)
    return rows


def _replay_requests(
    client: OpenAICompatibleClient,
    target: ExternalSystemTarget,
    requests: Sequence[Request],
    *,
    max_workers: int,
) -> list[ExternalRequestRecord]:
    if not requests:
        return []

    records: list[ExternalRequestRecord] = []
    worker_count = max(1, min(max_workers, len(requests)))
    run_origin = time.monotonic()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = []
        for request_obj in requests:
            target_submit = run_origin + request_obj.arrival_time
            delay = target_submit - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            futures.append(executor.submit(_issue_request, client, target, request_obj, run_origin))
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda record: record.started_at)
    return records


def _issue_request(
    client: OpenAICompatibleClient,
    target: ExternalSystemTarget,
    request_obj: Request,
    run_origin: float,
) -> ExternalRequestRecord:
    started = time.monotonic()
    try:
        result = client.complete(request_obj)
        finished = time.monotonic()
        return ExternalRequestRecord(
            system=target.name,
            family=target.family,
            request_id=request_obj.request_id,
            arrival_time=request_obj.arrival_time,
            started_at=started - run_origin,
            finished_at=finished - run_origin,
            actual_ttft=result.ttft,
            actual_latency=result.total_latency,
            input_length=request_obj.input_length,
            continuation_tokens=request_obj.continuation_tokens,
            success=True,
            status_code=result.status_code,
            traffic_class=request_obj.traffic_class,
            session_id=request_obj.session_id,
            source_id=request_obj.source_id,
        )
    except error.HTTPError as exc:
        finished = time.monotonic()
        return ExternalRequestRecord(
            system=target.name,
            family=target.family,
            request_id=request_obj.request_id,
            arrival_time=request_obj.arrival_time,
            started_at=started - run_origin,
            finished_at=finished - run_origin,
            actual_ttft=finished - started,
            actual_latency=finished - started,
            input_length=request_obj.input_length,
            continuation_tokens=request_obj.continuation_tokens,
            success=False,
            status_code=exc.code,
            error=str(exc),
            traffic_class=request_obj.traffic_class,
            session_id=request_obj.session_id,
            source_id=request_obj.source_id,
        )
    except Exception as exc:  # noqa: BLE001
        finished = time.monotonic()
        return ExternalRequestRecord(
            system=target.name,
            family=target.family,
            request_id=request_obj.request_id,
            arrival_time=request_obj.arrival_time,
            started_at=started - run_origin,
            finished_at=finished - run_origin,
            actual_ttft=finished - started,
            actual_latency=finished - started,
            input_length=request_obj.input_length,
            continuation_tokens=request_obj.continuation_tokens,
            success=False,
            status_code=None,
            error=str(exc),
            traffic_class=request_obj.traffic_class,
            session_id=request_obj.session_id,
            source_id=request_obj.source_id,
        )


def _run_command(
    command: Sequence[str],
    *,
    cwd: str | None,
    env: Mapping[str, str],
    timeout: float,
) -> None:
    merged_env = os.environ.copy()
    merged_env.update(env)
    subprocess.run(
        list(command),
        cwd=cwd,
        env=merged_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
        timeout=timeout,
    )


def _tuple_of_strings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"expected list of strings, got {value!r}")
    return tuple(str(item) for item in value)


def _dict_of_strings(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"expected string map, got {value!r}")
    return {str(key): str(item) for key, item in value.items()}


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


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
