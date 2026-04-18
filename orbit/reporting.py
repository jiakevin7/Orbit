from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, fields
from pathlib import Path
from typing import Mapping, Sequence

from .models import ExecutionRecord, Request, SimulationMetrics


EXECUTION_RECORD_FIELDS = tuple(field.name for field in fields(ExecutionRecord))
REQUEST_FIELDS = tuple(field.name for field in fields(Request))


def execution_records_as_dicts(records: Sequence[ExecutionRecord]) -> list[dict[str, object]]:
    return [asdict(record) for record in records]


def requests_as_dicts(requests: Sequence[Request]) -> list[dict[str, object]]:
    return [asdict(request_obj) for request_obj in requests]


def metrics_as_dict(metrics: SimulationMetrics) -> dict[str, object]:
    return asdict(metrics)


def metrics_rows_by_policy(
    metrics_by_policy: Mapping[str, SimulationMetrics],
) -> list[dict[str, object]]:
    cluster_ids = sorted(
        {
            cluster_id
            for metrics in metrics_by_policy.values()
            for cluster_id in metrics.cluster_request_counts
        }
    )
    rows: list[dict[str, object]] = []
    for policy_name in sorted(metrics_by_policy):
        metrics = metrics_by_policy[policy_name]
        row = metrics_as_dict(metrics)
        cluster_counts = row.pop("cluster_request_counts")
        for cluster_id in cluster_ids:
            row[f"cluster_requests_{cluster_id}"] = cluster_counts.get(cluster_id, 0)
        rows.append(row)
    return rows


def write_json(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_execution_records_csv(path: str | Path, records: Sequence[ExecutionRecord]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXECUTION_RECORD_FIELDS)
        writer.writeheader()
        for row in execution_records_as_dicts(records):
            writer.writerow(row)


def write_rows_csv(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_execution_records(
    records: Sequence[ExecutionRecord],
    policy_name: str,
    group_field: str | None = None,
) -> list[dict[str, object]]:
    grouped: dict[str | None, list[ExecutionRecord]] = {}
    for record in records:
        group_value = getattr(record, group_field) if group_field else None
        grouped.setdefault(group_value, []).append(record)

    rows: list[dict[str, object]] = []
    for group_value, group_records in sorted(grouped.items(), key=lambda item: (str(item[0]) if item[0] is not None else "")):
        latencies = [record.actual_latency for record in group_records]
        ttfts = [record.actual_ttft for record in group_records]
        reusable_prefixes = [record.actual_reusable_tokens for record in group_records]
        reuse_fractions = [record.reuse_fraction for record in group_records]
        failover_count = sum(1 for record in group_records if record.had_failover)
        row = {
            "policy": policy_name,
            "request_count": len(group_records),
            "mean_reusable_prefix": statistics.fmean(reusable_prefixes) if reusable_prefixes else 0.0,
            "mean_reuse_fraction": statistics.fmean(reuse_fractions) if reuse_fractions else 0.0,
            "ttft_p50": _percentile(ttfts, 0.50),
            "ttft_p95": _percentile(ttfts, 0.95),
            "latency_p50": _percentile(latencies, 0.50),
            "latency_p95": _percentile(latencies, 0.95),
            "failover_count": failover_count,
            "failover_rate": (failover_count / len(group_records)) if group_records else 0.0,
        }
        if group_field is not None:
            row[group_field] = group_value
        rows.append(row)
    return rows


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
