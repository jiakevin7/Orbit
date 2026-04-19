from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "external_benchmark_matrix.json"


@dataclass(frozen=True)
class MatrixScenario:
    name: str
    description: str
    workload_kind: str
    traffic_mix_chat: float
    traffic_mix_rag: float
    traffic_mix_agent: float
    traffic_mix_bursty: float
    required_paths: tuple[str, ...] = ()


def load_external_benchmark_matrix(path: Path | None = None) -> tuple[str, str, tuple[MatrixScenario, ...]]:
    resolved = path or CONFIG_PATH
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    scenarios = tuple(
        MatrixScenario(
            name=str(item["name"]),
            description=str(item["description"]),
            workload_kind=str(item.get("workload_kind", "mixed_realistic")),
            traffic_mix_chat=float(item.get("traffic_mix_chat", 0.0)),
            traffic_mix_rag=float(item.get("traffic_mix_rag", 0.0)),
            traffic_mix_agent=float(item.get("traffic_mix_agent", 0.0)),
            traffic_mix_bursty=float(item.get("traffic_mix_bursty", 0.0)),
            required_paths=tuple(str(path_name) for path_name in item.get("required_paths", [])),
        )
        for item in payload.get("scenarios", [])
    )
    return str(payload.get("name", "external_standard_matrix")), str(payload.get("description", "")), scenarios


def scenario_source_resolution(
    scenario: MatrixScenario,
    sharegpt_path: str | None,
    rag_path: str | None,
    agent_path: str | None,
) -> dict[str, str]:
    provided = {
        "sharegpt_path": sharegpt_path,
        "rag_path": rag_path,
        "agent_path": agent_path,
    }
    resolution: dict[str, str] = {}
    for field_name, path_value in provided.items():
        resolution[field_name] = "external" if path_value else "fallback"
    for field_name in scenario.required_paths:
        resolution.setdefault(field_name, "fallback")
    return resolution


def collect_matrix_summary_rows(output_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        aggregate_path = scenario_dir / "summary_aggregate.json"
        if aggregate_path.exists():
            payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                for row in payload:
                    if isinstance(row, dict):
                        rows.append({"scenario": scenario_dir.name, **row})
            continue

        summary_path = scenario_dir / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        for policy_name, metrics in sorted(payload.items()):
            if isinstance(metrics, dict):
                rows.append({"scenario": scenario_dir.name, "policy": policy_name, **metrics})
    return rows


def matrix_manifest(
    matrix_name: str,
    description: str,
    scenarios: Sequence[MatrixScenario],
    *,
    backend: str,
    control_plane_mode: str,
    router_count: int,
    cluster_count: int,
    topology_mode: str,
    reachable_clusters_per_router: int | None,
    seeds: Sequence[int],
    measured_requests: int,
    warmup_requests: int,
    validation_requests: int,
    sharegpt_path: str | None,
    rag_path: str | None,
    agent_path: str | None,
) -> dict[str, object]:
    return {
        "matrix_name": matrix_name,
        "description": description,
        "backend": backend,
        "control_plane_mode": control_plane_mode,
        "router_count": router_count,
        "cluster_count": cluster_count,
        "topology_mode": topology_mode,
        "reachable_clusters_per_router": reachable_clusters_per_router,
        "seeds": list(seeds),
        "measured_requests": measured_requests,
        "warmup_requests": warmup_requests,
        "validation_requests": validation_requests,
        "datasets": {
            "sharegpt_path": sharegpt_path,
            "rag_path": rag_path,
            "agent_path": agent_path,
        },
        "scenarios": [
            {
                "name": scenario.name,
                "description": scenario.description,
                "workload_kind": scenario.workload_kind,
                "traffic_mix_chat": scenario.traffic_mix_chat,
                "traffic_mix_rag": scenario.traffic_mix_rag,
                "traffic_mix_agent": scenario.traffic_mix_agent,
                "traffic_mix_bursty": scenario.traffic_mix_bursty,
                "required_paths": list(scenario.required_paths),
                "source_resolution": scenario_source_resolution(
                    scenario,
                    sharegpt_path=sharegpt_path,
                    rag_path=rag_path,
                    agent_path=agent_path,
                ),
            }
            for scenario in scenarios
        ],
    }
