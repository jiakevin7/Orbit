from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Sequence

from .models import ExecutionRecord
from .router import RouterConfig


_FEATURE_NAMES = (
    "fixed_overhead",
    "prefill_cost_per_token",
    "decode_cost_per_token",
    "queue_depth_penalty",
    "stale_penalty_per_second",
    "uncertainty_penalty_per_token",
    "missing_summary_penalty",
)


@dataclass(frozen=True)
class RouterCalibration:
    source_policy: str
    record_count: int
    applied: bool
    reason: str | None
    coefficients: dict[str, float]
    mae: float
    rmse: float
    baseline_mae: float
    baseline_rmse: float
    base_config: dict[str, float | int | tuple[int, ...]]
    calibrated_config: dict[str, float | int | tuple[int, ...]]
    scope: str = "global"
    cluster_calibrations: dict[str, dict[str, object]] = field(default_factory=dict)
    applied_clusters: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClusterRouterCalibration:
    cluster_id: str
    record_count: int
    applied: bool
    reason: str | None
    coefficients: dict[str, float]
    mae: float
    rmse: float
    baseline_mae: float
    baseline_rmse: float


def fit_router_config(
    records: Sequence[ExecutionRecord],
    base_config: RouterConfig,
    source_policy: str = "summary",
    cluster_specific: bool = False,
) -> tuple[RouterConfig, RouterCalibration]:
    if cluster_specific:
        return _fit_cluster_specific_router_config(records, base_config, source_policy)
    return _fit_global_router_config(records, base_config, source_policy)


def _fit_global_router_config(
    records: Sequence[ExecutionRecord],
    base_config: RouterConfig,
    source_policy: str,
) -> tuple[RouterConfig, RouterCalibration]:
    if not records:
        payload = RouterCalibration(
            source_policy=source_policy,
            record_count=0,
            applied=False,
            reason="no_records",
            coefficients={name: float(getattr(base_config, name)) for name in _FEATURE_NAMES},
            mae=0.0,
            rmse=0.0,
            baseline_mae=0.0,
            baseline_rmse=0.0,
            base_config=asdict(base_config),
            calibrated_config=asdict(base_config),
        )
        return base_config, payload

    coefficients, baseline_mae, baseline_rmse, mae, rmse, applied, reason = _fit_coefficients(records, base_config)
    if not applied:
        payload = RouterCalibration(
            source_policy=source_policy,
            record_count=len(records),
            applied=False,
            reason=reason,
            coefficients={name: float(getattr(base_config, name)) for name in _FEATURE_NAMES},
            mae=baseline_mae,
            rmse=baseline_rmse,
            baseline_mae=baseline_mae,
            baseline_rmse=baseline_rmse,
            base_config=asdict(base_config),
            calibrated_config=asdict(base_config),
        )
        return base_config, payload

    calibrated_config = replace(
        base_config,
        fixed_overhead=coefficients["fixed_overhead"],
        prefill_cost_per_token=coefficients["prefill_cost_per_token"],
        decode_cost_per_token=coefficients["decode_cost_per_token"],
        queue_depth_penalty=coefficients["queue_depth_penalty"],
        stale_penalty_per_second=coefficients["stale_penalty_per_second"],
        uncertainty_penalty_per_token=coefficients["uncertainty_penalty_per_token"],
        missing_summary_penalty=coefficients["missing_summary_penalty"],
    )

    payload = RouterCalibration(
        source_policy=source_policy,
        record_count=len(records),
        applied=True,
        reason=None,
        coefficients=coefficients,
        mae=mae,
        rmse=rmse,
        baseline_mae=baseline_mae,
        baseline_rmse=baseline_rmse,
        base_config=asdict(base_config),
        calibrated_config=asdict(calibrated_config),
    )
    return calibrated_config, payload


def _fit_cluster_specific_router_config(
    records: Sequence[ExecutionRecord],
    base_config: RouterConfig,
    source_policy: str,
) -> tuple[RouterConfig, RouterCalibration]:
    if not records:
        payload = RouterCalibration(
            source_policy=source_policy,
            record_count=0,
            applied=False,
            reason="no_records",
            coefficients={name: float(getattr(base_config, name)) for name in _FEATURE_NAMES},
            mae=0.0,
            rmse=0.0,
            baseline_mae=0.0,
            baseline_rmse=0.0,
            base_config=asdict(base_config),
            calibrated_config=asdict(base_config),
            scope="per_cluster",
        )
        return base_config, payload

    grouped_records: dict[str, list[ExecutionRecord]] = {}
    for record in records:
        grouped_records.setdefault(record.cluster_id, []).append(record)

    cluster_payloads: dict[str, ClusterRouterCalibration] = {}
    cluster_overrides: dict[str, dict[str, float]] = {}
    for cluster_id, cluster_records in sorted(grouped_records.items()):
        coefficients, baseline_mae, baseline_rmse, mae, rmse, applied, reason = _fit_coefficients(cluster_records, base_config)
        cluster_payloads[cluster_id] = ClusterRouterCalibration(
            cluster_id=cluster_id,
            record_count=len(cluster_records),
            applied=applied,
            reason=reason,
            coefficients=coefficients if applied else {name: float(getattr(base_config, name)) for name in _FEATURE_NAMES},
            mae=mae if applied else baseline_mae,
            rmse=rmse if applied else baseline_rmse,
            baseline_mae=baseline_mae,
            baseline_rmse=baseline_rmse,
        )
        if applied:
            cluster_overrides[cluster_id] = coefficients

    baseline_predictions = [_predict_record(record, base_config) for record in records]
    fitted_config = replace(base_config, cluster_overrides=cluster_overrides)
    calibrated_predictions = [_predict_record(record, fitted_config) for record in records]
    targets = [_target_latency(record) for record in records]
    baseline_mae = _mae(targets, baseline_predictions)
    baseline_rmse = _rmse(targets, baseline_predictions)
    mae = _mae(targets, calibrated_predictions)
    rmse = _rmse(targets, calibrated_predictions)

    if not cluster_overrides:
        payload = RouterCalibration(
            source_policy=source_policy,
            record_count=len(records),
            applied=False,
            reason="no_cluster_improvement",
            coefficients={name: float(getattr(base_config, name)) for name in _FEATURE_NAMES},
            mae=baseline_mae,
            rmse=baseline_rmse,
            baseline_mae=baseline_mae,
            baseline_rmse=baseline_rmse,
            base_config=asdict(base_config),
            calibrated_config=asdict(base_config),
            scope="per_cluster",
            cluster_calibrations={cluster_id: asdict(payload) for cluster_id, payload in cluster_payloads.items()},
        )
        return base_config, payload

    if mae >= baseline_mae:
        payload = RouterCalibration(
            source_policy=source_policy,
            record_count=len(records),
            applied=False,
            reason="no_improvement",
            coefficients={name: float(getattr(base_config, name)) for name in _FEATURE_NAMES},
            mae=baseline_mae,
            rmse=baseline_rmse,
            baseline_mae=baseline_mae,
            baseline_rmse=baseline_rmse,
            base_config=asdict(base_config),
            calibrated_config=asdict(base_config),
            scope="per_cluster",
            cluster_calibrations={cluster_id: asdict(payload) for cluster_id, payload in cluster_payloads.items()},
        )
        return base_config, payload

    payload = RouterCalibration(
        source_policy=source_policy,
        record_count=len(records),
        applied=True,
        reason=None,
        coefficients={name: float(getattr(base_config, name)) for name in _FEATURE_NAMES},
        mae=mae,
        rmse=rmse,
        baseline_mae=baseline_mae,
        baseline_rmse=baseline_rmse,
        base_config=asdict(base_config),
        calibrated_config=asdict(fitted_config),
        scope="per_cluster",
        cluster_calibrations={cluster_id: asdict(payload) for cluster_id, payload in cluster_payloads.items()},
        applied_clusters=tuple(sorted(cluster_overrides)),
    )
    return fitted_config, payload


def _fit_coefficients(
    records: Sequence[ExecutionRecord],
    base_config: RouterConfig,
) -> tuple[dict[str, float], float, float, float, float, bool, str | None]:
    design_matrix: list[list[float]] = []
    targets: list[float] = []
    baseline_predictions: list[float] = []
    for record in records:
        design_matrix.append(
            [
                1.0,
                float(record.estimated_remaining_prefill_tokens),
                float(record.continuation_tokens),
                float(record.route_queue_depth),
                float(record.metadata_age),
                float(record.uncertainty_gap),
                1.0 if record.missing_summary else 0.0,
            ]
        )
        targets.append(_target_latency(record))
        baseline_predictions.append(_predict_record(record, base_config))

    baseline_mae = _mae(targets, baseline_predictions)
    baseline_rmse = _rmse(targets, baseline_predictions)

    minimum_records = max(8, len(_FEATURE_NAMES) + 1)
    if len(records) < minimum_records:
        return (
            {name: float(getattr(base_config, name)) for name in _FEATURE_NAMES},
            baseline_mae,
            baseline_rmse,
            baseline_mae,
            baseline_rmse,
            False,
            f"need_at_least_{minimum_records}_records",
        )

    coefficients = _solve_ridge(design_matrix, targets, ridge=1e-6)
    clipped_coefficients = [max(0.0, coefficient) for coefficient in coefficients]
    coefficient_map = {
        "fixed_overhead": clipped_coefficients[0],
        "prefill_cost_per_token": clipped_coefficients[1],
        "decode_cost_per_token": clipped_coefficients[2],
        "queue_depth_penalty": clipped_coefficients[3],
        "stale_penalty_per_second": clipped_coefficients[4],
        "uncertainty_penalty_per_token": clipped_coefficients[5],
        "missing_summary_penalty": clipped_coefficients[6],
    }
    fitted_predictions = [_predict_latency(row, clipped_coefficients) for row in design_matrix]
    mae = _mae(targets, fitted_predictions)
    rmse = _rmse(targets, fitted_predictions)
    if mae >= baseline_mae:
        return (
            {name: float(getattr(base_config, name)) for name in _FEATURE_NAMES},
            baseline_mae,
            baseline_rmse,
            baseline_mae,
            baseline_rmse,
            False,
            "no_improvement",
        )
    return coefficient_map, baseline_mae, baseline_rmse, mae, rmse, True, None


def _predict_latency(features: Sequence[float], coefficients: Sequence[float]) -> float:
    return max(0.0, sum(feature * coefficient for feature, coefficient in zip(features, coefficients)))


def _target_latency(record: ExecutionRecord) -> float:
    return max(0.0, record.actual_latency - record.network_cost)


def _predict_record(record: ExecutionRecord, config: RouterConfig) -> float:
    overrides = config.cluster_overrides.get(record.cluster_id, {})
    fixed_overhead = float(overrides.get("fixed_overhead", config.fixed_overhead))
    prefill_cost = float(overrides.get("prefill_cost_per_token", config.prefill_cost_per_token))
    decode_cost = float(overrides.get("decode_cost_per_token", config.decode_cost_per_token))
    queue_penalty = float(overrides.get("queue_depth_penalty", config.queue_depth_penalty))
    stale_penalty = float(overrides.get("stale_penalty_per_second", config.stale_penalty_per_second))
    uncertainty_penalty = float(overrides.get("uncertainty_penalty_per_token", config.uncertainty_penalty_per_token))
    missing_penalty = float(overrides.get("missing_summary_penalty", config.missing_summary_penalty))
    return max(
        0.0,
        fixed_overhead
        + record.estimated_remaining_prefill_tokens * prefill_cost
        + record.continuation_tokens * decode_cost
        + record.route_queue_depth * queue_penalty
        + record.metadata_age * stale_penalty
        + record.uncertainty_gap * uncertainty_penalty
        + (missing_penalty if record.missing_summary else 0.0),
    )


def _mae(targets: Sequence[float], predictions: Sequence[float]) -> float:
    if not targets:
        return 0.0
    return sum(abs(target - prediction) for target, prediction in zip(targets, predictions)) / len(targets)


def _rmse(targets: Sequence[float], predictions: Sequence[float]) -> float:
    if not targets:
        return 0.0
    squared_error = sum((target - prediction) ** 2 for target, prediction in zip(targets, predictions))
    return math.sqrt(squared_error / len(targets))


def _solve_ridge(
    features: Sequence[Sequence[float]],
    targets: Sequence[float],
    ridge: float,
) -> list[float]:
    feature_count = len(features[0])
    normal_matrix = [[0.0 for _ in range(feature_count)] for _ in range(feature_count)]
    normal_targets = [0.0 for _ in range(feature_count)]

    for row, target in zip(features, targets):
        for row_index in range(feature_count):
            normal_targets[row_index] += row[row_index] * target
            for col_index in range(feature_count):
                normal_matrix[row_index][col_index] += row[row_index] * row[col_index]

    for index in range(feature_count):
        normal_matrix[index][index] += ridge

    return _gaussian_elimination(normal_matrix, normal_targets)


def _gaussian_elimination(matrix: list[list[float]], values: list[float]) -> list[float]:
    size = len(values)
    augmented = [row[:] + [values[index]] for index, row in enumerate(matrix)]

    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda row_index: abs(augmented[row_index][pivot_index]))
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            continue
        if pivot_row != pivot_index:
            augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]

        pivot_value = augmented[pivot_index][pivot_index]
        for col_index in range(pivot_index, size + 1):
            augmented[pivot_index][col_index] /= pivot_value

        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            if factor == 0.0:
                continue
            for col_index in range(pivot_index, size + 1):
                augmented[row_index][col_index] -= factor * augmented[pivot_index][col_index]

    return [augmented[index][size] for index in range(size)]
