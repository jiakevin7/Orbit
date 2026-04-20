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
    "queue_quadratic_penalty",
    "queue_prefill_interaction",
    "stale_penalty_per_second",
    "uncertainty_penalty_per_token",
    "missing_summary_penalty",
)
_TTFT_FEATURE_NAMES = (
    "ttft_fixed_overhead",
    "ttft_prefill_cost_per_token",
    "ttft_queue_depth_penalty",
    "ttft_queue_quadratic_penalty",
    "ttft_queue_prefill_interaction",
    "ttft_stale_penalty_per_second",
    "ttft_uncertainty_penalty_per_token",
    "ttft_missing_summary_penalty",
)
_REUSE_FEATURE_NAMES = (
    "reuse_intercept",
    "reuse_estimate_scale",
    "reuse_match_level_bonus",
    "reuse_hotset_match_bonus",
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
    ttft_coefficients: dict[str, float] = field(default_factory=dict)
    ttft_mae: float = 0.0
    ttft_rmse: float = 0.0
    ttft_baseline_mae: float = 0.0
    ttft_baseline_rmse: float = 0.0
    reuse_coefficients: dict[str, float] = field(default_factory=dict)
    reuse_mae: float = 0.0
    reuse_rmse: float = 0.0
    reuse_baseline_mae: float = 0.0
    reuse_baseline_rmse: float = 0.0
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
    ttft_coefficients: dict[str, float] = field(default_factory=dict)
    ttft_mae: float = 0.0
    ttft_rmse: float = 0.0
    ttft_baseline_mae: float = 0.0
    ttft_baseline_rmse: float = 0.0
    reuse_coefficients: dict[str, float] = field(default_factory=dict)
    reuse_mae: float = 0.0
    reuse_rmse: float = 0.0
    reuse_baseline_mae: float = 0.0
    reuse_baseline_rmse: float = 0.0


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
            ttft_coefficients={name: float(getattr(base_config, name)) for name in _TTFT_FEATURE_NAMES},
            reuse_coefficients={name: float(getattr(base_config, name)) for name in _REUSE_FEATURE_NAMES},
            mae=0.0,
            rmse=0.0,
            baseline_mae=0.0,
            baseline_rmse=0.0,
            ttft_mae=0.0,
            ttft_rmse=0.0,
            ttft_baseline_mae=0.0,
            ttft_baseline_rmse=0.0,
            reuse_mae=0.0,
            reuse_rmse=0.0,
            reuse_baseline_mae=0.0,
            reuse_baseline_rmse=0.0,
            base_config=asdict(base_config),
            calibrated_config=asdict(base_config),
        )
        return base_config, payload

    reuse_coefficients, reuse_baseline_mae, reuse_baseline_rmse, reuse_mae, reuse_rmse, reuse_applied, reuse_reason = _fit_reuse_coefficients(
        records,
        base_config,
    )
    reuse_config = base_config
    if reuse_applied:
        reuse_config = replace(
            base_config,
            reuse_intercept=reuse_coefficients["reuse_intercept"],
            reuse_estimate_scale=reuse_coefficients["reuse_estimate_scale"],
            reuse_match_level_bonus=reuse_coefficients["reuse_match_level_bonus"],
            reuse_hotset_match_bonus=reuse_coefficients["reuse_hotset_match_bonus"],
        )

    reuse_calibrated_records = [_apply_reuse_model_to_record(record, reuse_config) for record in records]
    coefficients, baseline_mae, baseline_rmse, mae, rmse, applied, reason = _fit_coefficients(
        reuse_calibrated_records,
        reuse_config,
    )
    ttft_coefficients, ttft_baseline_mae, ttft_baseline_rmse, ttft_mae, ttft_rmse, ttft_applied, ttft_reason = _fit_ttft_coefficients(
        reuse_calibrated_records,
        reuse_config,
    )
    if not applied and not reuse_applied and not ttft_applied:
        payload = RouterCalibration(
            source_policy=source_policy,
            record_count=len(records),
            applied=False,
            reason=reason or ttft_reason or reuse_reason,
            coefficients={name: float(getattr(base_config, name)) for name in _FEATURE_NAMES},
            ttft_coefficients={name: float(getattr(base_config, name)) for name in _TTFT_FEATURE_NAMES},
            reuse_coefficients={name: float(getattr(base_config, name)) for name in _REUSE_FEATURE_NAMES},
            mae=baseline_mae,
            rmse=baseline_rmse,
            baseline_mae=baseline_mae,
            baseline_rmse=baseline_rmse,
            ttft_mae=ttft_baseline_mae,
            ttft_rmse=ttft_baseline_rmse,
            ttft_baseline_mae=ttft_baseline_mae,
            ttft_baseline_rmse=ttft_baseline_rmse,
            reuse_mae=reuse_baseline_mae,
            reuse_rmse=reuse_baseline_rmse,
            reuse_baseline_mae=reuse_baseline_mae,
            reuse_baseline_rmse=reuse_baseline_rmse,
            base_config=asdict(base_config),
            calibrated_config=asdict(base_config),
        )
        return base_config, payload

    calibrated_config = reuse_config
    if applied or ttft_applied:
        calibrated_config = replace(
            reuse_config,
            fixed_overhead=coefficients["fixed_overhead"] if applied else reuse_config.fixed_overhead,
            prefill_cost_per_token=coefficients["prefill_cost_per_token"] if applied else reuse_config.prefill_cost_per_token,
            decode_cost_per_token=coefficients["decode_cost_per_token"] if applied else reuse_config.decode_cost_per_token,
            queue_depth_penalty=coefficients["queue_depth_penalty"] if applied else reuse_config.queue_depth_penalty,
            queue_quadratic_penalty=coefficients["queue_quadratic_penalty"] if applied else reuse_config.queue_quadratic_penalty,
            queue_prefill_interaction=coefficients["queue_prefill_interaction"] if applied else reuse_config.queue_prefill_interaction,
            stale_penalty_per_second=coefficients["stale_penalty_per_second"] if applied else reuse_config.stale_penalty_per_second,
            uncertainty_penalty_per_token=coefficients["uncertainty_penalty_per_token"] if applied else reuse_config.uncertainty_penalty_per_token,
            missing_summary_penalty=coefficients["missing_summary_penalty"] if applied else reuse_config.missing_summary_penalty,
            ttft_fixed_overhead=ttft_coefficients["ttft_fixed_overhead"] if ttft_applied else reuse_config.ttft_fixed_overhead,
            ttft_prefill_cost_per_token=ttft_coefficients["ttft_prefill_cost_per_token"] if ttft_applied else reuse_config.ttft_prefill_cost_per_token,
            ttft_queue_depth_penalty=ttft_coefficients["ttft_queue_depth_penalty"] if ttft_applied else reuse_config.ttft_queue_depth_penalty,
            ttft_queue_quadratic_penalty=ttft_coefficients["ttft_queue_quadratic_penalty"] if ttft_applied else reuse_config.ttft_queue_quadratic_penalty,
            ttft_queue_prefill_interaction=ttft_coefficients["ttft_queue_prefill_interaction"] if ttft_applied else reuse_config.ttft_queue_prefill_interaction,
            ttft_stale_penalty_per_second=ttft_coefficients["ttft_stale_penalty_per_second"] if ttft_applied else reuse_config.ttft_stale_penalty_per_second,
            ttft_uncertainty_penalty_per_token=ttft_coefficients["ttft_uncertainty_penalty_per_token"] if ttft_applied else reuse_config.ttft_uncertainty_penalty_per_token,
            ttft_missing_summary_penalty=ttft_coefficients["ttft_missing_summary_penalty"] if ttft_applied else reuse_config.ttft_missing_summary_penalty,
        )

    payload = RouterCalibration(
        source_policy=source_policy,
        record_count=len(records),
        applied=True,
        reason=None,
        coefficients=(
            coefficients
            if applied
            else {name: float(getattr(reuse_config, name)) for name in _FEATURE_NAMES}
        ),
        ttft_coefficients=(
            ttft_coefficients
            if ttft_applied
            else {name: float(getattr(reuse_config, name)) for name in _TTFT_FEATURE_NAMES}
        ),
        reuse_coefficients=(
            reuse_coefficients
            if reuse_applied
            else {name: float(getattr(base_config, name)) for name in _REUSE_FEATURE_NAMES}
        ),
        mae=mae if applied else baseline_mae,
        rmse=rmse if applied else baseline_rmse,
        baseline_mae=baseline_mae,
        baseline_rmse=baseline_rmse,
        ttft_mae=ttft_mae if ttft_applied else ttft_baseline_mae,
        ttft_rmse=ttft_rmse if ttft_applied else ttft_baseline_rmse,
        ttft_baseline_mae=ttft_baseline_mae,
        ttft_baseline_rmse=ttft_baseline_rmse,
        reuse_mae=reuse_mae if reuse_applied else reuse_baseline_mae,
        reuse_rmse=reuse_rmse if reuse_applied else reuse_baseline_rmse,
        reuse_baseline_mae=reuse_baseline_mae,
        reuse_baseline_rmse=reuse_baseline_rmse,
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
            ttft_coefficients={name: float(getattr(base_config, name)) for name in _TTFT_FEATURE_NAMES},
            reuse_coefficients={name: float(getattr(base_config, name)) for name in _REUSE_FEATURE_NAMES},
            mae=0.0,
            rmse=0.0,
            baseline_mae=0.0,
            baseline_rmse=0.0,
            ttft_mae=0.0,
            ttft_rmse=0.0,
            ttft_baseline_mae=0.0,
            ttft_baseline_rmse=0.0,
            reuse_mae=0.0,
            reuse_rmse=0.0,
            reuse_baseline_mae=0.0,
            reuse_baseline_rmse=0.0,
            base_config=asdict(base_config),
            calibrated_config=asdict(base_config),
            scope="per_cluster",
        )
        return base_config, payload

    reuse_coefficients, reuse_baseline_mae, reuse_baseline_rmse, reuse_mae, reuse_rmse, reuse_applied, reuse_reason = _fit_reuse_coefficients(
        records,
        base_config,
    )
    reuse_config = base_config
    if reuse_applied:
        reuse_config = replace(
            base_config,
            reuse_intercept=reuse_coefficients["reuse_intercept"],
            reuse_estimate_scale=reuse_coefficients["reuse_estimate_scale"],
            reuse_match_level_bonus=reuse_coefficients["reuse_match_level_bonus"],
            reuse_hotset_match_bonus=reuse_coefficients["reuse_hotset_match_bonus"],
        )
    reuse_calibrated_records = [_apply_reuse_model_to_record(record, reuse_config) for record in records]

    grouped_records: dict[str, list[ExecutionRecord]] = {}
    for record in reuse_calibrated_records:
        grouped_records.setdefault(record.cluster_id, []).append(record)

    cluster_payloads: dict[str, ClusterRouterCalibration] = {}
    cluster_overrides: dict[str, dict[str, float]] = {}
    for cluster_id, cluster_records in sorted(grouped_records.items()):
        coefficients, baseline_mae, baseline_rmse, mae, rmse, applied, reason = _fit_coefficients(
            cluster_records,
            reuse_config,
        )
        ttft_coefficients, ttft_baseline_mae, ttft_baseline_rmse, ttft_mae, ttft_rmse, ttft_applied, ttft_reason = _fit_ttft_coefficients(
            cluster_records,
            reuse_config,
        )
        cluster_payloads[cluster_id] = ClusterRouterCalibration(
            cluster_id=cluster_id,
            record_count=len(cluster_records),
            applied=(applied or ttft_applied or reuse_applied),
            reason=reason if applied else ttft_reason,
            coefficients=coefficients if applied else {name: float(getattr(base_config, name)) for name in _FEATURE_NAMES},
            ttft_coefficients=(
                ttft_coefficients
                if ttft_applied
                else {name: float(getattr(base_config, name)) for name in _TTFT_FEATURE_NAMES}
            ),
            mae=mae if applied else baseline_mae,
            rmse=rmse if applied else baseline_rmse,
            baseline_mae=baseline_mae,
            baseline_rmse=baseline_rmse,
            ttft_mae=ttft_mae if ttft_applied else ttft_baseline_mae,
            ttft_rmse=ttft_rmse if ttft_applied else ttft_baseline_rmse,
            ttft_baseline_mae=ttft_baseline_mae,
            ttft_baseline_rmse=ttft_baseline_rmse,
            reuse_coefficients=(
                reuse_coefficients
                if reuse_applied
                else {name: float(getattr(base_config, name)) for name in _REUSE_FEATURE_NAMES}
            ),
            reuse_mae=reuse_mae if reuse_applied else reuse_baseline_mae,
            reuse_rmse=reuse_rmse if reuse_applied else reuse_baseline_rmse,
            reuse_baseline_mae=reuse_baseline_mae,
            reuse_baseline_rmse=reuse_baseline_rmse,
        )
        override: dict[str, float] = {}
        if applied:
            override.update(coefficients)
        if ttft_applied:
            override.update(ttft_coefficients)
        if override:
            cluster_overrides[cluster_id] = override

    baseline_predictions = [_predict_record(record, reuse_config) for record in reuse_calibrated_records]
    fitted_config = replace(reuse_config, cluster_overrides=cluster_overrides)
    calibrated_predictions = [_predict_record(record, fitted_config) for record in reuse_calibrated_records]
    targets = [_target_latency(record) for record in reuse_calibrated_records]
    baseline_mae = _mae(targets, baseline_predictions)
    baseline_rmse = _rmse(targets, baseline_predictions)
    mae = _mae(targets, calibrated_predictions)
    rmse = _rmse(targets, calibrated_predictions)
    baseline_ttft_predictions = [_predict_ttft_record(record, reuse_config) for record in reuse_calibrated_records]
    calibrated_ttft_predictions = [_predict_ttft_record(record, fitted_config) for record in reuse_calibrated_records]
    ttft_targets = [_target_ttft(record) for record in reuse_calibrated_records]
    ttft_baseline_mae = _mae(ttft_targets, baseline_ttft_predictions)
    ttft_baseline_rmse = _rmse(ttft_targets, baseline_ttft_predictions)
    ttft_mae = _mae(ttft_targets, calibrated_ttft_predictions)
    ttft_rmse = _rmse(ttft_targets, calibrated_ttft_predictions)

    if not cluster_overrides and not reuse_applied:
        payload = RouterCalibration(
            source_policy=source_policy,
            record_count=len(records),
            applied=False,
            reason="no_cluster_improvement" if not reuse_reason else reuse_reason,
            coefficients={name: float(getattr(base_config, name)) for name in _FEATURE_NAMES},
            ttft_coefficients={name: float(getattr(base_config, name)) for name in _TTFT_FEATURE_NAMES},
            reuse_coefficients={name: float(getattr(base_config, name)) for name in _REUSE_FEATURE_NAMES},
            mae=baseline_mae,
            rmse=baseline_rmse,
            baseline_mae=baseline_mae,
            baseline_rmse=baseline_rmse,
            ttft_mae=ttft_baseline_mae,
            ttft_rmse=ttft_baseline_rmse,
            ttft_baseline_mae=ttft_baseline_mae,
            ttft_baseline_rmse=ttft_baseline_rmse,
            reuse_mae=reuse_baseline_mae,
            reuse_rmse=reuse_baseline_rmse,
            reuse_baseline_mae=reuse_baseline_mae,
            reuse_baseline_rmse=reuse_baseline_rmse,
            base_config=asdict(base_config),
            calibrated_config=asdict(base_config),
            scope="per_cluster",
            cluster_calibrations={cluster_id: asdict(payload) for cluster_id, payload in cluster_payloads.items()},
        )
        return base_config, payload

    if mae >= baseline_mae and ttft_mae >= ttft_baseline_mae and not reuse_applied:
        payload = RouterCalibration(
            source_policy=source_policy,
            record_count=len(records),
            applied=False,
            reason="no_improvement",
            coefficients={name: float(getattr(base_config, name)) for name in _FEATURE_NAMES},
            ttft_coefficients={name: float(getattr(base_config, name)) for name in _TTFT_FEATURE_NAMES},
            reuse_coefficients={name: float(getattr(base_config, name)) for name in _REUSE_FEATURE_NAMES},
            mae=baseline_mae,
            rmse=baseline_rmse,
            baseline_mae=baseline_mae,
            baseline_rmse=baseline_rmse,
            ttft_mae=ttft_baseline_mae,
            ttft_rmse=ttft_baseline_rmse,
            ttft_baseline_mae=ttft_baseline_mae,
            ttft_baseline_rmse=ttft_baseline_rmse,
            reuse_mae=reuse_baseline_mae,
            reuse_rmse=reuse_baseline_rmse,
            reuse_baseline_mae=reuse_baseline_mae,
            reuse_baseline_rmse=reuse_baseline_rmse,
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
        coefficients={name: float(getattr(fitted_config, name)) for name in _FEATURE_NAMES},
        ttft_coefficients={name: float(getattr(fitted_config, name)) for name in _TTFT_FEATURE_NAMES},
        reuse_coefficients=(
            reuse_coefficients
            if reuse_applied
            else {name: float(getattr(base_config, name)) for name in _REUSE_FEATURE_NAMES}
        ),
        mae=mae,
        rmse=rmse,
        baseline_mae=baseline_mae,
        baseline_rmse=baseline_rmse,
        ttft_mae=ttft_mae,
        ttft_rmse=ttft_rmse,
        ttft_baseline_mae=ttft_baseline_mae,
        ttft_baseline_rmse=ttft_baseline_rmse,
        reuse_mae=reuse_mae if reuse_applied else reuse_baseline_mae,
        reuse_rmse=reuse_rmse if reuse_applied else reuse_baseline_rmse,
        reuse_baseline_mae=reuse_baseline_mae,
        reuse_baseline_rmse=reuse_baseline_rmse,
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
        design_matrix.append(_latency_design_row(record))
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
        "queue_quadratic_penalty": clipped_coefficients[4],
        "queue_prefill_interaction": clipped_coefficients[5],
        "stale_penalty_per_second": clipped_coefficients[6],
        "uncertainty_penalty_per_token": clipped_coefficients[7],
        "missing_summary_penalty": clipped_coefficients[8],
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


def _fit_ttft_coefficients(
    records: Sequence[ExecutionRecord],
    base_config: RouterConfig,
) -> tuple[dict[str, float], float, float, float, float, bool, str | None]:
    design_matrix: list[list[float]] = []
    targets: list[float] = []
    baseline_predictions: list[float] = []
    for record in records:
        design_matrix.append(_ttft_design_row(record))
        targets.append(_target_ttft(record))
        baseline_predictions.append(_predict_ttft_record(record, base_config))

    baseline_mae = _mae(targets, baseline_predictions)
    baseline_rmse = _rmse(targets, baseline_predictions)

    minimum_records = max(8, len(_TTFT_FEATURE_NAMES) + 1)
    if len(records) < minimum_records:
        return (
            {name: float(getattr(base_config, name)) for name in _TTFT_FEATURE_NAMES},
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
        "ttft_fixed_overhead": clipped_coefficients[0],
        "ttft_prefill_cost_per_token": clipped_coefficients[1],
        "ttft_queue_depth_penalty": clipped_coefficients[2],
        "ttft_queue_quadratic_penalty": clipped_coefficients[3],
        "ttft_queue_prefill_interaction": clipped_coefficients[4],
        "ttft_stale_penalty_per_second": clipped_coefficients[5],
        "ttft_uncertainty_penalty_per_token": clipped_coefficients[6],
        "ttft_missing_summary_penalty": clipped_coefficients[7],
    }
    fitted_predictions = [_predict_latency(row, clipped_coefficients) for row in design_matrix]
    mae = _mae(targets, fitted_predictions)
    rmse = _rmse(targets, fitted_predictions)
    if mae >= baseline_mae:
        return (
            {name: float(getattr(base_config, name)) for name in _TTFT_FEATURE_NAMES},
            baseline_mae,
            baseline_rmse,
            baseline_mae,
            baseline_rmse,
            False,
            "no_improvement",
        )
    return coefficient_map, baseline_mae, baseline_rmse, mae, rmse, True, None


def _fit_reuse_coefficients(
    records: Sequence[ExecutionRecord],
    base_config: RouterConfig,
) -> tuple[dict[str, float], float, float, float, float, bool, str | None]:
    design_matrix: list[list[float]] = []
    targets: list[float] = []
    baseline_predictions: list[float] = []
    for record in records:
        raw_reuse = _raw_reuse_signal(record)
        design_matrix.append(
            [
                1.0,
                float(raw_reuse),
                float(record.summary_matched_levels),
                float(record.hotset_matched_levels),
            ]
        )
        targets.append(float(record.actual_reusable_tokens))
        baseline_predictions.append(float(record.estimated_reusable_tokens))

    baseline_mae = _mae(targets, baseline_predictions)
    baseline_rmse = _rmse(targets, baseline_predictions)

    minimum_records = max(8, len(_REUSE_FEATURE_NAMES) + 1)
    if len(records) < minimum_records:
        return (
            {name: float(getattr(base_config, name)) for name in _REUSE_FEATURE_NAMES},
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
        "reuse_intercept": clipped_coefficients[0],
        "reuse_estimate_scale": clipped_coefficients[1],
        "reuse_match_level_bonus": clipped_coefficients[2],
        "reuse_hotset_match_bonus": clipped_coefficients[3],
    }
    fitted_predictions = [_predict_reuse(row, clipped_coefficients) for row in design_matrix]
    mae = _mae(targets, fitted_predictions)
    rmse = _rmse(targets, fitted_predictions)
    if mae >= baseline_mae:
        return (
            {name: float(getattr(base_config, name)) for name in _REUSE_FEATURE_NAMES},
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


def _predict_reuse(features: Sequence[float], coefficients: Sequence[float]) -> float:
    return max(0.0, sum(feature * coefficient for feature, coefficient in zip(features, coefficients)))


def _target_latency(record: ExecutionRecord) -> float:
    return max(0.0, record.actual_latency - record.network_cost)


def _target_ttft(record: ExecutionRecord) -> float:
    return max(0.0, record.actual_ttft - record.network_cost)


def _latency_design_row(record: ExecutionRecord) -> list[float]:
    queue_depth = float(record.route_queue_depth)
    remaining_prefill = float(record.estimated_remaining_prefill_tokens)
    return [
        1.0,
        remaining_prefill,
        float(record.continuation_tokens),
        queue_depth,
        queue_depth ** 2,
        queue_depth * remaining_prefill,
        float(record.metadata_age),
        float(record.uncertainty_gap),
        1.0 if record.missing_summary else 0.0,
    ]


def _ttft_design_row(record: ExecutionRecord) -> list[float]:
    queue_depth = float(record.route_queue_depth)
    remaining_prefill = float(record.estimated_remaining_prefill_tokens)
    return [
        1.0,
        remaining_prefill,
        queue_depth,
        queue_depth ** 2,
        queue_depth * remaining_prefill,
        float(record.metadata_age),
        float(record.uncertainty_gap),
        1.0 if record.missing_summary else 0.0,
    ]


def _predict_record(record: ExecutionRecord, config: RouterConfig) -> float:
    overrides = config.cluster_overrides.get(record.cluster_id, {})
    fixed_overhead = float(overrides.get("fixed_overhead", config.fixed_overhead))
    prefill_cost = float(overrides.get("prefill_cost_per_token", config.prefill_cost_per_token))
    decode_cost = float(overrides.get("decode_cost_per_token", config.decode_cost_per_token))
    queue_penalty = float(overrides.get("queue_depth_penalty", config.queue_depth_penalty))
    queue_quadratic_penalty = float(overrides.get("queue_quadratic_penalty", config.queue_quadratic_penalty))
    queue_prefill_interaction = float(overrides.get("queue_prefill_interaction", config.queue_prefill_interaction))
    stale_penalty = float(overrides.get("stale_penalty_per_second", config.stale_penalty_per_second))
    uncertainty_penalty = float(overrides.get("uncertainty_penalty_per_token", config.uncertainty_penalty_per_token))
    missing_penalty = float(overrides.get("missing_summary_penalty", config.missing_summary_penalty))
    queue_depth = float(record.route_queue_depth)
    remaining_prefill = float(record.estimated_remaining_prefill_tokens)
    return max(
        0.0,
        fixed_overhead
        + remaining_prefill * prefill_cost
        + record.continuation_tokens * decode_cost
        + queue_depth * queue_penalty
        + (queue_depth ** 2) * queue_quadratic_penalty
        + (queue_depth * remaining_prefill) * queue_prefill_interaction
        + record.metadata_age * stale_penalty
        + record.uncertainty_gap * uncertainty_penalty
        + (missing_penalty if record.missing_summary else 0.0),
    )


def _predict_ttft_record(record: ExecutionRecord, config: RouterConfig) -> float:
    overrides = config.cluster_overrides.get(record.cluster_id, {})
    fixed_overhead = float(overrides.get("ttft_fixed_overhead", config.ttft_fixed_overhead))
    prefill_cost = float(overrides.get("ttft_prefill_cost_per_token", config.ttft_prefill_cost_per_token))
    queue_penalty = float(overrides.get("ttft_queue_depth_penalty", config.ttft_queue_depth_penalty))
    queue_quadratic_penalty = float(overrides.get("ttft_queue_quadratic_penalty", config.ttft_queue_quadratic_penalty))
    queue_prefill_interaction = float(
        overrides.get("ttft_queue_prefill_interaction", config.ttft_queue_prefill_interaction)
    )
    stale_penalty = float(overrides.get("ttft_stale_penalty_per_second", config.ttft_stale_penalty_per_second))
    uncertainty_penalty = float(
        overrides.get("ttft_uncertainty_penalty_per_token", config.ttft_uncertainty_penalty_per_token)
    )
    missing_penalty = float(overrides.get("ttft_missing_summary_penalty", config.ttft_missing_summary_penalty))
    queue_depth = float(record.route_queue_depth)
    remaining_prefill = float(record.estimated_remaining_prefill_tokens)
    return max(
        0.0,
        fixed_overhead
        + remaining_prefill * prefill_cost
        + queue_depth * queue_penalty
        + (queue_depth ** 2) * queue_quadratic_penalty
        + (queue_depth * remaining_prefill) * queue_prefill_interaction
        + record.metadata_age * stale_penalty
        + record.uncertainty_gap * uncertainty_penalty
        + (missing_penalty if record.missing_summary else 0.0),
    )


def _predict_reuse_record(record: ExecutionRecord, config: RouterConfig) -> int:
    overrides = config.cluster_overrides.get(record.cluster_id, {})
    coefficients = [
        float(overrides.get("reuse_intercept", config.reuse_intercept)),
        float(overrides.get("reuse_estimate_scale", config.reuse_estimate_scale)),
        float(overrides.get("reuse_match_level_bonus", config.reuse_match_level_bonus)),
        float(overrides.get("reuse_hotset_match_bonus", config.reuse_hotset_match_bonus)),
    ]
    if _raw_reuse_signal(record) <= 0 and record.summary_matched_levels <= 0 and record.hotset_matched_levels <= 0:
        return 0
    predicted = _predict_reuse(
        [
            1.0,
            float(_raw_reuse_signal(record)),
            float(record.summary_matched_levels),
            float(record.hotset_matched_levels),
        ],
        coefficients,
    )
    return max(0, min(record.input_length, int(round(predicted))))


def _apply_reuse_model_to_record(record: ExecutionRecord, config: RouterConfig) -> ExecutionRecord:
    predicted_reuse = _predict_reuse_record(record, config)
    return replace(
        record,
        estimated_reusable_tokens=predicted_reuse,
        estimated_remaining_prefill_tokens=max(0, record.input_length - predicted_reuse),
    )


def _raw_reuse_signal(record: ExecutionRecord) -> int:
    if record.raw_estimated_reusable_tokens > 0 or record.summary_matched_levels > 0 or record.hotset_matched_levels > 0:
        return int(record.raw_estimated_reusable_tokens)
    return int(record.estimated_reusable_tokens)


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
