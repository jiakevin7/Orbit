import math
from dataclasses import asdict, dataclass, field, replace

# Calibration is intentionally global: the router can learn one reuse model and
# one latency model, but it cannot learn per-cluster service coefficients.
_FEATURE_NAMES = (
    "fixed_overhead",
    "prefill_cost_per_token",
    "decode_cost_per_token",
    "queue_depth_penalty",
    "stale_penalty_per_second",
    "uncertainty_penalty_per_token",
    "missing_summary_penalty",
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


def fit_router_config(records, base_config, source_policy="orbit"):
    return _fit_global_router_config(records, base_config, source_policy)


def _fit_global_router_config(records, base_config, source_policy):
    # Fit reuse first, then recompute remaining prefill tokens before fitting
    # the latency model that depends on that prefill estimate.
    if not records:
        payload = RouterCalibration(
            source_policy=source_policy,
            record_count=0,
            applied=False,
            reason="no_records",
            coefficients={
                name: float(getattr(base_config, name)) for name in _FEATURE_NAMES
            },
            ttft_coefficients={
                name: float(getattr(base_config, name)) for name in _FEATURE_NAMES
            },
            reuse_coefficients={
                name: float(getattr(base_config, name)) for name in _REUSE_FEATURE_NAMES
            },
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
        return (base_config, payload)

    # Stage 1: calibrate summary-derived reusable-token estimates against exact
    # cache observations from the warm-up run.
    (
        reuse_coefficients,
        reuse_baseline_mae,
        reuse_baseline_rmse,
        reuse_mae,
        reuse_rmse,
        reuse_applied,
        reuse_reason,
    ) = _fit_reuse_coefficients(records, base_config)
    reuse_config = base_config
    if reuse_applied:
        reuse_config = replace(
            base_config,
            reuse_intercept=reuse_coefficients["reuse_intercept"],
            reuse_estimate_scale=reuse_coefficients["reuse_estimate_scale"],
            reuse_match_level_bonus=reuse_coefficients["reuse_match_level_bonus"],
            reuse_hotset_match_bonus=reuse_coefficients["reuse_hotset_match_bonus"],
        )
    reuse_calibrated_records = [
        _apply_reuse_model_to_record(record, reuse_config) for record in records
    ]

    # Stage 2: fit latency after applying the reuse model so the prefill feature
    # reflects the tokens Orbit would actually expect to compute.
    coefficients, baseline_mae, baseline_rmse, mae, rmse, applied, reason = (
        _fit_coefficients(reuse_calibrated_records, reuse_config)
    )
    calibrated_config = reuse_config
    if applied:
        calibrated_config = replace(
            reuse_config,
            fixed_overhead=coefficients["fixed_overhead"],
            prefill_cost_per_token=coefficients["prefill_cost_per_token"],
            decode_cost_per_token=coefficients["decode_cost_per_token"],
            queue_depth_penalty=coefficients["queue_depth_penalty"],
            stale_penalty_per_second=coefficients["stale_penalty_per_second"],
            uncertainty_penalty_per_token=coefficients["uncertainty_penalty_per_token"],
            missing_summary_penalty=coefficients["missing_summary_penalty"],
        )
    ttft_baseline_predictions = [
        _predict_ttft_record(record, reuse_config)
        for record in reuse_calibrated_records
    ]
    ttft_calibrated_predictions = [
        _predict_ttft_record(record, calibrated_config)
        for record in reuse_calibrated_records
    ]
    ttft_targets = [_target_ttft(record) for record in reuse_calibrated_records]
    ttft_baseline_mae = _mae(ttft_targets, ttft_baseline_predictions)
    ttft_baseline_rmse = _rmse(ttft_targets, ttft_baseline_predictions)
    ttft_mae = _mae(ttft_targets, ttft_calibrated_predictions)
    ttft_rmse = _rmse(ttft_targets, ttft_calibrated_predictions)

    # If neither stage improved, return the original config but still report the
    # baseline error so the artifact explains why calibration was skipped.
    if not applied and (not reuse_applied):
        payload = RouterCalibration(
            source_policy=source_policy,
            record_count=len(records),
            applied=False,
            reason=reason or reuse_reason,
            coefficients={
                name: float(getattr(base_config, name)) for name in _FEATURE_NAMES
            },
            ttft_coefficients={
                name: float(getattr(base_config, name)) for name in _FEATURE_NAMES
            },
            reuse_coefficients={
                name: float(getattr(base_config, name)) for name in _REUSE_FEATURE_NAMES
            },
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
        return (base_config, payload)
    payload = RouterCalibration(
        source_policy=source_policy,
        record_count=len(records),
        applied=True,
        reason=None,
        coefficients=coefficients
        if applied
        else {name: float(getattr(reuse_config, name)) for name in _FEATURE_NAMES},
        ttft_coefficients=coefficients
        if applied
        else {name: float(getattr(reuse_config, name)) for name in _FEATURE_NAMES},
        reuse_coefficients=reuse_coefficients
        if reuse_applied
        else {name: float(getattr(base_config, name)) for name in _REUSE_FEATURE_NAMES},
        mae=mae if applied else baseline_mae,
        rmse=rmse if applied else baseline_rmse,
        baseline_mae=baseline_mae,
        baseline_rmse=baseline_rmse,
        ttft_mae=ttft_mae if applied else ttft_baseline_mae,
        ttft_rmse=ttft_rmse if applied else ttft_baseline_rmse,
        ttft_baseline_mae=ttft_baseline_mae,
        ttft_baseline_rmse=ttft_baseline_rmse,
        reuse_mae=reuse_mae if reuse_applied else reuse_baseline_mae,
        reuse_rmse=reuse_rmse if reuse_applied else reuse_baseline_rmse,
        reuse_baseline_mae=reuse_baseline_mae,
        reuse_baseline_rmse=reuse_baseline_rmse,
        base_config=asdict(base_config),
        calibrated_config=asdict(calibrated_config),
    )
    return (calibrated_config, payload)


def _fit_coefficients(records, base_config):
    # Use a small ridge term for numerical stability and reject coefficients
    # unless they improve warm-up prediction MAE over the current defaults.
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
    coefficients = _solve_ridge(design_matrix, targets, ridge=1e-06)
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
    fitted_predictions = [
        _predict_latency(row, clipped_coefficients) for row in design_matrix
    ]
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
    return (coefficient_map, baseline_mae, baseline_rmse, mae, rmse, True, None)


def _fit_reuse_coefficients(records, base_config):
    # Reuse calibration maps approximate summary matches to observed exact
    # reusable tokens collected from the warm-up run.
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
    coefficients = _solve_ridge(design_matrix, targets, ridge=1e-06)
    clipped_coefficients = [max(0.0, coefficient) for coefficient in coefficients]
    coefficient_map = {
        "reuse_intercept": clipped_coefficients[0],
        "reuse_estimate_scale": clipped_coefficients[1],
        "reuse_match_level_bonus": clipped_coefficients[2],
        "reuse_hotset_match_bonus": clipped_coefficients[3],
    }
    fitted_predictions = [
        _predict_reuse(row, clipped_coefficients) for row in design_matrix
    ]
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
    return (coefficient_map, baseline_mae, baseline_rmse, mae, rmse, True, None)


def _predict_latency(features, coefficients):
    return max(
        0.0,
        sum(
            (
                feature * coefficient
                for feature, coefficient in zip(features, coefficients)
            )
        ),
    )


def _predict_reuse(features, coefficients):
    return max(
        0.0,
        sum(
            (
                feature * coefficient
                for feature, coefficient in zip(features, coefficients)
            )
        ),
    )


def _target_latency(record):
    return max(0.0, record.actual_latency - record.network_cost)


def _target_ttft(record):
    return max(0.0, record.actual_ttft - record.network_cost)


def _latency_design_row(record):
    queue_depth = float(record.route_queue_depth)
    remaining_prefill = float(record.estimated_remaining_prefill_tokens)
    return [
        1.0,
        remaining_prefill,
        float(record.continuation_tokens),
        queue_depth,
        float(record.metadata_age),
        float(record.uncertainty_gap),
        1.0 if record.missing_summary else 0.0,
    ]


def _predict_record(record, config):
    fixed_overhead = float(config.fixed_overhead)
    prefill_cost = float(config.prefill_cost_per_token)
    decode_cost = float(config.decode_cost_per_token)
    queue_penalty = float(config.queue_depth_penalty)
    stale_penalty = float(config.stale_penalty_per_second)
    uncertainty_penalty = float(config.uncertainty_penalty_per_token)
    missing_penalty = float(config.missing_summary_penalty)
    queue_depth = float(record.route_queue_depth)
    remaining_prefill = float(record.estimated_remaining_prefill_tokens)
    return max(
        0.0,
        fixed_overhead
        + remaining_prefill * prefill_cost
        + record.continuation_tokens * decode_cost
        + queue_depth * queue_penalty
        + record.metadata_age * stale_penalty
        + record.uncertainty_gap * uncertainty_penalty
        + (missing_penalty if record.missing_summary else 0.0),
    )


def _predict_ttft_record(record, config):
    fixed_overhead = float(config.fixed_overhead)
    prefill_cost = float(config.prefill_cost_per_token)
    queue_penalty = float(config.queue_depth_penalty)
    stale_penalty = float(config.stale_penalty_per_second)
    uncertainty_penalty = float(config.uncertainty_penalty_per_token)
    missing_penalty = float(config.missing_summary_penalty)
    queue_depth = float(record.route_queue_depth)
    remaining_prefill = float(record.estimated_remaining_prefill_tokens)
    return max(
        0.0,
        fixed_overhead
        + remaining_prefill * prefill_cost
        + queue_depth * queue_penalty
        + record.metadata_age * stale_penalty
        + record.uncertainty_gap * uncertainty_penalty
        + (missing_penalty if record.missing_summary else 0.0),
    )


def _predict_reuse_record(record, config):
    coefficients = [
        float(config.reuse_intercept),
        float(config.reuse_estimate_scale),
        float(config.reuse_match_level_bonus),
        float(config.reuse_hotset_match_bonus),
    ]
    if (
        _raw_reuse_signal(record) <= 0
        and record.summary_matched_levels <= 0
        and (record.hotset_matched_levels <= 0)
    ):
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


def _apply_reuse_model_to_record(record, config):
    predicted_reuse = _predict_reuse_record(record, config)
    return replace(
        record,
        estimated_reusable_tokens=predicted_reuse,
        estimated_remaining_prefill_tokens=max(
            0, record.input_length - predicted_reuse
        ),
    )


def _raw_reuse_signal(record):
    if (
        record.raw_estimated_reusable_tokens > 0
        or record.summary_matched_levels > 0
        or record.hotset_matched_levels > 0
    ):
        return int(record.raw_estimated_reusable_tokens)
    return int(record.estimated_reusable_tokens)


def _mae(targets, predictions):
    if not targets:
        return 0.0
    return sum(
        (abs(target - prediction) for target, prediction in zip(targets, predictions))
    ) / len(targets)


def _rmse(targets, predictions):
    if not targets:
        return 0.0
    squared_error = sum(
        ((target - prediction) ** 2 for target, prediction in zip(targets, predictions))
    )
    return math.sqrt(squared_error / len(targets))


def _solve_ridge(features, targets, ridge):
    # Solve the normal equations directly to keep the project dependency-free.
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


def _gaussian_elimination(matrix, values):
    size = len(values)
    augmented = [row[:] + [values[index]] for index, row in enumerate(matrix)]
    for pivot_index in range(size):
        pivot_row = max(
            range(pivot_index, size),
            key=lambda row_index: abs(augmented[row_index][pivot_index]),
        )
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            continue
        if pivot_row != pivot_index:
            augmented[pivot_index], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[pivot_index],
            )
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
                augmented[row_index][col_index] -= (
                    factor * augmented[pivot_index][col_index]
                )
    return [augmented[index][size] for index in range(size)]
