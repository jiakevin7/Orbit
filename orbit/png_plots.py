from __future__ import annotations

import json
import os
from importlib.util import find_spec
from pathlib import Path
from typing import Mapping


POLICY_ORDER = (
    "orbit",
    "least_loaded",
    "random",
    "round_robin",
    "summary",
    "vllm_prefix_mock",
    "vllm_kv_mock",
    "load_only",
    "exact_prefix",
    "oracle",
)
POLICY_LABELS = {
    "orbit": "Orbit",
    "summary": "Orbit",
    "least_loaded": "Least Loaded",
    "load_only": "Least Loaded",
    "round_robin": "Round Robin",
    "vllm_prefix_mock": "vLLM Prefix-Style Mock",
    "vllm_kv_mock": "vLLM KV-Style Mock",
    "random": "Random",
    "exact_prefix": "Exact Prefix",
    "oracle": "Oracle",
}
POLICY_PALETTE = {
    "orbit": "#1d4ed8",
    "summary": "#1d4ed8",
    "least_loaded": "#ea580c",
    "round_robin": "#2563eb",
    "vllm_prefix_mock": "#0f766e",
    "vllm_kv_mock": "#7c3aed",
    "load_only": "#ea580c",
    "random": "#b91c1c",
    "exact_prefix": "#047857",
    "oracle": "#6d28d9",
}
POLICY_LINESTYLES = {
    "orbit": "-",
    "summary": "-",
    "least_loaded": "-.",
    "round_robin": (0, (2, 2)),
    "vllm_prefix_mock": (0, (6, 2)),
    "vllm_kv_mock": (0, (3, 2)),
    "load_only": "-.",
    "random": ":",
    "exact_prefix": "--",
    "oracle": (0, (5, 2)),
}

def generate_run_plots(run_dir: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(run_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"run directory not found: {root}")

    generated: list[Path] = []
    if recursive:
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name.startswith("seed-") and _is_run_directory(child):
                generated.extend(generate_run_plots(child, recursive=False))

    if _is_run_directory(root):
        generated.extend(_generate_single_run_plots(root))
    return generated


def plotting_available() -> bool:
    return all(find_spec(name) is not None for name in ("matplotlib", "pandas", "seaborn"))


def _generate_single_run_plots(run_dir: Path) -> list[Path]:
    records = _load_records(run_dir)
    summary_runs = _load_csv_rows(run_dir / "summary_runs.csv")
    traffic_rows = _load_csv_rows(run_dir / "summary_by_traffic.csv")
    if not records:
        return _generate_summary_only_plots(run_dir, summary_runs, traffic_rows)

    mpl_dir = run_dir / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    cache_dir = run_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "legend.frameon": False,
        },
    )

    dataframe = pd.DataFrame.from_records(records)
    if dataframe.empty:
        return []

    dataframe = _prepare_dataframe(dataframe)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _clear_existing_plots(plots_dir)
    created: list[Path] = []

    aggregate_mode = bool(summary_runs)
    created.append(
        _save_ecdf_plot(
            dataframe,
            x_field="actual_ttft",
            output_path=plots_dir / "ttft_cdf.png",
            title="TTFT ECDF by Routing Policy",
            x_label="TTFT (s)",
            subtitle="Pooled measured requests across seeds" if aggregate_mode else None,
        )
    )
    created.append(
        _save_ecdf_plot(
            dataframe,
            x_field="actual_latency",
            output_path=plots_dir / "latency_cdf.png",
            title="End-to-End Latency ECDF by Routing Policy",
            x_label="Latency (s)",
            subtitle="Pooled measured requests across seeds" if aggregate_mode else None,
        )
    )
    created.append(
        _save_policy_interval_plot(
            dataframe,
            summary_runs,
            value_field="actual_ttft",
            p50_field="ttft_p50",
            p95_field="ttft_p95",
            output_path=plots_dir / "ttft_by_policy.png",
            title="TTFT by Policy",
            x_label="TTFT (s)",
        )
    )
    created.append(
        _save_policy_interval_plot(
            dataframe,
            summary_runs,
            value_field="actual_latency",
            p50_field="latency_p50",
            p95_field="latency_p95",
            output_path=plots_dir / "latency_by_policy.png",
            title="Latency by Policy",
            x_label="Latency (s)",
        )
    )
    created.append(
        _save_tradeoff_plot(
            dataframe,
            summary_runs,
            output_path=plots_dir / "reuse_latency_tradeoff.png",
            title="Reuse vs Median Latency",
        )
    )
    if traffic_rows:
        created.append(
            _save_traffic_point_plot(
                traffic_rows,
                metric="latency_p50",
                output_path=plots_dir / "latency_by_traffic.png",
                title="Median Latency by Traffic Class",
                y_label="Latency p50 (s)",
            )
        )
        created.append(
            _save_traffic_point_plot(
                traffic_rows,
                metric="mean_reusable_prefix",
                output_path=plots_dir / "reuse_by_traffic.png",
                title="Reusable Prefix by Traffic Class",
                y_label="Mean reusable prefix (tokens)",
            )
        )

    plt.close("all")
    return [path for path in created if path.exists()]


def _generate_summary_only_plots(
    run_dir: Path,
    summary_runs: list[dict[str, str]],
    traffic_rows: list[dict[str, str]],
) -> list[Path]:
    if not summary_runs:
        return []

    mpl_dir = run_dir / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    cache_dir = run_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "legend.frameon": False,
        },
    )

    summary_frame = _prepare_dataframe(pd.DataFrame.from_records(summary_runs))
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _clear_existing_plots(plots_dir)
    created: list[Path] = []
    created.append(
        _save_policy_interval_plot(
            None,
            summary_runs,
            value_field="actual_ttft",
            p50_field="ttft_p50",
            p95_field="ttft_p95",
            output_path=plots_dir / "ttft_by_policy.png",
            title="TTFT by Policy",
            x_label="TTFT (s)",
        )
    )
    created.append(
        _save_policy_interval_plot(
            None,
            summary_runs,
            value_field="actual_latency",
            p50_field="latency_p50",
            p95_field="latency_p95",
            output_path=plots_dir / "latency_by_policy.png",
            title="Latency by Policy",
            x_label="Latency (s)",
        )
    )
    created.append(
        _save_tradeoff_plot(
            summary_frame,
            summary_runs,
            output_path=plots_dir / "reuse_latency_tradeoff.png",
            title="Reuse vs Median Latency",
        )
    )
    if traffic_rows:
        created.append(
            _save_traffic_point_plot(
                traffic_rows,
                metric="latency_p50",
                output_path=plots_dir / "latency_by_traffic.png",
                title="Median Latency by Traffic Class",
                y_label="Latency p50 (s)",
            )
        )
        created.append(
            _save_traffic_point_plot(
                traffic_rows,
                metric="mean_reusable_prefix",
                output_path=plots_dir / "reuse_by_traffic.png",
                title="Reusable Prefix by Traffic Class",
                y_label="Mean reusable prefix (tokens)",
            )
        )
    plt.close("all")
    return [path for path in created if path.exists()]


def _clear_existing_plots(plots_dir: Path) -> None:
    for path in plots_dir.glob("*.png"):
        path.unlink()


def _prepare_dataframe(dataframe):
    import pandas as pd

    normalized = dataframe.copy()
    if "policy" in normalized.columns:
        order = _policy_order(normalized)
        normalized["policy"] = pd.Categorical(normalized["policy"], categories=order, ordered=True)
    for column in (
        "actual_ttft",
        "actual_latency",
        "predicted_latency",
        "reuse_fraction",
        "failover_delay",
        "route_queue_depth",
    ):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def _load_records(run_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    direct_paths = sorted(run_dir.glob("*_records.json"))
    if direct_paths:
        for path in direct_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                records.extend(dict(row) for row in payload if isinstance(row, Mapping))
        return records

    for child in sorted(run_dir.iterdir()) if run_dir.exists() else []:
        if not child.is_dir() or not child.name.startswith("seed-"):
            continue
        for row in _load_records(child):
            normalized = dict(row)
            normalized.setdefault("seed", child.name)
            records.append(normalized)
    return records


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _save_ecdf_plot(
    dataframe,
    x_field: str,
    output_path: Path,
    title: str,
    x_label: str,
    subtitle: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    import seaborn as sns

    order = _policy_order(dataframe)
    palette = _policy_palette(order)
    figure, axis = plt.subplots(figsize=(10.5, 7.2), dpi=150)
    sns.ecdfplot(data=dataframe, x=x_field, hue="policy", hue_order=order, palette=palette, linewidth=2.5, ax=axis)
    axis.set_title(title)
    if subtitle:
        axis.text(0.0, 1.02, subtitle, transform=axis.transAxes, ha="left", va="bottom", fontsize=11, color="#4b5563")
    axis.set_xlabel(x_label)
    axis.set_ylabel("Cumulative probability")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlim(left=0.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.2)

    stats = _policy_quantiles(dataframe, x_field, order)
    for line, policy in zip(axis.lines, order):
        line.set_linestyle(POLICY_LINESTYLES.get(policy, "-"))
        if policy == "summary":
            line.set_linewidth(3.2)

    handles = [
        Line2D(
            [0],
            [0],
            color=palette[policy],
            linestyle=POLICY_LINESTYLES.get(policy, "-"),
            linewidth=3.2 if policy == "summary" else 2.5,
        )
        for policy in order
    ]
    labels = [
        f"{_display_name(policy)}  p50={stats[policy]['p50']:.3f}s  p95={stats[policy]['p95']:.3f}s"
        for policy in order
    ]
    axis.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.19), borderaxespad=0.0)
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _save_policy_interval_plot(
    dataframe,
    summary_runs: list[dict[str, str]],
    value_field: str,
    p50_field: str,
    p95_field: str,
    output_path: Path,
    title: str,
    x_label: str,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import pandas as pd

    if summary_runs:
        stats_frame = pd.DataFrame.from_records(summary_runs)
        stats_frame = _prepare_dataframe(stats_frame)
        order = _policy_order(stats_frame)
        stats = _summary_interval_stats(stats_frame, order, p50_field, p95_field)
    else:
        order = _policy_order(dataframe)
        stats = _policy_quantiles(dataframe, value_field, order)
        stats = {
            policy: {
                "p50_mean": values["p50"],
                "p95_mean": values["p95"],
                "p50_values": [values["p50"]],
                "p95_values": [values["p95"]],
            }
            for policy, values in stats.items()
        }

    figure, axis = plt.subplots(figsize=(10.2, 5.0), dpi=150)
    for index, policy in enumerate(order):
        color = POLICY_PALETTE.get(policy, "#1f6feb")
        p50 = stats[policy]["p50_mean"]
        p95 = stats[policy]["p95_mean"]
        p50_values = stats[policy]["p50_values"]
        p95_values = stats[policy]["p95_values"]
        if len(p50_values) > 1:
            offsets = _seed_offsets(len(p50_values))
            for offset, run_p50, run_p95 in zip(offsets, p50_values, p95_values):
                axis.hlines(index + offset, run_p50, run_p95, color=color, linewidth=1.2, alpha=0.28)
                axis.scatter([run_p50], [index + offset], color=color, s=26, marker="o", alpha=0.35, zorder=2)
                axis.scatter([run_p95], [index + offset], color=color, s=32, marker="D", alpha=0.35, zorder=2)
        axis.hlines(index, p50, p95, color=color, linewidth=3)
        axis.scatter([p50], [index], color=color, s=80, marker="o", zorder=3)
        axis.scatter([p95], [index], color=color, s=95, marker="D", zorder=3)

    axis.set_yticks(range(len(order)))
    axis.set_yticklabels([_display_name(policy) for policy in order])
    axis.set_xlabel(x_label)
    axis.set_title(f"{title} (p50 circle, p95 diamond)")
    axis.set_xlim(left=0.0)
    axis.grid(True, axis="x", alpha=0.2)
    axis.invert_yaxis()
    note = "Circle = p50, diamond = p95"
    if summary_runs:
        note += ", faint intervals = per-seed runs"
    axis.text(0.0, -0.16, note, transform=axis.transAxes, ha="left", va="top", fontsize=10, color="#4b5563")
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _save_tradeoff_plot(dataframe, summary_runs: list[dict[str, str]], output_path: Path, title: str) -> Path:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    if summary_runs:
        summary_frame = _prepare_dataframe(pd.DataFrame.from_records(summary_runs))
        summary_frame["reuse_metric"] = pd.to_numeric(summary_frame.get("mean_reusable_prefix"), errors="coerce")
        summary_frame["latency_metric"] = pd.to_numeric(summary_frame.get("latency_p50"), errors="coerce")
    else:
        reuse_field = next(
            (
                field
                for field in ("true_reusable_prefix", "actual_reusable_tokens", "estimated_reusable_tokens", "reuse_fraction")
                if field in dataframe.columns
            ),
            None,
        )
        if reuse_field is None:
            return output_path
        summary_frame = (
            dataframe.groupby("policy", observed=False)
            .agg(reuse_metric=(reuse_field, "mean"), latency_metric=("actual_latency", "median"))
            .reset_index()
        )
    summary_frame = summary_frame.dropna(subset=["reuse_metric", "latency_metric"])
    order = _policy_order(summary_frame)
    palette = _policy_palette(order)

    figure, axis = plt.subplots(figsize=(8.8, 6.8), dpi=150)
    if "seed" in summary_frame.columns:
        sns.scatterplot(
            data=summary_frame,
            x="reuse_metric",
            y="latency_metric",
            hue="policy",
            hue_order=order,
            palette=palette,
            s=60,
            alpha=0.35,
            legend=False,
            ax=axis,
        )
        means = (
            summary_frame.groupby("policy", observed=False)[["reuse_metric", "latency_metric"]]
            .mean()
            .reset_index()
        )
    else:
        means = summary_frame
    sns.scatterplot(data=means, x="reuse_metric", y="latency_metric", hue="policy", hue_order=order, palette=palette, s=170, marker="X", legend=False, ax=axis)
    for _, row in means.iterrows():
        axis.annotate(
            _display_name(str(row["policy"])),
            (float(row["reuse_metric"]), float(row["latency_metric"])),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=10,
        )
    axis.set_title(title)
    axis.set_xlabel("Mean reusable prefix (tokens)")
    axis.set_ylabel("Median latency (s)")
    axis.grid(True, alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _save_traffic_point_plot(
    traffic_rows: list[dict[str, str]],
    metric: str,
    output_path: Path,
    title: str,
    y_label: str,
) -> Path:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    frame = pd.DataFrame.from_records(traffic_rows)
    if metric not in frame.columns or "policy" not in frame.columns or "traffic_class" not in frame.columns:
        return output_path
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.dropna(subset=[metric])
    if frame.empty:
        return output_path
    order = _policy_order(frame)
    traffic_order = [
        name for name in ("sharegpt_chat", "rag", "agent", "bursty") if name in set(frame["traffic_class"].astype(str))
    ]
    traffic_order.extend(
        sorted(name for name in frame["traffic_class"].astype(str).unique() if name not in set(traffic_order))
    )
    frame["traffic_class"] = pd.Categorical(frame["traffic_class"], categories=traffic_order, ordered=True)

    figure, axis = plt.subplots(figsize=(10.5, 6.3), dpi=150)
    sns.lineplot(
        data=frame,
        x="traffic_class",
        y=metric,
        hue="policy",
        hue_order=order,
        style="policy",
        style_order=order,
        markers=True,
        dashes=True,
        sort=False,
        palette=_policy_palette(order),
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Traffic class")
    axis.set_ylabel(y_label)
    plt.setp(axis.get_xticklabels(), rotation=15, ha="right")
    axis.grid(True, axis="y", alpha=0.2)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    figure.tight_layout(rect=(0.0, 0.0, 0.8, 1.0))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _policy_order(dataframe) -> list[str]:
    present = [str(value) for value in dataframe["policy"].dropna().astype(str).unique()] if "policy" in dataframe.columns else []
    ordered = [policy for policy in POLICY_ORDER if policy in present]
    extras = sorted(policy for policy in present if policy not in POLICY_ORDER)
    return ordered + extras


def _policy_palette(order: list[str]) -> dict[str, str]:
    return {policy: POLICY_PALETTE.get(policy, "#4b5563") for policy in order}


def _policy_quantiles(dataframe, value_field: str, order: list[str]) -> dict[str, dict[str, float]]:
    quantiles: dict[str, dict[str, float]] = {}
    for policy in order:
        subset = dataframe.loc[dataframe["policy"] == policy, value_field].dropna()
        if subset.empty:
            quantiles[policy] = {"p50": 0.0, "p95": 0.0}
            continue
        quantiles[policy] = {
            "p50": float(subset.quantile(0.50)),
            "p95": float(subset.quantile(0.95)),
        }
    return quantiles


def _summary_interval_stats(summary_frame, order: list[str], p50_field: str, p95_field: str) -> dict[str, dict[str, object]]:
    stats: dict[str, dict[str, object]] = {}
    for policy in order:
        subset = summary_frame.loc[summary_frame["policy"] == policy]
        p50_values = [float(value) for value in subset[p50_field].dropna().tolist()] if p50_field in subset else []
        p95_values = [float(value) for value in subset[p95_field].dropna().tolist()] if p95_field in subset else []
        stats[policy] = {
            "p50_mean": float(sum(p50_values) / len(p50_values)) if p50_values else 0.0,
            "p95_mean": float(sum(p95_values) / len(p95_values)) if p95_values else 0.0,
            "p50_values": p50_values,
            "p95_values": p95_values,
        }
    return stats


def _seed_offsets(count: int) -> list[float]:
    if count <= 1:
        return [0.0]
    span = 0.18
    step = (span * 2) / (count - 1)
    return [(-span + index * step) for index in range(count)]


def _display_name(policy: str) -> str:
    return POLICY_LABELS.get(policy, policy.replace("_", " ").title())


def _is_run_directory(path: Path) -> bool:
    return any((path / name).exists() for name in ("manifest.json", "summary.json", "summary_aggregate.json"))
