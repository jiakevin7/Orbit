import json
import os
from importlib.util import find_spec
from pathlib import Path

POLICY_ORDER = (
    "orbit",
    "least_loaded",
    "random",
    "round_robin",
)
POLICY_LABELS = {
    "orbit": "Orbit",
    "least_loaded": "Least-Loaded",
    "round_robin": "Round-Robin",
    "random": "Random",
}
POLICY_ALIASES = {
    # Compatibility for older result artifacts. These are display aliases only;
    # runtime policy registration remains limited to POLICY_ORDER.
    "summary": "orbit",
    "load_only": "least_loaded",
}
ORBIT_COLOR = "#1F4E79"
POLICY_PALETTE = {
    "orbit": ORBIT_COLOR,
    "least_loaded": "#9A9A9A",
    "random": "#6E6E6E",
    "round_robin": "#3F3F3F",
}
POLICY_LINESTYLES = {
    "orbit": "-",
    "least_loaded": "-.",
    "round_robin": (0, (2, 2)),
    "random": ":",
}
TRAFFIC_LABELS = {
    "agent": "Agent / Tool",
    "rag": "RAG",
    "sharegpt_chat": "Chat",
    "bursty": "Bursty",
}
TRAFFIC_ORDER = ("agent", "rag", "sharegpt_chat", "bursty")
FOOTER_TEMPLATE = "{workload} · {nseeds} seeds × {nreq} requests · error bars = 95% CI"


def generate_run_plots(run_dir, recursive=True):
    # A top-level benchmark directory contains seed subdirectories; recurse so
    # one command refreshes both per-seed and aggregate figures.
    root = Path(run_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"run directory not found: {root}")
    generated: list[Path] = []
    if recursive:
        for child in sorted(root.iterdir()):
            if (
                child.is_dir()
                and child.name.startswith("seed-")
                and _is_run_directory(child)
            ):
                generated.extend(generate_run_plots(child, recursive=False))
    if _is_run_directory(root):
        generated.extend(_generate_single_run_plots(root))
    return generated


def plotting_available():
    return all(
        (find_spec(name) is not None for name in ("matplotlib", "pandas", "seaborn"))
    )


def _generate_single_run_plots(run_dir):
    # Prefer per-request traces for ECDFs; fall back to aggregate CSVs when a
    # run directory only has summary artifacts.
    records = _load_records(run_dir)
    summary_runs = _load_csv_rows(run_dir / "summary_runs.csv")
    traffic_rows = _load_traffic_rows(run_dir)
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

    _setup_plot_style(matplotlib, sns)
    dataframe = pd.DataFrame.from_records(records)
    if dataframe.empty:
        return []
    dataframe = _prepare_dataframe(dataframe)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _clear_existing_plots(plots_dir)
    created: list[Path] = []
    aggregate_mode = bool(summary_runs)
    run_frame = _build_run_frame(summary_runs, dataframe)
    traffic_frame = _build_traffic_frame(traffic_rows, dataframe)
    footer_text = _footer_text(run_frame)
    created.append(
        _save_ecdf_plot(
            dataframe,
            x_field="actual_ttft",
            output_path=plots_dir / "ttft_cdf.png",
            title="TTFT ECDF by Routing Policy",
            x_label="TTFT (s)",
            subtitle="Pooled measured requests across seeds"
            if aggregate_mode
            else None,
        )
    )
    created.append(
        _save_ecdf_plot(
            dataframe,
            x_field="actual_latency",
            output_path=plots_dir / "latency_cdf.png",
            title="End-to-End Latency ECDF by Routing Policy",
            x_label="Latency (s)",
            subtitle="Pooled measured requests across seeds"
            if aggregate_mode
            else None,
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
    created.extend(
        _save_orbit_impact_plots(
            run_frame=run_frame,
            traffic_frame=traffic_frame,
            plots_dir=plots_dir,
            footer_text=footer_text,
        )
    )
    plt.close("all")
    return [path for path in created if path.exists()]


def _generate_summary_only_plots(run_dir, summary_runs, traffic_rows):
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

    _setup_plot_style(matplotlib, sns)
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
    run_frame = _build_run_frame(summary_runs, summary_frame)
    traffic_frame = _build_traffic_frame(traffic_rows, summary_frame)
    created.extend(
        _save_orbit_impact_plots(
            run_frame=run_frame,
            traffic_frame=traffic_frame,
            plots_dir=plots_dir,
            footer_text=_footer_text(run_frame),
        )
    )
    plt.close("all")
    return [path for path in created if path.exists()]


def _clear_existing_plots(plots_dir):
    for path in plots_dir.glob("*.png"):
        path.unlink()


def _setup_plot_style(matplotlib, sns):
    sns.set_style(
        "whitegrid",
        {
            "grid.color": "#B8B8B8",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "axes.facecolor": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 20,
            "axes.titlesize": 24,
            "axes.titleweight": "bold",
            "axes.labelsize": 22,
            "axes.labelweight": "bold",
            "axes.linewidth": 1.6,
            "axes.edgecolor": "#222222",
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "xtick.major.width": 1.4,
            "ytick.major.width": 1.4,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "legend.fontsize": 18,
            "legend.title_fontsize": 19,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.facecolor": "white",
            "legend.edgecolor": "#222222",
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.transparent": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _prepare_dataframe(dataframe):
    import pandas as pd

    normalized = dataframe.copy()
    if "policy" in normalized.columns:
        normalized["policy"] = normalized["policy"].map(_canonical_policy)
        normalized = normalized.loc[normalized["policy"].isin(POLICY_ORDER)].copy()
        order = _policy_order(normalized)
        normalized["policy"] = pd.Categorical(
            normalized["policy"], categories=order, ordered=True
        )
    for column in (
        "actual_ttft",
        "actual_latency",
        "predicted_latency",
        "reuse_fraction",
        "actual_reusable_tokens",
        "mean_reusable_prefix",
        "mean_reuse_fraction",
        "ttft_p50",
        "ttft_p95",
        "latency_p50",
        "latency_p95",
        "request_count",
        "failover_delay",
        "route_queue_depth",
    ):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if "traffic_class" in normalized.columns:
        normalized["traffic_class"] = normalized["traffic_class"].astype(str)
    return normalized


def _load_records(run_dir):
    records: list[dict[str, object]] = []
    direct_paths = sorted(run_dir.glob("*_records.json"))
    if direct_paths:
        for path in direct_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                records.extend((dict(row) for row in payload if isinstance(row, dict)))
        return records
    for child in sorted(run_dir.iterdir()) if run_dir.exists() else []:
        if not child.is_dir() or not child.name.startswith("seed-"):
            continue
        for row in _load_records(child):
            normalized = dict(row)
            normalized.setdefault("seed", child.name)
            records.append(normalized)
    return records


def _load_csv_rows(path):
    import csv

    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_traffic_rows(run_dir):
    for name in (
        "summary_by_traffic_runs.csv",
        "summary_by_traffic.csv",
        "summary_by_traffic_aggregate.csv",
    ):
        rows = _load_csv_rows(run_dir / name)
        if rows:
            return rows
    return []


def _build_run_frame(summary_runs, dataframe):
    import pandas as pd

    if summary_runs:
        frame = _prepare_dataframe(pd.DataFrame.from_records(summary_runs))
    elif dataframe is None or dataframe.empty or "policy" not in dataframe.columns:
        return pd.DataFrame()
    else:
        group_fields = ["policy"]
        if "seed" in dataframe.columns:
            group_fields.insert(0, "seed")
        rows = []
        for keys, group in dataframe.groupby(group_fields, observed=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_fields, keys))
            row["request_count"] = int(len(group))
            if "actual_reusable_tokens" in group.columns:
                row["mean_reusable_prefix"] = float(
                    group["actual_reusable_tokens"].mean()
                )
            if "reuse_fraction" in group.columns:
                row["mean_reuse_fraction"] = float(group["reuse_fraction"].mean())
            if "actual_ttft" in group.columns:
                row["ttft_p50"] = float(group["actual_ttft"].quantile(0.5))
                row["ttft_p95"] = float(group["actual_ttft"].quantile(0.95))
            if "actual_latency" in group.columns:
                row["latency_p50"] = float(group["actual_latency"].quantile(0.5))
                row["latency_p95"] = float(group["actual_latency"].quantile(0.95))
            rows.append(row)
        frame = _prepare_dataframe(pd.DataFrame.from_records(rows))
    return _with_policy_labels(frame)


def _build_traffic_frame(traffic_rows, dataframe):
    import pandas as pd

    if traffic_rows:
        frame = _prepare_dataframe(pd.DataFrame.from_records(traffic_rows))
    elif (
        dataframe is not None
        and not dataframe.empty
        and {"policy", "traffic_class"}.issubset(dataframe.columns)
    ):
        group_fields = ["policy", "traffic_class"]
        if "seed" in dataframe.columns:
            group_fields.insert(0, "seed")
        rows = []
        for keys, group in dataframe.groupby(group_fields, observed=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_fields, keys))
            row["request_count"] = int(len(group))
            if "actual_reusable_tokens" in group.columns:
                row["mean_reusable_prefix"] = float(
                    group["actual_reusable_tokens"].mean()
                )
            if "reuse_fraction" in group.columns:
                row["mean_reuse_fraction"] = float(group["reuse_fraction"].mean())
            if "actual_ttft" in group.columns:
                row["ttft_p50"] = float(group["actual_ttft"].quantile(0.5))
                row["ttft_p95"] = float(group["actual_ttft"].quantile(0.95))
            if "actual_latency" in group.columns:
                row["latency_p50"] = float(group["actual_latency"].quantile(0.5))
                row["latency_p95"] = float(group["actual_latency"].quantile(0.95))
            rows.append(row)
        frame = _prepare_dataframe(pd.DataFrame.from_records(rows))
    else:
        return pd.DataFrame()
    return _with_traffic_labels(_with_policy_labels(frame))


def _with_policy_labels(frame):
    if frame.empty or "policy" not in frame.columns:
        return frame
    labeled = frame.copy()
    labeled["policy_label"] = labeled["policy"].astype(str).map(_display_name)
    return labeled


def _with_traffic_labels(frame):
    import pandas as pd

    if frame.empty or "traffic_class" not in frame.columns:
        return frame
    labeled = frame.copy()
    traffic_order = _traffic_order(labeled)
    labeled["traffic_class"] = pd.Categorical(
        labeled["traffic_class"], categories=traffic_order, ordered=True
    )
    labeled["traffic_class_label"] = (
        labeled["traffic_class"].astype(str).map(_traffic_display_name)
    )
    label_order = [_traffic_display_name(name) for name in traffic_order]
    labeled["traffic_class_label"] = pd.Categorical(
        labeled["traffic_class_label"], categories=label_order, ordered=True
    )
    return labeled


def _footer_text(run_frame):
    if run_frame is None or run_frame.empty:
        return FOOTER_TEMPLATE.format(
            workload="Mixed realistic workload", nseeds=1, nreq="unknown"
        )
    nseeds = (
        int(run_frame["seed"].nunique())
        if "seed" in run_frame.columns and run_frame["seed"].notna().any()
        else 1
    )
    nreq = "unknown"
    if (
        "request_count" in run_frame.columns
        and run_frame["request_count"].notna().any()
    ):
        nreq = int(round(float(run_frame["request_count"].dropna().median())))
    return FOOTER_TEMPLATE.format(
        workload="Mixed realistic workload",
        nseeds=nseeds,
        nreq=nreq,
    )


def _save_figure(figure, output_path):
    figure.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    return output_path


def _save_orbit_impact_plots(run_frame, traffic_frame, plots_dir, footer_text):
    if run_frame is None or run_frame.empty or "policy" not in run_frame.columns:
        return []
    import matplotlib.pyplot as plt

    created: list[Path] = []
    created.append(
        _save_poster_ttft_p50(
            run_frame, plots_dir / "orbit_01_ttft_p50.png", footer_text
        )
    )
    created.append(
        _save_poster_reuse_fraction(
            run_frame, plots_dir / "orbit_02_reuse_fraction.png", footer_text
        )
    )
    if traffic_frame is not None and not traffic_frame.empty:
        created.append(
            _save_poster_prefix_by_traffic(
                traffic_frame,
                plots_dir / "orbit_03_reusable_prefix_by_traffic.png",
                footer_text,
            )
        )
    created.append(
        _save_poster_latency_vs_reuse(
            run_frame, plots_dir / "orbit_04_latency_vs_reuse.png", footer_text
        )
    )
    if traffic_frame is not None and not traffic_frame.empty:
        created.append(
            _save_poster_combined(
                run_frame,
                traffic_frame,
                plots_dir / "orbit_combined.png",
                footer_text,
            )
        )
    plt.close("all")
    return [path for path in created if path.exists()]


def _save_poster_ttft_p50(run_frame, output_path, footer_text, axis=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if "ttft_p50" not in run_frame.columns:
        return output_path
    own_figure = axis is None
    if own_figure:
        figure, axis = plt.subplots(figsize=(11, 7.5), dpi=300)
    frame = _with_policy_labels(run_frame.dropna(subset=["ttft_p50"]))
    order = _policy_order(frame)
    label_order = [_display_name(policy) for policy in order]
    sns.barplot(
        data=frame,
        x="policy_label",
        y="ttft_p50",
        hue="policy_label",
        order=label_order,
        hue_order=label_order,
        palette=_label_palette(order),
        errorbar=("ci", 95),
        capsize=0.18,
        err_kws={"linewidth": 2.2, "color": "#333333"},
        edgecolor="#222222",
        linewidth=1.4,
        legend=False,
        ax=axis,
    )
    sns.stripplot(
        data=frame,
        x="policy_label",
        y="ttft_p50",
        order=label_order,
        color="#111111",
        size=8,
        alpha=0.75,
        jitter=0.16,
        ax=axis,
    )
    _annotate_bars(axis, fmt="{:.2f}s")
    _style_policy_axis(axis)
    axis.set_title("Time-to-First-Token (p50)", pad=14)
    axis.set_ylabel("TTFT p50  (seconds)")
    axis.set_xlabel("")
    axis.set_ylim(top=axis.get_ylim()[1] * 1.15)
    _add_ttft_callout(axis, frame)
    axis.text(
        0.0, -0.18, footer_text, transform=axis.transAxes, fontsize=14, color="#555555"
    )
    if own_figure:
        figure.tight_layout()
        _save_figure(figure, output_path)
        plt.close(figure)
    return output_path


def _save_poster_reuse_fraction(run_frame, output_path, footer_text, axis=None):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import seaborn as sns

    if "mean_reuse_fraction" not in run_frame.columns:
        return output_path
    own_figure = axis is None
    if own_figure:
        figure, axis = plt.subplots(figsize=(11, 7.5), dpi=300)
    frame = _with_policy_labels(run_frame.dropna(subset=["mean_reuse_fraction"]))
    order = _policy_order(frame)
    label_order = [_display_name(policy) for policy in order]
    sns.barplot(
        data=frame,
        x="policy_label",
        y="mean_reuse_fraction",
        hue="policy_label",
        order=label_order,
        hue_order=label_order,
        palette=_label_palette(order),
        errorbar=("ci", 95),
        capsize=0.18,
        err_kws={"linewidth": 2.2, "color": "#333333"},
        edgecolor="#222222",
        linewidth=1.4,
        legend=False,
        ax=axis,
    )
    sns.stripplot(
        data=frame,
        x="policy_label",
        y="mean_reuse_fraction",
        order=label_order,
        color="#111111",
        size=8,
        alpha=0.75,
        jitter=0.16,
        ax=axis,
    )
    _annotate_bars(axis, fmt="{:.0%}")
    _style_policy_axis(axis)
    axis.set_title("KV-Cache Reuse Fraction", pad=14)
    axis.set_ylabel("Mean reuse fraction")
    axis.set_xlabel("")
    axis.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    axis.set_ylim(top=axis.get_ylim()[1] * 1.15)
    _add_reuse_callout(axis, frame)
    axis.text(
        0.0, -0.18, footer_text, transform=axis.transAxes, fontsize=14, color="#555555"
    )
    if own_figure:
        figure.tight_layout()
        _save_figure(figure, output_path)
        plt.close(figure)
    return output_path


def _save_poster_prefix_by_traffic(traffic_frame, output_path, footer_text, axis=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not {"traffic_class_label", "mean_reusable_prefix"}.issubset(
        traffic_frame.columns
    ):
        return output_path
    own_figure = axis is None
    if own_figure:
        figure, axis = plt.subplots(figsize=(13, 7.5), dpi=300)
    frame = traffic_frame.dropna(subset=["mean_reusable_prefix"])
    order = _policy_order(frame)
    traffic_order = _traffic_label_order(frame)
    sns.barplot(
        data=frame,
        x="traffic_class_label",
        y="mean_reusable_prefix",
        hue="policy_label",
        order=traffic_order,
        hue_order=[_display_name(policy) for policy in order],
        palette=_label_palette(order),
        errorbar=("ci", 95),
        capsize=0.12,
        err_kws={"linewidth": 1.8, "color": "#333333"},
        edgecolor="#222222",
        linewidth=1.2,
        ax=axis,
    )
    axis.set_title("Reusable Prefix Length by Traffic Class", pad=14)
    axis.set_ylabel("Mean reusable prefix  (tokens)")
    axis.set_xlabel("")
    _style_policy_legend(axis)
    axis.margins(y=0.15)
    axis.text(
        0.0, -0.18, footer_text, transform=axis.transAxes, fontsize=14, color="#555555"
    )
    if own_figure:
        figure.tight_layout()
        _save_figure(figure, output_path)
        plt.close(figure)
    return output_path


def _save_poster_latency_vs_reuse(run_frame, output_path, footer_text, axis=None):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    if not {"mean_reuse_fraction", "ttft_p95"}.issubset(run_frame.columns):
        return output_path
    own_figure = axis is None
    if own_figure:
        figure, axis = plt.subplots(figsize=(11.5, 8), dpi=300)
    frame = _with_policy_labels(
        run_frame.dropna(subset=["mean_reuse_fraction", "ttft_p95"])
    )
    order = _policy_order(frame)
    for policy in order:
        subset = frame.loc[frame["policy"] == policy]
        is_orbit = policy == "orbit"
        axis.scatter(
            subset["mean_reuse_fraction"],
            subset["ttft_p95"],
            s=420 if is_orbit else 200,
            color=POLICY_PALETTE.get(policy, "#4b5563"),
            edgecolor="black",
            linewidth=2.0 if is_orbit else 1.0,
            marker="*" if is_orbit else "o",
            label=_display_name(policy),
            alpha=0.95,
            zorder=4 if is_orbit else 2,
        )
    centroids = frame.groupby("policy", observed=False).agg(
        reuse=("mean_reuse_fraction", "mean"),
        ttft=("ttft_p95", "mean"),
    )
    for policy, row in centroids.iterrows():
        axis.scatter(
            row["reuse"],
            row["ttft"],
            marker="X",
            s=320,
            color=POLICY_PALETTE.get(str(policy), "#4b5563"),
            edgecolor="black",
            linewidth=2.0,
            zorder=5,
        )
    _add_better_region(axis, centroids)
    axis.set_title("Latency vs. Cache Reuse  (per seed)", pad=14)
    axis.set_xlabel("Mean reuse fraction  →")
    axis.set_ylabel("TTFT p95  (seconds)")
    axis.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    _style_policy_legend(axis, title="Router  (X = mean)")
    axis.text(
        0.0, -0.16, footer_text, transform=axis.transAxes, fontsize=14, color="#555555"
    )
    if own_figure:
        figure.tight_layout()
        _save_figure(figure, output_path)
        plt.close(figure)
    return output_path


def _save_poster_combined(run_frame, traffic_frame, output_path, footer_text):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(22, 15), dpi=300)
    figure.suptitle(
        "Orbit: Higher KV-Cache Reuse, Lower Time-to-First-Token",
        fontsize=30,
        fontweight="bold",
        y=0.995,
    )
    _save_poster_ttft_p50(run_frame, output_path, footer_text, axis=axes[0, 0])
    _save_poster_reuse_fraction(run_frame, output_path, footer_text, axis=axes[0, 1])
    _save_poster_prefix_by_traffic(
        traffic_frame, output_path, footer_text, axis=axes[1, 0]
    )
    _save_poster_latency_vs_reuse(run_frame, output_path, footer_text, axis=axes[1, 1])
    for label, axis in zip(["(a)", "(b)", "(c)", "(d)"], axes.flat):
        axis.text(
            -0.10,
            1.05,
            label,
            transform=axis.transAxes,
            fontsize=26,
            fontweight="bold",
            color="#222222",
            ha="left",
            va="bottom",
        )
    for axis in axes.flat:
        footer_artists = [
            artist
            for artist in axis.texts
            if artist.get_text() == footer_text and artist.get_position()[1] < 0
        ]
        for artist in footer_artists:
            artist.remove()
    figure.text(0.5, 0.005, footer_text, ha="center", fontsize=16, color="#555555")
    figure.tight_layout(rect=(0, 0.02, 1, 0.965))
    _save_figure(figure, output_path)
    plt.close(figure)
    return output_path


def _annotate_bars(axis, fmt="{:.2f}"):
    import pandas as pd

    bars = [
        patch
        for patch in axis.patches
        if patch.get_height() > 0 and not pd.isna(patch.get_height())
    ]
    for patch in bars:
        height = patch.get_height()
        center = patch.get_x() + patch.get_width() / 2
        axis.text(
            center,
            height * 0.06,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
            color="white",
        )


def _style_policy_axis(axis):
    for tick in axis.get_xticklabels():
        if tick.get_text() == POLICY_LABELS["orbit"]:
            tick.set_fontweight("bold")
            tick.set_color(ORBIT_COLOR)


def _style_policy_legend(axis, title="Router"):
    legend = axis.legend(title=title, loc="upper right", ncol=1, borderpad=0.6)
    if legend is None:
        return
    for text in legend.get_texts():
        if text.get_text() == POLICY_LABELS["orbit"]:
            text.set_fontweight("bold")
            text.set_color(ORBIT_COLOR)


def _add_ttft_callout(axis, frame):
    if "orbit" not in set(frame["policy"].astype(str)):
        return
    orbit = frame.loc[frame["policy"] == "orbit", "ttft_p50"].mean()
    alternatives = frame.loc[frame["policy"] != "orbit"]
    if alternatives.empty:
        return
    best_alt = alternatives.groupby("policy", observed=False)["ttft_p50"].mean().min()
    if best_alt <= 0:
        return
    axis.text(
        0.98,
        0.97,
        f"{(1 - orbit / best_alt) * 100:.0f}% lower TTFT\nvs. best baseline",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=20,
        fontweight="bold",
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor=ORBIT_COLOR, edgecolor=ORBIT_COLOR
        ),
    )


def _add_reuse_callout(axis, frame):
    if "orbit" not in set(frame["policy"].astype(str)):
        return
    orbit = frame.loc[frame["policy"] == "orbit", "mean_reuse_fraction"].mean()
    alternatives = frame.loc[frame["policy"] != "orbit"]
    if alternatives.empty:
        return
    best_alt = (
        alternatives.groupby("policy", observed=False)["mean_reuse_fraction"]
        .mean()
        .max()
    )
    axis.text(
        0.98,
        0.97,
        f"+{(orbit - best_alt) * 100:.0f} pts\nvs. best baseline",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=20,
        fontweight="bold",
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor=ORBIT_COLOR, edgecolor=ORBIT_COLOR
        ),
    )


def _add_better_region(axis, centroids):
    if "orbit" not in set(centroids.index.astype(str)):
        return
    xmin, xmax = axis.get_xlim()
    _, ymax = axis.get_ylim()
    orbit_reuse = float(centroids.loc["orbit", "reuse"])
    orbit_ttft = float(centroids.loc["orbit", "ttft"])
    if ymax <= 0:
        return
    axis.axvspan(
        orbit_reuse - 0.02,
        xmax,
        ymin=0,
        ymax=min(0.6, orbit_ttft / ymax + 0.1),
        alpha=0.07,
        color=ORBIT_COLOR,
        zorder=0,
    )
    axis.annotate(
        "",
        xy=(min(orbit_reuse + 0.05, xmax - 0.01), orbit_ttft * 0.6),
        xytext=(xmin + 0.02 * (xmax - xmin), ymax * 0.92),
        arrowprops=dict(arrowstyle="-|>", color=ORBIT_COLOR, lw=3.0, mutation_scale=28),
    )
    axis.text(
        xmin + 0.03 * (xmax - xmin),
        ymax * 0.96,
        "better",
        color=ORBIT_COLOR,
        fontsize=22,
        fontweight="bold",
        style="italic",
    )


def _save_ecdf_plot(dataframe, x_field, output_path, title, x_label, subtitle=None):
    # ECDFs are the primary latency plots because they show tail behavior and
    # stochastic dominance without choosing arbitrary histogram bins.
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    import seaborn as sns

    order = _policy_order(dataframe)
    palette = _policy_palette(order)
    figure, axis = plt.subplots(figsize=(11, 7.5), dpi=300)
    sns.ecdfplot(
        data=dataframe,
        x=x_field,
        hue="policy",
        hue_order=order,
        palette=palette,
        linewidth=3.2,
        ax=axis,
    )
    axis.set_title(title, pad=14)
    if subtitle:
        axis.text(
            0.0,
            1.02,
            subtitle,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=14,
            color="#4b5563",
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel("Cumulative probability")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlim(left=0.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.28)
    stats = _policy_quantiles(dataframe, x_field, order)
    for line, policy in zip(axis.lines, order):
        line.set_linestyle(POLICY_LINESTYLES.get(policy, "-"))
        if policy == "orbit":
            line.set_linewidth(4.2)
    handles = [
        Line2D(
            [0],
            [0],
            color=palette[policy],
            linestyle=POLICY_LINESTYLES.get(policy, "-"),
            linewidth=4.2 if policy == "orbit" else 3.2,
        )
        for policy in order
    ]
    labels = [
        f"{_display_name(policy)}  p50={stats[policy]['p50']:.3f}s  p95={stats[policy]['p95']:.3f}s"
        for policy in order
    ]
    axis.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        borderaxespad=0.0,
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    _save_figure(figure, output_path)
    plt.close(figure)
    return output_path


def _save_policy_interval_plot(
    dataframe,
    summary_runs,
    value_field,
    p50_field,
    p95_field,
    output_path,
    title,
    x_label,
):
    # Show p50 and p95 together; in aggregate mode faint intervals expose
    # seed-to-seed variance behind the mean line.
    import matplotlib.pyplot as plt
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
    figure, axis = plt.subplots(figsize=(11, 6.5), dpi=300)
    for index, policy in enumerate(order):
        color = POLICY_PALETTE.get(policy, "#1f6feb")
        p50 = stats[policy]["p50_mean"]
        p95 = stats[policy]["p95_mean"]
        p50_values = stats[policy]["p50_values"]
        p95_values = stats[policy]["p95_values"]
        if len(p50_values) > 1:
            offsets = _seed_offsets(len(p50_values))
            for offset, run_p50, run_p95 in zip(offsets, p50_values, p95_values):
                axis.hlines(
                    index + offset,
                    run_p50,
                    run_p95,
                    color=color,
                    linewidth=1.8,
                    alpha=0.35,
                )
                axis.scatter(
                    [run_p50],
                    [index + offset],
                    color=color,
                    s=46,
                    marker="o",
                    alpha=0.35,
                    zorder=2,
                )
                axis.scatter(
                    [run_p95],
                    [index + offset],
                    color=color,
                    s=56,
                    marker="D",
                    alpha=0.35,
                    zorder=2,
                )
        axis.hlines(index, p50, p95, color=color, linewidth=4)
        axis.scatter([p50], [index], color=color, s=130, marker="o", zorder=3)
        axis.scatter([p95], [index], color=color, s=150, marker="D", zorder=3)
    axis.set_yticks(range(len(order)))
    axis.set_yticklabels([_display_name(policy) for policy in order])
    axis.set_xlabel(x_label)
    axis.set_title(f"{title} (p50 circle, p95 diamond)", pad=14)
    axis.set_xlim(left=0.0)
    axis.grid(True, axis="x", alpha=0.28)
    axis.invert_yaxis()
    note = "Circle = p50, diamond = p95"
    if summary_runs:
        note += ", faint intervals = per-seed runs"
    axis.text(
        0.0,
        -0.16,
        note,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        color="#4b5563",
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    _save_figure(figure, output_path)
    plt.close(figure)
    return output_path


def _save_tradeoff_plot(dataframe, summary_runs, output_path, title):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    if summary_runs:
        summary_frame = _prepare_dataframe(pd.DataFrame.from_records(summary_runs))
        summary_frame["reuse_metric"] = pd.to_numeric(
            summary_frame.get("mean_reusable_prefix"), errors="coerce"
        )
        summary_frame["latency_metric"] = pd.to_numeric(
            summary_frame.get("latency_p50"), errors="coerce"
        )
    else:
        reuse_field = next(
            (
                field
                for field in (
                    "true_reusable_prefix",
                    "actual_reusable_tokens",
                    "estimated_reusable_tokens",
                    "reuse_fraction",
                )
                if field in dataframe.columns
            ),
            None,
        )
        if reuse_field is None:
            return output_path
        summary_frame = (
            dataframe.groupby("policy", observed=False)
            .agg(
                reuse_metric=(reuse_field, "mean"),
                latency_metric=("actual_latency", "median"),
            )
            .reset_index()
        )
    summary_frame = summary_frame.dropna(subset=["reuse_metric", "latency_metric"])
    order = _policy_order(summary_frame)
    summary_frame = _with_policy_labels(summary_frame)
    figure, axis = plt.subplots(figsize=(11, 7.5), dpi=300)
    if "seed" in summary_frame.columns:
        sns.scatterplot(
            data=summary_frame,
            x="reuse_metric",
            y="latency_metric",
            hue="policy_label",
            hue_order=[_display_name(policy) for policy in order],
            palette=_label_palette(order),
            s=115,
            alpha=0.45,
            legend=False,
            ax=axis,
        )
        means = (
            summary_frame.groupby("policy", observed=False)[
                ["reuse_metric", "latency_metric"]
            ]
            .mean()
            .reset_index()
        )
    else:
        means = summary_frame
    means = _with_policy_labels(means)
    sns.scatterplot(
        data=means,
        x="reuse_metric",
        y="latency_metric",
        hue="policy_label",
        hue_order=[_display_name(policy) for policy in order],
        palette=_label_palette(order),
        s=320,
        marker="X",
        legend=False,
        ax=axis,
    )
    for _, row in means.iterrows():
        axis.annotate(
            _display_name(str(row["policy"])),
            (float(row["reuse_metric"]), float(row["latency_metric"])),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=16,
            fontweight="bold" if str(row["policy"]) == "orbit" else "normal",
            color=ORBIT_COLOR if str(row["policy"]) == "orbit" else "#222222",
        )
    axis.set_title(title, pad=14)
    axis.set_xlabel("Mean reusable prefix (tokens)")
    axis.set_ylabel("Median latency (s)")
    axis.grid(True, alpha=0.28)
    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)
    return output_path


def _save_traffic_point_plot(traffic_rows, metric, output_path, title, y_label):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    frame = _prepare_dataframe(pd.DataFrame.from_records(traffic_rows))
    if (
        metric not in frame.columns
        or "policy" not in frame.columns
        or "traffic_class" not in frame.columns
    ):
        return output_path
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.dropna(subset=[metric])
    if frame.empty:
        return output_path
    frame = _with_traffic_labels(_with_policy_labels(frame))
    order = _policy_order(frame)
    figure, axis = plt.subplots(figsize=(11, 7.5), dpi=300)
    sns.barplot(
        data=frame,
        x="traffic_class_label",
        y=metric,
        hue="policy_label",
        order=_traffic_label_order(frame),
        hue_order=[_display_name(policy) for policy in order],
        palette=_label_palette(order),
        errorbar=("ci", 95),
        capsize=0.12,
        err_kws={"linewidth": 1.8, "color": "#333333"},
        edgecolor="#222222",
        linewidth=1.2,
        ax=axis,
    )
    axis.set_title(title, pad=14)
    axis.set_xlabel("Traffic class")
    axis.set_ylabel(y_label)
    axis.grid(True, axis="y", alpha=0.28)
    _style_policy_legend(axis)
    figure.tight_layout()
    _save_figure(figure, output_path)
    plt.close(figure)
    return output_path


def _policy_order(dataframe):
    present = (
        [str(value) for value in dataframe["policy"].dropna().astype(str).unique()]
        if "policy" in dataframe.columns
        else []
    )
    return [policy for policy in POLICY_ORDER if policy in present]


def _canonical_policy(policy):
    raw = str(policy)
    return POLICY_ALIASES.get(raw, raw)


def _policy_palette(order):
    return {policy: POLICY_PALETTE.get(policy, "#4b5563") for policy in order}


def _label_palette(order):
    return {
        _display_name(policy): POLICY_PALETTE.get(policy, "#4b5563") for policy in order
    }


def _traffic_order(dataframe):
    present = (
        [
            str(value)
            for value in dataframe["traffic_class"].dropna().astype(str).unique()
        ]
        if "traffic_class" in dataframe.columns
        else []
    )
    ordered = [traffic for traffic in TRAFFIC_ORDER if traffic in present]
    extras = sorted((traffic for traffic in present if traffic not in TRAFFIC_ORDER))
    return ordered + extras


def _traffic_label_order(dataframe):
    order = _traffic_order(dataframe)
    if not order and "traffic_class_label" in dataframe.columns:
        return [
            str(value)
            for value in dataframe["traffic_class_label"].dropna().astype(str).unique()
        ]
    return [_traffic_display_name(traffic) for traffic in order]


def _policy_quantiles(dataframe, value_field, order):
    quantiles: dict[str, dict[str, float]] = {}
    for policy in order:
        subset = dataframe.loc[dataframe["policy"] == policy, value_field].dropna()
        if subset.empty:
            quantiles[policy] = {"p50": 0.0, "p95": 0.0}
            continue
        quantiles[policy] = {
            "p50": float(subset.quantile(0.5)),
            "p95": float(subset.quantile(0.95)),
        }
    return quantiles


def _summary_interval_stats(summary_frame, order, p50_field, p95_field):
    stats: dict[str, dict[str, object]] = {}
    for policy in order:
        subset = summary_frame.loc[summary_frame["policy"] == policy]
        p50_values = (
            [float(value) for value in subset[p50_field].dropna().tolist()]
            if p50_field in subset
            else []
        )
        p95_values = (
            [float(value) for value in subset[p95_field].dropna().tolist()]
            if p95_field in subset
            else []
        )
        stats[policy] = {
            "p50_mean": float(sum(p50_values) / len(p50_values)) if p50_values else 0.0,
            "p95_mean": float(sum(p95_values) / len(p95_values)) if p95_values else 0.0,
            "p50_values": p50_values,
            "p95_values": p95_values,
        }
    return stats


def _seed_offsets(count):
    if count <= 1:
        return [0.0]
    span = 0.18
    step = span * 2 / (count - 1)
    return [-span + index * step for index in range(count)]


def _display_name(policy):
    return POLICY_LABELS.get(
        _canonical_policy(policy), _canonical_policy(policy).replace("_", " ").title()
    )


def _traffic_display_name(traffic_class):
    return TRAFFIC_LABELS.get(
        str(traffic_class), str(traffic_class).replace("_", " ").title()
    )


def _is_run_directory(path):
    return any(
        (
            (path / name).exists()
            for name in ("manifest.json", "summary.json", "summary_aggregate.json")
        )
    )
