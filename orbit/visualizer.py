from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Mapping, Sequence


SUMMARY_FIELDS = (
    "ttft_p50",
    "ttft_p95",
    "latency_p50",
    "latency_p95",
    "mean_reusable_prefix",
    "mean_reuse_fraction",
    "failover_rate",
)


def generate_reports(run_dir: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(run_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"run directory not found: {root}")

    reports: list[Path] = []
    if recursive:
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name.startswith("seed-") and _is_run_directory(child):
                reports.extend(generate_reports(child, recursive=False))

    if _is_run_directory(root):
        reports.append(write_report(root))
    return reports


def write_report(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    root = Path(run_dir).resolve()
    destination = Path(output_path).resolve() if output_path is not None else root / "report.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_report_html(root), encoding="utf-8")
    return destination


def build_report_html(run_dir: Path) -> str:
    manifest = _load_json(run_dir / "manifest.json", default={})
    summary_rows, aggregate_mode = _load_summary_rows(run_dir)
    calibration = _load_json(run_dir / "calibration.json", default=None)
    selection = _load_json(run_dir / "selection.json", default=None)
    traffic_rows = _load_csv(run_dir / "summary_by_traffic.csv")
    source_rows = _load_csv(run_dir / "summary_by_source.csv")
    records_by_policy = _load_policy_records(run_dir)
    seed_links = _seed_links(run_dir)
    plot_images = _plot_images(run_dir)

    title = f"Orbit Report: {run_dir.name}"
    header_cards = _render_definition_list(
        [
            ("Backend", str(manifest.get("backend", "unknown"))),
            ("Workload", str(manifest.get("workload_kind", "unknown"))),
            ("Policies", ", ".join(manifest.get("policies", [])) if isinstance(manifest.get("policies"), list) else "n/a"),
            ("Requests", str(manifest.get("request_count", manifest.get("requests", "n/a")))),
            ("Warmup", str(manifest.get("warmup_requests", 0))),
            ("Validation", str(manifest.get("validation_requests", 0))),
            ("Generated", str(manifest.get("generated_at", "n/a"))),
            ("Path", str(run_dir)),
        ]
    )

    parts = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        _styles(),
        "</style>",
        "</head>",
        "<body>",
        "<main>",
        f"<h1>{html.escape(title)}</h1>",
        "<p class='subtitle'>Self-contained report generated from benchmark artifacts.</p>",
        header_cards,
    ]

    if seed_links:
        parts.append("<section>")
        parts.append("<h2>Seed Runs</h2>")
        parts.append("<ul class='seed-links'>")
        for seed_name, report_name in seed_links:
            parts.append(f"<li><a href='{html.escape(report_name)}'>{html.escape(seed_name)}</a></li>")
        parts.append("</ul>")
        parts.append("</section>")

    if summary_rows:
        parts.append("<section>")
        parts.append("<h2>Policy Summary</h2>")
        if aggregate_mode:
            parts.append("<p class='note'>Aggregate report using mean metrics across multiple seeds.</p>")
        parts.append(_render_policy_cards(summary_rows))
        parts.append(_render_metric_chart(summary_rows, "latency_p50", "Latency P50"))
        parts.append(_render_metric_chart(summary_rows, "ttft_p50", "TTFT P50"))
        parts.append(_render_metric_chart(summary_rows, "mean_reusable_prefix", "Mean Reusable Prefix"))
        parts.append(_render_summary_table(summary_rows, aggregate_mode))
        parts.append("</section>")

    if calibration or selection:
        parts.append("<section>")
        parts.append("<h2>Model Selection</h2>")
        if calibration:
            parts.append(_render_definition_list(_flatten_mapping("Calibration", calibration)))
        if selection:
            parts.append(_render_definition_list(_flatten_mapping("Selection", selection)))
        parts.append("</section>")

    if traffic_rows:
        parts.append("<section>")
        parts.append("<h2>Traffic Breakdown</h2>")
        parts.append(
            _render_grouped_bar_chart(
                traffic_rows,
                group_field="traffic_class",
                metric="latency_p50",
                title="Latency P50 by Traffic Class",
            )
        )
        parts.append(
            _render_grouped_bar_chart(
                traffic_rows,
                group_field="traffic_class",
                metric="mean_reusable_prefix",
                title="Reusable Prefix by Traffic Class",
            )
        )
        parts.append(_render_table(traffic_rows))
        parts.append("</section>")

    if source_rows:
        parts.append("<section>")
        parts.append("<h2>Source Breakdown</h2>")
        parts.append(
            _render_grouped_bar_chart(
                source_rows,
                group_field="source_id",
                metric="latency_p50",
                title="Latency P50 by Source",
                limit=8,
            )
        )
        parts.append(
            _render_grouped_bar_chart(
                source_rows,
                group_field="source_id",
                metric="request_count",
                title="Request Volume by Source",
                limit=8,
            )
        )
        parts.append(_render_table(source_rows))
        parts.append("</section>")

    if plot_images:
        parts.append("<section>")
        parts.append("<h2>PNG Plots</h2>")
        parts.append(_render_plot_gallery(plot_images))
        parts.append("</section>")

    for policy_name in sorted(records_by_policy):
        policy_records = records_by_policy[policy_name]
        parts.append("<section>")
        parts.append(f"<h2>Per-Request Trace: {html.escape(policy_name)}</h2>")
        parts.append(_render_trace_chart(policy_records, policy_name))
        parts.append(
            _render_chart_grid(
                [
                    _render_scatter_plot(
                        policy_records,
                        x_field="predicted_latency",
                        y_field="actual_latency",
                        title="Predicted vs Actual Latency",
                    ),
                    _render_scatter_plot(
                        policy_records,
                        x_field="route_queue_depth",
                        y_field="actual_latency",
                        title="Route Queue Depth vs Actual Latency",
                    ),
                ]
            )
        )
        parts.append(
            _render_chart_grid(
                [
                    _render_histogram(
                        policy_records,
                        field="reuse_fraction",
                        title="Reuse Fraction Distribution",
                    ),
                    _render_distribution_chart(
                        policy_records,
                        key_field="cluster_id",
                        title="Cluster Assignment",
                    ),
                    _render_distribution_chart(
                        policy_records,
                        key_field="had_failover",
                        title="Failover Outcomes",
                    ),
                ]
            )
        )
        parts.append(_render_record_snapshot(policy_records))
        parts.append("</section>")

    parts.extend(["</main>", "</body>", "</html>"])
    return "\n".join(parts)


def _is_run_directory(path: Path) -> bool:
    return any((path / name).exists() for name in ("manifest.json", "summary.json", "summary_aggregate.json"))


def _load_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_summary_rows(run_dir: Path) -> tuple[list[dict[str, object]], bool]:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        payload = _load_json(summary_path, default={})
        if isinstance(payload, dict):
            rows = []
            for policy_name, metrics in sorted(payload.items()):
                if not isinstance(metrics, Mapping):
                    continue
                row = {"policy": policy_name}
                row.update(metrics)
                rows.append(_normalize_summary_row(row))
            return rows, False

    aggregate_path = run_dir / "summary_aggregate.json"
    payload = _load_json(aggregate_path, default=[])
    if isinstance(payload, list):
        return [_normalize_summary_row(dict(row)) for row in payload if isinstance(row, Mapping)], True
    return [], False


def _normalize_summary_row(row: dict[str, object]) -> dict[str, object]:
    normalized = dict(row)
    for field in SUMMARY_FIELDS:
        mean_key = f"{field}_mean"
        if field not in normalized and mean_key in normalized:
            normalized[field] = normalized[mean_key]
    return normalized


def _load_policy_records(run_dir: Path) -> dict[str, list[dict[str, object]]]:
    records: dict[str, list[dict[str, object]]] = {}
    for path in sorted(run_dir.glob("*_records.json")):
        payload = _load_json(path, default=[])
        if isinstance(payload, list):
            policy_name = path.name[: -len("_records.json")]
            records[policy_name] = [dict(row) for row in payload if isinstance(row, Mapping)]
    return records


def _seed_links(run_dir: Path) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    for child in sorted(run_dir.iterdir()) if run_dir.exists() else []:
        if not child.is_dir() or not child.name.startswith("seed-"):
            continue
        report_path = child / "report.html"
        if report_path.exists():
            links.append((child.name, f"{child.name}/report.html"))
    return links


def _plot_images(run_dir: Path) -> list[str]:
    plots_dir = run_dir / "plots"
    if not plots_dir.exists():
        return []
    return [f"plots/{path.name}" for path in sorted(plots_dir.glob("*.png"))]


def _render_policy_cards(summary_rows: Sequence[Mapping[str, object]]) -> str:
    cards = ["<div class='policy-grid'>"]
    for row in summary_rows:
        policy = html.escape(str(row.get("policy", "unknown")))
        cards.append("<article class='card'>")
        cards.append(f"<h3>{policy}</h3>")
        cards.append("<dl>")
        for label, field in (
            ("TTFT P50", "ttft_p50"),
            ("Latency P50", "latency_p50"),
            ("Reuse", "mean_reusable_prefix"),
            ("Failover Rate", "failover_rate"),
        ):
            cards.append(f"<dt>{html.escape(label)}</dt><dd>{_format_value(row.get(field))}</dd>")
        cards.append("</dl>")
        cards.append("</article>")
    cards.append("</div>")
    return "\n".join(cards)


def _render_metric_chart(summary_rows: Sequence[Mapping[str, object]], field: str, title: str) -> str:
    values = [max(0.0, _to_float(row.get(field))) for row in summary_rows]
    if not values:
        return ""
    maximum = max(values) or 1.0
    bar_width = 520
    bar_height = 24
    gap = 14
    label_width = 140
    total_height = len(summary_rows) * (bar_height + gap) + 24
    lines = [
        "<div class='chart'>",
        f"<h3>{html.escape(title)}</h3>",
        f"<svg viewBox='0 0 {label_width + bar_width + 120} {total_height}' role='img' aria-label='{html.escape(title)}'>",
    ]
    for index, row in enumerate(summary_rows):
        y = 12 + index * (bar_height + gap)
        policy = html.escape(str(row.get("policy", "unknown")))
        value = max(0.0, _to_float(row.get(field)))
        width = 0.0 if maximum <= 0 else (value / maximum) * bar_width
        lines.append(f"<text x='0' y='{y + 17}' class='axis-label'>{policy}</text>")
        lines.append(f"<rect x='{label_width}' y='{y}' width='{bar_width}' height='{bar_height}' class='bar-bg' rx='4' />")
        lines.append(f"<rect x='{label_width}' y='{y}' width='{width:.2f}' height='{bar_height}' class='bar' rx='4' />")
        ci_low = _to_float(row.get(f"{field}_ci_low"))
        ci_high = _to_float(row.get(f"{field}_ci_high"))
        if ci_high > 0.0 and ci_high >= ci_low:
            ci_low_x = label_width + (ci_low / maximum) * bar_width
            ci_high_x = label_width + (ci_high / maximum) * bar_width
            mid_y = y + (bar_height / 2.0)
            lines.append(f"<line x1='{ci_low_x:.2f}' y1='{mid_y:.2f}' x2='{ci_high_x:.2f}' y2='{mid_y:.2f}' class='ci-line' />")
            lines.append(f"<line x1='{ci_low_x:.2f}' y1='{y + 4:.2f}' x2='{ci_low_x:.2f}' y2='{y + bar_height - 4:.2f}' class='ci-line' />")
            lines.append(f"<line x1='{ci_high_x:.2f}' y1='{y + 4:.2f}' x2='{ci_high_x:.2f}' y2='{y + bar_height - 4:.2f}' class='ci-line' />")
        lines.append(f"<text x='{label_width + bar_width + 12}' y='{y + 17}' class='value-label'>{html.escape(_format_value(value))}</text>")
    lines.extend(["</svg>", "</div>"])
    return "\n".join(lines)


def _render_grouped_bar_chart(
    rows: Sequence[Mapping[str, object]],
    group_field: str,
    metric: str,
    title: str,
    limit: int | None = None,
) -> str:
    filtered_rows = list(rows)
    if limit is not None:
        filtered_rows = _limit_groups(filtered_rows, group_field, limit)
    if not filtered_rows:
        return ""

    groups = sorted({str(row.get(group_field, "n/a")) for row in filtered_rows})
    policies = sorted({str(row.get("policy", "unknown")) for row in filtered_rows})
    if not groups or not policies:
        return ""

    max_value = max(_to_float(row.get(metric)) for row in filtered_rows)
    if max_value <= 0.0:
        max_value = 1.0

    label_width = 120
    group_width = 110
    group_gap = 28
    bar_gap = 8
    inner_width = group_width - bar_gap * max(0, len(policies) - 1)
    bar_width = max(14.0, inner_width / max(1, len(policies)))
    chart_height = 220
    top_pad = 28
    bottom_pad = 54
    width = label_width + len(groups) * group_width + max(0, len(groups) - 1) * group_gap + 36
    height = top_pad + chart_height + bottom_pad

    value_map = {
        (str(row.get(group_field, "n/a")), str(row.get("policy", "unknown"))): _to_float(row.get(metric))
        for row in filtered_rows
    }
    lines = [
        "<div class='chart'>",
        f"<h3>{html.escape(title)}</h3>",
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>",
    ]
    for tick_index in range(5):
        tick_value = (max_value / 4.0) * tick_index
        y = top_pad + chart_height - (tick_value / max_value) * chart_height
        lines.append(f"<line x1='{label_width}' y1='{y:.2f}' x2='{width - 20}' y2='{y:.2f}' class='grid' />")
        lines.append(f"<text x='4' y='{y + 4:.2f}' class='axis-label'>{html.escape(_format_value(tick_value))}</text>")

    for group_index, group in enumerate(groups):
        group_x = label_width + group_index * (group_width + group_gap)
        used_width = len(policies) * bar_width + max(0, len(policies) - 1) * bar_gap
        for policy_index, policy in enumerate(policies):
            value = value_map.get((group, policy), 0.0)
            current_bar_height = (value / max_value) * chart_height
            x = group_x + policy_index * (bar_width + bar_gap)
            y = top_pad + chart_height - current_bar_height
            color = _series_color(policy_index)
            lines.append(
                f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_width:.2f}' height='{current_bar_height:.2f}' "
                f"rx='4' fill='{color}'>"
                f"<title>{html.escape(policy)} / {html.escape(group)}: {html.escape(_format_value(value))}</title>"
                "</rect>"
            )
        label_x = group_x + (used_width / 2.0)
        lines.append(f"<text x='{label_x:.2f}' y='{height - 18}' class='group-label' text-anchor='middle'>{html.escape(_short_label(group))}</text>")

    legend_x = label_width
    for policy_index, policy in enumerate(policies):
        color = _series_color(policy_index)
        entry_x = legend_x + policy_index * 160
        lines.append(f"<rect x='{entry_x}' y='4' width='14' height='14' rx='3' fill='{color}' />")
        lines.append(f"<text x='{entry_x + 20}' y='15' class='legend'>{html.escape(policy)}</text>")

    lines.extend(["</svg>", "</div>"])
    return "\n".join(lines)


def _render_summary_table(summary_rows: Sequence[Mapping[str, object]], aggregate_mode: bool) -> str:
    preferred_columns = ["policy", *SUMMARY_FIELDS]
    if aggregate_mode:
        preferred_columns.extend(["runs"])
    available_columns = [column for column in preferred_columns if any(column in row for row in summary_rows)]
    return _render_table(summary_rows, columns=available_columns)


def _render_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str] | None = None) -> str:
    if not rows:
        return "<p class='note'>No data available.</p>"
    ordered_columns = list(columns or rows[0].keys())
    parts = ["<div class='table-wrap'><table><thead><tr>"]
    for column in ordered_columns:
        parts.append(f"<th>{html.escape(_prettify_name(column))}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for column in ordered_columns:
            parts.append(f"<td>{html.escape(_format_value(row.get(column)))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "\n".join(parts)


def _render_trace_chart(records: Sequence[Mapping[str, object]], policy_name: str) -> str:
    if not records:
        return "<p class='note'>No per-request records available.</p>"
    latency_points = _polyline_points(records, "actual_latency")
    ttft_points = _polyline_points(records, "actual_ttft")
    max_value = max(
        [_to_float(record.get("actual_latency")) for record in records]
        + [_to_float(record.get("actual_ttft")) for record in records]
    )
    if max_value <= 0:
        max_value = 1.0

    width = 860
    height = 280
    left_pad = 44
    right_pad = 20
    top_pad = 20
    bottom_pad = 32
    chart_width = width - left_pad - right_pad
    chart_height = height - top_pad - bottom_pad

    def to_svg_points(points: Sequence[tuple[int, float]]) -> str:
        transformed: list[str] = []
        count = max(1, len(points) - 1)
        for index, value in points:
            x = left_pad + (index / count) * chart_width
            y = top_pad + chart_height - (value / max_value) * chart_height
            transformed.append(f"{x:.2f},{y:.2f}")
        return " ".join(transformed)

    lines = [
        "<div class='chart'>",
        f"<h3>{html.escape(policy_name)} request trace</h3>",
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(policy_name)} request trace'>",
        f"<line x1='{left_pad}' y1='{top_pad + chart_height}' x2='{width - right_pad}' y2='{top_pad + chart_height}' class='axis' />",
        f"<line x1='{left_pad}' y1='{top_pad}' x2='{left_pad}' y2='{top_pad + chart_height}' class='axis' />",
    ]
    for tick_index in range(5):
        tick_value = (max_value / 4.0) * tick_index
        y = top_pad + chart_height - (tick_value / max_value) * chart_height
        lines.append(f"<line x1='{left_pad}' y1='{y:.2f}' x2='{width - right_pad}' y2='{y:.2f}' class='grid' />")
        lines.append(f"<text x='4' y='{y + 4:.2f}' class='axis-label'>{html.escape(_format_value(tick_value))}</text>")
    lines.append(f"<polyline points='{to_svg_points(latency_points)}' class='trace latency-trace' />")
    lines.append(f"<polyline points='{to_svg_points(ttft_points)}' class='trace ttft-trace' />")
    lines.append(
        f"<text x='{left_pad}' y='{height - 8}' class='legend'>Request index</text>"
    )
    lines.append(
        f"<text x='{width - 220}' y='{top_pad + 12}' class='legend latency-trace-text'>Latency</text>"
    )
    lines.append(
        f"<text x='{width - 110}' y='{top_pad + 12}' class='legend ttft-trace-text'>TTFT</text>"
    )
    lines.extend(["</svg>", "</div>"])
    return "\n".join(lines)


def _render_scatter_plot(
    records: Sequence[Mapping[str, object]],
    x_field: str,
    y_field: str,
    title: str,
) -> str:
    points = [
        (record, _to_float(record.get(x_field)), _to_float(record.get(y_field)))
        for record in records
        if x_field in record and y_field in record
    ]
    if not points:
        return ""

    max_x = max(x for _, x, _ in points)
    max_y = max(y for _, _, y in points)
    if max_x <= 0.0:
        max_x = 1.0
    if max_y <= 0.0:
        max_y = 1.0

    width = 520
    height = 300
    left_pad = 50
    right_pad = 20
    top_pad = 20
    bottom_pad = 44
    chart_width = width - left_pad - right_pad
    chart_height = height - top_pad - bottom_pad
    diagonal_limit = min(max_x, max_y)

    lines = [
        "<div class='chart'>",
        f"<h3>{html.escape(title)}</h3>",
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>",
        f"<line x1='{left_pad}' y1='{top_pad + chart_height}' x2='{width - right_pad}' y2='{top_pad + chart_height}' class='axis' />",
        f"<line x1='{left_pad}' y1='{top_pad}' x2='{left_pad}' y2='{top_pad + chart_height}' class='axis' />",
    ]
    for tick_index in range(5):
        x_tick = (max_x / 4.0) * tick_index
        x = left_pad + (x_tick / max_x) * chart_width
        lines.append(f"<line x1='{x:.2f}' y1='{top_pad}' x2='{x:.2f}' y2='{top_pad + chart_height}' class='grid' />")
        lines.append(f"<text x='{x:.2f}' y='{height - 8}' class='axis-label' text-anchor='middle'>{html.escape(_format_value(x_tick))}</text>")
        y_tick = (max_y / 4.0) * tick_index
        y = top_pad + chart_height - (y_tick / max_y) * chart_height
        lines.append(f"<line x1='{left_pad}' y1='{y:.2f}' x2='{width - right_pad}' y2='{y:.2f}' class='grid' />")
        lines.append(f"<text x='4' y='{y + 4:.2f}' class='axis-label'>{html.escape(_format_value(y_tick))}</text>")

    if diagonal_limit > 0:
        x2 = left_pad + (diagonal_limit / max_x) * chart_width
        y2 = top_pad + chart_height - (diagonal_limit / max_y) * chart_height
        lines.append(f"<line x1='{left_pad}' y1='{top_pad + chart_height}' x2='{x2:.2f}' y2='{y2:.2f}' class='reference-line' />")

    for record, x_value, y_value in points:
        x = left_pad + (x_value / max_x) * chart_width
        y = top_pad + chart_height - (y_value / max_y) * chart_height
        lines.append(
            f"<circle cx='{x:.2f}' cy='{y:.2f}' r='4.5' class='scatter-dot'>"
            f"<title>{html.escape(str(record.get('request_id', 'request')))}: {html.escape(_prettify_name(x_field))}={html.escape(_format_value(x_value))}, {html.escape(_prettify_name(y_field))}={html.escape(_format_value(y_value))}</title>"
            "</circle>"
        )

    lines.append(f"<text x='{left_pad}' y='{height - 24}' class='legend'>{html.escape(_prettify_name(x_field))}</text>")
    lines.append(f"<text x='4' y='{top_pad - 4}' class='legend'>{html.escape(_prettify_name(y_field))}</text>")
    lines.extend(["</svg>", "</div>"])
    return "\n".join(lines)


def _render_histogram(
    records: Sequence[Mapping[str, object]],
    field: str,
    title: str,
    bins: int = 8,
) -> str:
    values = [_to_float(record.get(field)) for record in records if field in record]
    if not values:
        return ""

    minimum = min(values)
    maximum = max(values)
    bucket_count = max(1, bins)
    if minimum == maximum:
        counts = [len(values)]
        labels = [f"{_format_value(minimum)}"]
    else:
        bucket_width = (maximum - minimum) / bucket_count
        counts = [0 for _ in range(bucket_count)]
        for value in values:
            if value == maximum:
                bucket_index = bucket_count - 1
            else:
                bucket_index = int((value - minimum) / bucket_width)
            counts[bucket_index] += 1
        labels = []
        for index in range(bucket_count):
            start = minimum + index * bucket_width
            end = start + bucket_width
            labels.append(f"{_format_value(start)}-{_format_value(end)}")

    chart_width = 420
    chart_height = 220
    left_pad = 34
    right_pad = 14
    top_pad = 20
    bottom_pad = 64
    inner_width = chart_width - left_pad - right_pad
    inner_height = chart_height - top_pad - bottom_pad
    max_count = max(counts) or 1
    bar_width = inner_width / max(1, len(counts))

    lines = [
        "<div class='chart'>",
        f"<h3>{html.escape(title)}</h3>",
        f"<svg viewBox='0 0 {chart_width} {chart_height}' role='img' aria-label='{html.escape(title)}'>",
    ]
    for index, count in enumerate(counts):
        current_bar_height = (count / max_count) * inner_height
        x = left_pad + index * bar_width + 3
        y = top_pad + inner_height - current_bar_height
        actual_bar_width = max(8.0, bar_width - 6)
        lines.append(f"<rect x='{x:.2f}' y='{y:.2f}' width='{actual_bar_width:.2f}' height='{current_bar_height:.2f}' rx='3' class='histogram-bar' />")
        lines.append(f"<text x='{x + actual_bar_width / 2.0:.2f}' y='{top_pad + inner_height + 16}' class='group-label' text-anchor='middle'>{html.escape(_short_label(labels[index], max_length=14))}</text>")
        lines.append(f"<text x='{x + actual_bar_width / 2.0:.2f}' y='{y - 6:.2f}' class='value-label' text-anchor='middle'>{count}</text>")
    lines.append(f"<line x1='{left_pad}' y1='{top_pad + inner_height}' x2='{chart_width - right_pad}' y2='{top_pad + inner_height}' class='axis' />")
    lines.append(f"<text x='{left_pad}' y='{chart_height - 8}' class='legend'>{html.escape(_prettify_name(field))}</text>")
    lines.extend(["</svg>", "</div>"])
    return "\n".join(lines)


def _render_distribution_chart(
    records: Sequence[Mapping[str, object]],
    key_field: str,
    title: str,
) -> str:
    counts: dict[str, int] = {}
    for record in records:
        if key_field not in record:
            continue
        raw_value = record.get(key_field)
        if isinstance(raw_value, bool):
            key = "yes" if raw_value else "no"
        else:
            key = str(raw_value)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return ""

    labels = sorted(counts)
    max_count = max(counts.values()) or 1
    width = 420
    height = 44 + len(labels) * 34
    label_width = 118
    max_bar_width = width - label_width - 80
    lines = [
        "<div class='chart'>",
        f"<h3>{html.escape(title)}</h3>",
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>",
    ]
    for index, label in enumerate(labels):
        y = 12 + index * 34
        count = counts[label]
        current_width = (count / max_count) * max_bar_width
        lines.append(f"<text x='0' y='{y + 15}' class='axis-label'>{html.escape(_short_label(label))}</text>")
        lines.append(f"<rect x='{label_width}' y='{y}' width='{max_bar_width}' height='20' class='bar-bg' rx='4' />")
        lines.append(f"<rect x='{label_width}' y='{y}' width='{current_width:.2f}' height='20' rx='4' class='distribution-bar' />")
        lines.append(f"<text x='{label_width + max_bar_width + 10}' y='{y + 15}' class='value-label'>{count}</text>")
    lines.extend(["</svg>", "</div>"])
    return "\n".join(lines)


def _render_chart_grid(charts: Sequence[str]) -> str:
    present = [chart for chart in charts if chart]
    if not present:
        return ""
    return "<div class='chart-grid'>\n" + "\n".join(present) + "\n</div>"


def _polyline_points(records: Sequence[Mapping[str, object]], field: str) -> list[tuple[int, float]]:
    return [(index, max(0.0, _to_float(record.get(field)))) for index, record in enumerate(records)]


def _render_record_snapshot(records: Sequence[Mapping[str, object]]) -> str:
    snapshot_rows: list[dict[str, object]] = []
    for record in records[:10]:
        snapshot_rows.append(
            {
                "request_id": record.get("request_id"),
                "cluster_id": record.get("cluster_id"),
                "actual_latency": record.get("actual_latency"),
                "actual_ttft": record.get("actual_ttft"),
                "actual_reusable_tokens": record.get("actual_reusable_tokens"),
                "had_failover": record.get("had_failover"),
            }
        )
    return _render_table(snapshot_rows)


def _render_plot_gallery(images: Sequence[str]) -> str:
    figures = ["<div class='plot-gallery'>"]
    for image_path in images:
        label = _prettify_name(Path(image_path).stem)
        figures.append("<figure class='plot-card'>")
        figures.append(f"<img src='{html.escape(image_path)}' alt='{html.escape(label)}' />")
        figures.append(f"<figcaption>{html.escape(label)}</figcaption>")
        figures.append("</figure>")
    figures.append("</div>")
    return "\n".join(figures)


def _render_definition_list(items: Sequence[tuple[str, str]]) -> str:
    parts = ["<div class='definition-grid'>"]
    for label, value in items:
        parts.append("<article class='definition-card'>")
        parts.append(f"<div class='definition-label'>{html.escape(label)}</div>")
        parts.append(f"<div class='definition-value'>{html.escape(value)}</div>")
        parts.append("</article>")
    parts.append("</div>")
    return "\n".join(parts)


def _flatten_mapping(prefix: str, mapping: Mapping[str, object]) -> list[tuple[str, str]]:
    flattened: list[tuple[str, str]] = []
    for key, value in sorted(mapping.items()):
        if isinstance(value, Mapping):
            flattened.append((f"{prefix} {key}", json.dumps(value, sort_keys=True)))
        else:
            flattened.append((f"{prefix} {key}", _format_value(value)))
    return flattened


def _prettify_name(name: str) -> str:
    return name.replace("_", " ").title()


def _short_label(label: str, max_length: int = 18) -> str:
    if len(label) <= max_length:
        return label
    return f"{label[: max_length - 1]}…"


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if value is None:
        return "n/a"
    return str(value)


def _to_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _limit_groups(
    rows: Sequence[Mapping[str, object]],
    group_field: str,
    limit: int,
) -> list[Mapping[str, object]]:
    if limit <= 0:
        return list(rows)
    totals: dict[str, float] = {}
    for row in rows:
        key = str(row.get(group_field, "n/a"))
        totals[key] = totals.get(key, 0.0) + _to_float(row.get("request_count"))
    selected = {
        group
        for group, _ in sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:limit]
    }
    return [row for row in rows if str(row.get(group_field, "n/a")) in selected]


def _series_color(index: int) -> str:
    palette = (
        "#1f6feb",
        "#d97706",
        "#059669",
        "#dc2626",
        "#7c3aed",
        "#0f766e",
        "#b45309",
    )
    return palette[index % len(palette)]


def _styles() -> str:
    return """
body {
  margin: 0;
  font-family: "SF Pro Text", "Segoe UI", Helvetica, Arial, sans-serif;
  background: linear-gradient(180deg, #f8f7f2 0%, #eef3f8 100%);
  color: #1b2733;
}
main {
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 24px 64px;
}
h1, h2, h3 {
  margin: 0 0 12px;
}
section {
  margin-top: 32px;
  padding: 24px;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid #d9e2ec;
  border-radius: 18px;
  box-shadow: 0 10px 24px rgba(44, 62, 80, 0.08);
}
.subtitle,
.note {
  color: #52606d;
}
.definition-grid,
.policy-grid,
.chart-grid,
.plot-gallery {
  display: grid;
  gap: 16px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
}
.chart-grid {
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  align-items: start;
}
.plot-gallery {
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
}
.definition-card,
.card {
  padding: 16px 18px;
  border-radius: 14px;
  background: linear-gradient(180deg, #ffffff 0%, #f5f8fb 100%);
  border: 1px solid #d9e2ec;
}
.definition-label {
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #52606d;
}
.definition-value {
  margin-top: 8px;
  font-size: 0.98rem;
  line-height: 1.4;
  word-break: break-word;
}
.plot-card {
  margin: 0;
  padding: 14px;
  border-radius: 14px;
  border: 1px solid #d9e2ec;
  background: linear-gradient(180deg, #ffffff 0%, #f5f8fb 100%);
}
.plot-card img {
  display: block;
  width: 100%;
  height: auto;
  border-radius: 10px;
}
.plot-card figcaption {
  margin-top: 10px;
  font-size: 0.92rem;
  color: #52606d;
}
.card dl {
  margin: 0;
}
.card dt {
  margin-top: 10px;
  font-size: 0.78rem;
  text-transform: uppercase;
  color: #7b8794;
}
.card dd {
  margin: 4px 0 0;
  font-size: 1rem;
}
.chart {
  margin-top: 18px;
}
.axis,
.grid {
  stroke: #cbd2d9;
  stroke-width: 1;
}
.grid {
  stroke-dasharray: 4 4;
}
.axis-label,
.value-label,
.legend,
.group-label {
  font-size: 12px;
  fill: #52606d;
}
.bar-bg {
  fill: #e4ebf1;
}
.bar {
  fill: #1f6feb;
}
.distribution-bar {
  fill: #0f766e;
}
.histogram-bar {
  fill: #7c3aed;
}
.ci-line,
.reference-line {
  stroke: #2d3748;
  stroke-width: 1.5;
}
.reference-line {
  stroke-dasharray: 6 4;
}
.scatter-dot {
  fill: #1f6feb;
  opacity: 0.75;
}
.trace {
  fill: none;
  stroke-width: 3;
}
.latency-trace {
  stroke: #1f6feb;
}
.ttft-trace {
  stroke: #d97706;
}
.latency-trace-text {
  fill: #1f6feb;
}
.ttft-trace-text {
  fill: #d97706;
}
.table-wrap {
  overflow-x: auto;
}
table {
  width: 100%;
  border-collapse: collapse;
}
th,
td {
  padding: 10px 12px;
  border-bottom: 1px solid #e4ebf1;
  text-align: left;
  white-space: nowrap;
}
th {
  font-size: 0.84rem;
  color: #52606d;
  background: #f8fafc;
}
.seed-links {
  margin: 0;
  padding-left: 20px;
}
a {
  color: #1f6feb;
}
"""
