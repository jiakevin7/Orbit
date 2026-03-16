from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class MetricsSummary:
    router_name: str
    workload_name: str
    num_requests: int
    success_rate: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    ttft_mean_ms: float
    total_latency_p50_ms: float
    total_latency_p95_ms: float
    prefill_savings_mean: float
    cache_hit_rate: float
    goodput_rps: float  # successful requests per second


def compute_metrics(result_path: str) -> MetricsSummary:
    """Compute metrics from a benchmark result JSON file."""
    with open(result_path) as f:
        data = json.load(f)

    results = data["results"]
    successful = [r for r in results if r["success"]]

    if not successful:
        return MetricsSummary(
            router_name=data["router_name"],
            workload_name=data["workload_name"],
            num_requests=data["num_requests"],
            success_rate=0.0,
            ttft_p50_ms=0.0,
            ttft_p95_ms=0.0,
            ttft_p99_ms=0.0,
            ttft_mean_ms=0.0,
            total_latency_p50_ms=0.0,
            total_latency_p95_ms=0.0,
            prefill_savings_mean=0.0,
            cache_hit_rate=0.0,
            goodput_rps=0.0,
        )

    ttfts = np.array([r["ttft_ms"] for r in successful])
    latencies = np.array([r["total_ms"] for r in successful])

    # Prefill savings: fraction of tokens that were cached
    savings = []
    cache_hits = 0
    for r in successful:
        if r["prompt_tokens"] > 0:
            savings.append(r["cached_tokens"] / r["prompt_tokens"])
        if r["cached_tokens"] > 0:
            cache_hits += 1

    return MetricsSummary(
        router_name=data["router_name"],
        workload_name=data["workload_name"],
        num_requests=data["num_requests"],
        success_rate=len(successful) / data["num_requests"],
        ttft_p50_ms=float(np.percentile(ttfts, 50)),
        ttft_p95_ms=float(np.percentile(ttfts, 95)),
        ttft_p99_ms=float(np.percentile(ttfts, 99)),
        ttft_mean_ms=float(np.mean(ttfts)),
        total_latency_p50_ms=float(np.percentile(latencies, 50)),
        total_latency_p95_ms=float(np.percentile(latencies, 95)),
        prefill_savings_mean=float(np.mean(savings)) if savings else 0.0,
        cache_hit_rate=cache_hits / len(successful) if successful else 0.0,
        goodput_rps=len(successful) / data["total_time_s"] if data["total_time_s"] > 0 else 0.0,
    )


def _discover_result_files(results_dir: str) -> dict[str, list[str]]:
    """Discover result JSON files, supporting both flat and subfolder layouts.

    Returns {workload_name: [file_paths]}.
    """
    workloads: dict[str, list[str]] = {}

    for entry in sorted(os.listdir(results_dir)):
        full = os.path.join(results_dir, entry)

        if os.path.isdir(full):
            # Subfolder layout: results/<workload>/<router>.json
            wl_name = entry
            for fname in sorted(os.listdir(full)):
                if fname.endswith(".json"):
                    workloads.setdefault(wl_name, []).append(os.path.join(full, fname))

        elif entry.endswith(".json"):
            # Flat layout: results/<router>_<workload>.json (legacy)
            path = full
            with open(path) as f:
                data = json.load(f)
            wl_name = data.get("workload_name", "unknown")
            workloads.setdefault(wl_name, []).append(path)

    return workloads


def generate_report(results_dir: str) -> str:
    """Generate a text report from all result files in a directory."""
    lines = ["=" * 80, "ORBIT BENCHMARK REPORT", "=" * 80, ""]

    workloads_files = _discover_result_files(results_dir)

    for wl_name, paths in workloads_files.items():
        summaries = [compute_metrics(p) for p in paths]

        lines.append(f"Workload: {wl_name}")
        lines.append("-" * 60)
        lines.append(
            f"{'Router':<15} {'TTFT p50':>10} {'TTFT p95':>10} "
            f"{'Cache Hit%':>10} {'Savings%':>10} {'Goodput':>10}"
        )
        lines.append("-" * 60)

        for s in sorted(summaries, key=lambda x: x.ttft_p50_ms):
            lines.append(
                f"{s.router_name:<15} {s.ttft_p50_ms:>9.1f}ms {s.ttft_p95_ms:>9.1f}ms "
                f"{s.cache_hit_rate * 100:>9.1f}% {s.prefill_savings_mean * 100:>9.1f}% "
                f"{s.goodput_rps:>9.1f}"
            )
        lines.append("")

    return "\n".join(lines)


def plot_results(results_dir: str, output_dir: str | None = None) -> None:
    """Generate plots from benchmark results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    if output_dir is None:
        output_dir = results_dir

    workloads_files = _discover_result_files(results_dir)

    for wl_name, paths in workloads_files.items():
        # Build per-router TTFT lists
        routers: dict[str, list] = {}
        for path in paths:
            with open(path) as f:
                data = json.load(f)
            router = data["router_name"]
            ttfts = [r["ttft_ms"] for r in data["results"] if r["success"]]
            routers[router] = ttfts

        # Plots go into the workload subfolder if it exists, else output_dir
        wl_output = os.path.join(output_dir, wl_name)
        if not os.path.isdir(wl_output):
            wl_output = output_dir
        os.makedirs(wl_output, exist_ok=True)

        # CDF plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for router_name, ttfts in sorted(routers.items()):
            sorted_ttfts = np.sort(ttfts)
            cdf = np.arange(1, len(sorted_ttfts) + 1) / len(sorted_ttfts)
            ax.plot(sorted_ttfts, cdf, label=router_name, linewidth=2)

        ax.set_xlabel("Time to First Token (ms)")
        ax.set_ylabel("CDF")
        ax.set_title(f"TTFT CDF — {wl_name} workload")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(wl_output, "ttft_cdf.png"), dpi=150)
        plt.close(fig)

        # Bar chart: cache hit rate
        fig, ax = plt.subplots(figsize=(8, 5))
        metrics = [compute_metrics(p) for p in paths]
        metrics.sort(key=lambda m: m.router_name)

        names = [m.router_name for m in metrics]
        hits = [m.cache_hit_rate * 100 for m in metrics]
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
        ax.bar(names, hits, color=colors[: len(names)])
        ax.set_ylabel("Cache Hit Rate (%)")
        ax.set_title(f"Cache Hit Rate — {wl_name} workload")
        fig.tight_layout()
        fig.savefig(os.path.join(wl_output, "cache_hit.png"), dpi=150)
        plt.close(fig)
