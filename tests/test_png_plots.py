from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from orbit.png_plots import generate_run_plots


@unittest.skipUnless(importlib.util.find_spec("seaborn") is not None, "seaborn is not installed")
class PngPlotTests(unittest.TestCase):
    def test_generate_run_plots_writes_ttft_cdf_and_related_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "manifest.json").write_text(
                json.dumps({"backend": "synthetic", "policies": ["summary", "random"]}),
                encoding="utf-8",
            )
            (run_dir / "summary_records.json").write_text(
                json.dumps(
                    [
                        {
                            "request_id": "summary-0",
                            "policy": "summary",
                            "cluster_id": "cluster-0",
                            "predicted_latency": 1.5,
                            "actual_latency": 1.8,
                            "actual_ttft": 0.8,
                            "reuse_fraction": 0.6,
                            "had_failover": False,
                        },
                        {
                            "request_id": "summary-1",
                            "policy": "summary",
                            "cluster_id": "cluster-1",
                            "predicted_latency": 1.9,
                            "actual_latency": 2.1,
                            "actual_ttft": 1.0,
                            "reuse_fraction": 0.5,
                            "had_failover": True,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "random_records.json").write_text(
                json.dumps(
                    [
                        {
                            "request_id": "random-0",
                            "policy": "random",
                            "cluster_id": "cluster-1",
                            "predicted_latency": 2.4,
                            "actual_latency": 2.6,
                            "actual_ttft": 1.4,
                            "reuse_fraction": 0.2,
                            "had_failover": False,
                        },
                        {
                            "request_id": "random-1",
                            "policy": "random",
                            "cluster_id": "cluster-0",
                            "predicted_latency": 2.8,
                            "actual_latency": 3.0,
                            "actual_ttft": 1.7,
                            "reuse_fraction": 0.1,
                            "had_failover": False,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            created = generate_run_plots(run_dir)

            expected = {
                (run_dir / "plots" / "ttft_cdf.png").resolve(),
                (run_dir / "plots" / "latency_cdf.png").resolve(),
                (run_dir / "plots" / "reuse_fraction_distribution.png").resolve(),
                (run_dir / "plots" / "predicted_vs_actual_latency.png").resolve(),
                (run_dir / "plots" / "ttft_by_policy.png").resolve(),
                (run_dir / "plots" / "cluster_assignment.png").resolve(),
                (run_dir / "plots" / "failover_distribution.png").resolve(),
            }
            self.assertTrue(expected.issubset(set(created)))
            for path in expected:
                self.assertTrue(path.exists(), f"missing plot {path}")


if __name__ == "__main__":
    unittest.main()
