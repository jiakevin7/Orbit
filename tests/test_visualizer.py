from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from orbit.visualizer import generate_reports


class VisualizerTests(unittest.TestCase):
    def test_generate_reports_writes_html_for_single_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "backend": "synthetic",
                        "workload_kind": "mixed_realistic",
                        "policies": ["summary"],
                        "request_count": 4,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "policy": "summary",
                            "request_count": 4,
                            "mean_reusable_prefix": 18.0,
                            "mean_reuse_fraction": 0.25,
                            "ttft_p50": 1.2,
                            "ttft_p95": 1.8,
                            "latency_p50": 2.2,
                            "latency_p95": 3.0,
                            "failover_rate": 0.25,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "summary_records.json").write_text(
                json.dumps(
                    [
                        {
                            "request_id": "req-0",
                            "cluster_id": "cluster-0",
                            "predicted_latency": 1.8,
                            "actual_latency": 2.0,
                            "actual_ttft": 1.0,
                            "actual_reusable_tokens": 16,
                            "reuse_fraction": 0.4,
                            "had_failover": False,
                        },
                        {
                            "request_id": "req-1",
                            "cluster_id": "cluster-1",
                            "predicted_latency": 2.3,
                            "actual_latency": 2.5,
                            "actual_ttft": 1.3,
                            "actual_reusable_tokens": 20,
                            "reuse_fraction": 0.5,
                            "had_failover": True,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            plots_dir = run_dir / "plots"
            plots_dir.mkdir()
            (plots_dir / "ttft_cdf.png").write_bytes(b"\x89PNG\r\n\x1a\n")

            report_paths = generate_reports(run_dir)

            expected_report = (run_dir / "report.html").resolve()
            self.assertEqual(report_paths, [expected_report])
            report_html = expected_report.read_text(encoding="utf-8")
            self.assertIn("Orbit Report", report_html)
            self.assertIn("summary", report_html)
            self.assertIn("Latency P50", report_html)
            self.assertIn("Predicted vs Actual Latency", report_html)
            self.assertIn("Reuse Fraction Distribution", report_html)
            self.assertIn("Cluster Assignment", report_html)
            self.assertIn("PNG Plots", report_html)
            self.assertIn("plots/ttft_cdf.png", report_html)


if __name__ == "__main__":
    unittest.main()
