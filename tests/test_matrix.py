from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from orbit.matrix import collect_matrix_summary_rows, load_external_benchmark_matrix, matrix_manifest


class MatrixTests(unittest.TestCase):
    def test_load_external_benchmark_matrix_reads_config(self) -> None:
        matrix_name, description, scenarios = load_external_benchmark_matrix()

        self.assertEqual(matrix_name, "external_standard_matrix")
        self.assertIn("multi-seed", description)
        self.assertEqual([scenario.name for scenario in scenarios[:3]], ["sharegpt_chat", "rag_retrieval", "agent_tools"])

    def test_collect_matrix_summary_rows_reads_aggregate_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scenario_dir = root / "sharegpt_chat"
            scenario_dir.mkdir()
            (scenario_dir / "summary_aggregate.json").write_text(
                json.dumps(
                    [
                        {
                            "policy": "summary",
                            "runs": 3,
                            "ttft_p50_mean": 0.5,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            rows = collect_matrix_summary_rows(root)

            self.assertEqual(rows, [{"scenario": "sharegpt_chat", "policy": "summary", "runs": 3, "ttft_p50_mean": 0.5}])

    def test_matrix_manifest_records_dataset_resolution(self) -> None:
        _, description, scenarios = load_external_benchmark_matrix()
        manifest = matrix_manifest(
            "external_standard_matrix",
            description,
            scenarios[:1],
            backend="synthetic",
            control_plane_mode="multiprocess",
            router_count=4,
            cluster_count=6,
            topology_mode="sparse_overlap",
            reachable_clusters_per_router=3,
            seeds=(7, 11),
            measured_requests=64,
            warmup_requests=16,
            validation_requests=16,
            sharegpt_path="/tmp/sharegpt.json",
            rag_path=None,
            agent_path=None,
        )

        self.assertEqual(manifest["backend"], "synthetic")
        self.assertEqual(manifest["control_plane_mode"], "multiprocess")
        self.assertEqual(manifest["router_count"], 4)
        self.assertEqual(manifest["topology_mode"], "sparse_overlap")
        self.assertEqual(manifest["scenarios"][0]["source_resolution"]["sharegpt_path"], "external")


if __name__ == "__main__":
    unittest.main()
