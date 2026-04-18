from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from orbit.models import ExecutionRecord, SimulationMetrics
from orbit.reporting import (
    execution_records_as_dicts,
    metrics_rows_by_policy,
    summarize_execution_records,
    write_execution_records_csv,
    write_json,
    write_rows_csv,
)


class ReportingTests(unittest.TestCase):
    def test_execution_record_csv_export_writes_expected_columns(self) -> None:
        record = ExecutionRecord(
            request_id="req-00001",
            policy="summary",
            router_id="router-0",
            cluster_id="cluster-0",
            arrival_time=1.5,
            started_at=2.0,
            finished_at=3.0,
            predicted_latency=1.2,
            actual_latency=1.5,
            actual_ttft=0.75,
            estimated_reusable_tokens=64,
            actual_reusable_tokens=96,
            estimated_remaining_prefill_tokens=32,
            input_length=128,
            continuation_tokens=16,
            reuse_fraction=0.5,
            network_cost=10.0,
            queue_delay=0.5,
            queue_depth_before=1,
            route_queue_depth=1,
            metadata_age=0.25,
            uncertainty_gap=16,
            missing_summary=False,
            initial_cluster_id="cluster-0",
            had_failover=False,
            failover_delay=0.0,
            attempt_count=1,
            service_time=1.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "records.csv"
            write_execution_records_csv(path, [record])
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["request_id"], "req-00001")
        self.assertEqual(rows[0]["actual_ttft"], "0.75")
        self.assertEqual(rows[0]["actual_latency"], "1.5")

    def test_metrics_rows_flatten_cluster_counts(self) -> None:
        metrics_by_policy = {
            "summary": SimulationMetrics(
                policy="summary",
                request_count=10,
                mean_reusable_prefix=12.0,
                mean_reuse_fraction=0.2,
                ttft_p50=1.0,
                ttft_p95=2.0,
                latency_p50=3.0,
                latency_p95=4.0,
                control_plane_bytes=100,
                summary_memory_bytes=200,
                load_stddev=0.5,
                failover_count=1,
                failover_rate=0.1,
                cluster_request_counts={"cluster-0": 6, "cluster-1": 4},
            )
        }

        rows = metrics_rows_by_policy(metrics_by_policy)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["cluster_requests_cluster-0"], 6)
        self.assertEqual(rows[0]["cluster_requests_cluster-1"], 4)

    def test_json_and_generic_csv_writers_emit_files(self) -> None:
        rows = [{"policy": "summary", "ttft_p50": 1.2}]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            json_path = root / "summary.json"
            csv_path = root / "summary.csv"
            write_json(json_path, {"records": execution_records_as_dicts([])})
            write_rows_csv(csv_path, rows)

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(payload, {"records": []})
        self.assertEqual(csv_rows[0]["policy"], "summary")

    def test_summarize_execution_records_groups_by_traffic_class(self) -> None:
        records = [
            ExecutionRecord(
                request_id="req-1",
                policy="summary",
                router_id="router-0",
                cluster_id="cluster-0",
                arrival_time=0.0,
                started_at=0.0,
                finished_at=1.0,
                predicted_latency=1.0,
                actual_latency=1.0,
                actual_ttft=0.5,
                estimated_reusable_tokens=8,
                actual_reusable_tokens=8,
                estimated_remaining_prefill_tokens=0,
                input_length=8,
                continuation_tokens=4,
                reuse_fraction=1.0,
                network_cost=0.0,
                queue_delay=0.0,
                queue_depth_before=0,
                route_queue_depth=0,
                metadata_age=0.0,
                uncertainty_gap=0,
                missing_summary=False,
                initial_cluster_id="cluster-0",
                had_failover=False,
                failover_delay=0.0,
                attempt_count=1,
                service_time=1.0,
                traffic_class="rag",
                source_id="doc-1",
            ),
            ExecutionRecord(
                request_id="req-2",
                policy="summary",
                router_id="router-0",
                cluster_id="cluster-1",
                arrival_time=0.0,
                started_at=0.0,
                finished_at=2.0,
                predicted_latency=2.0,
                actual_latency=2.0,
                actual_ttft=1.0,
                estimated_reusable_tokens=0,
                actual_reusable_tokens=0,
                estimated_remaining_prefill_tokens=8,
                input_length=8,
                continuation_tokens=4,
                reuse_fraction=0.0,
                network_cost=0.0,
                queue_delay=0.0,
                queue_depth_before=0,
                route_queue_depth=0,
                metadata_age=0.0,
                uncertainty_gap=0,
                missing_summary=False,
                initial_cluster_id="cluster-1",
                had_failover=True,
                failover_delay=1.0,
                attempt_count=2,
                service_time=2.0,
                traffic_class="rag",
                source_id="doc-1",
            ),
        ]

        rows = summarize_execution_records(records, "summary", group_field="traffic_class")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["traffic_class"], "rag")
        self.assertEqual(rows[0]["request_count"], 2)
        self.assertEqual(rows[0]["failover_count"], 1)
