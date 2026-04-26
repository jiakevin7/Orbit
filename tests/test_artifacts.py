import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from orbit.models import ExecutionRecord, SimulationMetrics
from orbit.png_plots import generate_run_plots
from orbit.reporting import execution_records_as_dicts, metrics_rows_by_policy, summarize_execution_records, write_execution_records_csv, write_json, write_rows_csv

class ArtifactTests(unittest.TestCase):

    def test_reporting_csv_json_and_grouped_summaries(self):
        records = [ExecutionRecord(request_id='req-1', policy='summary', router_id='router-0', cluster_id='cluster-0', arrival_time=0.0, started_at=0.0, finished_at=1.0, predicted_latency=1.0, actual_latency=1.0, actual_ttft=0.5, estimated_reusable_tokens=8, actual_reusable_tokens=8, estimated_remaining_prefill_tokens=0, input_length=8, continuation_tokens=4, reuse_fraction=1.0, network_cost=0.0, queue_delay=0.0, queue_depth_before=0, route_queue_depth=0, metadata_age=0.0, uncertainty_gap=0, missing_summary=False, initial_cluster_id='cluster-0', had_failover=False, failover_delay=0.0, attempt_count=1, service_time=1.0, traffic_class='rag', source_id='doc-1'), ExecutionRecord(request_id='req-2', policy='summary', router_id='router-0', cluster_id='cluster-1', arrival_time=0.0, started_at=0.0, finished_at=2.0, predicted_latency=2.0, actual_latency=2.0, actual_ttft=1.0, estimated_reusable_tokens=0, actual_reusable_tokens=0, estimated_remaining_prefill_tokens=8, input_length=8, continuation_tokens=4, reuse_fraction=0.0, network_cost=0.0, queue_delay=0.0, queue_depth_before=0, route_queue_depth=0, metadata_age=0.0, uncertainty_gap=0, missing_summary=False, initial_cluster_id='cluster-1', had_failover=True, failover_delay=1.0, attempt_count=2, service_time=2.0, traffic_class='rag', source_id='doc-1')]
        metrics_by_policy = {'summary': SimulationMetrics(policy='summary', request_count=2, mean_reusable_prefix=4.0, mean_reuse_fraction=0.5, ttft_p50=0.75, ttft_p95=1.0, latency_p50=1.5, latency_p95=2.0, control_plane_bytes=100, summary_memory_bytes=200, load_stddev=0.5, failover_count=1, failover_rate=0.5, cluster_request_counts={'cluster-0': 1, 'cluster-1': 1})}
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            records_path = root / 'records.csv'
            json_path = root / 'records.json'
            summary_path = root / 'summary.csv'
            write_execution_records_csv(records_path, records)
            write_json(json_path, {'records': execution_records_as_dicts(records)})
            write_rows_csv(summary_path, metrics_rows_by_policy(metrics_by_policy))
            with records_path.open('r', encoding='utf-8', newline='') as handle:
                record_rows = list(csv.DictReader(handle))
            with summary_path.open('r', encoding='utf-8', newline='') as handle:
                summary_rows = list(csv.DictReader(handle))
            payload = json.loads(json_path.read_text(encoding='utf-8'))
        grouped = summarize_execution_records(records, 'summary', group_field='traffic_class')
        self.assertEqual(record_rows[0]['request_id'], 'req-1')
        self.assertEqual(record_rows[1]['actual_latency'], '2.0')
        self.assertEqual(payload['records'][0]['actual_ttft'], 0.5)
        self.assertEqual(summary_rows[0]['cluster_requests_cluster-0'], '1')
        self.assertEqual(grouped[0]['traffic_class'], 'rag')
        self.assertEqual(grouped[0]['failover_count'], 1)

    def test_png_generation_for_run_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / 'manifest.json').write_text(json.dumps({'backend': 'synthetic', 'policies': ['summary', 'random']}), encoding='utf-8')
            (run_dir / 'summary.json').write_text(json.dumps({'summary': {'policy': 'summary', 'request_count': 2, 'mean_reusable_prefix': 18.0, 'mean_reuse_fraction': 0.25, 'ttft_p50': 1.2, 'ttft_p95': 1.8, 'latency_p50': 2.2, 'latency_p95': 3.0, 'failover_rate': 0.25}}), encoding='utf-8')
            (run_dir / 'summary_records.json').write_text(json.dumps([{'request_id': 'summary-0', 'policy': 'summary', 'cluster_id': 'cluster-0', 'predicted_latency': 1.5, 'actual_latency': 1.8, 'actual_ttft': 0.8, 'actual_reusable_tokens': 16, 'reuse_fraction': 0.6, 'had_failover': False}, {'request_id': 'summary-1', 'policy': 'summary', 'cluster_id': 'cluster-1', 'predicted_latency': 1.9, 'actual_latency': 2.1, 'actual_ttft': 1.0, 'actual_reusable_tokens': 20, 'reuse_fraction': 0.5, 'had_failover': True}]), encoding='utf-8')
            (run_dir / 'random_records.json').write_text(json.dumps([{'request_id': 'random-0', 'policy': 'random', 'cluster_id': 'cluster-1', 'predicted_latency': 2.4, 'actual_latency': 2.6, 'actual_ttft': 1.4, 'reuse_fraction': 0.2, 'had_failover': False}]), encoding='utf-8')
            if importlib.util.find_spec('seaborn') is not None:
                created = set(generate_run_plots(run_dir))
                for name in ('ttft_cdf.png', 'latency_cdf.png', 'ttft_by_policy.png', 'latency_by_policy.png', 'reuse_latency_tradeoff.png'):
                    self.assertIn((run_dir / 'plots' / name).resolve(), created)
if __name__ == '__main__':
    unittest.main()
