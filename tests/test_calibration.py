import unittest
from orbit.calibration import fit_router_config
from orbit.models import ExecutionRecord
from orbit.router import RouterConfig

class CalibrationTests(unittest.TestCase):

    def test_fit_router_config_can_calibrate_reuse_before_latency(self):
        base_config = RouterConfig()
        records: list[ExecutionRecord] = []
        for index in range(12):
            raw_estimate = 8 if index % 2 == 0 else 32
            actual_reuse = 12 if index % 2 == 0 else 48
            input_length = actual_reuse + 24
            records.append(ExecutionRecord(request_id=f'req-{index}', policy='summary', router_id='router-0', cluster_id='cluster-0', arrival_time=float(index), started_at=float(index), finished_at=float(index) + 1.0, predicted_latency=1.0, actual_latency=1.0, actual_ttft=0.5, estimated_reusable_tokens=raw_estimate, actual_reusable_tokens=actual_reuse, estimated_remaining_prefill_tokens=input_length - raw_estimate, input_length=input_length, continuation_tokens=4, reuse_fraction=actual_reuse / input_length, network_cost=0.0, queue_delay=0.0, queue_depth_before=0, route_queue_depth=0, metadata_age=0.0, uncertainty_gap=0, missing_summary=False, initial_cluster_id='cluster-0', had_failover=False, failover_delay=0.0, attempt_count=1, service_time=1.0, raw_estimated_reusable_tokens=raw_estimate, summary_matched_levels=2, hotset_matched_levels=1))
        calibrated_config, calibration = fit_router_config(records, base_config)
        self.assertTrue(calibration.applied)
        self.assertGreater(calibrated_config.reuse_estimate_scale, 1.0)
        self.assertLess(calibration.reuse_mae, calibration.reuse_baseline_mae)

    def test_fit_router_config_recovers_latency_coefficients_from_records(self):
        target_config = RouterConfig(fixed_overhead=0.3, prefill_cost_per_token=0.02, decode_cost_per_token=0.05, queue_depth_penalty=0.4, stale_penalty_per_second=0.1, uncertainty_penalty_per_token=0.03, missing_summary_penalty=1.2)
        base_config = RouterConfig()
        feature_rows = [(10, 8, 0, 0.0, 0, False), (20, 4, 1, 0.5, 8, False), (0, 16, 2, 1.0, 0, False), (32, 6, 0, 0.0, 16, False), (8, 12, 3, 0.0, 4, False), (24, 10, 1, 2.0, 12, True), (4, 20, 2, 1.5, 6, False), (12, 14, 0, 0.5, 10, True), (28, 5, 4, 0.25, 2, False), (16, 9, 2, 0.75, 14, False), (6, 18, 5, 1.25, 3, True), (30, 7, 1, 0.1, 9, False)]
        records: list[ExecutionRecord] = []
        for index, (remaining_prefill, continuation, queue_depth, metadata_age, uncertainty_gap, missing_summary) in enumerate(feature_rows):
            residual_latency = target_config.fixed_overhead + remaining_prefill * target_config.prefill_cost_per_token + continuation * target_config.decode_cost_per_token + queue_depth * target_config.queue_depth_penalty + metadata_age * target_config.stale_penalty_per_second + uncertainty_gap * target_config.uncertainty_penalty_per_token + (target_config.missing_summary_penalty if missing_summary else 0.0)
            records.append(ExecutionRecord(request_id=f'req-{index}', policy='summary', router_id='router-0', cluster_id='cluster-0', arrival_time=float(index), started_at=float(index), finished_at=float(index) + residual_latency + 0.01, predicted_latency=10.0, actual_latency=0.005 + residual_latency, actual_ttft=0.005 + residual_latency / 2.0, estimated_reusable_tokens=64, actual_reusable_tokens=64, estimated_remaining_prefill_tokens=remaining_prefill, input_length=64 + remaining_prefill, continuation_tokens=continuation, reuse_fraction=0.5, network_cost=0.005, queue_delay=queue_depth * target_config.queue_depth_penalty, queue_depth_before=queue_depth, route_queue_depth=queue_depth, metadata_age=metadata_age, uncertainty_gap=uncertainty_gap, missing_summary=missing_summary, initial_cluster_id='cluster-0', had_failover=False, failover_delay=0.0, attempt_count=1, service_time=residual_latency))
        calibrated_config, calibration = fit_router_config(records, base_config)
        self.assertAlmostEqual(calibrated_config.fixed_overhead, target_config.fixed_overhead, places=4)
        self.assertAlmostEqual(calibrated_config.prefill_cost_per_token, target_config.prefill_cost_per_token, places=4)
        self.assertAlmostEqual(calibrated_config.decode_cost_per_token, target_config.decode_cost_per_token, places=4)
        self.assertAlmostEqual(calibrated_config.queue_depth_penalty, target_config.queue_depth_penalty, places=4)
        self.assertAlmostEqual(calibrated_config.stale_penalty_per_second, target_config.stale_penalty_per_second, places=4)
        self.assertAlmostEqual(calibrated_config.uncertainty_penalty_per_token, target_config.uncertainty_penalty_per_token, places=4)
        self.assertAlmostEqual(calibrated_config.missing_summary_penalty, target_config.missing_summary_penalty, places=4)
        self.assertLess(calibration.mae, calibration.baseline_mae)
        self.assertEqual(calibration.record_count, len(records))
        self.assertTrue(calibration.applied)

    def test_fit_router_config_skips_when_too_few_records_are_available(self):
        base_config = RouterConfig()
        records = [ExecutionRecord(request_id='req-0', policy='summary', router_id='router-0', cluster_id='cluster-0', arrival_time=0.0, started_at=0.0, finished_at=1.0, predicted_latency=1.0, actual_latency=1.0, actual_ttft=0.5, estimated_reusable_tokens=0, actual_reusable_tokens=0, estimated_remaining_prefill_tokens=10, input_length=10, continuation_tokens=4, reuse_fraction=0.0, network_cost=0.0, queue_delay=0.0, queue_depth_before=0, route_queue_depth=0, metadata_age=0.0, uncertainty_gap=0, missing_summary=False, initial_cluster_id='cluster-0', had_failover=False, failover_delay=0.0, attempt_count=1, service_time=1.0) for _ in range(3)]
        calibrated_config, calibration = fit_router_config(records, base_config)
        self.assertEqual(calibrated_config, base_config)
        self.assertFalse(calibration.applied)
        self.assertEqual(calibration.reason, 'need_at_least_8_records')

    def test_fit_router_config_uses_one_global_policy(self):
        base_config = RouterConfig()
        records: list[ExecutionRecord] = []
        for index in range(10):
            records.append(ExecutionRecord(request_id=f'cluster-0-{index}', policy='summary', router_id='router-0', cluster_id='cluster-0', arrival_time=float(index), started_at=float(index), finished_at=float(index) + 0.5, predicted_latency=10.0, actual_latency=0.1 + 0.2 + index * 0.001, actual_ttft=0.2, estimated_reusable_tokens=32, actual_reusable_tokens=32, estimated_remaining_prefill_tokens=5, input_length=37, continuation_tokens=2, reuse_fraction=0.5, network_cost=0.1, queue_delay=0.0, queue_depth_before=0, route_queue_depth=0, metadata_age=0.0, uncertainty_gap=0, missing_summary=False, initial_cluster_id='cluster-0', had_failover=False, failover_delay=0.0, attempt_count=1, service_time=0.4))
        for index in range(10):
            records.append(ExecutionRecord(request_id=f'cluster-1-{index}', policy='summary', router_id='router-0', cluster_id='cluster-1', arrival_time=float(index), started_at=float(index), finished_at=float(index) + 1.1, predicted_latency=10.0, actual_latency=0.1 + 0.9 + index * 0.001, actual_ttft=0.3, estimated_reusable_tokens=32, actual_reusable_tokens=32, estimated_remaining_prefill_tokens=5, input_length=37, continuation_tokens=2, reuse_fraction=0.5, network_cost=0.1, queue_delay=0.0, queue_depth_before=0, route_queue_depth=0, metadata_age=0.0, uncertainty_gap=0, missing_summary=False, initial_cluster_id='cluster-1', had_failover=False, failover_delay=0.0, attempt_count=1, service_time=1.0))
        calibrated_config, calibration = fit_router_config(records, base_config)
        self.assertTrue(calibration.applied)
        self.assertEqual(calibration.scope, 'global')
        self.assertEqual(calibration.applied_clusters, ())
        self.assertEqual(calibration.cluster_calibrations, {})
        self.assertNotEqual(calibrated_config, base_config)
if __name__ == '__main__':
    unittest.main()
