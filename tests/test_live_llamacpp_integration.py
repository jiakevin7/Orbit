import json
import os
import shutil
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from orbit.benchmark import main as benchmark_main
from orbit.models import Request

def _default_model_path():
    return Path(__file__).resolve().parents[1] / 'models' / 'qwen2.5-3b-instruct-q4_k_m.gguf'

def _live_test_model_path():
    override = os.environ.get('ORBIT_LIVE_TEST_MODEL')
    if override:
        return Path(override).expanduser().resolve()
    default_model = _default_model_path()
    if default_model.exists():
        return default_model.resolve()
    return None

def _live_test_skip_reason():
    if os.environ.get('ORBIT_RUN_LIVE_TESTS') != '1':
        return 'set ORBIT_RUN_LIVE_TESTS=1 to enable live llama.cpp integration tests'
    return ''

def _live_test_prerequisite_error():
    if shutil.which('llama-server') is None:
        return 'llama-server executable is not available'
    model_path = _live_test_model_path()
    if model_path is None or not model_path.exists():
        return 'set ORBIT_LIVE_TEST_MODEL to a local GGUF path or place the default test model under models/'
    return None

def _reserve_contiguous_ports(count):
    for _ in range(128):
        sockets: list[socket.socket] = []
        first = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        first.bind(('127.0.0.1', 0))
        base_port = first.getsockname()[1]
        sockets.append(first)
        try:
            for offset in range(1, count):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('127.0.0.1', base_port + offset))
                sockets.append(sock)
            return (base_port, sockets)
        except OSError:
            for sock in sockets:
                sock.close()
    raise RuntimeError('unable to reserve a contiguous localhost port block')

def _build_live_requests():
    prompts = ['System:\nYou route inference traffic across data centers.\n\nUser:\nSummarize the queue state for tenant alpha.', 'System:\nYou route inference traffic across data centers.\n\nUser:\nSummarize the queue state for tenant beta.', 'System:\nYou route inference traffic across data centers.\n\nUser:\nSummarize the cache reuse state for tenant alpha.']
    requests: list[Request] = []
    for index in range(10):
        prompt = prompts[index % len(prompts)]
        requests.append(Request(request_id=f'live-req-{index:02d}', arrival_time=float(index), router_id='router-0', prefix_tokens=(index + 1,), continuation_tokens=8, prompt_prefix_text=prompt, traffic_class='synthetic', session_id='live-test-session', source_id=f'fixture-{index % len(prompts)}'))
    return requests

@unittest.skipUnless(os.environ.get('ORBIT_RUN_LIVE_TESTS') == '1', _live_test_skip_reason())
class LiveLlamaCppIntegrationTests(unittest.TestCase):

    def test_benchmark_runner_exercises_validation_and_fault_injection_live(self):
        prerequisite_error = _live_test_prerequisite_error()
        if prerequisite_error is not None:
            self.fail(prerequisite_error)
        model_path = _live_test_model_path()
        if model_path is None:
            self.fail('live integration test requires a GGUF model path')
        base_port, reservations = _reserve_contiguous_ports(2)
        for sock in reservations:
            sock.close()
        requests = _build_live_requests()
        with tempfile.TemporaryDirectory(prefix='orbit-live-integration-') as tmpdir, mock.patch('orbit.benchmark.generate_workload', return_value=requests):
            result = benchmark_main(['--backend', 'llama_cpp', '--model', str(model_path), '--policies', 'load_only', '--calibration-policy', 'load_only', '--requests', '10', '--warmup-requests', '8', '--validation-requests', '1', '--calibrate-router', '--routers', '1', '--clusters', '2', '--seeds', '7', '--record-format', 'json', '--llama-port-base', str(base_port), '--llama-threads', '2', '--llama-ctx-size', '2048', '--llama-parallel', '1', '--llama-timeout', '60', '--llama-startup-timeout', '180', '--live-arrival-scale', '0.001', '--failed-clusters', 'cluster-0', '--failure-start', '0.0', '--failure-duration', '60.0', '--retry-penalty', '0.001', '--output-dir', tmpdir])
            self.assertEqual(result, 0)
            output_dir = Path(tmpdir)
            expected_paths = [output_dir / 'manifest.json', output_dir / 'calibration.json', output_dir / 'selection.json', output_dir / 'validation_workload.json', output_dir / 'summary.json', output_dir / 'summary_by_traffic.csv', output_dir / 'summary_by_source.csv', output_dir / 'load_only_records.json']
            for path in expected_paths:
                self.assertTrue(path.exists(), f'missing artifact: {path}')
            manifest = json.loads((output_dir / 'manifest.json').read_text())
            self.assertEqual(manifest['backend'], 'llama_cpp')
            self.assertEqual(manifest['validation_requests'], 1)
            self.assertEqual(manifest['faults']['failed_clusters'], ['cluster-0'])
            calibration = json.loads((output_dir / 'calibration.json').read_text())
            self.assertEqual(calibration['record_count'], 8)
            selection = json.loads((output_dir / 'selection.json').read_text())
            self.assertIn(selection['selected_config'], {'base', 'calibrated'})
            self.assertEqual(selection['selection_metric'], 'validation_prediction_mae_with_p95_guardrail')
            self.assertIn('base_validation_error', selection)
            self.assertIn('calibrated_validation_error', selection)
            workload = json.loads((output_dir / 'workload.json').read_text())
            self.assertEqual(len(workload), 10)
            self.assertTrue(all((entry['prefix_token_source'] == 'llama_cpp' for entry in workload)))
            summary = json.loads((output_dir / 'summary.json').read_text())
            self.assertGreaterEqual(summary['load_only']['failover_count'], 1)
            records = json.loads((output_dir / 'load_only_records.json').read_text())
            self.assertEqual(len(records), 1)
            self.assertTrue(records[0]['had_failover'])
            self.assertEqual(records[0]['attempt_count'], 2)
if __name__ == '__main__':
    unittest.main()
