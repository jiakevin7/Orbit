import unittest
from unittest import mock
from orbit.cluster import ClusterConfig
from orbit.llamacpp import (
    LlamaCppClient,
    LlamaCppCluster,
    LlamaCppClusterConfig,
    LlamaCppResult,
)
from orbit.models import Request


class _FakeResponse:
    def __init__(self, lines=None, status=200, body=b""):
        self._lines = lines or []
        self.status = status
        self._body = body

    def __iter__(self):
        return iter(self._lines)

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class LlamaCppClientTests(unittest.TestCase):
    def test_client_waits_for_health_and_measures_stream_ttft(self):
        client = LlamaCppClient(base_url="http://127.0.0.1:8081", request_timeout=5.0)
        responses = [
            _FakeResponse(status=200),
            _FakeResponse(
                lines=[
                    b'data: {"content":"hello","stop":false}\n',
                    b'data: {"content":"","stop":true}\n',
                    b"data: [DONE]\n",
                ],
                status=200,
            ),
        ]
        with mock.patch("orbit.llamacpp.request.urlopen", side_effect=responses):
            client.wait_until_ready(timeout=1.0)
            result = client.complete("prompt", max_tokens=4)
        self.assertGreaterEqual(result.ttft, 0.0)
        self.assertGreaterEqual(result.total_latency, 0.0)
        self.assertLessEqual(result.ttft, result.total_latency)
        self.assertAlmostEqual(result.prompt_eval_latency, result.ttft)

    def test_client_reads_prompt_eval_latency_from_final_timings(self):
        client = LlamaCppClient(base_url="http://127.0.0.1:8081", request_timeout=5.0)
        responses = [
            _FakeResponse(
                lines=[
                    b'data: {"content":"hello","stop":false}\n',
                    b'data: {"content":"","stop":true,"timings":{"prompt_ms":123.0,"predicted_ms":456.0}}\n',
                    b"data: [DONE]\n",
                ],
                status=200,
            )
        ]
        with (
            mock.patch("orbit.llamacpp.request.urlopen", side_effect=responses),
            mock.patch(
                "orbit.llamacpp.time.perf_counter",
                side_effect=(0.0, 0.2, 0.3, 0.4, 0.5),
            ),
        ):
            result = client.complete("prompt", max_tokens=4)
        self.assertAlmostEqual(result.prompt_eval_latency, 0.123)
        self.assertAlmostEqual(result.processing_started_latency, 0.0)
        self.assertAlmostEqual(result.cache_ready_latency, 0.123)

    def test_client_reads_processing_and_cache_ready_from_prompt_progress(self):
        client = LlamaCppClient(base_url="http://127.0.0.1:8081", request_timeout=5.0)
        responses = [
            _FakeResponse(
                lines=[
                    b'data: {"content":"","stop":false,"prompt_progress":{"total":16,"processed":0,"time_ms":0}}\n',
                    b'data: {"content":"","stop":false,"prompt_progress":{"total":16,"processed":16,"time_ms":23.0}}\n',
                    b'data: {"content":"hello","stop":false}\n',
                    b'data: {"content":"","stop":true}\n',
                    b"data: [DONE]\n",
                ],
                status=200,
            )
        ]
        with (
            mock.patch("orbit.llamacpp.request.urlopen", side_effect=responses),
            mock.patch(
                "orbit.llamacpp.time.perf_counter",
                side_effect=(0.0, 0.01, 0.04, 0.06, 0.08, 0.09, 0.1),
            ),
        ):
            result = client.complete("prompt", max_tokens=4)
        self.assertAlmostEqual(result.ttft, 0.06)
        self.assertAlmostEqual(result.processing_started_latency, 0.01)
        self.assertAlmostEqual(result.cache_ready_latency, 0.04)
        self.assertAlmostEqual(result.prompt_eval_latency, 0.023)

    def test_client_can_read_tokenize_and_slots_endpoints(self):
        client = LlamaCppClient(base_url="http://127.0.0.1:8081", request_timeout=5.0)
        responses = [
            _FakeResponse(body=b'{"tokens":[10,20,30]}', status=200),
            _FakeResponse(
                body=b'[{"id":0,"n_ctx":2048,"is_processing":true,"id_task":17},{"id":1,"n_ctx":2048,"is_processing":false}]',
                status=200,
            ),
        ]
        with mock.patch("orbit.llamacpp.request.urlopen", side_effect=responses):
            tokens = client.tokenize("prompt")
            slots = client.slots()
        self.assertEqual(tokens, (10, 20, 30))
        self.assertEqual(len(slots), 2)
        self.assertEqual(slots[0].slot_id, 0)
        self.assertEqual(slots[0].task_id, 17)
        self.assertTrue(slots[0].is_processing)
        self.assertFalse(slots[1].is_processing)

    def test_cluster_prepares_requests_and_prefers_live_slot_depth(self):
        cluster = LlamaCppCluster(
            cluster_id="cluster-a",
            cluster_config=ClusterConfig(),
            backend_config=LlamaCppClusterConfig(
                model_path="/tmp/placeholder.gguf",
                manage_server=False,
                parallel=2,
                prompt_token_cap=256,
            ),
        )
        cluster._started = True
        request_obj = Request(
            request_id="req-0",
            arrival_time=0.0,
            router_id="router-a",
            prefix_tokens=(1, 2, 3),
            prompt_prefix_text="System:\nRespond briefly.\n\nUser:\nHello there.",
        )
        with mock.patch.object(
            cluster._client, "tokenize", return_value=(101, 102, 103)
        ):
            prepared = cluster.prepare_requests([request_obj])
        self.assertEqual(prepared[0].prefix_tokens, (101, 102, 103))
        self.assertEqual(prepared[0].prefix_token_source, "llama_cpp")
        with mock.patch.object(
            cluster._client,
            "slots",
            return_value=(
                mock.Mock(is_processing=True),
                mock.Mock(is_processing=False),
            ),
        ):
            self.assertEqual(cluster.queue_depth(now=3.5), 1)

    def test_cluster_prepare_requests_truncates_to_real_prompt_token_budget(self):
        cluster = LlamaCppCluster(
            cluster_id="cluster-a",
            cluster_config=ClusterConfig(),
            backend_config=LlamaCppClusterConfig(
                model_path="/tmp/placeholder.gguf",
                manage_server=False,
                parallel=1,
                ctx_size=256,
                prompt_token_cap=64,
            ),
        )
        cluster._started = True
        request_obj = Request(
            request_id="req-0",
            arrival_time=0.0,
            router_id="router-a",
            prefix_tokens=(1, 2, 3),
            prompt_prefix_text="User:\n" + "漢" * 200 + "\n\nAssistant:",
            continuation_tokens=16,
        )

        def fake_tokenize(text):
            if len(text) > 80:
                return tuple(range(200))
            return tuple(range(max(1, min(64, len(text) // 2))))

        with mock.patch.object(cluster._client, "tokenize", side_effect=fake_tokenize):
            prepared = cluster.prepare_requests([request_obj])
        self.assertEqual(prepared[0].prefix_token_source, "llama_cpp")
        self.assertLessEqual(len(prepared[0].prefix_tokens), 64)
        self.assertLess(len(prepared[0].prompt_text), len(request_obj.prompt_text))

    def test_cluster_makes_prefix_reusable_after_prompt_eval(self):
        cluster = LlamaCppCluster(
            cluster_id="cluster-a",
            cluster_config=ClusterConfig(),
            backend_config=LlamaCppClusterConfig(
                model_path="/tmp/placeholder.gguf", manage_server=False, parallel=1
            ),
        )
        cluster._started = True
        first_request = Request(
            request_id="req-0",
            arrival_time=0.0,
            router_id="router-a",
            prefix_tokens=(7, 8, 9),
            prompt_prefix_text="System:\nRespond briefly.\n\nUser:\nHello there.",
            prefix_token_source="llama_cpp",
        )
        second_request = Request(
            request_id="req-1",
            arrival_time=0.3,
            router_id="router-a",
            prefix_tokens=(7, 8, 9),
            prompt_prefix_text="System:\nRespond briefly.\n\nUser:\nHello there.",
            prefix_token_source="llama_cpp",
        )
        with (
            mock.patch.object(
                cluster._client,
                "complete",
                return_value=LlamaCppResult(
                    total_latency=1.0,
                    ttft=0.4,
                    prompt_eval_latency=0.2,
                    processing_started_latency=0.0,
                    cache_ready_latency=0.2,
                ),
            ),
            mock.patch.object(
                cluster._client, "slots", side_effect=RuntimeError("offline")
            ),
        ):
            first_execution = cluster.execute(first_request)
            self.assertAlmostEqual(first_execution.cache_ready_at, 0.2)
            self.assertGreater(first_execution.finished_at, second_request.arrival_time)
            second_execution = cluster.execute(second_request)
        self.assertEqual(second_execution.true_reusable_tokens, 3)


if __name__ == "__main__":
    unittest.main()
