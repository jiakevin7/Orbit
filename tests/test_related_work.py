from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from orbit.models import Request
from orbit.related_work import (
    ExternalRequestRecord,
    ExternalSystemTarget,
    OpenAICompatibleClient,
    cap_request_continuations,
    load_related_work_targets,
    prompt_text_to_messages,
    summarize_external_records,
)


class _FakeResponse:
    def __init__(
        self,
        *,
        lines: list[bytes] | None = None,
        status: int = 200,
        body: bytes = b"",
    ) -> None:
        self._lines = lines or []
        self.status = status
        self._body = body

    def __iter__(self):
        return iter(self._lines)

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        return False


class RelatedWorkTests(unittest.TestCase):
    def test_prompt_text_to_messages_parses_structured_prompt(self) -> None:
        prompt = (
            "System:\nYou are a routing assistant.\n\n"
            "User:\nSummarize queue depth.\n\n"
            "Assistant:\nSure.\n\n"
            "User:\nNow compare clusters.\n\n"
            "Assistant:"
        )

        messages = prompt_text_to_messages(prompt)

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "You are a routing assistant."},
                {"role": "user", "content": "Summarize queue depth."},
                {"role": "assistant", "content": "Sure."},
                {"role": "user", "content": "Now compare clusters."},
            ],
        )

    def test_openai_chat_client_measures_stream_ttft(self) -> None:
        target = ExternalSystemTarget(
            name="semantic-router",
            family="vllm_semantic_router",
            base_url="http://127.0.0.1:8080",
            model="MoM",
            request_format="chat",
        )
        client = OpenAICompatibleClient(target)
        request_obj = Request(
            request_id="req-0",
            arrival_time=0.0,
            router_id="router-0",
            prefix_tokens=(1, 2, 3),
            continuation_tokens=8,
            prompt_prefix_text="System:\nYou are helpful.\n\nUser:\nHello.\n\nAssistant:",
        )

        with mock.patch(
            "orbit.related_work.request.urlopen",
            return_value=_FakeResponse(
                lines=[
                    (
                        b'data: {"choices":[{"delta":{"role":"assistant","content":"Hel"},'
                        b'"index":0,"finish_reason":null}]}\n'
                    ),
                    b'data: {"choices":[{"delta":{"content":"lo"},"index":0,"finish_reason":null}]}\n',
                    b"data: [DONE]\n",
                ],
            ),
        ), mock.patch(
            "orbit.related_work.time.perf_counter",
            side_effect=(0.0, 0.05, 0.08, 0.09, 0.10),
        ):
            result = client.complete(request_obj)

        self.assertAlmostEqual(result.ttft, 0.05)
        self.assertAlmostEqual(result.total_latency, 0.10)
        self.assertEqual(result.status_code, 200)

    def test_openai_completion_client_detects_text_stream(self) -> None:
        target = ExternalSystemTarget(
            name="preble",
            family="preble",
            base_url="http://127.0.0.1:8081",
            model="default",
            request_format="completion",
        )
        client = OpenAICompatibleClient(target)
        request_obj = Request(
            request_id="req-1",
            arrival_time=0.0,
            router_id="router-0",
            prefix_tokens=(1, 2, 3),
            continuation_tokens=4,
            prompt_prefix_text="System:\nYou are helpful.\n\nUser:\nHello.",
        )

        with mock.patch(
            "orbit.related_work.request.urlopen",
            return_value=_FakeResponse(
                lines=[
                    b'data: {"choices":[{"text":"Hi","index":0,"finish_reason":null}]}\n',
                    b"data: [DONE]\n",
                ],
            ),
        ), mock.patch(
            "orbit.related_work.time.perf_counter",
            side_effect=(0.0, 0.03, 0.035, 0.04),
        ):
            result = client.complete(request_obj)

        self.assertAlmostEqual(result.ttft, 0.03)
        self.assertAlmostEqual(result.total_latency, 0.04)

    def test_load_targets_and_group_external_records(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "related_work_targets.example.json"

        targets = load_related_work_targets(config_path)

        self.assertGreaterEqual(len(targets), 7)
        self.assertEqual(targets[0].name, "vllm_prefix_aware")

        records = [
            ExternalRequestRecord(
                system="sglang",
                family="sglang",
                request_id="req-0",
                arrival_time=0.0,
                started_at=0.0,
                finished_at=1.0,
                actual_ttft=0.4,
                actual_latency=1.0,
                input_length=10,
                continuation_tokens=4,
                success=True,
                status_code=200,
                traffic_class="rag",
                source_id="dataset-a",
            ),
            ExternalRequestRecord(
                system="sglang",
                family="sglang",
                request_id="req-1",
                arrival_time=1.0,
                started_at=1.0,
                finished_at=2.5,
                actual_ttft=0.6,
                actual_latency=1.5,
                input_length=12,
                continuation_tokens=4,
                success=False,
                status_code=500,
                error="server error",
                traffic_class="rag",
                source_id="dataset-a",
            ),
        ]

        grouped = summarize_external_records(records, group_field="traffic_class")

        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0]["traffic_class"], "rag")
        self.assertEqual(grouped[0]["request_count"], 2)
        self.assertEqual(grouped[0]["failure_count"], 1)

    def test_cap_request_continuations_clamps_request_lengths(self) -> None:
        requests = [
            Request(
                request_id="req-0",
                arrival_time=0.0,
                router_id="router-0",
                prefix_tokens=(1, 2, 3),
                continuation_tokens=96,
            ),
            Request(
                request_id="req-1",
                arrival_time=1.0,
                router_id="router-0",
                prefix_tokens=(4, 5, 6),
                continuation_tokens=8,
            ),
        ]

        clamped = cap_request_continuations(requests, 16)

        self.assertEqual([request.continuation_tokens for request in clamped], [16, 8])
        self.assertEqual([request.request_id for request in clamped], ["req-0", "req-1"])


if __name__ == "__main__":
    unittest.main()
