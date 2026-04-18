from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from orbit.workload import (
    AGENT_TRAFFIC_CLASS,
    BURSTY_TRAFFIC_CLASS,
    CHAT_TRAFFIC_CLASS,
    RAG_TRAFFIC_CLASS,
    WorkloadConfig,
    generate_workload,
    load_agent_conversations,
    load_rag_examples,
    load_sharegpt_conversations,
    text_to_routing_tokens,
)


class WorkloadTests(unittest.TestCase):
    def test_generated_requests_include_structured_prompt_text(self) -> None:
        requests = generate_workload(
            WorkloadConfig(
                num_requests=3,
                prefix_length_choices=(32,),
                overlap_length_choices=(16,),
                seed=5,
            )
        )

        self.assertEqual(len(requests), 3)
        for request in requests:
            self.assertIsNotNone(request.prompt_prefix_text)
            self.assertIn("System:", request.prompt_prefix_text or "")
            self.assertIn("User:", request.prompt_prefix_text or "")
            self.assertEqual(request.traffic_class, "synthetic")
            self.assertEqual(
                request.prefix_tokens,
                text_to_routing_tokens(request.prompt_prefix_text or ""),
            )

    def test_mixed_realistic_workload_covers_all_supported_traffic_classes(self) -> None:
        sharegpt_payload = [
            {
                "id": "conv-1",
                "system": "You are a concise operations assistant.",
                "conversations": [
                    {"from": "human", "value": "Summarize why router metadata drift matters."},
                    {"from": "gpt", "value": "It can send requests to clusters with stale overlap estimates."},
                    {"from": "human", "value": "Give two concrete checks before rerouting."},
                    {"from": "gpt", "value": "Compare summary age and current busy slots."},
                ],
            },
            {
                "id": "conv-2",
                "conversations": [
                    {"from": "human", "value": "A customer package crossed regions and is now late. What should support gather first?"},
                    {"from": "gpt", "value": "Collect the order id, SLA promise, and last carrier scan."},
                    {"from": "human", "value": "Write a calm reply with one next step."},
                    {"from": "gpt", "value": "Acknowledge the delay and confirm the next tracking update."},
                ],
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "sharegpt.json"
            dataset_path.write_text(json.dumps(sharegpt_payload), encoding="utf-8")

            conversations = load_sharegpt_conversations(str(dataset_path))
            requests = generate_workload(
                WorkloadConfig(
                    num_requests=10,
                    workload_kind="mixed_realistic",
                    sharegpt_path=str(dataset_path),
                    router_ids=("router-a", "router-b"),
                    seed=9,
                )
            )

        self.assertEqual(len(conversations), 2)
        self.assertEqual(len(requests), 10)

        traffic_counts = Counter(request.traffic_class for request in requests)
        self.assertGreaterEqual(traffic_counts[CHAT_TRAFFIC_CLASS], 1)
        self.assertGreaterEqual(traffic_counts[RAG_TRAFFIC_CLASS], 1)
        self.assertGreaterEqual(traffic_counts[AGENT_TRAFFIC_CLASS], 1)
        self.assertGreaterEqual(traffic_counts[BURSTY_TRAFFIC_CLASS], 1)

        for request in requests:
            self.assertIsNotNone(request.prompt_prefix_text)
            self.assertTrue(request.prompt_prefix_text.endswith("Assistant:"))
            self.assertEqual(
                request.prefix_tokens,
                text_to_routing_tokens(request.prompt_prefix_text or ""),
            )
            if request.traffic_class in {CHAT_TRAFFIC_CLASS, BURSTY_TRAFFIC_CLASS, AGENT_TRAFFIC_CLASS}:
                self.assertIsNotNone(request.session_id)

    def test_external_rag_and_agent_datasets_are_loaded_and_used(self) -> None:
        sharegpt_payload = [
            {
                "id": "conv-1",
                "conversations": [
                    {"from": "human", "value": "Why does metadata staleness matter?"},
                    {"from": "gpt", "value": "It can misroute overlap-heavy requests."},
                    {"from": "human", "value": "Give one mitigation."},
                    {"from": "gpt", "value": "Refresh summaries before large shifts."},
                ],
            }
        ]
        rag_payload = [
            {
                "id": "rag-1",
                "query": "Which rule applies before shifting traffic cross-region?",
                "quotes": [
                    {"docid": "doc-a", "text": "Compare prefix overlap estimates with direct cluster health."},
                    {"docid": "doc-b", "text": "Do not shift more than twenty percent of traffic at once."},
                ],
                "answers": [{"answer": "Check overlap against health and limit the first shift."}],
            }
        ]
        agent_payload = [
            {
                "id": "agent-1",
                "tools": [
                    {
                        "name": "lookup_order",
                        "description": "Look up order status by order id.",
                        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
                    }
                ],
                "messages": [
                    {"role": "system", "content": "You are a tool-using support assistant."},
                    {"role": "user", "content": "Investigate order A1183 and explain the first tool call."},
                    {"role": "assistant", "content": "I should look up the order status first."},
                    {"role": "tool", "content": "{\"tool\":\"lookup_order\",\"order_id\":\"A1183\"}"},
                    {"role": "assistant", "content": "The shipment crossed regions and missed the SLA."},
                ],
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sharegpt_path = tmp / "sharegpt.json"
            rag_path = tmp / "rag.json"
            agent_path = tmp / "agent.jsonl"
            sharegpt_path.write_text(json.dumps(sharegpt_payload), encoding="utf-8")
            rag_path.write_text(json.dumps(rag_payload), encoding="utf-8")
            agent_path.write_text(json.dumps(agent_payload[0]) + "\n", encoding="utf-8")

            rag_examples = load_rag_examples(str(rag_path))
            agent_examples = load_agent_conversations(str(agent_path))
            requests = generate_workload(
                WorkloadConfig(
                    num_requests=12,
                    workload_kind="mixed_realistic",
                    sharegpt_path=str(sharegpt_path),
                    rag_path=str(rag_path),
                    agent_path=str(agent_path),
                    router_ids=("router-a", "router-b"),
                    seed=4,
                )
            )

        self.assertEqual(len(rag_examples), 1)
        self.assertEqual(rag_examples[0].example_id, "rag-1")
        self.assertEqual(rag_examples[0].contexts[0][0], "doc-a")
        self.assertEqual(len(agent_examples), 1)
        self.assertEqual(agent_examples[0].conversation_id, "agent-1")

        rag_requests = [request for request in requests if request.traffic_class == RAG_TRAFFIC_CLASS]
        agent_requests = [request for request in requests if request.traffic_class == AGENT_TRAFFIC_CLASS]
        self.assertTrue(rag_requests)
        self.assertTrue(agent_requests)
        self.assertTrue(any(request.source_id == "rag-1" for request in rag_requests))
        self.assertTrue(any(request.source_id == "agent-1" for request in agent_requests))
        self.assertTrue(any("doc-a" in (request.prompt_prefix_text or "") for request in rag_requests))
        self.assertTrue(any("lookup_order" in (request.prompt_prefix_text or "") for request in agent_requests))

    def test_lmsys_and_financebench_shapes_load_without_manual_rewrite(self) -> None:
        lmsys_payload = [
            {
                "conversation_id": "arena-1",
                "conversation": [
                    {"role": "user", "content": "Summarize the root cause of the incident."},
                    {"role": "assistant", "content": "The failover plan used stale capacity telemetry."},
                    {"role": "user", "content": "What should we verify before the next cutover?"},
                    {"role": "assistant", "content": "Verify slot pressure, backlog, and summary freshness."},
                ],
            }
        ]
        financebench_payload = [
            {
                "financebench_id": "financebench-1",
                "question": "What was the reported capital expenditure amount?",
                "evidence": [
                    {
                        "doc_name": "ExampleCo_2024_10K",
                        "evidence_text": "Purchases of property and equipment totaled $120 million.",
                    }
                ],
                "answer": "$120 million",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            lmsys_path = tmp / "lmsys.json"
            financebench_path = tmp / "financebench.json"
            lmsys_path.write_text(json.dumps(lmsys_payload), encoding="utf-8")
            financebench_path.write_text(json.dumps(financebench_payload), encoding="utf-8")

            chat_examples = load_sharegpt_conversations(str(lmsys_path))
            rag_examples = load_rag_examples(str(financebench_path))

        self.assertEqual(len(chat_examples), 1)
        self.assertEqual(chat_examples[0].conversation_id, "arena-1")
        self.assertEqual(chat_examples[0].messages[0][0], "user")
        self.assertEqual(len(rag_examples), 1)
        self.assertEqual(rag_examples[0].contexts[0][0], "ExampleCo_2024_10K")
        self.assertIn("capital expenditure", rag_examples[0].query.lower())

    def test_bfcl_and_tau_bench_agent_shapes_load_without_custom_conversion(self) -> None:
        bfcl_payload = [
            {
                "id": "bfcl-1",
                "question": "Book a refundable flight from Detroit to Seattle next Tuesday.",
                "function": [
                    {
                        "name": "search_flights",
                        "description": "Search for available flights by origin, destination, and date.",
                        "parameters": {"type": "object", "properties": {"origin": {"type": "string"}}},
                    }
                ],
                "ground_truth": "Call search_flights with the user's route and date.",
            }
        ]
        tau_payload = [
            {
                "task_id": "tau-1",
                "task": {
                    "domain": "retail",
                    "policy": "Verify the order id before making account changes.",
                    "goal": "Help the user update the shipping address for order A-18.",
                },
                "tools": [
                    {
                        "name": "lookup_order",
                        "description": "Look up order details by order id.",
                    }
                ],
                "history": [
                    {"role": "user", "content": "My order A-18 is headed to the wrong address."},
                    {"role": "assistant", "content": "I can help with that after I verify the order."},
                    {"role": "tool", "content": "{\"tool\": \"lookup_order\", \"order_id\": \"A-18\"}"},
                    {"role": "assistant", "content": "I found the order and can update the address."},
                ],
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bfcl_path = tmp / "bfcl.json"
            tau_path = tmp / "tau.jsonl"
            bfcl_path.write_text(json.dumps(bfcl_payload), encoding="utf-8")
            tau_path.write_text(json.dumps(tau_payload[0]) + "\n", encoding="utf-8")

            bfcl_examples = load_agent_conversations(str(bfcl_path))
            tau_examples = load_agent_conversations(str(tau_path))

        self.assertEqual(len(bfcl_examples), 1)
        self.assertEqual(bfcl_examples[0].conversation_id, "bfcl-1")
        self.assertTrue(any("search_flights" in message for _, message in bfcl_examples[0].messages))
        self.assertEqual(len(tau_examples), 1)
        self.assertEqual(tau_examples[0].conversation_id, "tau-1")
        self.assertTrue(any("retail" in message.lower() for _, message in tau_examples[0].messages))
        self.assertTrue(any("lookup_order" in message for _, message in tau_examples[0].messages))


if __name__ == "__main__":
    unittest.main()
