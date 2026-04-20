from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .models import Request

CHAT_TRAFFIC_CLASS = "sharegpt_chat"
RAG_TRAFFIC_CLASS = "rag"
AGENT_TRAFFIC_CLASS = "agent"
BURSTY_TRAFFIC_CLASS = "bursty"

_ROLE_LABELS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool",
}


@dataclass(frozen=True)
class WorkloadConfig:
    num_requests: int = 200
    num_templates: int = 8
    router_ids: tuple[str, ...] = ("router-a", "router-b")
    prefix_length_choices: tuple[int, ...] = (128, 192, 256)
    overlap_length_choices: tuple[int, ...] = (0, 64, 128, 192)
    continuation_token_range: tuple[int, int] = (8, 24)
    mean_interarrival: float = 24.0
    shared_template_probability: float = 0.85
    template_popularity_alpha: float = 1.2
    vocab_size: int = 50_000
    seed: int = 7
    workload_kind: str = "synthetic"
    sharegpt_path: str | None = None
    sharegpt_sample_limit: int = 2_000
    rag_path: str | None = None
    rag_sample_limit: int = 2_000
    agent_path: str | None = None
    agent_sample_limit: int = 2_000
    traffic_mix_chat: float = 0.4375
    traffic_mix_rag: float = 0.3125
    traffic_mix_agent: float = 0.25
    traffic_mix_bursty: float = 0.0
    session_affinity_probability: float = 0.85
    burst_size_choices: tuple[int, ...] = (2, 3)
    burst_interarrival_ratio: float = 0.05
    traffic_burst_probability: float = 0.0
    dataset_continuation_floor: int = 8
    dataset_continuation_cap: int = 96
    prompt_prefix_token_cap: int | None = 4096


@dataclass(frozen=True)
class ConversationExample:
    conversation_id: str
    messages: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RagExample:
    example_id: str
    query: str
    contexts: tuple[tuple[str, str], ...]
    answer: str | None = None


def generate_workload(config: WorkloadConfig) -> list[Request]:
    if config.workload_kind == "synthetic":
        return _generate_synthetic_workload(config)
    if config.workload_kind == "mixed_realistic":
        return _generate_mixed_realistic_workload(config)
    raise ValueError(f"unsupported workload kind: {config.workload_kind}")


def load_sharegpt_conversations(
    path: str | None,
    sample_limit: int = 2_000,
) -> list[ConversationExample]:
    if path is None:
        return []

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"ShareGPT dataset not found: {dataset_path}")

    raw_payload = _read_dataset_payload(dataset_path)
    if isinstance(raw_payload, dict):
        if isinstance(raw_payload.get("items"), list):
            records = raw_payload["items"]
        elif isinstance(raw_payload.get("conversations"), list):
            records = raw_payload["conversations"]
        else:
            records = [raw_payload]
    elif isinstance(raw_payload, list):
        records = raw_payload
    else:
        raise ValueError(f"unsupported ShareGPT payload type: {type(raw_payload)!r}")

    examples: list[ConversationExample] = []
    seen_signatures: set[str] = set()
    for index, record in enumerate(records):
        if len(examples) >= sample_limit:
            break
        example = _normalize_sharegpt_record(record, fallback_index=index)
        if example is None:
            continue
        signature = hashlib.blake2b(
            repr(example.messages).encode("utf-8"),
            digest_size=16,
        ).hexdigest()
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        examples.append(example)
    return examples


def load_rag_examples(
    path: str | None,
    sample_limit: int = 2_000,
) -> list[RagExample]:
    records = _load_records_from_path(path, missing_label="RAG dataset")
    if not records:
        return []

    examples: list[RagExample] = []
    seen_signatures: set[str] = set()
    for index, record in enumerate(records):
        if len(examples) >= sample_limit:
            break
        example = _normalize_rag_record(record, fallback_index=index)
        if example is None:
            continue
        signature = hashlib.blake2b(
            repr((example.query, example.contexts)).encode("utf-8"),
            digest_size=16,
        ).hexdigest()
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        examples.append(example)
    return examples


def load_agent_conversations(
    path: str | None,
    sample_limit: int = 2_000,
) -> list[ConversationExample]:
    records = _load_records_from_path(path, missing_label="agent dataset")
    if not records:
        return []

    examples: list[ConversationExample] = []
    seen_signatures: set[str] = set()
    for index, record in enumerate(records):
        if len(examples) >= sample_limit:
            break
        example = _normalize_agent_record(record, fallback_index=index)
        if example is None:
            continue
        signature = hashlib.blake2b(
            repr(example.messages).encode("utf-8"),
            digest_size=16,
        ).hexdigest()
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        examples.append(example)
    return examples


def text_to_routing_tokens(prompt_text: str) -> tuple[int, ...]:
    return tuple(_stable_token_id(piece) for piece in _lexical_tokens(prompt_text))


def render_prompt_from_words(words: tuple[str, ...]) -> str:
    if not words:
        return "System:\nRespond briefly.\n\nUser:\nEmpty request."

    total = len(words)
    system_count = max(8, total // 5)
    tool_count = max(4, total // 10)
    context_count = max(8, total // 3)
    if system_count + tool_count + context_count >= total:
        system_count = max(4, total // 4)
        tool_count = max(2, total // 8)
        context_count = max(4, total // 4)

    system_words = words[:system_count]
    tool_words = words[system_count : system_count + tool_count]
    context_words = words[system_count + tool_count : system_count + tool_count + context_count]
    user_words = words[system_count + tool_count + context_count :]
    if not user_words:
        user_words = words[-max(4, total // 5) :]

    sections = [
        "System:\n"
        "You are an operations assistant for a distributed inference platform. "
        + " ".join(system_words)
        + ".",
        "Tools:\n"
        + ", ".join(tool_words or ("inspect_cache", "fetch_policy"))
        + ".",
        "Context:\n"
        + " ".join(context_words or system_words)
        + ".",
        "User:\nPlease analyze the following request and answer succinctly: "
        + " ".join(user_words)
        + ".",
    ]
    return "\n\n".join(sections)


def _generate_synthetic_workload(config: WorkloadConfig) -> list[Request]:
    rng = random.Random(config.seed)
    max_prefix_length = max(config.prefix_length_choices)
    templates = [
        _build_template_words(rng, max_prefix_length, template_index=index)
        for index in range(config.num_templates)
    ]

    weights = [1.0 / ((index + 1) ** config.template_popularity_alpha) for index in range(config.num_templates)]
    total_weight = sum(weights)
    normalized_weights = [weight / total_weight for weight in weights]

    arrival_time = 0.0
    requests: list[Request] = []
    for index in range(config.num_requests):
        arrival_time += _sample_interarrival(config.mean_interarrival, rng)

        router_id = rng.choice(config.router_ids)
        template = rng.choices(templates, weights=normalized_weights, k=1)[0]
        prefix_length = rng.choice(config.prefix_length_choices)
        overlap = min(prefix_length, rng.choice(config.overlap_length_choices))
        if rng.random() > config.shared_template_probability:
            overlap = 0

        shared_prefix = template[:overlap]
        suffix = _build_unique_words(rng, prefix_length - overlap, config.vocab_size)
        prefix_words = tuple(shared_prefix + suffix)
        prompt_prefix_text = render_prompt_from_words(prefix_words)
        prefix_tokens = text_to_routing_tokens(prompt_prefix_text)

        continuation_tokens = rng.randint(*config.continuation_token_range)
        requests.append(
            Request(
                request_id=f"req-{index:05d}",
                arrival_time=arrival_time,
                router_id=router_id,
                prefix_tokens=prefix_tokens,
                continuation_tokens=continuation_tokens,
                prompt_prefix_text=prompt_prefix_text,
                traffic_class="synthetic",
                source_id=f"template-{templates.index(template)}",
            )
        )

    return requests


def _generate_mixed_realistic_workload(config: WorkloadConfig) -> list[Request]:
    rng = random.Random(config.seed)
    chat_examples = load_sharegpt_conversations(
        config.sharegpt_path,
        sample_limit=config.sharegpt_sample_limit,
    )
    rag_examples = load_rag_examples(
        config.rag_path,
        sample_limit=config.rag_sample_limit,
    )
    agent_examples = load_agent_conversations(
        config.agent_path,
        sample_limit=config.agent_sample_limit,
    )
    if not chat_examples:
        chat_examples = _default_chat_examples()

    rag_corpora = _default_rag_corpora()
    tool_catalogs = _default_tool_catalogs()
    session_router_map: dict[str, str] = {}
    source_variant_counts: dict[tuple[str, str], int] = {}
    chat_turn_pool = _build_chat_turn_pool(chat_examples, rng)
    if not chat_turn_pool:
        chat_examples = _default_chat_examples()
        chat_turn_pool = _build_chat_turn_pool(chat_examples, rng)
    rag_example_pool = _build_rag_example_pool(rag_examples, rng)
    agent_turn_pool = _build_agent_turn_pool(agent_examples, rng)
    if agent_examples and not agent_turn_pool:
        agent_examples = []

    positive_traffic_types = [
        traffic_type
        for traffic_type, weight in _traffic_mix(config).items()
        if weight > 0
    ]
    if not positive_traffic_types:
        raise ValueError("mixed_realistic workload requires at least one positive traffic weight")
    rng.shuffle(positive_traffic_types)

    requests: list[Request] = []
    request_index = 0
    arrival_time = 0.0

    while len(requests) < config.num_requests:
        remaining = config.num_requests - len(requests)
        reserve_for_required = len(positive_traffic_types) - 1 if positive_traffic_types else 0

        if positive_traffic_types:
            traffic_type = positive_traffic_types.pop(0)
        else:
            traffic_type = rng.choices(
                list(_traffic_mix(config).keys()),
                weights=list(_traffic_mix(config).values()),
                k=1,
            )[0]

        arrival_time += _sample_interarrival(config.mean_interarrival, rng)

        if traffic_type == CHAT_TRAFFIC_CLASS:
            request_obj = _build_sharegpt_chat_request(
                config=config,
                rng=rng,
                request_index=request_index,
                arrival_time=arrival_time,
                chat_examples=chat_examples,
                chat_turn_pool=chat_turn_pool,
                session_router_map=session_router_map,
                source_variant_counts=source_variant_counts,
            )
            requests.append(request_obj)
            request_index += 1
            burst_requests = _build_followup_burst_requests(
                base_request=request_obj,
                config=config,
                rng=rng,
                request_index_start=request_index,
                max_requests=max(0, remaining - reserve_for_required - 1),
            )
            requests.extend(burst_requests)
            request_index += len(burst_requests)
            if burst_requests:
                arrival_time = burst_requests[-1].arrival_time
            continue

        if traffic_type == RAG_TRAFFIC_CLASS:
            request_obj = _build_rag_request(
                config=config,
                rng=rng,
                request_index=request_index,
                arrival_time=arrival_time,
                rag_examples=rag_examples,
                rag_example_pool=rag_example_pool,
                rag_corpora=rag_corpora,
                source_variant_counts=source_variant_counts,
            )
            requests.append(request_obj)
            request_index += 1
            burst_requests = _build_followup_burst_requests(
                base_request=request_obj,
                config=config,
                rng=rng,
                request_index_start=request_index,
                max_requests=max(0, remaining - reserve_for_required - 1),
            )
            requests.extend(burst_requests)
            request_index += len(burst_requests)
            if burst_requests:
                arrival_time = burst_requests[-1].arrival_time
            continue

        if traffic_type == AGENT_TRAFFIC_CLASS:
            request_obj = _build_agent_request(
                config=config,
                rng=rng,
                request_index=request_index,
                arrival_time=arrival_time,
                agent_examples=agent_examples,
                agent_turn_pool=agent_turn_pool,
                tool_catalogs=tool_catalogs,
                session_router_map=session_router_map,
                source_variant_counts=source_variant_counts,
            )
            requests.append(request_obj)
            request_index += 1
            burst_requests = _build_followup_burst_requests(
                base_request=request_obj,
                config=config,
                rng=rng,
                request_index_start=request_index,
                max_requests=max(0, remaining - reserve_for_required - 1),
            )
            requests.extend(burst_requests)
            request_index += len(burst_requests)
            if burst_requests:
                arrival_time = burst_requests[-1].arrival_time
            continue

        burst_requests = _build_bursty_requests(
            config=config,
            rng=rng,
            request_index_start=request_index,
            base_arrival_time=arrival_time,
            max_requests=max(1, remaining - reserve_for_required),
            chat_examples=chat_examples,
            session_router_map=session_router_map,
        )
        requests.extend(burst_requests)
        request_index += len(burst_requests)
        if burst_requests:
            arrival_time = burst_requests[-1].arrival_time

    return sorted(requests[: config.num_requests], key=lambda request_obj: request_obj.arrival_time)


def _build_sharegpt_chat_request(
    config: WorkloadConfig,
    rng: random.Random,
    request_index: int,
    arrival_time: float,
    chat_examples: Sequence[ConversationExample],
    chat_turn_pool: list[tuple[ConversationExample, int]],
    session_router_map: dict[str, str],
    source_variant_counts: dict[tuple[str, str], int],
) -> Request:
    example, user_index = _next_conversation_turn(chat_turn_pool, chat_examples, rng)
    assistant_response = _next_message_content(example.messages, user_index + 1, "assistant") or ""
    prompt_prefix_text = render_message_prompt(example.messages[: user_index + 1])
    session_id = f"chat-{example.conversation_id}"
    base_source_id = f"{example.conversation_id}:turn-{user_index}"
    prompt_prefix_text, source_id = _apply_request_variant(
        prompt_prefix_text=prompt_prefix_text,
        traffic_class=CHAT_TRAFFIC_CLASS,
        base_source_id=base_source_id,
        source_variant_counts=source_variant_counts,
    )
    router_id = _choose_router_for_session(
        router_ids=config.router_ids,
        session_id=session_id,
        session_router_map=session_router_map,
        affinity_probability=config.session_affinity_probability,
        rng=rng,
    )
    return _build_request(
        config=config,
        request_id=f"req-{request_index:05d}",
        arrival_time=arrival_time,
        router_id=router_id,
        prompt_prefix_text=prompt_prefix_text,
        continuation_tokens=_continuation_tokens_from_text(assistant_response, config),
        traffic_class=CHAT_TRAFFIC_CLASS,
        session_id=session_id,
        source_id=source_id,
    )


def _build_rag_request(
    config: WorkloadConfig,
    rng: random.Random,
    request_index: int,
    arrival_time: float,
    rag_examples: Sequence[RagExample],
    rag_example_pool: list[RagExample],
    rag_corpora: dict[str, tuple[tuple[str, str], ...]],
    source_variant_counts: dict[tuple[str, str], int],
) -> Request:
    if rag_examples:
        example = _next_rag_example(rag_example_pool, rag_examples, rng)
        chosen_docs = list(example.contexts)
        if len(chosen_docs) > 3:
            chosen_docs = rng.sample(chosen_docs, k=rng.randint(2, 3))
        prompt_sections = [
            (
                "System",
                "You are a retrieval-grounded assistant. Answer only from retrieved context and note uncertainty when evidence is weak.",
            ),
            (
                "Instructions",
                "Use the retrieved passages below. Prefer exact operational details, citations, and explicit caveats over generic advice.",
            ),
            (
                "Retrieved Context",
                "\n\n".join(
                    f"[{doc_id}]\n{doc_text}"
                    for doc_id, doc_text in chosen_docs
                ),
            ),
            ("User", example.query),
            ("Assistant", ""),
        ]
        prompt_prefix_text = render_section_prompt(prompt_sections)
        prompt_prefix_text, source_id = _apply_request_variant(
            prompt_prefix_text=prompt_prefix_text,
            traffic_class=RAG_TRAFFIC_CLASS,
            base_source_id=example.example_id,
            source_variant_counts=source_variant_counts,
        )
        continuation_seed = example.answer or " ".join(doc_text for _, doc_text in chosen_docs[:1])
        return _build_request(
            config=config,
            request_id=f"req-{request_index:05d}",
            arrival_time=arrival_time,
            router_id=rng.choice(config.router_ids),
            prompt_prefix_text=prompt_prefix_text,
            continuation_tokens=_continuation_tokens_from_text(continuation_seed, config),
            traffic_class=RAG_TRAFFIC_CLASS,
            source_id=source_id,
        )

    corpus_id = rng.choice(sorted(rag_corpora))
    documents = rag_corpora[corpus_id]
    chosen_docs = rng.sample(list(documents), k=min(len(documents), rng.randint(2, 3)))
    query = rng.choice(_RAG_QUERIES[corpus_id])
    prompt_sections = [
        (
            "System",
            "You are a retrieval-grounded assistant. Answer only from retrieved context and note uncertainty when evidence is weak.",
        ),
        (
            "Instructions",
            "Use the retrieved passages below. Prefer exact operational details, retention windows, and escalation boundaries over generic advice.",
        ),
        (
            "Retrieved Context",
            "\n\n".join(
                f"[{doc_id}]\n{doc_text}"
                for doc_id, doc_text in chosen_docs
            ),
        ),
        ("User", query),
        ("Assistant", ""),
    ]
    prompt_prefix_text = render_section_prompt(prompt_sections)
    prompt_prefix_text, source_id = _apply_request_variant(
        prompt_prefix_text=prompt_prefix_text,
        traffic_class=RAG_TRAFFIC_CLASS,
        base_source_id=corpus_id,
        source_variant_counts=source_variant_counts,
    )
    continuation_seed = " ".join(doc_text for _, doc_text in chosen_docs[:1])
    return _build_request(
        config=config,
        request_id=f"req-{request_index:05d}",
        arrival_time=arrival_time,
        router_id=rng.choice(config.router_ids),
        prompt_prefix_text=prompt_prefix_text,
        continuation_tokens=_continuation_tokens_from_text(continuation_seed, config),
        traffic_class=RAG_TRAFFIC_CLASS,
        source_id=source_id,
    )


def _build_agent_request(
    config: WorkloadConfig,
    rng: random.Random,
    request_index: int,
    arrival_time: float,
    agent_examples: Sequence[ConversationExample],
    agent_turn_pool: list[tuple[ConversationExample, int]],
    tool_catalogs: dict[str, tuple[tuple[str, str], ...]],
    session_router_map: dict[str, str],
    source_variant_counts: dict[tuple[str, str], int],
) -> Request:
    if agent_examples:
        example, user_index = _next_conversation_turn(agent_turn_pool, agent_examples, rng, agent_mode=True)
        prompt_prefix_text = render_message_prompt(example.messages[: user_index + 1])
        continuation_seed = _next_agent_turn_seed(example.messages, user_index + 1) or example.messages[user_index][1]

        session_id = f"agent-{example.conversation_id}"
        base_source_id = f"{example.conversation_id}:turn-{user_index}"
        prompt_prefix_text, source_id = _apply_request_variant(
            prompt_prefix_text=prompt_prefix_text,
            traffic_class=AGENT_TRAFFIC_CLASS,
            base_source_id=base_source_id,
            source_variant_counts=source_variant_counts,
        )
        router_id = _choose_router_for_session(
            router_ids=config.router_ids,
            session_id=session_id,
            session_router_map=session_router_map,
            affinity_probability=config.session_affinity_probability,
            rng=rng,
        )
        return _build_request(
            config=config,
            request_id=f"req-{request_index:05d}",
            arrival_time=arrival_time,
            router_id=router_id,
            prompt_prefix_text=prompt_prefix_text,
            continuation_tokens=_continuation_tokens_from_text(continuation_seed, config),
            traffic_class=AGENT_TRAFFIC_CLASS,
            session_id=session_id,
            source_id=source_id,
        )

    catalog_id = rng.choice(sorted(tool_catalogs))
    catalog = tool_catalogs[catalog_id]
    session_id = f"agent-{catalog_id}-{rng.randint(0, 4)}"
    router_id = _choose_router_for_session(
        router_ids=config.router_ids,
        session_id=session_id,
        session_router_map=session_router_map,
        affinity_probability=config.session_affinity_probability,
        rng=rng,
    )
    prior_context = rng.choice(_AGENT_PRIOR_CONTEXT[catalog_id])
    user_task = rng.choice(_AGENT_TASKS[catalog_id])
    prompt_sections = [
        (
            "System",
            "You are an orchestration agent. Plan safely, decide which tools are needed, and return a concise operator-facing answer.",
        ),
        (
            "Tools",
            "\n".join(
                f"- {tool_name}: {schema_text}"
                for tool_name, schema_text in catalog
            ),
        ),
        ("Context", prior_context),
        ("User", user_task),
        ("Assistant", ""),
    ]
    prompt_prefix_text = render_section_prompt(prompt_sections)
    prompt_prefix_text, source_id = _apply_request_variant(
        prompt_prefix_text=prompt_prefix_text,
        traffic_class=AGENT_TRAFFIC_CLASS,
        base_source_id=catalog_id,
        source_variant_counts=source_variant_counts,
    )
    return _build_request(
        config=config,
        request_id=f"req-{request_index:05d}",
        arrival_time=arrival_time,
        router_id=router_id,
        prompt_prefix_text=prompt_prefix_text,
        continuation_tokens=_continuation_tokens_from_text(prior_context, config),
        traffic_class=AGENT_TRAFFIC_CLASS,
        session_id=session_id,
        source_id=source_id,
    )


def _build_bursty_requests(
    config: WorkloadConfig,
    rng: random.Random,
    request_index_start: int,
    base_arrival_time: float,
    max_requests: int,
    chat_examples: Sequence[ConversationExample],
    session_router_map: dict[str, str],
) -> list[Request]:
    burst_size = min(rng.choice(config.burst_size_choices), max_requests)
    example = rng.choice(tuple(chat_examples))
    assistant_indices = [
        index
        for index, (role, _) in enumerate(example.messages)
        if role == "assistant"
    ]
    if assistant_indices:
        cut_index = rng.choice(assistant_indices)
        base_messages = example.messages[: cut_index + 1]
    else:
        base_messages = example.messages[:1]

    session_id = f"burst-{example.conversation_id}-{request_index_start}"
    router_id = _choose_router_for_session(
        router_ids=config.router_ids,
        session_id=session_id,
        session_router_map=session_router_map,
        affinity_probability=1.0,
        rng=rng,
    )

    requests: list[Request] = []
    current_arrival = base_arrival_time
    previous_prompt: str | None = None
    previous_continuation = max(config.dataset_continuation_floor, config.continuation_token_range[1])
    for offset in range(burst_size):
        if previous_prompt is not None and rng.random() < 0.30:
            prompt_prefix_text = previous_prompt
            continuation_tokens = previous_continuation
        else:
            followup = rng.choice(_BURSTY_FOLLOWUPS)
            prompt_prefix_text = render_message_prompt(
                list(base_messages) + [("user", followup)]
            )
            continuation_tokens = max(
                config.dataset_continuation_floor,
                min(config.dataset_continuation_cap, len(_lexical_tokens(followup)) * 3),
            )
            previous_prompt = prompt_prefix_text
            previous_continuation = continuation_tokens

        requests.append(
            _build_request(
                config=config,
                request_id=f"req-{request_index_start + offset:05d}",
                arrival_time=current_arrival,
                router_id=router_id,
                prompt_prefix_text=prompt_prefix_text,
                continuation_tokens=continuation_tokens,
                traffic_class=BURSTY_TRAFFIC_CLASS,
                session_id=session_id,
                source_id=example.conversation_id,
            )
        )
        current_arrival += max(
            0.01,
            config.mean_interarrival * config.burst_interarrival_ratio * rng.uniform(0.25, 1.25),
        )
    return requests


def _build_followup_burst_requests(
    *,
    base_request: Request,
    config: WorkloadConfig,
    rng: random.Random,
    request_index_start: int,
    max_requests: int,
) -> list[Request]:
    if max_requests <= 0 or config.traffic_burst_probability <= 0:
        return []
    if rng.random() >= config.traffic_burst_probability:
        return []

    burst_size = min(rng.choice(config.burst_size_choices), max_requests + 1)
    if burst_size <= 1:
        return []

    requests: list[Request] = []
    current_arrival = base_request.arrival_time
    for offset in range(1, burst_size):
        current_arrival += max(
            0.01,
            config.mean_interarrival * config.burst_interarrival_ratio * rng.uniform(0.25, 1.25),
        )
        followup_prompt = _build_followup_prompt_from_request(
            base_request=base_request,
            rng=rng,
            offset=offset,
        )
        followup_continuation = max(
            config.dataset_continuation_floor,
            min(
                config.dataset_continuation_cap,
                max(
                    base_request.continuation_tokens,
                    _continuation_tokens_from_text(followup_prompt, config),
                ),
            ),
        )
        requests.append(
            _build_request(
                config=config,
                request_id=f"req-{request_index_start + offset - 1:05d}",
                arrival_time=current_arrival,
                router_id=base_request.router_id,
                prompt_prefix_text=followup_prompt,
                continuation_tokens=followup_continuation,
                traffic_class=base_request.traffic_class,
                session_id=base_request.session_id,
                source_id=base_request.source_id,
            )
        )
    return requests


def _build_request(
    config: WorkloadConfig,
    request_id: str,
    arrival_time: float,
    router_id: str,
    prompt_prefix_text: str,
    continuation_tokens: int,
    traffic_class: str,
    session_id: str | None = None,
    source_id: str | None = None,
) -> Request:
    normalized_prompt_prefix_text = _truncate_prompt_prefix_text(
        prompt_prefix_text,
        config.prompt_prefix_token_cap,
    )
    return Request(
        request_id=request_id,
        arrival_time=arrival_time,
        router_id=router_id,
        prefix_tokens=text_to_routing_tokens(normalized_prompt_prefix_text),
        continuation_tokens=continuation_tokens,
        prompt_prefix_text=normalized_prompt_prefix_text,
        traffic_class=traffic_class,
        session_id=session_id,
        source_id=source_id,
    )


def render_message_prompt(messages: Sequence[tuple[str, str]]) -> str:
    blocks = []
    for role, content in messages:
        normalized_role = role if role in _ROLE_LABELS else "user"
        if normalized_role == "assistant" and not content.strip():
            continue
        blocks.append(f"{_ROLE_LABELS[normalized_role]}:\n{content.strip()}")
    blocks.append("Assistant:")
    return "\n\n".join(blocks)


def _truncate_prompt_prefix_text(prompt_prefix_text: str, token_cap: int | None) -> str:
    if token_cap is None or token_cap <= 0:
        return prompt_prefix_text
    matches = list(re.finditer(r"[A-Za-z0-9_-]+|[^\w\s]", prompt_prefix_text))
    if len(matches) <= token_cap:
        return prompt_prefix_text
    end_index = matches[token_cap - 1].end()
    return prompt_prefix_text[:end_index].rstrip()


def render_section_prompt(sections: Sequence[tuple[str, str]]) -> str:
    blocks = []
    for title, content in sections:
        if title == "Assistant" and not content.strip():
            blocks.append("Assistant:")
            continue
        blocks.append(f"{title}:\n{content.strip()}")
    if not blocks or not blocks[-1].rstrip().endswith("Assistant:"):
        blocks.append("Assistant:")
    return "\n\n".join(blocks)


def _read_dataset_payload(dataset_path: Path) -> object:
    if dataset_path.suffix == ".jsonl":
        rows = []
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        return rows
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def _load_records_from_path(
    path: str | None,
    missing_label: str,
) -> list[object]:
    if path is None:
        return []

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"{missing_label} not found: {dataset_path}")

    raw_payload = _read_dataset_payload(dataset_path)
    if isinstance(raw_payload, dict):
        for field_name in ("items", "rows", "examples", "data", "conversations", "messages", "turns"):
            field_value = raw_payload.get(field_name)
            if isinstance(field_value, list):
                return list(field_value)
        return [raw_payload]
    if isinstance(raw_payload, list):
        return list(raw_payload)
    raise ValueError(f"unsupported dataset payload type: {type(raw_payload)!r}")


def _normalize_sharegpt_record(
    record: object,
    fallback_index: int,
) -> ConversationExample | None:
    if not isinstance(record, dict):
        return None

    messages: list[tuple[str, str]] = []
    system_prompt = record.get("system") or record.get("system_prompt")
    if isinstance(system_prompt, str) and system_prompt.strip():
        messages.append(("system", _normalize_text(system_prompt)))

    raw_messages = (
        record.get("conversations")
        or record.get("messages")
        or record.get("turns")
        or record.get("conversation")
    )
    if not isinstance(raw_messages, list):
        return None

    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            continue
        role = _normalize_role(
            raw_message.get("from") or raw_message.get("role") or raw_message.get("speaker")
        )
        content = raw_message.get("value") or raw_message.get("content") or raw_message.get("text")
        if role is None or not isinstance(content, str):
            continue
        normalized_content = _normalize_text(content)
        if not normalized_content:
            continue
        if messages and messages[-1][0] == role and role != "tool":
            merged = f"{messages[-1][1]}\n\n{normalized_content}"
            messages[-1] = (role, merged)
            continue
        messages.append((role, normalized_content))

    if not any(role == "user" for role, _ in messages):
        return None
    if not any(role == "assistant" for role, _ in messages):
        return None

    conversation_id = str(record.get("id") or record.get("conversation_id") or f"sharegpt-{fallback_index}")
    return ConversationExample(conversation_id=conversation_id, messages=tuple(messages))


def _normalize_rag_record(
    record: object,
    fallback_index: int,
) -> RagExample | None:
    if not isinstance(record, dict):
        return None

    query = _first_text_value(
        record,
        ("query", "question", "input", "prompt", "instruction"),
    )
    if query is None:
        return None

    contexts = _extract_rag_contexts(record)
    if not contexts:
        return None

    answer = _extract_answer_text(record)
    example_id = str(
        record.get("id")
        or record.get("query_id")
        or record.get("_id")
        or f"rag-{fallback_index}"
    )
    return RagExample(
        example_id=example_id,
        query=query,
        contexts=tuple(contexts),
        answer=answer,
    )


def _normalize_agent_record(
    record: object,
    fallback_index: int,
) -> ConversationExample | None:
    if not isinstance(record, dict):
        return None

    bfcl_example = _normalize_bfcl_record(record, fallback_index)
    if bfcl_example is not None:
        return bfcl_example

    tau_example = _normalize_tau_bench_record(record, fallback_index)
    if tau_example is not None:
        return tau_example

    tools = _extract_tool_descriptions(record)

    if any(
        isinstance(record.get(field_name), list)
        for field_name in ("conversations", "messages", "turns")
    ):
        example = _normalize_sharegpt_record(record, fallback_index)
        if example is not None:
            if not tools:
                return example
            system_messages = [
                ("system", "Available tools:\n" + "\n".join(tools)),
            ]
            return ConversationExample(
                conversation_id=example.conversation_id,
                messages=tuple(system_messages + list(example.messages)),
            )

    user_prompt = _first_text_value(
        record,
        ("question", "query", "instruction", "user", "prompt"),
    )
    if user_prompt is None or not tools:
        return None

    messages = [("system", "You are a tool-using assistant. Choose safe tool calls before answering the user.")]
    messages.append(("system", "Available tools:\n" + "\n".join(tools)))
    messages.append(("user", user_prompt))
    assistant_seed = _extract_answer_text(record)
    if assistant_seed is not None:
        messages.append(("assistant", assistant_seed))

    conversation_id = str(
        record.get("id")
        or record.get("conversation_id")
        or record.get("question_id")
        or record.get("query_id")
        or f"agent-{fallback_index}"
    )
    return ConversationExample(conversation_id=conversation_id, messages=tuple(messages))


def _normalize_bfcl_record(
    record: dict[str, object],
    fallback_index: int,
) -> ConversationExample | None:
    if "question" not in record:
        return None
    if not any(
        key in record
        for key in ("function", "functions", "ground_truth", "possible_answer", "execution_result_type")
    ):
        return None

    messages = _extract_message_sequence(record.get("question"))
    if not messages:
        question_text = _first_text_value(record, ("question", "query", "instruction", "prompt"))
        if question_text is None:
            return None
        messages = [("user", question_text)]

    tools = _extract_bfcl_tools(record)
    normalized_messages: list[tuple[str, str]] = []
    if tools:
        normalized_messages.append(("system", "Available functions:\n" + "\n".join(tools)))
    normalized_messages.extend(messages)

    answer_text = _extract_answer_text(record)
    if answer_text is not None and not any(role == "assistant" for role, _ in normalized_messages):
        normalized_messages.append(("assistant", answer_text))

    if not any(role == "user" for role, _ in normalized_messages):
        return None

    conversation_id = str(record.get("id") or record.get("question_id") or f"bfcl-{fallback_index}")
    return ConversationExample(conversation_id=conversation_id, messages=tuple(normalized_messages))


def _normalize_tau_bench_record(
    record: dict[str, object],
    fallback_index: int,
) -> ConversationExample | None:
    if not any(key in record for key in ("task", "trajectory", "history", "trace", "env", "domain")):
        return None

    task = record.get("task")
    tools = _extract_tool_descriptions(record)
    messages: list[tuple[str, str]] = []

    if isinstance(task, dict):
        domain = _first_text_value(task, ("domain", "env"))
        policy = _first_text_value(task, ("policy", "policy_text", "guidelines"))
        goal = _first_text_value(task, ("instruction", "user_instruction", "goal", "description", "task"))
        task_tools = _extract_tool_descriptions(task)
        if task_tools:
            tools = task_tools + [tool for tool in tools if tool not in task_tools]
        if domain is not None:
            messages.append(("system", f"Domain: {domain}"))
        if policy is not None:
            messages.append(("system", "Policy:\n" + policy))
        if goal is not None:
            messages.append(("user", goal))

    if tools:
        messages.insert(0, ("system", "Available tools:\n" + "\n".join(tools)))

    for field_name in ("messages", "history", "trajectory", "trace", "turns", "conversation"):
        extracted = _extract_message_sequence(record.get(field_name))
        if extracted:
            messages.extend(extracted)
            break

    if not any(role == "user" for role, _ in messages):
        prompt_text = _first_text_value(record, ("instruction", "query", "prompt", "goal", "description"))
        if prompt_text is not None:
            messages.append(("user", prompt_text))

    if not any(role == "user" for role, _ in messages):
        return None

    conversation_id = str(
        record.get("id")
        or record.get("task_id")
        or record.get("trajectory_id")
        or f"tau-{fallback_index}"
    )
    return ConversationExample(conversation_id=conversation_id, messages=tuple(messages))


def _normalize_role(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered in {"human", "user"}:
        return "user"
    if lowered in {"gpt", "assistant", "model"}:
        return "assistant"
    if lowered in {"system", "developer"}:
        return "system"
    if lowered in {"tool", "function"}:
        return "tool"
    return None


def _extract_message_sequence(raw_messages: object) -> list[tuple[str, str]]:
    if not isinstance(raw_messages, list):
        return []

    messages: list[tuple[str, str]] = []
    for raw_message in raw_messages:
        normalized = _normalize_message_item(raw_message)
        if normalized is None:
            continue
        role, content = normalized
        if messages and messages[-1][0] == role and role != "tool":
            messages[-1] = (role, f"{messages[-1][1]}\n\n{content}")
            continue
        messages.append((role, content))
    return messages


def _normalize_message_item(raw_message: object) -> tuple[str, str] | None:
    if isinstance(raw_message, str):
        normalized = _normalize_text(raw_message)
        if normalized:
            return ("user", normalized)
        return None
    if not isinstance(raw_message, dict):
        return None

    role = _normalize_role(
        raw_message.get("from")
        or raw_message.get("role")
        or raw_message.get("speaker")
        or raw_message.get("actor")
        or raw_message.get("type")
    )
    content = raw_message.get("value") or raw_message.get("content") or raw_message.get("text")
    if content is None:
        for field_name in ("message", "utterance", "response", "observation", "action", "result"):
            if field_name in raw_message:
                content = raw_message[field_name]
                break

    if isinstance(content, dict):
        content = json.dumps(content, sort_keys=True)
    elif isinstance(content, list):
        if all(isinstance(part, str) for part in content):
            content = "\n".join(part for part in content if part.strip())
        else:
            content = json.dumps(content, sort_keys=True)

    if role is None:
        tool_name = raw_message.get("tool_name") or raw_message.get("tool") or raw_message.get("name")
        if tool_name is not None:
            role = "tool"
            if content is None:
                content = json.dumps(raw_message, sort_keys=True)

    if role is None or not isinstance(content, str):
        return None

    normalized_content = _normalize_text(content)
    if not normalized_content:
        return None
    return (role, normalized_content)


def _normalize_text(text: str) -> str:
    stripped = text.strip()
    stripped = stripped.replace("\r\n", "\n").replace("\r", "\n")
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped


def _first_text_value(
    record: dict[str, object],
    field_names: Sequence[str],
) -> str | None:
    for field_name in field_names:
        value = record.get(field_name)
        if isinstance(value, str):
            normalized = _normalize_text(value)
            if normalized:
                return normalized
    return None


def _extract_answer_text(record: dict[str, object]) -> str | None:
    direct = _first_text_value(record, ("answer", "response", "output", "target"))
    if direct is not None:
        return direct

    answers = record.get("answers")
    if isinstance(answers, list):
        for answer in answers:
            if isinstance(answer, str):
                normalized = _normalize_text(answer)
                if normalized:
                    return normalized
            if isinstance(answer, dict):
                for field_name in ("answer", "text", "value", "content"):
                    field_value = answer.get(field_name)
                    if isinstance(field_value, str):
                        normalized = _normalize_text(field_value)
                        if normalized:
                            return normalized
    return None


def _extract_rag_contexts(record: dict[str, object]) -> list[tuple[str, str]]:
    contexts: list[tuple[str, str]] = []
    for list_field in (
        "quotes",
        "contexts",
        "passages",
        "documents",
        "docs",
        "retrieved_contexts",
        "contexts_list",
        "evidence",
    ):
        field_value = record.get(list_field)
        if not isinstance(field_value, list):
            continue
        for index, item in enumerate(field_value):
            context = _normalize_context_item(item, fallback_index=index)
            if context is not None:
                contexts.append(context)
        if contexts:
            return contexts

    for field_name in ("context", "article", "document", "passage"):
        field_value = record.get(field_name)
        if isinstance(field_value, str):
            normalized = _normalize_text(field_value)
            if normalized:
                return [("doc-0", normalized)]
    return []


def _normalize_context_item(
    item: object,
    fallback_index: int,
) -> tuple[str, str] | None:
    if isinstance(item, str):
        normalized = _normalize_text(item)
        if normalized:
            return (f"doc-{fallback_index}", normalized)
        return None
    if not isinstance(item, dict):
        return None

    text = _first_text_value(
        item,
        ("text", "value", "content", "passage", "document", "body", "evidence_text", "quote", "snippet"),
    )
    if text is None:
        return None
    doc_id = str(
        item.get("docid")
        or item.get("id")
        or item.get("doc_name")
        or item.get("doc_link")
        or item.get("title")
        or item.get("name")
        or f"doc-{fallback_index}"
    )
    return (doc_id, text)


def _extract_tool_descriptions(record: dict[str, object]) -> list[str]:
    tools: list[str] = []
    for field_name in ("tools", "functions", "available_tools", "api_list"):
        field_value = record.get(field_name)
        if not isinstance(field_value, list):
            continue
        for index, tool in enumerate(field_value):
            description = _normalize_tool_description(tool, fallback_index=index)
            if description is not None:
                tools.append(description)
        if tools:
            return tools
    return []


def _extract_bfcl_tools(record: dict[str, object]) -> list[str]:
    tools = _extract_tool_descriptions(record)
    if tools:
        return tools

    function_payload = record.get("function")
    if isinstance(function_payload, str):
        normalized = _normalize_text(function_payload)
        return [normalized] if normalized else []
    if isinstance(function_payload, list):
        parsed_tools: list[str] = []
        for index, tool in enumerate(function_payload):
            description = _normalize_tool_description(tool, fallback_index=index)
            if description is not None:
                parsed_tools.append(description)
        return parsed_tools
    if isinstance(function_payload, dict):
        description = _normalize_tool_description(function_payload, fallback_index=0)
        return [description] if description is not None else []
    return []


def _normalize_tool_description(
    tool: object,
    fallback_index: int,
) -> str | None:
    if isinstance(tool, str):
        normalized = _normalize_text(tool)
        return normalized or None
    if not isinstance(tool, dict):
        return None

    name = str(
        tool.get("name")
        or tool.get("function")
        or tool.get("title")
        or tool.get("api_name")
        or f"tool_{fallback_index}"
    )
    description = _first_text_value(tool, ("description", "desc", "summary"))
    parameters = tool.get("parameters") or tool.get("schema") or tool.get("arguments")
    category_name = tool.get("category_name")
    tool_name = tool.get("tool_name")

    parts = [name]
    if isinstance(category_name, str) and category_name.strip():
        parts.append(f"category={category_name.strip()}")
    if isinstance(tool_name, str) and tool_name.strip():
        parts.append(f"tool={tool_name.strip()}")
    if description is not None:
        parts.append(description)
    if parameters is not None:
        parts.append(json.dumps(parameters, sort_keys=True))
    return ": ".join(parts)


def _next_message_content(
    messages: Sequence[tuple[str, str]],
    start_index: int,
    role: str,
) -> str | None:
    for next_role, content in messages[start_index:]:
        if next_role == role:
            return content
    return None


def _next_agent_turn_seed(
    messages: Sequence[tuple[str, str]],
    start_index: int,
) -> str | None:
    collected: list[str] = []
    for next_role, content in messages[start_index:]:
        if next_role == "user" and collected:
            break
        if next_role in {"assistant", "tool"}:
            collected.append(content)
    if not collected:
        return None
    return "\n\n".join(collected)


def _build_chat_turn_pool(
    chat_examples: Sequence[ConversationExample],
    rng: random.Random,
) -> list[tuple[ConversationExample, int]]:
    pool = [
        (example, index)
        for example in chat_examples
        for index, (role, _) in enumerate(example.messages)
        if role == "user" and _next_message_content(example.messages, index + 1, "assistant") is not None
    ]
    rng.shuffle(pool)
    return pool


def _build_agent_turn_pool(
    agent_examples: Sequence[ConversationExample],
    rng: random.Random,
) -> list[tuple[ConversationExample, int]]:
    pool = [
        (example, index)
        for example in agent_examples
        for index, (role, _) in enumerate(example.messages)
        if role == "user"
    ]
    rng.shuffle(pool)
    return pool


def _build_rag_example_pool(
    rag_examples: Sequence[RagExample],
    rng: random.Random,
) -> list[RagExample]:
    pool = list(rag_examples)
    rng.shuffle(pool)
    return pool


def _next_conversation_turn(
    pool: list[tuple[ConversationExample, int]],
    examples: Sequence[ConversationExample],
    rng: random.Random,
    *,
    agent_mode: bool = False,
) -> tuple[ConversationExample, int]:
    if not pool:
        refill = (
            _build_agent_turn_pool(examples, rng)
            if agent_mode
            else _build_chat_turn_pool(examples, rng)
        )
        if not refill:
            raise ValueError("conversation examples did not contain a usable user turn")
        pool.extend(refill)
    return pool.pop()


def _next_rag_example(
    pool: list[RagExample],
    examples: Sequence[RagExample],
    rng: random.Random,
) -> RagExample:
    if not pool:
        pool.extend(_build_rag_example_pool(examples, rng))
    if not pool:
        raise ValueError("rag examples did not contain a usable example")
    return pool.pop()


def _choose_router_for_session(
    router_ids: Sequence[str],
    session_id: str,
    session_router_map: dict[str, str],
    affinity_probability: float,
    rng: random.Random,
) -> str:
    existing_router = session_router_map.get(session_id)
    if existing_router is not None and rng.random() < affinity_probability:
        return existing_router
    chosen_router = rng.choice(tuple(router_ids))
    session_router_map[session_id] = chosen_router
    return chosen_router


def _continuation_tokens_from_text(text: str, config: WorkloadConfig) -> int:
    token_count = len(_lexical_tokens(text))
    return max(
        config.dataset_continuation_floor,
        min(config.dataset_continuation_cap, token_count),
    )


def _build_followup_prompt_from_request(
    *,
    base_request: Request,
    rng: random.Random,
    offset: int,
) -> str:
    return _build_followup_prompt_from_text(
        base_prompt_text=base_request.prompt_prefix_text or "",
        traffic_class=base_request.traffic_class,
        offset=offset,
    )


def _apply_request_variant(
    *,
    prompt_prefix_text: str,
    traffic_class: str,
    base_source_id: str,
    source_variant_counts: dict[tuple[str, str], int],
) -> tuple[str, str]:
    variant_key = (traffic_class, base_source_id)
    variant_index = source_variant_counts.get(variant_key, 0)
    source_variant_counts[variant_key] = variant_index + 1
    if variant_index == 0:
        return prompt_prefix_text, base_source_id
    return (
        _build_followup_prompt_from_text(
            base_prompt_text=prompt_prefix_text,
            traffic_class=traffic_class,
            offset=variant_index,
        ),
        f"{base_source_id}:variant-{variant_index}",
    )


def _build_followup_prompt_from_text(
    *,
    base_prompt_text: str,
    traffic_class: str,
    offset: int,
) -> str:
    followup_catalog = _FOLLOWUP_PROMPTS_BY_TRAFFIC.get(
        traffic_class,
        _FOLLOWUP_PROMPTS_BY_TRAFFIC[CHAT_TRAFFIC_CLASS],
    )
    followup_text = followup_catalog[(offset - 1) % len(followup_catalog)]
    base_prompt = base_prompt_text.rstrip()
    if base_prompt.endswith("Assistant:"):
        assistant_bridge = _ASSISTANT_BRIDGE_BY_TRAFFIC.get(
            traffic_class,
            _ASSISTANT_BRIDGE_BY_TRAFFIC[CHAT_TRAFFIC_CLASS],
        )
        base_prompt = f"{base_prompt}\n{assistant_bridge}"
    return f"{base_prompt}\n\nUser:\n{followup_text}\n\nAssistant:"


def _traffic_mix(config: WorkloadConfig) -> dict[str, float]:
    return {
        CHAT_TRAFFIC_CLASS: config.traffic_mix_chat,
        RAG_TRAFFIC_CLASS: config.traffic_mix_rag,
        AGENT_TRAFFIC_CLASS: config.traffic_mix_agent,
        BURSTY_TRAFFIC_CLASS: config.traffic_mix_bursty,
    }


def _sample_interarrival(mean_interarrival: float, rng: random.Random) -> float:
    if mean_interarrival <= 0:
        return 0.0
    return rng.expovariate(1.0 / mean_interarrival)


def _default_chat_examples() -> list[ConversationExample]:
    return [
        ConversationExample(
            conversation_id="fallback-chat-0",
            messages=(
                (
                    "system",
                    "You are an operations assistant for a global inference platform. Keep answers concise and reference concrete evidence.",
                ),
                (
                    "user",
                    "Summarize why requests from the Midwest are seeing intermittent latency spikes after the last cache registry deployment.",
                ),
                (
                    "assistant",
                    "The primary suspicion is stale router metadata combined with a cold cache on the failover cluster after deployment.",
                ),
                (
                    "user",
                    "List the first three checks an on-call engineer should run before draining traffic.",
                ),
                (
                    "assistant",
                    "Check summary freshness, current queue depth by cluster, and whether the affected prefixes are concentrated in one tenant or one route.",
                ),
                (
                    "user",
                    "Draft a short status update for the incident channel with those findings.",
                ),
            ),
        ),
        ConversationExample(
            conversation_id="fallback-chat-1",
            messages=(
                (
                    "system",
                    "You help a support team reason about retrieval quality, cache locality, and cross-region serving costs.",
                ),
                (
                    "user",
                    "We changed the document chunking strategy. Explain how that could change prompt overlap for follow-up questions.",
                ),
                (
                    "assistant",
                    "Chunking changes which retrieved passages repeat across turns, so it can either increase shared prefixes or fragment them into smaller overlaps.",
                ),
                (
                    "user",
                    "What metrics would confirm that the overlap actually improved after rollout?",
                ),
                (
                    "assistant",
                    "Track reusable-prefix length, TTFT, prefill-token reduction, and how often routers keep related requests in the same region.",
                ),
                (
                    "user",
                    "Turn that into a brief experiment plan for tomorrow's rollout review.",
                ),
            ),
        ),
        ConversationExample(
            conversation_id="fallback-chat-2",
            messages=(
                (
                    "system",
                    "You are a concise assistant for commerce operations and customer support agents.",
                ),
                (
                    "user",
                    "A customer says their expedited shipment missed the promised delivery window. What context should the agent gather first?",
                ),
                (
                    "assistant",
                    "The agent should verify the order id, promised delivery SLA, last carrier scan, warehouse exceptions, and whether any address correction was requested.",
                ),
                (
                    "user",
                    "How would you tailor the response if the package crossed regions and the handoff scan is missing?",
                ),
                (
                    "assistant",
                    "A missing inter-region handoff scan suggests the parcel is in transit but unconfirmed, so the agent should acknowledge uncertainty and avoid promising a specific new date.",
                ),
                (
                    "user",
                    "Write the reply in a calm tone with one concrete next step.",
                ),
            ),
        ),
    ]


def _default_rag_corpora() -> dict[str, tuple[tuple[str, str], ...]]:
    return {
        "incident_ops": (
            (
                "doc-incident-01",
                "Regional failover playbook: when router metadata is older than thirty seconds, compare prefix overlap estimates with direct cluster health before shifting more than twenty percent of traffic.",
            ),
            (
                "doc-incident-02",
                "Cache registry notes: repeated system prompts and repeated retrieval headers account for most stable overlap during incident review workflows, while customer-specific details change late in the prefix.",
            ),
            (
                "doc-incident-03",
                "Queue handling guidance: if the backup region is already above one busy slot per allocated worker, prioritize local recovery over cross-region spillover to protect TTFT.",
            ),
        ),
        "commerce_support": (
            (
                "doc-commerce-01",
                "Refund policy: expedited shipping credits apply when the promised delivery date is missed and no carrier exception or customer-requested hold is present.",
            ),
            (
                "doc-commerce-02",
                "Order lookup procedure: agents should confirm order id, tenant, destination region, and the most recent fulfillment event before offering compensation.",
            ),
            (
                "doc-commerce-03",
                "Support escalation matrix: repeated warehouse scan gaps usually route to logistics operations, while pricing or duplicate-billing issues route to finance support.",
            ),
        ),
        "analytics_governance": (
            (
                "doc-analytics-01",
                "Retention policy: user-level diagnostics older than thirty days must be summarized before inclusion in prompts sent across regions.",
            ),
            (
                "doc-analytics-02",
                "Dashboard rollout note: retrieval prompts for governance reviews should include the same compliance preamble so follow-up questions reuse prefix state.",
            ),
            (
                "doc-analytics-03",
                "Access control reminder: analyst-facing tools may expose aggregate counters but should redact tenant identifiers in shared reports.",
            ),
        ),
    }


def _default_tool_catalogs() -> dict[str, tuple[tuple[str, str], ...]]:
    return {
        "ops_agent": (
            (
                "lookup_cluster_health",
                '{"arguments":{"cluster_id":"string"},"returns":{"busy_slots":"integer","summary_age_s":"number","queue_depth":"integer"}}',
            ),
            (
                "fetch_prefix_report",
                '{"arguments":{"tenant":"string","window_min":"integer"},"returns":{"top_prefixes":"array","reuse_rate":"number"}}',
            ),
            (
                "reroute_traffic",
                '{"arguments":{"source_cluster":"string","target_cluster":"string","percentage":"number"},"returns":{"accepted":"boolean","change_id":"string"}}',
            ),
        ),
        "commerce_agent": (
            (
                "lookup_order",
                '{"arguments":{"order_id":"string"},"returns":{"status":"string","region":"string","sla_date":"string"}}',
            ),
            (
                "issue_credit",
                '{"arguments":{"order_id":"string","amount":"number","reason":"string"},"returns":{"approved":"boolean","credit_id":"string"}}',
            ),
            (
                "contact_carrier",
                '{"arguments":{"tracking_id":"string","priority":"string"},"returns":{"case_id":"string","eta_note":"string"}}',
            ),
        ),
        "analytics_agent": (
            (
                "run_dashboard_query",
                '{"arguments":{"dashboard":"string","filters":"object"},"returns":{"rows":"array","lag_min":"integer"}}',
            ),
            (
                "get_policy_snippet",
                '{"arguments":{"policy_id":"string"},"returns":{"title":"string","body":"string"}}',
            ),
            (
                "open_review_ticket",
                '{"arguments":{"owner":"string","summary":"string"},"returns":{"ticket_id":"string","queue":"string"}}',
            ),
        ),
    }


_RAG_QUERIES = {
    "incident_ops": (
        "Which step should an on-call engineer take first if prefix overlap looks good but cross-region queue depth is rising?",
        "Summarize the policy for routing additional incident traffic when summaries are stale and the backup region is already partially loaded.",
        "Draft a brief operator note explaining why metadata staleness matters during failover.",
    ),
    "commerce_support": (
        "Based on the policy, when should the agent issue an expedited shipping credit for a late package?",
        "What details should support confirm before escalating a missing handoff scan to logistics operations?",
        "Write a concise answer for an agent handling a late order with no carrier exception.",
    ),
    "analytics_governance": (
        "What prompt content should be removed before sending diagnostics across regions according to the retention policy?",
        "Why would a shared compliance preamble improve prefix reuse for governance dashboards?",
        "Summarize the access-control rule that applies to analyst-facing shared reports.",
    ),
}

_AGENT_PRIOR_CONTEXT = {
    "ops_agent": (
        "Tenant northwind has repeated cache misses on route us-midwest -> us-east after a failover rehearsal. The previous recommendation was to inspect summary freshness before rebalancing.",
        "Cluster cluster-2 reported elevated queue depth but acceptable slot utilization five minutes ago. An operator asked whether more traffic should shift to cluster-0.",
    ),
    "commerce_agent": (
        "Customer says order A1183 missed the expedited promise by one day. Prior lookup showed the parcel crossed regions and the carrier handoff scan is still missing.",
        "A support lead wants to know whether a shipping credit or a carrier escalation should happen first for order C7712.",
    ),
    "analytics_agent": (
        "A reviewer is preparing a governance dashboard for regional leaders and needs to avoid leaking tenant identifiers in the prompt context.",
        "The analytics team suspects a dashboard rollout changed retrieval overlap because the compliance header was shortened last week.",
    ),
}

_AGENT_TASKS = {
    "ops_agent": (
        "Decide which tool to call first and explain the recommendation in one short paragraph for the incident commander.",
        "Plan the minimum safe tool sequence for checking whether traffic should be rerouted.",
    ),
    "commerce_agent": (
        "Explain the next best action for the support agent and identify whether a credit or carrier contact is more appropriate.",
        "Produce a concise operator note that names the first tool call and why it is needed.",
    ),
    "analytics_agent": (
        "State the first tool call and summarize the compliance constraint the analyst must respect.",
        "Plan a short sequence of tool calls to verify the policy and open a review ticket if needed.",
    ),
}

_BURSTY_FOLLOWUPS = (
    "The dashboard still shows stale queue depth. Retry the routing recommendation and highlight anything that changed in the last minute.",
    "The first request timed out for the operator. Re-run the analysis for the same tenant and mention whether the preferred cluster changed.",
    "A teammate asked the same question from another region. Give the shortest operational answer with the safest next action.",
    "The incident channel needs an immediate update. Repeat the recommendation and trim it to three sentences.",
)

_ASSISTANT_BRIDGE_BY_TRAFFIC = {
    CHAT_TRAFFIC_CLASS: "The assistant responded with a concise operator-facing summary.",
    RAG_TRAFFIC_CLASS: "The assistant answered using the retrieved passages and cited the strongest evidence.",
    AGENT_TRAFFIC_CLASS: "The assistant proposed the initial tool plan and summarized the next action.",
}

_FOLLOWUP_PROMPTS_BY_TRAFFIC = {
    CHAT_TRAFFIC_CLASS: (
        "Clarify which assumption in the previous answer is most likely to break if router metadata is stale.",
        "Rewrite the recommendation for an on-call engineer who only has thirty seconds to act.",
        "State one concrete next check if the same issue appears again in another region.",
    ),
    RAG_TRAFFIC_CLASS: (
        "Answer the same question again, but emphasize the strongest cited passage and one caveat.",
        "If one retrieved passage were removed, explain which conclusion would become least certain.",
        "Summarize the retrieved evidence in two sentences for an operations review note.",
    ),
    AGENT_TRAFFIC_CLASS: (
        "Given the previous tool plan, explain the safest next tool call if the first result is inconclusive.",
        "Trim the plan to the minimum set of tools needed before escalating to a human operator.",
        "Describe how the tool sequence changes if the first lookup returns stale information.",
    ),
}


def _build_template_words(
    rng: random.Random,
    length: int,
    template_index: int,
) -> list[str]:
    topics = [
        "cache", "routing", "latency", "region", "tenant", "policy", "prefix", "cluster",
        "summary", "throughput", "queue", "failover", "billing", "search", "catalog",
        "compliance", "document", "inventory", "shipment", "fraud",
    ]
    actions = [
        "compare", "audit", "trace", "review", "summarize", "prioritize", "explain",
        "diagnose", "route", "measure", "monitor", "rebalance",
    ]
    qualifiers = [
        "gold", "silver", "priority", "global", "regional", "stale", "fresh", "burst",
        "cached", "streaming", "shared", "long", "short", "partial", "stable",
    ]
    words: list[str] = [
        "tenant",
        f"group{template_index}",
        rng.choice(actions),
        rng.choice(topics),
        "region",
        rng.choice(("us-east", "us-west", "eu-central", "ap-south")),
        "policy",
        rng.choice(qualifiers),
    ]
    while len(words) < length:
        bucket = rng.random()
        if bucket < 0.4:
            words.append(rng.choice(topics))
        elif bucket < 0.7:
            words.append(rng.choice(actions))
        elif bucket < 0.9:
            words.append(rng.choice(qualifiers))
        else:
            words.append(f"doc{rng.randint(10, 999)}")
    return words[:length]


def _build_unique_words(
    rng: random.Random,
    length: int,
    vocab_size: int,
) -> list[str]:
    suffix_words: list[str] = []
    while len(suffix_words) < length:
        suffix_words.extend(
            [
                rng.choice(
                    (
                        "investigate",
                        "summarize",
                        "prioritize",
                        "compare",
                        "explain",
                        "trace",
                    )
                ),
                rng.choice(
                    (
                        "incident",
                        "request",
                        "workflow",
                        "handoff",
                        "ticket",
                        "report",
                    )
                ),
                f"id{rng.randint(1, vocab_size)}",
            ]
        )
    return suffix_words[:length]


def _lexical_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_-]+|[^\w\s]", text)


def _stable_token_id(piece: str) -> int:
    digest = hashlib.blake2b(piece.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)
