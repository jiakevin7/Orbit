from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


# ── OpenAI-compatible chat types ──


class ChatMessage(BaseModel):
    role: str
    content: str


class InferenceRequest(BaseModel):
    model: str = "orbit-sim"
    messages: list[ChatMessage]
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = False


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class InferenceResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str = "orbit-sim"
    choices: list[Choice]
    usage: UsageInfo
    # Orbit-specific metadata
    orbit_replica_id: str | None = None
    orbit_cached_tokens: int = 0
    orbit_prefill_ms: float = 0.0
    orbit_decode_ms: float = 0.0
    orbit_queue_ms: float = 0.0


# ── Replica status ──


class ReplicaStatus(BaseModel):
    replica_id: str
    active_requests: int
    queued_requests: int
    max_concurrent: int
    cache_used_blocks: int
    cache_capacity_blocks: int
    cache_hit_rate: float = 0.0


# ── Cache update messages (replica → router) ──


class CacheUpdateType(str, Enum):
    INSERT = "insert"
    EVICT = "evict"


class CacheBlockInfo(BaseModel):
    block_hash: str
    block_index: int
    prefix_hash: str  # hash of the full prefix up to and including this block


class CacheUpdate(BaseModel):
    replica_id: str
    update_type: CacheUpdateType
    blocks: list[CacheBlockInfo]


# ── Routing decision ──


class RoutingDecision(BaseModel):
    selected_replica: str
    replica_url: str
    cached_tokens: int = 0
    total_prompt_tokens: int = 0
    score: float = 0.0
    reason: str = ""


# ── Load update (monitor → router) ──


class LoadUpdate(BaseModel):
    replica_id: str
    active_requests: int
    queued_requests: int
    max_concurrent: int
    cache_used_blocks: int
    cache_capacity_blocks: int
