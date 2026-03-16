from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from orbit.common.config import ReplicaConfig
from orbit.common.hashing import get_tokenizer
from orbit.common.schemas import (
    CacheBlockInfo,
    CacheUpdate,
    CacheUpdateType,
    ChatMessage,
    Choice,
    InferenceRequest,
    InferenceResponse,
    ReplicaStatus,
    UsageInfo,
)
from orbit.replica.backend_interface import BackendInterface
from orbit.replica.sim_backend import SimulatedBackend

logger = logging.getLogger("orbit.replica")

config = ReplicaConfig.from_env()
tokenizer = get_tokenizer(config.tokenizer_name)


def _create_backend(config: ReplicaConfig) -> BackendInterface:
    if config.backend == "llamacpp":
        from orbit.replica.llamacpp_backend import LlamaCppBackend
        logger.info(f"Using llama.cpp backend at {config.llamacpp_url}")
        return LlamaCppBackend(config)
    else:
        logger.info("Using simulated backend")
        return SimulatedBackend(config)


backend = _create_backend(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Replica {config.replica_id} starting on port {config.port}")
    yield
    logger.info(f"Replica {config.replica_id} shutting down")


app = FastAPI(title=f"Orbit Replica ({config.replica_id})", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: InferenceRequest) -> InferenceResponse:
    # Tokenize messages
    all_tokens: list[int] = []
    for msg in request.messages:
        text = f"<|{msg.role}|>{msg.content}"
        all_tokens.extend(tokenizer.encode(text))

    # Generate
    result = await backend.generate(
        token_ids=all_tokens,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    # Push cache updates to router (best-effort, non-blocking)
    await _push_cache_updates()

    return InferenceResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        model=request.model,
        choices=[
            Choice(
                message=ChatMessage(role="assistant", content=result.output_text),
            )
        ],
        usage=UsageInfo(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.output_tokens,
            total_tokens=result.prompt_tokens + result.output_tokens,
        ),
        orbit_replica_id=config.replica_id,
        orbit_cached_tokens=result.cached_tokens,
        orbit_prefill_ms=result.prefill_ms,
        orbit_decode_ms=result.decode_ms,
        orbit_queue_ms=result.queue_ms,
    )


@app.get("/v1/status")
async def get_status() -> ReplicaStatus:
    status = backend.get_status()
    return ReplicaStatus(
        replica_id=config.replica_id,
        **status,
    )


@app.get("/v1/cache/blocks")
async def get_cache_blocks() -> dict:
    """Return all block hashes in cache (for reconciliation)."""
    hashes = backend.kv_cache.get_all_block_hashes()
    return {"replica_id": config.replica_id, "block_hashes": hashes}


@app.post("/v1/cache/reset")
async def reset_cache() -> dict:
    """Clear the KV cache (used between benchmark runs)."""
    backend.kv_cache.clear()
    backend.kv_cache.total_lookups = 0
    backend.kv_cache.total_hits = 0
    await backend.reset_cache()
    return {"replica_id": config.replica_id, "status": "cache_cleared"}


async def _push_cache_updates() -> None:
    """Push pending cache insert/evict events to the router."""
    inserted, evicted = backend.kv_cache.drain_updates()

    updates: list[CacheUpdate] = []

    if inserted:
        updates.append(CacheUpdate(
            replica_id=config.replica_id,
            update_type=CacheUpdateType.INSERT,
            blocks=[
                CacheBlockInfo(
                    block_hash=b.block_hash,
                    block_index=b.block_index,
                    prefix_hash=b.block_hash,  # for chained hashing, block hash IS prefix hash
                )
                for b in inserted
            ],
        ))

    if evicted:
        updates.append(CacheUpdate(
            replica_id=config.replica_id,
            update_type=CacheUpdateType.EVICT,
            blocks=[
                CacheBlockInfo(
                    block_hash=b.block_hash,
                    block_index=b.block_index,
                    prefix_hash=b.block_hash,
                )
                for b in evicted
            ],
        ))

    if not updates:
        return

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            for update in updates:
                await client.post(
                    f"{config.router_url}/v1/cache/update",
                    json=update.model_dump(),
                )
    except Exception as e:
        logger.debug(f"Failed to push cache update to router: {e}")
