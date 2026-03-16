from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException

from orbit.common.config import RouterConfig
from orbit.common.schemas import (
    CacheUpdate,
    ChatMessage,
    Choice,
    InferenceRequest,
    InferenceResponse,
    LoadUpdate,
    RoutingDecision,
    UsageInfo,
)
from orbit.router.cache_registry import CacheRegistry
from orbit.router.routing_engine import STRATEGIES, RoutingEngine

logger = logging.getLogger("orbit.router")

config = RouterConfig.from_env()
engine = RoutingEngine(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Orbit Router starting on port {config.port}")
    logger.info(f"Registered replicas: {list(engine._replica_urls.items())}")
    yield
    logger.info("Orbit Router shutting down")


app = FastAPI(title="Orbit Router", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: InferenceRequest) -> InferenceResponse:
    """Active routing mode: route and proxy the request to the best replica."""
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Route
    decision = engine.route(messages)
    logger.info(
        f"Routed to {decision.selected_replica} "
        f"(cached={decision.cached_tokens}/{decision.total_prompt_tokens}, "
        f"score={decision.score:.3f}, reason={decision.reason})"
    )

    # Proxy to selected replica
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{decision.replica_url}/v1/chat/completions",
                json=request.model_dump(),
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Replica error: {e}")

    # Parse and augment response with routing metadata
    response = InferenceResponse(**data)
    return response


@app.post("/v1/route")
async def advisory_route(request: InferenceRequest) -> RoutingDecision:
    """Advisory mode: return routing decision without forwarding."""
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    decision = engine.route(messages)
    return decision


@app.post("/v1/cache/update")
async def cache_update(update: CacheUpdate) -> dict:
    """Receive cache update from a replica."""
    engine.cache_registry.apply_update(update)
    logger.debug(
        f"Cache update from {update.replica_id}: "
        f"{update.update_type.value} {len(update.blocks)} blocks"
    )
    return {"status": "ok"}


@app.post("/v1/load/update")
async def load_update(update: LoadUpdate) -> dict:
    """Receive load update from the monitor."""
    engine.load_tracker.update(update)
    return {"status": "ok"}


@app.post("/v1/cache/reset")
async def cache_reset() -> dict:
    """Clear the router's cache registry (used between benchmark runs)."""
    engine.cache_registry = CacheRegistry()
    engine._rr_counter = 0
    logger.info("Cache registry reset")
    return {"status": "cache_cleared"}


@app.post("/v1/strategy")
async def set_strategy(body: dict) -> dict:
    """Switch the routing strategy (used by benchmarks)."""
    strategy = body.get("strategy", "orbit")
    if strategy not in STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy '{strategy}'. Valid: {sorted(STRATEGIES)}",
        )
    engine.strategy = strategy
    engine._rr_counter = 0
    logger.info(f"Routing strategy set to: {strategy}")
    return {"strategy": strategy}


@app.get("/v1/status")
async def get_status() -> dict:
    """Return router status including registered replicas and cache state."""
    return {
        "replicas": list(engine._replica_urls.keys()),
        "load": {
            rid: {
                "active": load.active_requests,
                "queued": load.queued_requests,
                "congestion": load.normalized_congestion,
            }
            for rid, load in engine.load_tracker.get_all().items()
        },
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
