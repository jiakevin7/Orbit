from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from orbit.common.config import MonitorConfig
from orbit.monitor.collector import LoadCollector

logger = logging.getLogger("orbit.monitor")

config = MonitorConfig.from_env()
collector = LoadCollector(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Monitor starting on port {config.port}")
    task = asyncio.create_task(collector.start())
    yield
    collector.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    logger.info("Monitor shutting down")


app = FastAPI(title="Orbit Monitor", lifespan=lifespan)


@app.get("/v1/status")
async def get_status() -> dict:
    """Return latest collected status from all replicas."""
    latest = collector.get_latest()
    return {
        "replicas": {
            rid: status.model_dump()
            for rid, status in latest.items()
        }
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
