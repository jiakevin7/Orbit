from __future__ import annotations

import asyncio
import logging

import httpx

from orbit.common.config import MonitorConfig
from orbit.common.schemas import LoadUpdate, ReplicaStatus

logger = logging.getLogger("orbit.monitor")


class LoadCollector:
    """Polls replicas for load status and pushes to the router."""

    def __init__(self, config: MonitorConfig):
        self.config = config
        self._latest: dict[str, ReplicaStatus] = {}
        self._running = False

    async def start(self) -> None:
        self._running = True
        logger.info(
            f"Monitor starting: polling {len(self.config.replica_urls)} replicas "
            f"every {self.config.poll_interval_s}s"
        )
        while self._running:
            await self._poll_all()
            await asyncio.sleep(self.config.poll_interval_s)

    def stop(self) -> None:
        self._running = False

    async def _poll_all(self) -> None:
        async with httpx.AsyncClient(timeout=5.0) as client:
            tasks = [
                self._poll_replica(client, url)
                for url in self.config.replica_urls
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _poll_replica(self, client: httpx.AsyncClient, url: str) -> None:
        try:
            resp = await client.get(f"{url}/v1/status")
            resp.raise_for_status()
            status = ReplicaStatus(**resp.json())
            self._latest[status.replica_id] = status

            # Push to router
            load_update = LoadUpdate(
                replica_id=status.replica_id,
                active_requests=status.active_requests,
                queued_requests=status.queued_requests,
                max_concurrent=status.max_concurrent,
                cache_used_blocks=status.cache_used_blocks,
                cache_capacity_blocks=status.cache_capacity_blocks,
            )
            await client.post(
                f"{self.config.router_url}/v1/load/update",
                json=load_update.model_dump(),
            )
        except Exception as e:
            logger.debug(f"Failed to poll {url}: {e}")

    def get_latest(self) -> dict[str, ReplicaStatus]:
        return dict(self._latest)
