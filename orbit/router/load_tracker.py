from __future__ import annotations

import threading
from dataclasses import dataclass, field

from orbit.common.schemas import LoadUpdate


@dataclass
class ReplicaLoad:
    replica_id: str
    active_requests: int = 0
    queued_requests: int = 0
    max_concurrent: int = 4
    cache_used_blocks: int = 0
    cache_capacity_blocks: int = 1000

    @property
    def normalized_congestion(self) -> float:
        """Congestion as fraction of max capacity (0.0 = idle, 1.0+ = overloaded)."""
        if self.max_concurrent == 0:
            return 1.0
        return (self.active_requests + self.queued_requests) / self.max_concurrent


class LoadTracker:
    """Tracks per-replica load state, updated by the monitor."""

    def __init__(self):
        self._replicas: dict[str, ReplicaLoad] = {}
        self._lock = threading.Lock()

    def update(self, load: LoadUpdate) -> None:
        with self._lock:
            self._replicas[load.replica_id] = ReplicaLoad(
                replica_id=load.replica_id,
                active_requests=load.active_requests,
                queued_requests=load.queued_requests,
                max_concurrent=load.max_concurrent,
                cache_used_blocks=load.cache_used_blocks,
                cache_capacity_blocks=load.cache_capacity_blocks,
            )

    def get_load(self, replica_id: str) -> ReplicaLoad:
        with self._lock:
            return self._replicas.get(replica_id, ReplicaLoad(replica_id=replica_id))

    def get_all(self) -> dict[str, ReplicaLoad]:
        with self._lock:
            return dict(self._replicas)

    def get_least_loaded(self) -> str | None:
        with self._lock:
            if not self._replicas:
                return None
            return min(
                self._replicas.values(),
                key=lambda r: r.normalized_congestion,
            ).replica_id

    def get_min_congestion(self) -> float:
        with self._lock:
            if not self._replicas:
                return 0.0
            return min(r.normalized_congestion for r in self._replicas.values())
