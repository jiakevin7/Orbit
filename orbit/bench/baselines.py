from __future__ import annotations

import hashlib
import random
from abc import ABC, abstractmethod

from orbit.common.schemas import InferenceRequest


class BaselineRouter(ABC):
    """Base class for baseline routing strategies."""

    def __init__(self, replica_urls: list[str]):
        self.replica_urls = replica_urls
        self.name = self.__class__.__name__

    @abstractmethod
    def select(self, request: InferenceRequest) -> str:
        """Return the URL of the selected replica."""
        ...


class RoundRobinRouter(BaselineRouter):
    """Simple round-robin routing."""

    def __init__(self, replica_urls: list[str]):
        super().__init__(replica_urls)
        self.name = "round_robin"
        self._counter = 0

    def select(self, request: InferenceRequest) -> str:
        url = self.replica_urls[self._counter % len(self.replica_urls)]
        self._counter += 1
        return url


class RandomRouter(BaselineRouter):
    """Random routing."""

    def __init__(self, replica_urls: list[str]):
        super().__init__(replica_urls)
        self.name = "random"

    def select(self, request: InferenceRequest) -> str:
        return random.choice(self.replica_urls)


class HashRouter(BaselineRouter):
    """Hash-based routing on the full prompt content."""

    def __init__(self, replica_urls: list[str]):
        super().__init__(replica_urls)
        self.name = "hash_based"

    def select(self, request: InferenceRequest) -> str:
        # Hash the full message content
        content = "".join(m.content for m in request.messages)
        h = hashlib.sha256(content.encode()).hexdigest()
        idx = int(h, 16) % len(self.replica_urls)
        return self.replica_urls[idx]


class LeastLoadedRouter(BaselineRouter):
    """Least-loaded routing (queries replica status each time)."""

    def __init__(self, replica_urls: list[str]):
        super().__init__(replica_urls)
        self.name = "least_loaded"
        self._loads: dict[str, int] = {url: 0 for url in replica_urls}

    def select(self, request: InferenceRequest) -> str:
        # Pick the URL with lowest tracked load
        url = min(self._loads, key=lambda u: self._loads[u])
        self._loads[url] += 1
        return url

    def report_done(self, url: str) -> None:
        self._loads[url] = max(0, self._loads.get(url, 0) - 1)


BASELINE_ROUTERS = {
    "round_robin": RoundRobinRouter,
    "random": RandomRouter,
    "hash_based": HashRouter,
    "least_loaded": LeastLoadedRouter,
}
