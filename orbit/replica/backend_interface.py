from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BackendResult:
    output_text: str
    output_tokens: int
    prompt_tokens: int
    cached_tokens: int
    prefill_ms: float
    decode_ms: float
    queue_ms: float


class BackendInterface(ABC):
    @abstractmethod
    async def generate(
        self,
        token_ids: list[int],
        max_tokens: int,
        temperature: float,
    ) -> BackendResult:
        ...

    @abstractmethod
    def get_status(self) -> dict:
        ...

    async def reset_cache(self) -> None:
        """Reset the backend's KV cache. Override for real backends."""
        pass
