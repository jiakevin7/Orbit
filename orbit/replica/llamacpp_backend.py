from __future__ import annotations

import asyncio
import logging

import httpx

from orbit.common.config import ReplicaConfig
from orbit.replica.backend_interface import BackendInterface, BackendResult
from orbit.replica.kv_cache import KVCache

logger = logging.getLogger("orbit.replica.llamacpp")


class LlamaCppBackend(BackendInterface):
    """Backend that proxies to a real llama.cpp server instance.

    Uses llama.cpp for actual inference and timing, but maintains a shadow
    KV cache for block hash tracking (so the router knows what's cached where).

    llama.cpp's own prompt caching handles the real KV reuse — the shadow
    cache just mirrors what should be cached for routing decisions.
    """

    def __init__(self, config: ReplicaConfig):
        self.config = config
        self.llamacpp_url = config.llamacpp_url

        # Shadow cache: tracks block hashes for the router's prefix trie.
        # Does NOT affect inference timing — llama.cpp handles that.
        self.kv_cache = KVCache(
            capacity_blocks=config.cache_capacity_blocks,
            block_size=config.block_size,
        )

        self._active_requests = 0
        self._queued_requests = 0

    async def generate(
        self,
        token_ids: list[int],
        max_tokens: int,
        temperature: float,
    ) -> BackendResult:
        self._active_requests += 1
        try:
            result = await self._call_llamacpp(token_ids, max_tokens, temperature)
        finally:
            self._active_requests -= 1

        # Update shadow cache so the router knows this prefix is cached here
        self.kv_cache.insert_prefix(token_ids)

        return result

    async def _call_llamacpp(
        self,
        token_ids: list[int],
        max_tokens: int,
        temperature: float,
    ) -> BackendResult:
        """Call llama.cpp's /completion endpoint for detailed timing."""
        # Use the native /completion endpoint — it returns prompt_eval and
        # generation timing that the OpenAI-compat endpoint doesn't expose.
        payload = {
            "prompt": token_ids,  # llama.cpp accepts token ID arrays directly
            "n_predict": max_tokens,
            "temperature": temperature,
            "cache_prompt": True,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            # Retry on transient errors (e.g. llama.cpp restarting between runs)
            for attempt in range(3):
                resp = await client.post(
                    f"{self.llamacpp_url}/completion",
                    json=payload,
                )
                if resp.status_code < 500:
                    break
                logger.warning(f"llama.cpp returned {resp.status_code}, retrying ({attempt+1}/3)")
                await asyncio.sleep(2)
            resp.raise_for_status()
            data = resp.json()

        # Extract timing from llama.cpp response
        timings = data.get("timings", {})
        prompt_tokens = timings.get("prompt_n", len(token_ids))
        prompt_ms = timings.get("prompt_ms", 0.0)
        predicted_tokens = timings.get("predicted_n", 0)
        predicted_ms = timings.get("predicted_ms", 0.0)

        # llama.cpp reports prompt_n as the number of tokens actually evaluated
        # (not cached). If prefix caching kicked in, prompt_n < len(token_ids).
        cached_tokens = len(token_ids) - prompt_tokens

        output_text = data.get("content", "")

        return BackendResult(
            output_text=output_text,
            output_tokens=predicted_tokens,
            prompt_tokens=len(token_ids),
            cached_tokens=max(0, cached_tokens),
            prefill_ms=prompt_ms,
            decode_ms=predicted_ms,
            queue_ms=0.0,
        )

    def get_status(self) -> dict:
        return {
            "active_requests": self._active_requests,
            "queued_requests": self._queued_requests,
            "max_concurrent": self.config.max_concurrent,
            "cache_used_blocks": self.kv_cache.used_blocks,
            "cache_capacity_blocks": self.config.cache_capacity_blocks,
            "cache_hit_rate": self.kv_cache.hit_rate,
        }

    async def reset_cache(self) -> None:
        """Reset shadow cache. Real llama.cpp cache requires process restart."""
        pass
