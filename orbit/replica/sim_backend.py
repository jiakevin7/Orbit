from __future__ import annotations

import asyncio
import random
import string

from orbit.common.config import ReplicaConfig
from orbit.common.hashing import get_tokenizer
from orbit.replica.backend_interface import BackendInterface, BackendResult
from orbit.replica.kv_cache import KVCache


class SimulatedBackend(BackendInterface):
    """Simulated LLM backend with realistic timing and KV cache behavior."""

    def __init__(self, config: ReplicaConfig):
        self.config = config
        self.kv_cache = KVCache(
            capacity_blocks=config.cache_capacity_blocks,
            block_size=config.block_size,
        )
        self._tokenizer = get_tokenizer(config.tokenizer_name)

        # Concurrency control
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._active_requests = 0
        self._queued_requests = 0
        self._lock = asyncio.Lock()

    async def generate(
        self,
        token_ids: list[int],
        max_tokens: int,
        temperature: float,
    ) -> BackendResult:
        # Track queued requests
        async with self._lock:
            self._queued_requests += 1

        queue_start = asyncio.get_event_loop().time()

        async with self._semaphore:
            async with self._lock:
                self._queued_requests -= 1
                self._active_requests += 1

            queue_end = asyncio.get_event_loop().time()
            queue_ms = (queue_end - queue_start) * 1000

            try:
                result = await self._do_generate(token_ids, max_tokens, temperature, queue_ms)
            finally:
                async with self._lock:
                    self._active_requests -= 1

        return result

    async def _do_generate(
        self,
        token_ids: list[int],
        max_tokens: int,
        temperature: float,
        queue_ms: float,
    ) -> BackendResult:
        total_prompt_tokens = len(token_ids)

        # Check KV cache for prefix match
        cached_tokens = self.kv_cache.lookup_prefix(token_ids)

        # Simulate prefill time: only uncached tokens need prefill
        uncached_tokens = total_prompt_tokens - cached_tokens
        prefill_ms = self.config.prefill_ms_per_token * uncached_tokens

        # Simulate decode time
        output_tokens = min(max_tokens, random.randint(max(1, max_tokens // 2), max_tokens))
        decode_ms = self.config.decode_ms_per_token * output_tokens

        # Simulate the computation time
        total_sim_ms = prefill_ms + decode_ms
        await asyncio.sleep(total_sim_ms / 1000.0)

        # Insert prompt into KV cache
        self.kv_cache.insert_prefix(token_ids)

        # Generate dummy output text
        output_text = self._generate_dummy_text(output_tokens)

        return BackendResult(
            output_text=output_text,
            output_tokens=output_tokens,
            prompt_tokens=total_prompt_tokens,
            cached_tokens=cached_tokens,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            queue_ms=queue_ms,
        )

    def _generate_dummy_text(self, num_tokens: int) -> str:
        words = []
        for _ in range(num_tokens):
            word_len = random.randint(2, 8)
            words.append("".join(random.choices(string.ascii_lowercase, k=word_len)))
        return " ".join(words)

    def get_status(self) -> dict:
        return {
            "active_requests": self._active_requests,
            "queued_requests": self._queued_requests,
            "max_concurrent": self.config.max_concurrent,
            "cache_used_blocks": self.kv_cache.used_blocks,
            "cache_capacity_blocks": self.config.cache_capacity_blocks,
            "cache_hit_rate": self.kv_cache.hit_rate,
        }
