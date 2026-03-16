import asyncio

import pytest

from orbit.common.config import ReplicaConfig
from orbit.replica.kv_cache import KVCache
from orbit.replica.sim_backend import SimulatedBackend


class TestKVCache:
    def test_empty_cache(self):
        cache = KVCache(capacity_blocks=10, block_size=16)
        assert cache.used_blocks == 0
        assert cache.hit_rate == 0.0

    def test_insert_and_lookup(self):
        cache = KVCache(capacity_blocks=100, block_size=16)
        tokens = list(range(48))  # 3 full blocks

        cache.insert_prefix(tokens)
        assert cache.used_blocks == 3

        cached = cache.lookup_prefix(tokens)
        assert cached == 48  # all 3 blocks cached

    def test_partial_match(self):
        cache = KVCache(capacity_blocks=100, block_size=16)
        tokens_a = list(range(48))  # 3 blocks
        tokens_b = list(range(32)) + [999] * 16  # first 2 blocks match, 3rd differs

        cache.insert_prefix(tokens_a)
        cached = cache.lookup_prefix(tokens_b)
        assert cached == 32  # only first 2 blocks match

    def test_no_match(self):
        cache = KVCache(capacity_blocks=100, block_size=16)
        cache.insert_prefix(list(range(32)))

        cached = cache.lookup_prefix([999] * 32)
        assert cached == 0

    def test_lru_eviction(self):
        cache = KVCache(capacity_blocks=3, block_size=16)

        # Insert 3 blocks (fills capacity)
        cache.insert_prefix(list(range(48)))
        assert cache.used_blocks == 3

        # Insert 1 more block (different prefix) — should evict oldest
        cache.insert_prefix([999] * 16)
        assert cache.used_blocks == 3

    def test_drain_updates(self):
        cache = KVCache(capacity_blocks=100, block_size=16)
        cache.insert_prefix(list(range(32)))

        inserted, evicted = cache.drain_updates()
        assert len(inserted) == 2
        assert len(evicted) == 0

        # Second drain should be empty
        inserted2, evicted2 = cache.drain_updates()
        assert len(inserted2) == 0

    def test_eviction_tracked(self):
        cache = KVCache(capacity_blocks=2, block_size=16)
        cache.insert_prefix(list(range(32)))  # 2 blocks
        cache.drain_updates()  # clear

        # This forces eviction
        cache.insert_prefix([999] * 16)
        inserted, evicted = cache.drain_updates()
        assert len(inserted) == 1
        assert len(evicted) == 1

    def test_short_input(self):
        cache = KVCache(capacity_blocks=100, block_size=16)
        # Less than one block
        cache.insert_prefix([1, 2, 3])
        assert cache.used_blocks == 0

    def test_get_all_block_hashes(self):
        cache = KVCache(capacity_blocks=100, block_size=16)
        cache.insert_prefix(list(range(32)))
        hashes = cache.get_all_block_hashes()
        assert len(hashes) == 2

    def test_clear(self):
        cache = KVCache(capacity_blocks=100, block_size=16)
        cache.insert_prefix(list(range(32)))
        cache.clear()
        assert cache.used_blocks == 0


class TestSimulatedBackend:
    @pytest.fixture
    def config(self):
        return ReplicaConfig(
            cache_capacity_blocks=100,
            block_size=16,
            prefill_ms_per_token=0.1,  # faster for tests
            decode_ms_per_token=1.0,
            max_concurrent=2,
        )

    @pytest.fixture
    def backend(self, config):
        return SimulatedBackend(config)

    async def test_basic_generate(self, backend):
        tokens = list(range(32))
        result = await backend.generate(tokens, max_tokens=10, temperature=0.0)

        assert result.prompt_tokens == 32
        assert result.cached_tokens == 0  # first request, nothing cached
        assert result.output_tokens > 0
        assert result.prefill_ms > 0
        assert result.decode_ms > 0
        assert len(result.output_text) > 0

    async def test_cache_hit_on_repeat(self, backend):
        tokens = list(range(32))

        # First request
        r1 = await backend.generate(tokens, max_tokens=5, temperature=0.0)
        assert r1.cached_tokens == 0

        # Second request with same tokens
        r2 = await backend.generate(tokens, max_tokens=5, temperature=0.0)
        assert r2.cached_tokens == 32  # full prefix cached
        assert r2.prefill_ms < r1.prefill_ms  # should be faster

    async def test_get_status(self, backend):
        status = backend.get_status()
        assert "active_requests" in status
        assert "cache_used_blocks" in status
        assert status["max_concurrent"] == 2

    async def test_concurrent_limit(self, backend):
        """Requests beyond max_concurrent should queue."""
        tokens = list(range(32))

        # Launch 4 concurrent requests (max_concurrent=2)
        tasks = [
            asyncio.create_task(
                backend.generate(tokens, max_tokens=5, temperature=0.0)
            )
            for _ in range(4)
        ]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r.output_tokens > 0 for r in results)
        # Some should have queue wait
        queue_times = [r.queue_ms for r in results]
        assert any(q > 0 for q in queue_times)  # at least one was queued
