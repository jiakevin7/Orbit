import pytest

from orbit.common.config import RouterConfig
from orbit.common.schemas import (
    CacheBlockInfo,
    CacheUpdate,
    CacheUpdateType,
    LoadUpdate,
)
from orbit.router.routing_engine import RoutingEngine


@pytest.fixture
def config():
    return RouterConfig(
        replica_urls=[
            "http://replica-1:8001",
            "http://replica-2:8001",
            "http://replica-3:8001",
        ],
        alpha=1.0,
        beta=0.5,
        block_size=16,
    )


@pytest.fixture
def engine(config):
    return RoutingEngine(config)


class TestRoutingEngine:
    def test_round_robin_fallback(self, engine):
        """With no cache data, should round-robin."""
        messages = [{"role": "user", "content": "hello"}]
        d1 = engine.route(messages)
        d2 = engine.route(messages)
        d3 = engine.route(messages)

        # Should cycle through replicas
        replicas = {d1.selected_replica, d2.selected_replica, d3.selected_replica}
        assert len(replicas) >= 2  # at least 2 different replicas chosen

    def test_short_prompt_fallback(self, engine):
        """Prompts shorter than one block should fallback gracefully."""
        messages = [{"role": "user", "content": "hi"}]
        decision = engine.route(messages)
        assert decision.selected_replica in ["replica-1", "replica-2", "replica-3"]
        assert "fallback" in decision.reason

    def test_cache_affinity(self, engine):
        """After cache update, requests with matching prefix should route to cached replica."""
        # Generate a long system prompt that will produce block hashes
        sys_content = "You are a helpful assistant. " * 50
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": "What is Python?"},
        ]

        # Analyze to get block hashes
        analyzed = engine.prompt_analyzer.analyze(messages)
        assert len(analyzed.block_hashes) > 0, "Need block hashes for this test"

        # Simulate cache update from replica-2
        block_infos = [
            CacheBlockInfo(block_hash=bh, block_index=i, prefix_hash=bh)
            for i, bh in enumerate(analyzed.block_hashes)
        ]
        engine.cache_registry.apply_update(CacheUpdate(
            replica_id="replica-2",
            update_type=CacheUpdateType.INSERT,
            blocks=block_infos,
        ))

        # Set up load tracking
        for rid in ["replica-1", "replica-2", "replica-3"]:
            engine.load_tracker.update(LoadUpdate(
                replica_id=rid,
                active_requests=1,
                queued_requests=0,
                max_concurrent=4,
                cache_used_blocks=0,
                cache_capacity_blocks=1000,
            ))

        # Route — should prefer replica-2
        decision = engine.route(messages)
        assert decision.selected_replica == "replica-2"
        assert decision.cached_tokens > 0

    def test_least_loaded_fallback(self, engine):
        """With load data but no cache, should pick least-loaded."""
        engine.load_tracker.update(LoadUpdate(
            replica_id="replica-1",
            active_requests=4,
            queued_requests=2,
            max_concurrent=4,
            cache_used_blocks=0,
            cache_capacity_blocks=1000,
        ))
        engine.load_tracker.update(LoadUpdate(
            replica_id="replica-2",
            active_requests=0,
            queued_requests=0,
            max_concurrent=4,
            cache_used_blocks=0,
            cache_capacity_blocks=1000,
        ))
        engine.load_tracker.update(LoadUpdate(
            replica_id="replica-3",
            active_requests=2,
            queued_requests=0,
            max_concurrent=4,
            cache_used_blocks=0,
            cache_capacity_blocks=1000,
        ))

        messages = [
            {"role": "system", "content": "Be helpful. " * 50},
            {"role": "user", "content": "Hello"},
        ]
        decision = engine.route(messages)
        assert decision.selected_replica == "replica-2"
        assert "fallback_least_loaded" in decision.reason

    def test_cache_convergence(self, engine):
        """Repeated requests with same system prompt should converge to one replica."""
        sys_content = "You are a specialized coding assistant for data science. " * 30
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": "Help me with pandas"},
        ]

        # Set equal load
        for rid in ["replica-1", "replica-2", "replica-3"]:
            engine.load_tracker.update(LoadUpdate(
                replica_id=rid,
                active_requests=1,
                queued_requests=0,
                max_concurrent=4,
                cache_used_blocks=0,
                cache_capacity_blocks=1000,
            ))

        # First request goes somewhere (round-robin/least-loaded)
        d1 = engine.route(messages)
        first_replica = d1.selected_replica

        # Simulate that replica caching the prefix
        analyzed = engine.prompt_analyzer.analyze(messages)
        block_infos = [
            CacheBlockInfo(block_hash=bh, block_index=i, prefix_hash=bh)
            for i, bh in enumerate(analyzed.block_hashes)
        ]
        engine.cache_registry.apply_update(CacheUpdate(
            replica_id=first_replica,
            update_type=CacheUpdateType.INSERT,
            blocks=block_infos,
        ))

        # Subsequent requests should converge to the same replica
        for _ in range(5):
            d = engine.route(messages)
            assert d.selected_replica == first_replica
            assert d.cached_tokens > 0
