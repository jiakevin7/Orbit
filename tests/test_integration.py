import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from orbit.common.config import ReplicaConfig
from orbit.replica.sim_backend import SimulatedBackend


class TestReplicaEndToEnd:
    """Integration tests using FastAPI test client against the replica."""

    @pytest.fixture
    def replica_config(self, monkeypatch):
        monkeypatch.setenv("ORBIT_PORT", "8001")
        monkeypatch.setenv("ORBIT_REPLICA_ID", "test-replica")
        monkeypatch.setenv("ORBIT_ROUTER_URL", "http://localhost:9999")  # no real router
        monkeypatch.setenv("ORBIT_CACHE_CAPACITY", "100")
        monkeypatch.setenv("ORBIT_BLOCK_SIZE", "16")
        monkeypatch.setenv("ORBIT_PREFILL_MS", "0.1")
        monkeypatch.setenv("ORBIT_DECODE_MS", "0.5")
        monkeypatch.setenv("ORBIT_MAX_CONCURRENT", "4")

    def test_chat_completion(self, replica_config):
        # Re-import to pick up env vars
        import importlib
        import orbit.replica.app as replica_app
        importlib.reload(replica_app)

        client = TestClient(replica_app.app)

        response = client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. " * 20},
                {"role": "user", "content": "What is Python?"},
            ],
            "max_tokens": 10,
        })

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["orbit_replica_id"] == "test-replica"

    def test_status_endpoint(self, replica_config):
        import importlib
        import orbit.replica.app as replica_app
        importlib.reload(replica_app)

        client = TestClient(replica_app.app)
        response = client.get("/v1/status")

        assert response.status_code == 200
        data = response.json()
        assert "replica_id" in data
        assert "active_requests" in data
        assert "cache_used_blocks" in data

    def test_cache_blocks_endpoint(self, replica_config):
        import importlib
        import orbit.replica.app as replica_app
        importlib.reload(replica_app)

        client = TestClient(replica_app.app)
        response = client.get("/v1/cache/blocks")

        assert response.status_code == 200
        data = response.json()
        assert "block_hashes" in data

    def test_repeated_requests_show_cache_hits(self, replica_config):
        import importlib
        import orbit.replica.app as replica_app
        importlib.reload(replica_app)

        client = TestClient(replica_app.app)

        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant. " * 20},
                {"role": "user", "content": "Write a hello world program."},
            ],
            "max_tokens": 10,
        }

        # First request — no cache
        r1 = client.post("/v1/chat/completions", json=payload)
        assert r1.status_code == 200
        assert r1.json()["orbit_cached_tokens"] == 0

        # Second request — should have cache hit
        r2 = client.post("/v1/chat/completions", json=payload)
        assert r2.status_code == 200
        assert r2.json()["orbit_cached_tokens"] > 0
        assert r2.json()["orbit_prefill_ms"] < r1.json()["orbit_prefill_ms"]
