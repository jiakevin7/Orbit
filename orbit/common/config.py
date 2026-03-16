from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class RouterConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    replica_urls: list[str] = field(default_factory=list)
    monitor_url: str = "http://orbit-monitor:8080"

    # Scoring weights
    alpha: float = 1.0  # weight for prefill savings
    beta: float = 0.5   # weight for congestion penalty

    # Semantic pre-filter
    semantic_top_k: int = 3
    semantic_threshold: float = 0.5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Strong hit override: if cache overlap > this fraction AND load < 2x min, always pick it
    strong_hit_threshold: float = 0.9

    # Block hashing
    block_size: int = 16
    tokenizer_name: str = "cl100k_base"

    @classmethod
    def from_env(cls) -> RouterConfig:
        replica_urls_str = os.environ.get("ORBIT_REPLICA_URLS", "")
        replica_urls = [u.strip() for u in replica_urls_str.split(",") if u.strip()]
        return cls(
            host=os.environ.get("ORBIT_HOST", "0.0.0.0"),
            port=int(os.environ.get("ORBIT_PORT", "8000")),
            replica_urls=replica_urls,
            monitor_url=os.environ.get("ORBIT_MONITOR_URL", "http://orbit-monitor:8080"),
            alpha=float(os.environ.get("ORBIT_ALPHA", "1.0")),
            beta=float(os.environ.get("ORBIT_BETA", "0.5")),
            semantic_top_k=int(os.environ.get("ORBIT_SEMANTIC_TOP_K", "3")),
            semantic_threshold=float(os.environ.get("ORBIT_SEMANTIC_THRESHOLD", "0.5")),
            strong_hit_threshold=float(os.environ.get("ORBIT_STRONG_HIT_THRESHOLD", "0.9")),
            block_size=int(os.environ.get("ORBIT_BLOCK_SIZE", "16")),
            tokenizer_name=os.environ.get("ORBIT_TOKENIZER", "cl100k_base"),
        )


@dataclass
class ReplicaConfig:
    host: str = "0.0.0.0"
    port: int = 8001
    replica_id: str = "replica-1"
    router_url: str = "http://orbit-router:8000"

    # Backend selection: "sim" or "llamacpp"
    backend: str = "sim"

    # llama.cpp backend
    llamacpp_url: str = "http://localhost:9001"

    # KV cache (used by sim backend; shadow cache for llamacpp)
    cache_capacity_blocks: int = 1000
    block_size: int = 16

    # Simulated timing (sim backend only)
    prefill_ms_per_token: float = 0.5
    decode_ms_per_token: float = 10.0
    max_concurrent: int = 4

    # Tokenizer
    tokenizer_name: str = "cl100k_base"

    @classmethod
    def from_env(cls) -> ReplicaConfig:
        return cls(
            host=os.environ.get("ORBIT_HOST", "0.0.0.0"),
            port=int(os.environ.get("ORBIT_PORT", "8001")),
            replica_id=os.environ.get("ORBIT_REPLICA_ID", "replica-1"),
            router_url=os.environ.get("ORBIT_ROUTER_URL", "http://orbit-router:8000"),
            backend=os.environ.get("ORBIT_BACKEND", "sim"),
            llamacpp_url=os.environ.get("ORBIT_LLAMACPP_URL", "http://localhost:9001"),
            cache_capacity_blocks=int(os.environ.get("ORBIT_CACHE_CAPACITY", "1000")),
            block_size=int(os.environ.get("ORBIT_BLOCK_SIZE", "16")),
            prefill_ms_per_token=float(os.environ.get("ORBIT_PREFILL_MS", "0.5")),
            decode_ms_per_token=float(os.environ.get("ORBIT_DECODE_MS", "10.0")),
            max_concurrent=int(os.environ.get("ORBIT_MAX_CONCURRENT", "4")),
            tokenizer_name=os.environ.get("ORBIT_TOKENIZER", "cl100k_base"),
        )


@dataclass
class MonitorConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    router_url: str = "http://orbit-router:8000"
    replica_urls: list[str] = field(default_factory=list)
    poll_interval_s: float = 1.0

    @classmethod
    def from_env(cls) -> MonitorConfig:
        replica_urls_str = os.environ.get("ORBIT_REPLICA_URLS", "")
        replica_urls = [u.strip() for u in replica_urls_str.split(",") if u.strip()]
        return cls(
            host=os.environ.get("ORBIT_HOST", "0.0.0.0"),
            port=int(os.environ.get("ORBIT_PORT", "8080")),
            router_url=os.environ.get("ORBIT_ROUTER_URL", "http://orbit-router:8000"),
            replica_urls=replica_urls,
            poll_interval_s=float(os.environ.get("ORBIT_POLL_INTERVAL", "1.0")),
        )
