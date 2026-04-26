import re
from dataclasses import dataclass, field
from .bloom import BloomFilter

@dataclass(frozen=True)
class Request:
    request_id: str
    arrival_time: float
    router_id: str
    prefix_tokens: tuple[int, ...]
    continuation_tokens: int = 32
    prompt_prefix_text: str | None = None
    prefix_token_source: str = 'synthetic_lexical'
    arrival_scale_applied: float = 1.0
    traffic_class: str = 'synthetic'
    session_id: str | None = None
    source_id: str | None = None

    @property
    def input_length(self):
        return len(self.prefix_tokens)

    @property
    def prompt_text(self):
        if self.prompt_prefix_text is not None:
            normalized = self.prompt_prefix_text.rstrip()
            if re.search('assistant:\\s*$', normalized, re.IGNORECASE):
                return normalized
            return f'{normalized}\nAssistant:'
        token_text = ' '.join((f'tok_{token}' for token in self.prefix_tokens))
        return f'You are participating in a routing benchmark.\nTreat the following prefix as reusable context.\n{token_text}\nAssistant:'

@dataclass(frozen=True)
class ClusterSummary:
    cluster_id: str
    version: int
    created_at: float
    queue_depth: int
    depths: tuple[int, ...]
    filters: dict[int, BloomFilter]
    byte_size: int
    hot_prefix_hashes: dict[int, tuple[int, ...]] = field(default_factory=dict)

@dataclass(frozen=True)
class ClusterExecution:
    cluster_id: str
    queue_depth_before: int
    true_reusable_tokens: int
    service_time: float
    queue_delay: float
    started_at: float
    finished_at: float
    time_to_first_token: float
    cache_ready_at: float

@dataclass(frozen=True)
class RouteDecision:
    policy: str
    cluster_id: str
    estimated_reusable_tokens: int
    predicted_latency: float
    used_fallback: bool = False
    details: dict[str, float] = field(default_factory=dict)

@dataclass(frozen=True)
class ExecutionRecord:
    request_id: str
    policy: str
    router_id: str
    cluster_id: str
    arrival_time: float
    started_at: float
    finished_at: float
    predicted_latency: float
    actual_latency: float
    actual_ttft: float
    estimated_reusable_tokens: int
    actual_reusable_tokens: int
    estimated_remaining_prefill_tokens: int
    input_length: int
    continuation_tokens: int
    reuse_fraction: float
    network_cost: float
    queue_delay: float
    queue_depth_before: int
    route_queue_depth: int
    metadata_age: float
    uncertainty_gap: int
    missing_summary: bool
    initial_cluster_id: str
    had_failover: bool
    failover_delay: float
    attempt_count: int
    service_time: float
    traffic_class: str = 'synthetic'
    session_id: str | None = None
    source_id: str | None = None
    predicted_ttft: float = 0.0
    predicted_route_cost: float = 0.0
    raw_estimated_reusable_tokens: int = 0
    summary_matched_levels: int = 0
    hotset_matched_levels: int = 0

@dataclass(frozen=True)
class SimulationMetrics:
    policy: str
    request_count: int
    mean_reusable_prefix: float
    mean_reuse_fraction: float
    ttft_p50: float
    ttft_p95: float
    latency_p50: float
    latency_p95: float
    control_plane_bytes: int
    summary_memory_bytes: int
    load_stddev: float
    failover_count: int
    failover_rate: float
    cluster_request_counts: dict[str, int]
