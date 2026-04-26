import heapq
from collections import OrderedDict
from dataclasses import dataclass
from .bloom import BloomFilter
from .hashing import hash_prefix, hot_prefix_hashes
from .models import ClusterExecution, ClusterSummary
from .trie import PrefixTrie

@dataclass(frozen=True)
class ClusterConfig:
    cache_capacity: int = 256
    cache_capacity_tokens: int | None = None
    summary_depths: tuple[int, ...] = (4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 64, 96, 128, 192, 256, 384, 512)
    hotset_depths: tuple[int, ...] = (4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48)
    hotset_capacity_per_depth: int = 512
    bloom_bits: int = 16384
    bloom_hashes: int = 5
    summary_interval: float = 5.0
    concurrency: int = 1
    prefill_cost_per_token: float = 1.0
    decode_cost_per_token: float = 2.0

class Cluster:

    def __init__(self, cluster_id, config=None):
        self.cluster_id = cluster_id
        self.config = config or ClusterConfig()
        self.trie = PrefixTrie()
        self._cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self._cached_token_total = 0
        self._pending_insertions: list[tuple[float, int, str, tuple[int, ...]]] = []
        self._workers = [0.0 for _ in range(self.config.concurrency)]
        heapq.heapify(self._workers)
        self._cache_sequence = 0
        self._summary_version = 0

    def advance_time(self, now):
        while self._pending_insertions and self._pending_insertions[0][0] <= now:
            _, _, request_id, tokens = heapq.heappop(self._pending_insertions)
            self._insert_into_cache(request_id, tokens)

    def queue_depth(self, now):
        return sum((1 for free_at in self._workers if free_at > now))

    def exact_prefix_match(self, tokens, now):
        self.advance_time(now)
        return self.trie.contains(tokens)

    def true_reusable_prefix(self, tokens, now):
        self.advance_time(now)
        return self.trie.longest_prefix(tokens)

    def execute(self, request):
        now = request.arrival_time
        self.advance_time(now)
        true_reusable = self.trie.longest_prefix(request.prefix_tokens)
        prefill_tokens = len(request.prefix_tokens) - true_reusable
        prefill_time = prefill_tokens * self.config.prefill_cost_per_token
        service_time = prefill_time + request.continuation_tokens * self.config.decode_cost_per_token
        queue_depth_before = self.queue_depth(now)
        earliest_worker = heapq.heappop(self._workers)
        started_at = max(now, earliest_worker)
        cache_ready_at = started_at + prefill_time
        finished_at = started_at + service_time
        heapq.heappush(self._workers, finished_at)
        self._cache_sequence += 1
        heapq.heappush(self._pending_insertions, (cache_ready_at, self._cache_sequence, request.request_id, request.prefix_tokens))
        return ClusterExecution(cluster_id=self.cluster_id, queue_depth_before=queue_depth_before, true_reusable_tokens=true_reusable, service_time=service_time, queue_delay=started_at - now, started_at=started_at, finished_at=finished_at, time_to_first_token=started_at - now + prefill_time + self.config.decode_cost_per_token, cache_ready_at=cache_ready_at)

    def publish_summary(self, now):
        self.advance_time(now)
        filters = {depth: BloomFilter(num_bits=self.config.bloom_bits, num_hashes=self.config.bloom_hashes) for depth in self.config.summary_depths}
        for tokens in self._cache.values():
            for depth in self.config.summary_depths:
                if len(tokens) >= depth:
                    filters[depth].add(hash_prefix(tokens, depth))
        hotsets = hot_prefix_hashes(self._cache.values(), self.config.hotset_depths, self.config.hotset_capacity_per_depth)
        self._summary_version += 1
        byte_size = sum((bloom.byte_size for bloom in filters.values())) + sum((8 * len(values) for values in hotsets.values())) + 64
        return ClusterSummary(cluster_id=self.cluster_id, version=self._summary_version, created_at=now, queue_depth=self.queue_depth(now), depths=self.config.summary_depths, filters=filters, byte_size=byte_size, hot_prefix_hashes=hotsets)

    def _insert_into_cache(self, request_id, tokens):
        if request_id in self._cache:
            old_tokens = self._cache.pop(request_id)
            self._remove_cached_tokens(old_tokens)
        self._cache[request_id] = tuple(tokens)
        self._add_cached_tokens(tokens)
        while self._cache_limit_exceeded():
            evicted_request_id, evicted_tokens = self._cache.popitem(last=False)
            self._remove_cached_tokens(evicted_tokens)
            if evicted_request_id == request_id and (not self._cache_limit_exceeded()):
                break

    def _cache_limit_exceeded(self):
        if len(self._cache) > self.config.cache_capacity:
            return True
        if self.config.cache_capacity_tokens is not None and self._cached_token_total > self.config.cache_capacity_tokens:
            return True
        return False

    def _add_cached_tokens(self, tokens):
        self.trie.insert(tokens)
        self._cached_token_total += len(tokens)

    def _remove_cached_tokens(self, tokens):
        self.trie.remove(tokens)
        self._cached_token_total -= len(tokens)
