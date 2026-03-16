from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field

from orbit.common.hashing import compute_block_hashes


@dataclass
class CacheBlock:
    block_hash: str
    block_index: int  # position in the original sequence
    token_ids: list[int]


class KVCache:
    """Simulated block-level KV cache with LRU eviction and prefix lookup."""

    def __init__(self, capacity_blocks: int = 1000, block_size: int = 16):
        self.capacity = capacity_blocks
        self.block_size = block_size
        # OrderedDict for LRU: most recently used at the end
        self._blocks: OrderedDict[str, CacheBlock] = OrderedDict()
        self._lock = threading.Lock()

        # Stats
        self.total_lookups = 0
        self.total_hits = 0
        self._evicted: list[CacheBlock] = []
        self._inserted: list[CacheBlock] = []

    @property
    def used_blocks(self) -> int:
        return len(self._blocks)

    @property
    def hit_rate(self) -> float:
        if self.total_lookups == 0:
            return 0.0
        return self.total_hits / self.total_lookups

    def lookup_prefix(self, token_ids: list[int]) -> int:
        """Find the longest cached prefix match for the given token sequence.

        Returns the number of tokens that are cached (always a multiple of block_size).
        """
        block_hashes = compute_block_hashes(token_ids, self.block_size)
        if not block_hashes:
            return 0

        cached_blocks = 0
        with self._lock:
            self.total_lookups += 1
            for bh in block_hashes:
                if bh in self._blocks:
                    # Touch for LRU
                    self._blocks.move_to_end(bh)
                    cached_blocks += 1
                else:
                    # Prefix match breaks at first miss
                    break

            if cached_blocks > 0:
                self.total_hits += 1

        return cached_blocks * self.block_size

    def insert_prefix(self, token_ids: list[int]) -> tuple[list[CacheBlock], list[CacheBlock]]:
        """Insert token sequence into cache as blocks.

        Returns (inserted_blocks, evicted_blocks).
        """
        block_hashes = compute_block_hashes(token_ids, self.block_size)
        if not block_hashes:
            return [], []

        inserted: list[CacheBlock] = []
        evicted: list[CacheBlock] = []

        with self._lock:
            for i, bh in enumerate(block_hashes):
                if bh in self._blocks:
                    # Already cached, just touch for LRU
                    self._blocks.move_to_end(bh)
                    continue

                # Evict if at capacity
                while len(self._blocks) >= self.capacity:
                    _, evicted_block = self._blocks.popitem(last=False)
                    evicted.append(evicted_block)

                start = i * self.block_size
                end = start + self.block_size
                block = CacheBlock(
                    block_hash=bh,
                    block_index=i,
                    token_ids=token_ids[start:end],
                )
                self._blocks[bh] = block
                inserted.append(block)

            self._evicted.extend(evicted)
            self._inserted.extend(inserted)

        return inserted, evicted

    def drain_updates(self) -> tuple[list[CacheBlock], list[CacheBlock]]:
        """Drain accumulated insert/evict events since last drain."""
        with self._lock:
            inserted = self._inserted.copy()
            evicted = self._evicted.copy()
            self._inserted.clear()
            self._evicted.clear()
        return inserted, evicted

    def get_all_block_hashes(self) -> list[str]:
        """Return all block hashes currently in cache (for reconciliation)."""
        with self._lock:
            return list(self._blocks.keys())

    def clear(self) -> None:
        with self._lock:
            self._blocks.clear()
            self._inserted.clear()
            self._evicted.clear()
