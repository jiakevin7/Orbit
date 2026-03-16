from __future__ import annotations

import threading

from orbit.common.schemas import CacheUpdate, CacheUpdateType
from orbit.router.prefix_trie import PrefixTrie


class CacheRegistry:
    """Central metadata store: tracks which block hashes live on which replicas.

    Maintains both a prefix trie (for longest-prefix lookups) and a flat index
    (block_hash → set of replica_ids) for eviction handling.
    """

    def __init__(self):
        self.trie = PrefixTrie()
        # Flat index: block_hash → set of replica_ids
        self._block_to_replicas: dict[str, set[str]] = {}
        # Reverse: replica_id → set of block_hashes
        self._replica_to_blocks: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def apply_update(self, update: CacheUpdate) -> None:
        """Apply a cache update (insert or evict) from a replica."""
        if update.update_type == CacheUpdateType.INSERT:
            self._handle_insert(update)
        elif update.update_type == CacheUpdateType.EVICT:
            self._handle_evict(update)

    def _handle_insert(self, update: CacheUpdate) -> None:
        block_hashes = [b.block_hash for b in update.blocks]
        self.trie.insert(block_hashes, update.replica_id)

        with self._lock:
            if update.replica_id not in self._replica_to_blocks:
                self._replica_to_blocks[update.replica_id] = set()

            for b in update.blocks:
                if b.block_hash not in self._block_to_replicas:
                    self._block_to_replicas[b.block_hash] = set()
                self._block_to_replicas[b.block_hash].add(update.replica_id)
                self._replica_to_blocks[update.replica_id].add(b.block_hash)

    def _handle_evict(self, update: CacheUpdate) -> None:
        with self._lock:
            for b in update.blocks:
                self.trie.remove_replica_block(b.block_hash, update.replica_id)

                if b.block_hash in self._block_to_replicas:
                    self._block_to_replicas[b.block_hash].discard(update.replica_id)
                    if not self._block_to_replicas[b.block_hash]:
                        del self._block_to_replicas[b.block_hash]

                if update.replica_id in self._replica_to_blocks:
                    self._replica_to_blocks[update.replica_id].discard(b.block_hash)

    def lookup_prefix(self, block_hashes: list[str]) -> dict[str, int]:
        """Find longest prefix match per replica.

        Returns {replica_id: num_matched_blocks}.
        """
        return self.trie.lookup(block_hashes)

    def get_replicas_for_block(self, block_hash: str) -> set[str]:
        with self._lock:
            return set(self._block_to_replicas.get(block_hash, set()))

    def get_blocks_for_replica(self, replica_id: str) -> set[str]:
        with self._lock:
            return set(self._replica_to_blocks.get(replica_id, set()))
