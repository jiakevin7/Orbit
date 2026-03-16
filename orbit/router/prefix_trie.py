from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class TrieNode:
    """A node in the prefix trie.

    Each node corresponds to a block hash. Children are keyed by the next block hash.
    replicas tracks which replicas have this prefix cached.
    """
    block_hash: str
    replicas: set[str] = field(default_factory=set)
    children: dict[str, TrieNode] = field(default_factory=dict)


class PrefixTrie:
    """Radix trie tracking which block-hash prefixes are cached on which replicas.

    Keys are sequences of block hashes. Each path from root to a node represents
    a cached prefix.
    """

    def __init__(self):
        self.root: dict[str, TrieNode] = {}  # first block hash → TrieNode
        self._lock = threading.Lock()

    def insert(self, block_hashes: list[str], replica_id: str) -> None:
        """Record that replica_id has cached the prefix described by block_hashes."""
        if not block_hashes:
            return

        with self._lock:
            # First block
            first_hash = block_hashes[0]
            if first_hash not in self.root:
                self.root[first_hash] = TrieNode(block_hash=first_hash)
            node = self.root[first_hash]
            node.replicas.add(replica_id)

            # Subsequent blocks
            for bh in block_hashes[1:]:
                if bh not in node.children:
                    node.children[bh] = TrieNode(block_hash=bh)
                node = node.children[bh]
                node.replicas.add(replica_id)

    def lookup(self, block_hashes: list[str]) -> dict[str, int]:
        """Find the longest prefix match for each replica.

        Returns {replica_id: num_matched_blocks}.
        A replica's match stops at the first block where it is not present.
        """
        if not block_hashes:
            return {}

        with self._lock:
            first_hash = block_hashes[0]
            if first_hash not in self.root:
                return {}

            node = self.root[first_hash]
            # Track per-replica match depth; active = still matching contiguously
            replica_matches: dict[str, int] = {}
            active_replicas: set[str] = set()
            for rid in node.replicas:
                replica_matches[rid] = 1
                active_replicas.add(rid)

            for bh in block_hashes[1:]:
                if bh not in node.children:
                    break
                node = node.children[bh]
                # Only continue counting replicas that are present at this node
                still_active: set[str] = set()
                for rid in active_replicas:
                    if rid in node.replicas:
                        replica_matches[rid] += 1
                        still_active.add(rid)
                # Also pick up any new replicas that appear at this depth
                for rid in node.replicas:
                    if rid not in replica_matches:
                        replica_matches[rid] = 1
                        still_active.add(rid)
                active_replicas = still_active

            return replica_matches

    def remove_replica(self, block_hashes: list[str], replica_id: str) -> None:
        """Remove a replica from the prefix path described by block_hashes."""
        if not block_hashes:
            return

        with self._lock:
            first_hash = block_hashes[0]
            if first_hash not in self.root:
                return

            node = self.root[first_hash]
            node.replicas.discard(replica_id)

            for bh in block_hashes[1:]:
                if bh not in node.children:
                    break
                node = node.children[bh]
                node.replicas.discard(replica_id)

    def remove_replica_block(self, block_hash: str, replica_id: str) -> None:
        """Remove a replica from a specific block hash at any position in the trie."""
        with self._lock:
            self._remove_block_recursive(self.root, block_hash, replica_id)

    def _remove_block_recursive(
        self, nodes: dict[str, TrieNode], block_hash: str, replica_id: str
    ) -> None:
        for bh, node in nodes.items():
            if bh == block_hash:
                node.replicas.discard(replica_id)
            self._remove_block_recursive(node.children, block_hash, replica_id)

    def get_all_replicas(self) -> set[str]:
        """Return the set of all replica IDs referenced in the trie."""
        replicas: set[str] = set()
        with self._lock:
            self._collect_replicas(self.root, replicas)
        return replicas

    def _collect_replicas(self, nodes: dict[str, TrieNode], replicas: set[str]) -> None:
        for node in nodes.values():
            replicas.update(node.replicas)
            self._collect_replicas(node.children, replicas)
