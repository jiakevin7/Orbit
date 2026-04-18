from __future__ import annotations

import hashlib
from typing import Iterable, Sequence


def hash_prefix(tokens: Sequence[int], depth: int) -> int:
    if depth < 0:
        raise ValueError("depth must be non-negative")
    if depth > len(tokens):
        raise ValueError("depth cannot exceed the prefix length")

    hasher = hashlib.blake2b(digest_size=8)
    for token in tokens[:depth]:
        hasher.update(int(token).to_bytes(8, byteorder="little", signed=False))
    return int.from_bytes(hasher.digest(), byteorder="big", signed=False)


def prefix_hashes(tokens: Sequence[int], depths: Iterable[int]) -> dict[int, int]:
    result: dict[int, int] = {}
    for depth in sorted(set(depths)):
        if 0 < depth <= len(tokens):
            result[depth] = hash_prefix(tokens, depth)
    return result
