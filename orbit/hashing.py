import hashlib
from collections import Counter, defaultdict

def hash_prefix(tokens, depth):
    if depth < 0:
        raise ValueError('depth must be non-negative')
    if depth > len(tokens):
        raise ValueError('depth cannot exceed the prefix length')
    hasher = hashlib.blake2b(digest_size=8)
    for token in tokens[:depth]:
        hasher.update(int(token).to_bytes(8, byteorder='little', signed=False))
    return int.from_bytes(hasher.digest(), byteorder='big', signed=False)

def prefix_hashes(tokens, depths):
    result: dict[int, int] = {}
    for depth in sorted(set(depths)):
        if 0 < depth <= len(tokens):
            result[depth] = hash_prefix(tokens, depth)
    return result

def hot_prefix_hashes(token_sequences, depths, per_depth_limit):
    if per_depth_limit <= 0:
        return {}
    counters: dict[int, Counter[int]] = defaultdict(Counter)
    unique_depths = tuple(sorted(set((depth for depth in depths if depth > 0))))
    for tokens in token_sequences:
        for depth in unique_depths:
            if len(tokens) >= depth:
                counters[depth][hash_prefix(tokens, depth)] += 1
    hotsets: dict[int, tuple[int, ...]] = {}
    for depth in unique_depths:
        counts = counters.get(depth)
        if not counts:
            continue
        hottest = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:per_depth_limit]
        hotsets[depth] = tuple((hash_value for hash_value, _ in hottest))
    return hotsets
