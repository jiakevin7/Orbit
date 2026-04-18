from __future__ import annotations

import hashlib


class BloomFilter:
    def __init__(self, num_bits: int = 8192, num_hashes: int = 5) -> None:
        if num_bits <= 0:
            raise ValueError("num_bits must be positive")
        if num_hashes <= 0:
            raise ValueError("num_hashes must be positive")
        self.num_bits = num_bits
        self.num_hashes = num_hashes
        self._bits = bytearray((num_bits + 7) // 8)

    @property
    def byte_size(self) -> int:
        return len(self._bits)

    def _positions(self, item: int) -> list[int]:
        payload = int(item).to_bytes(16, byteorder="big", signed=False)
        digest = hashlib.blake2b(payload, digest_size=16).digest()
        h1 = int.from_bytes(digest[:8], byteorder="big", signed=False)
        h2 = int.from_bytes(digest[8:], byteorder="big", signed=False) or 0x9E3779B97F4A7C15
        return [((h1 + i * h2 + i * i) % self.num_bits) for i in range(self.num_hashes)]

    def add(self, item: int) -> None:
        for position in self._positions(item):
            byte_index, bit_index = divmod(position, 8)
            self._bits[byte_index] |= 1 << bit_index

    def contains(self, item: int) -> bool:
        for position in self._positions(item):
            byte_index, bit_index = divmod(position, 8)
            if not (self._bits[byte_index] & (1 << bit_index)):
                return False
        return True

    def __contains__(self, item: int) -> bool:
        return self.contains(item)

