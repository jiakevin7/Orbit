import hashlib


class BloomFilter:
    def __init__(self, num_bits=8192, num_hashes=5):
        if num_bits <= 0:
            raise ValueError("num_bits must be positive")
        if num_hashes <= 0:
            raise ValueError("num_hashes must be positive")
        self.num_bits = num_bits
        self.num_hashes = num_hashes
        self._bits = bytearray((num_bits + 7) // 8)

    @property
    def byte_size(self):
        return len(self._bits)

    def _positions(self, item):
        # Derive multiple stable bit positions from one digest so summaries are
        # deterministic and compact without extra dependencies.
        payload = int(item).to_bytes(16, byteorder="big", signed=False)
        digest = hashlib.blake2b(payload, digest_size=16).digest()
        h1 = int.from_bytes(digest[:8], byteorder="big", signed=False)
        h2 = (
            int.from_bytes(digest[8:], byteorder="big", signed=False)
            or 11400714819323198485
        )
        return [(h1 + i * h2 + i * i) % self.num_bits for i in range(self.num_hashes)]

    def add(self, item):
        for position in self._positions(item):
            byte_index, bit_index = divmod(position, 8)
            self._bits[byte_index] |= 1 << bit_index

    def contains(self, item):
        for position in self._positions(item):
            byte_index, bit_index = divmod(position, 8)
            if not self._bits[byte_index] & 1 << bit_index:
                return False
        return True

    def __contains__(self, item):
        return self.contains(item)
