import pytest

from orbit.common.hashing import compute_block_hashes, compute_prefix_hashes, tokenize


class TestTokenize:
    def test_basic_tokenize(self):
        tokens = tokenize("hello world")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

    def test_empty_string(self):
        tokens = tokenize("")
        assert tokens == []


class TestComputeBlockHashes:
    def test_empty_input(self):
        assert compute_block_hashes([]) == []

    def test_too_short(self):
        # Less than one block
        assert compute_block_hashes([1, 2, 3], block_size=16) == []

    def test_single_block(self):
        tokens = list(range(16))
        hashes = compute_block_hashes(tokens, block_size=16)
        assert len(hashes) == 1
        assert isinstance(hashes[0], str)
        assert len(hashes[0]) == 64  # SHA-256 hex digest

    def test_multiple_blocks(self):
        tokens = list(range(48))
        hashes = compute_block_hashes(tokens, block_size=16)
        assert len(hashes) == 3
        # All hashes should be unique
        assert len(set(hashes)) == 3

    def test_trailing_tokens_ignored(self):
        tokens = list(range(20))  # 16 + 4 trailing
        hashes = compute_block_hashes(tokens, block_size=16)
        assert len(hashes) == 1  # only 1 full block

    def test_deterministic(self):
        tokens = list(range(32))
        h1 = compute_block_hashes(tokens, block_size=16)
        h2 = compute_block_hashes(tokens, block_size=16)
        assert h1 == h2

    def test_chaining(self):
        """Block hashes are chained: changing block 0 changes block 1's hash."""
        tokens_a = list(range(32))
        tokens_b = [999] + list(range(1, 32))

        hashes_a = compute_block_hashes(tokens_a, block_size=16)
        hashes_b = compute_block_hashes(tokens_b, block_size=16)

        # First block differs
        assert hashes_a[0] != hashes_b[0]
        # Second block also differs (due to chaining), even though block 1 tokens are same
        assert hashes_a[1] != hashes_b[1]

    def test_prefix_stability(self):
        """Adding tokens at the end doesn't change existing block hashes."""
        tokens_short = list(range(32))
        tokens_long = list(range(48))

        hashes_short = compute_block_hashes(tokens_short, block_size=16)
        hashes_long = compute_block_hashes(tokens_long, block_size=16)

        # First two blocks should match
        assert hashes_short[0] == hashes_long[0]
        assert hashes_short[1] == hashes_long[1]


class TestComputePrefixHashes:
    def test_basic(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant. " * 10},
            {"role": "user", "content": "What is Python? " * 5},
        ]
        token_ids, block_hashes = compute_prefix_hashes(messages)
        assert len(token_ids) > 0
        assert len(block_hashes) > 0

    def test_same_messages_same_hashes(self):
        messages = [
            {"role": "system", "content": "You are helpful. " * 10},
            {"role": "user", "content": "Hello! " * 5},
        ]
        _, h1 = compute_prefix_hashes(messages)
        _, h2 = compute_prefix_hashes(messages)
        assert h1 == h2

    def test_different_messages_different_hashes(self):
        m1 = [{"role": "system", "content": "A " * 20}]
        m2 = [{"role": "system", "content": "B " * 20}]
        _, h1 = compute_prefix_hashes(m1)
        _, h2 = compute_prefix_hashes(m2)
        assert h1 != h2
