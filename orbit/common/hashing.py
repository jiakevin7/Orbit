from __future__ import annotations

import hashlib
import struct

import tiktoken


def get_tokenizer(name: str = "cl100k_base") -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def tokenize(text: str, tokenizer_name: str = "cl100k_base") -> list[int]:
    enc = get_tokenizer(tokenizer_name)
    return enc.encode(text)


def compute_block_hashes(
    token_ids: list[int],
    block_size: int = 16,
) -> list[str]:
    """Compute chained SHA-256 block hashes over a token ID sequence.

    Block N's hash = SHA256(block_{N-1}_hash || token_ids_in_block_N).
    Only full blocks are hashed — trailing tokens that don't fill a block are ignored.

    Returns a list of hex-digest strings, one per full block.
    """
    if not token_ids:
        return []

    num_full_blocks = len(token_ids) // block_size
    if num_full_blocks == 0:
        return []

    hashes: list[str] = []
    prev_hash = b""  # empty bytes for the first block

    for i in range(num_full_blocks):
        start = i * block_size
        end = start + block_size
        block_tokens = token_ids[start:end]

        hasher = hashlib.sha256()
        hasher.update(prev_hash)
        for tid in block_tokens:
            hasher.update(struct.pack("<I", tid))

        digest = hasher.digest()
        hex_digest = digest.hex()
        hashes.append(hex_digest)
        prev_hash = digest

    return hashes


def compute_prefix_hashes(
    messages: list[dict[str, str]],
    tokenizer_name: str = "cl100k_base",
    block_size: int = 16,
) -> tuple[list[int], list[str]]:
    """Tokenize a list of chat messages and compute block hashes.

    Returns (token_ids, block_hashes).
    """
    enc = get_tokenizer(tokenizer_name)
    all_tokens: list[int] = []
    for msg in messages:
        # Tokenize role and content together in a simple format
        text = f"<|{msg['role']}|>{msg['content']}"
        all_tokens.extend(enc.encode(text))

    block_hashes = compute_block_hashes(all_tokens, block_size)
    return all_tokens, block_hashes
