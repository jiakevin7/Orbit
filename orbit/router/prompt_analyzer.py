from __future__ import annotations

from dataclasses import dataclass, field

from orbit.common.hashing import compute_block_hashes, get_tokenizer


@dataclass
class PromptSegment:
    role: str
    content: str
    token_ids: list[int] = field(default_factory=list)
    is_stable: bool = False  # system prompts, tools = stable; user queries = volatile


@dataclass
class AnalyzedPrompt:
    segments: list[PromptSegment]
    all_token_ids: list[int]
    block_hashes: list[str]
    stable_text: str  # concatenated stable segment text for semantic embedding


class PromptAnalyzer:
    """Parses chat messages into segments, tokenizes, and computes block hashes."""

    def __init__(self, tokenizer_name: str = "cl100k_base", block_size: int = 16):
        self.tokenizer_name = tokenizer_name
        self.block_size = block_size
        self._enc = get_tokenizer(tokenizer_name)

    def analyze(self, messages: list[dict[str, str]]) -> AnalyzedPrompt:
        segments: list[PromptSegment] = []
        all_tokens: list[int] = []
        stable_parts: list[str] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Tokenize in the same format as the replica
            text = f"<|{role}|>{content}"
            token_ids = self._enc.encode(text)

            # System and tool messages are considered stable (high reuse)
            is_stable = role in ("system", "tool", "function")

            segments.append(PromptSegment(
                role=role,
                content=content,
                token_ids=token_ids,
                is_stable=is_stable,
            ))

            all_tokens.extend(token_ids)
            if is_stable:
                stable_parts.append(content)

        block_hashes = compute_block_hashes(all_tokens, self.block_size)

        return AnalyzedPrompt(
            segments=segments,
            all_token_ids=all_tokens,
            block_hashes=block_hashes,
            stable_text=" ".join(stable_parts),
        )
