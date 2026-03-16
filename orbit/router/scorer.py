from __future__ import annotations

from dataclasses import dataclass

from orbit.common.config import RouterConfig
from orbit.router.load_tracker import LoadTracker


@dataclass
class CandidateScore:
    replica_id: str
    cached_blocks: int
    total_blocks: int
    cached_tokens: int
    total_tokens: int
    prefill_savings: float
    congestion: float
    score: float
    reason: str


class Scorer:
    """Scores replica candidates based on cache overlap and congestion.

    score = alpha * (cached_tokens / total_tokens) - beta * normalized_congestion
    """

    def __init__(self, config: RouterConfig, load_tracker: LoadTracker):
        self.alpha = config.alpha
        self.beta = config.beta
        self.strong_hit_threshold = config.strong_hit_threshold
        self.block_size = config.block_size
        self.load_tracker = load_tracker

    def score_candidates(
        self,
        candidates: dict[str, int],  # replica_id → matched_blocks
        total_blocks: int,
        total_tokens: int,
    ) -> list[CandidateScore]:
        """Score each candidate replica.

        Args:
            candidates: {replica_id: num_matched_blocks}
            total_blocks: total blocks in the prompt
            total_tokens: total tokens in the prompt

        Returns:
            List of CandidateScore, sorted by score descending.
        """
        if not candidates or total_tokens == 0:
            return []

        scores: list[CandidateScore] = []
        min_congestion = self.load_tracker.get_min_congestion()

        for replica_id, matched_blocks in candidates.items():
            cached_tokens = matched_blocks * self.block_size
            prefill_savings = cached_tokens / total_tokens if total_tokens > 0 else 0.0

            load = self.load_tracker.get_load(replica_id)
            congestion = load.normalized_congestion

            score = self.alpha * prefill_savings - self.beta * congestion

            # Strong hit override
            reason = "scored"
            if (
                prefill_savings >= self.strong_hit_threshold
                and congestion < 2.0 * max(min_congestion, 0.01)
            ):
                score += 100.0  # effectively always pick this one
                reason = "strong_hit_override"

            scores.append(CandidateScore(
                replica_id=replica_id,
                cached_blocks=matched_blocks,
                total_blocks=total_blocks,
                cached_tokens=cached_tokens,
                total_tokens=total_tokens,
                prefill_savings=prefill_savings,
                congestion=congestion,
                score=score,
                reason=reason,
            ))

        scores.sort(key=lambda s: s.score, reverse=True)
        return scores
