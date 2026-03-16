import pytest

from orbit.common.config import RouterConfig
from orbit.common.schemas import LoadUpdate
from orbit.router.load_tracker import LoadTracker
from orbit.router.scorer import Scorer


@pytest.fixture
def load_tracker():
    tracker = LoadTracker()
    tracker.update(LoadUpdate(
        replica_id="replica-1",
        active_requests=1,
        queued_requests=0,
        max_concurrent=4,
        cache_used_blocks=100,
        cache_capacity_blocks=1000,
    ))
    tracker.update(LoadUpdate(
        replica_id="replica-2",
        active_requests=3,
        queued_requests=1,
        max_concurrent=4,
        cache_used_blocks=200,
        cache_capacity_blocks=1000,
    ))
    return tracker


@pytest.fixture
def scorer(load_tracker):
    config = RouterConfig(alpha=1.0, beta=0.5, strong_hit_threshold=0.9, block_size=16)
    return Scorer(config, load_tracker)


class TestScorer:
    def test_empty_candidates(self, scorer):
        scores = scorer.score_candidates({}, total_blocks=10, total_tokens=160)
        assert scores == []

    def test_basic_scoring(self, scorer):
        candidates = {"replica-1": 5, "replica-2": 8}
        scores = scorer.score_candidates(candidates, total_blocks=10, total_tokens=160)

        assert len(scores) == 2
        # Scores should be sorted descending
        assert scores[0].score >= scores[1].score

    def test_cache_overlap_preferred(self, scorer):
        """Replica with more cache overlap should score higher (same load)."""
        # Set equal load
        scorer.load_tracker.update(LoadUpdate(
            replica_id="replica-1",
            active_requests=1,
            queued_requests=0,
            max_concurrent=4,
            cache_used_blocks=100,
            cache_capacity_blocks=1000,
        ))
        scorer.load_tracker.update(LoadUpdate(
            replica_id="replica-2",
            active_requests=1,
            queued_requests=0,
            max_concurrent=4,
            cache_used_blocks=200,
            cache_capacity_blocks=1000,
        ))

        candidates = {"replica-1": 2, "replica-2": 8}
        scores = scorer.score_candidates(candidates, total_blocks=10, total_tokens=160)

        assert scores[0].replica_id == "replica-2"
        assert scores[0].cached_blocks == 8

    def test_high_load_penalized(self, scorer):
        """Heavily loaded replica should score lower even with good cache overlap."""
        scorer.load_tracker.update(LoadUpdate(
            replica_id="replica-1",
            active_requests=1,
            queued_requests=0,
            max_concurrent=4,
            cache_used_blocks=100,
            cache_capacity_blocks=1000,
        ))
        scorer.load_tracker.update(LoadUpdate(
            replica_id="replica-2",
            active_requests=4,
            queued_requests=4,
            max_concurrent=4,
            cache_used_blocks=200,
            cache_capacity_blocks=1000,
        ))

        # replica-2 has slightly more cache but much more load
        candidates = {"replica-1": 5, "replica-2": 6}
        scores = scorer.score_candidates(candidates, total_blocks=10, total_tokens=160)

        # replica-1 should win due to lower congestion
        assert scores[0].replica_id == "replica-1"

    def test_strong_hit_override(self, scorer):
        """Replica with >90% cache overlap should always win if load is reasonable."""
        candidates = {"replica-1": 9, "replica-2": 1}  # 9/10 = 90%
        scores = scorer.score_candidates(candidates, total_blocks=10, total_tokens=160)

        assert scores[0].replica_id == "replica-1"
        assert scores[0].reason == "strong_hit_override"

    def test_prefill_savings_calculation(self, scorer):
        candidates = {"replica-1": 5}
        scores = scorer.score_candidates(candidates, total_blocks=10, total_tokens=160)

        assert len(scores) == 1
        assert scores[0].cached_tokens == 80  # 5 blocks * 16 tokens
        assert scores[0].prefill_savings == pytest.approx(0.5, abs=0.01)
