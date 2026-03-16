from __future__ import annotations

import hashlib
import logging
import random

from orbit.common.config import RouterConfig
from orbit.common.schemas import RoutingDecision
from orbit.router.cache_registry import CacheRegistry
from orbit.router.load_tracker import LoadTracker
from orbit.router.prompt_analyzer import AnalyzedPrompt, PromptAnalyzer
from orbit.router.scorer import Scorer
from orbit.router.semantic_index import SemanticIndex

logger = logging.getLogger("orbit.router.engine")

STRATEGIES = {"orbit", "round_robin", "random", "hash_based", "least_loaded"}


class RoutingEngine:
    """Core routing orchestration: parse → semantic filter → exact match → score → route."""

    def __init__(self, config: RouterConfig):
        self.config = config
        self.strategy = "orbit"
        self.prompt_analyzer = PromptAnalyzer(
            tokenizer_name=config.tokenizer_name,
            block_size=config.block_size,
        )
        self.cache_registry = CacheRegistry()
        self.load_tracker = LoadTracker()
        self.scorer = Scorer(config, self.load_tracker)
        self.semantic_index = SemanticIndex(model_name=config.embedding_model)

        # Replica URL mapping
        self._replica_urls: dict[str, str] = {}
        for i, url in enumerate(config.replica_urls):
            replica_id = f"replica-{i + 1}"
            self._replica_urls[replica_id] = url

        # Round-robin counter (fallback and baseline)
        self._rr_counter = 0

    def register_replica(self, replica_id: str, url: str) -> None:
        self._replica_urls[replica_id] = url

    def route(self, messages: list[dict[str, str]]) -> RoutingDecision:
        """Main routing entry point. Dispatches to the active strategy."""
        if not self._replica_urls:
            raise ValueError("No replicas registered")

        if self.strategy == "orbit":
            return self._route_orbit(messages)
        elif self.strategy == "round_robin":
            return self._route_round_robin(messages)
        elif self.strategy == "random":
            return self._route_random(messages)
        elif self.strategy == "hash_based":
            return self._route_hash_based(messages)
        elif self.strategy == "least_loaded":
            return self._route_least_loaded(messages)
        else:
            return self._route_orbit(messages)

    # ── Orbit (full two-stage routing) ────────────────────────────────────

    def _route_orbit(self, messages: list[dict[str, str]]) -> RoutingDecision:
        """Two-stage routing: semantic pre-filter → exact hash match → score."""
        analyzed = self.prompt_analyzer.analyze(messages)

        if not analyzed.block_hashes:
            return self._fallback_route(analyzed, "prompt_too_short")

        # Stage 1: Semantic pre-filter (optional)
        candidate_replicas = self._semantic_prefilter(analyzed)

        # Stage 2: Exact match via prefix trie
        prefix_matches = self.cache_registry.lookup_prefix(analyzed.block_hashes)

        if candidate_replicas:
            filtered = {k: v for k, v in prefix_matches.items() if k in candidate_replicas}
            for rid, blocks in prefix_matches.items():
                if rid not in filtered:
                    filtered[rid] = blocks
            prefix_matches = filtered

        if not prefix_matches:
            return self._fallback_route(analyzed, "no_cache_overlap")

        total_blocks = len(analyzed.block_hashes)
        total_tokens = len(analyzed.all_token_ids)
        scored = self.scorer.score_candidates(prefix_matches, total_blocks, total_tokens)

        if not scored:
            return self._fallback_route(analyzed, "scoring_empty")

        best = scored[0]
        replica_url = self._replica_urls.get(best.replica_id, "")

        if analyzed.stable_text:
            self.semantic_index.add_entry(best.replica_id, analyzed.stable_text)

        return RoutingDecision(
            selected_replica=best.replica_id,
            replica_url=replica_url,
            cached_tokens=best.cached_tokens,
            total_prompt_tokens=total_tokens,
            score=best.score,
            reason=best.reason,
        )

    # ── Baseline strategies ───────────────────────────────────────────────

    def _route_round_robin(self, messages: list[dict[str, str]]) -> RoutingDecision:
        replica_ids = list(self._replica_urls.keys())
        selected = replica_ids[self._rr_counter % len(replica_ids)]
        self._rr_counter += 1
        return RoutingDecision(
            selected_replica=selected,
            replica_url=self._replica_urls[selected],
            score=0.0,
            reason="round_robin",
        )

    def _route_random(self, messages: list[dict[str, str]]) -> RoutingDecision:
        replica_ids = list(self._replica_urls.keys())
        selected = random.choice(replica_ids)
        return RoutingDecision(
            selected_replica=selected,
            replica_url=self._replica_urls[selected],
            score=0.0,
            reason="random",
        )

    def _route_hash_based(self, messages: list[dict[str, str]]) -> RoutingDecision:
        content = "".join(m["content"] for m in messages)
        h = hashlib.sha256(content.encode()).hexdigest()
        replica_ids = list(self._replica_urls.keys())
        idx = int(h, 16) % len(replica_ids)
        selected = replica_ids[idx]
        return RoutingDecision(
            selected_replica=selected,
            replica_url=self._replica_urls[selected],
            score=0.0,
            reason="hash_based",
        )

    def _route_least_loaded(self, messages: list[dict[str, str]]) -> RoutingDecision:
        least = self.load_tracker.get_least_loaded()
        if least and least in self._replica_urls:
            selected = least
        else:
            # Fallback to round-robin if no load data yet
            replica_ids = list(self._replica_urls.keys())
            selected = replica_ids[self._rr_counter % len(replica_ids)]
            self._rr_counter += 1
        return RoutingDecision(
            selected_replica=selected,
            replica_url=self._replica_urls[selected],
            score=0.0,
            reason="least_loaded",
        )

    def _semantic_prefilter(self, analyzed: AnalyzedPrompt) -> set[str] | None:
        """Stage 1: Use semantic similarity to narrow candidate set."""
        if not self.semantic_index.is_available or not analyzed.stable_text:
            return None

        candidates = self.semantic_index.search(
            analyzed.stable_text,
            top_k=self.config.semantic_top_k,
            threshold=self.config.semantic_threshold,
        )

        if not candidates:
            return None

        return set(candidates)

    def _fallback_route(self, analyzed: AnalyzedPrompt, reason: str) -> RoutingDecision:
        """Fallback: least-loaded or round-robin."""
        replica_ids = list(self._replica_urls.keys())

        # Try least-loaded first
        least_loaded = self.load_tracker.get_least_loaded()
        if least_loaded and least_loaded in self._replica_urls:
            selected = least_loaded
            reason = f"fallback_least_loaded ({reason})"
        else:
            # Round-robin
            selected = replica_ids[self._rr_counter % len(replica_ids)]
            self._rr_counter += 1
            reason = f"fallback_round_robin ({reason})"

        return RoutingDecision(
            selected_replica=selected,
            replica_url=self._replica_urls[selected],
            cached_tokens=0,
            total_prompt_tokens=len(analyzed.all_token_ids),
            score=0.0,
            reason=reason,
        )
