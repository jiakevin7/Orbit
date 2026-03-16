from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger("orbit.router.semantic")


@dataclass
class SemanticEntry:
    replica_id: str
    text: str
    embedding: list[float]


class SemanticIndex:
    """Embedding-based similarity search for candidate pre-filtering.

    Uses sentence-transformers MiniLM to embed stable prompt segments.
    Lazily loads the model on first use.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._entries: list[SemanticEntry] = []
        self._lock = threading.Lock()

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info(f"Loaded semantic model: {self._model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed; semantic index disabled")
            self._model = None

    def add_entry(self, replica_id: str, text: str) -> None:
        if not text.strip():
            return
        self._load_model()
        if self._model is None:
            return

        embedding = self._model.encode(text).tolist()
        with self._lock:
            self._entries.append(SemanticEntry(
                replica_id=replica_id,
                text=text,
                embedding=embedding,
            ))

    def search(self, query_text: str, top_k: int = 3, threshold: float = 0.5) -> list[str]:
        """Return replica IDs with semantically similar cached prefixes.

        Returns at most top_k replica IDs above the similarity threshold.
        """
        if not query_text.strip():
            return []
        self._load_model()
        if self._model is None:
            return []

        with self._lock:
            if not self._entries:
                return []
            entries = list(self._entries)

        import numpy as np

        query_emb = self._model.encode(query_text)
        query_norm = np.linalg.norm(query_emb)
        if query_norm == 0:
            return []

        scores: list[tuple[str, float]] = []
        for entry in entries:
            entry_emb = np.array(entry.embedding)
            entry_norm = np.linalg.norm(entry_emb)
            if entry_norm == 0:
                continue
            similarity = float(np.dot(query_emb, entry_emb) / (query_norm * entry_norm))
            if similarity >= threshold:
                scores.append((entry.replica_id, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)

        # Deduplicate replica IDs, keep highest score
        seen: set[str] = set()
        result: list[str] = []
        for rid, _ in scores:
            if rid not in seen:
                seen.add(rid)
                result.append(rid)
                if len(result) >= top_k:
                    break

        return result

    @property
    def is_available(self) -> bool:
        self._load_model()
        return self._model is not None
