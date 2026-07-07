"""
World Awareness Service — scheduled hourly interest-driven news scan.

Derives user interests from traits and topic frequency, queries the news tool,
and feeds results into world state. Zero LLM calls.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from services.embedding_service import EmbeddingService
    from services.news_service import NewsService

logger = logging.getLogger(__name__)

LOG_PREFIX = "[world-awareness]"

# ── Configuration ─────────────────────────────────────────────
POLL_INTERVAL = 3600          # 1 hour
MAX_INTERESTS = 8             # top N interests to query
SIMILARITY_DEDUP_THRESHOLD = 0.85  # embedding cosine sim above this = same interest
TRAIT_MIN_CONFIDENCE = 0.7
TRAIT_MIN_REINFORCEMENTS = 3
TOPIC_LOOKBACK_DAYS = 14
MAX_ARTICLES_PER_INTEREST = 3
SIGNAL_SOURCE = "world_awareness"


class WorldAwarenessService:
    def __init__(self) -> None:
        self._embedding_svc: "EmbeddingService | None" = None
        self._news_service: "NewsService | None" = None

    # ── Lazy accessors ────────────────────────────────────────

    def _get_embedding_service(self) -> "EmbeddingService":
        if self._embedding_svc is None:
            from services.embedding_service import get_embedding_service
            self._embedding_svc = get_embedding_service()
        return self._embedding_svc

    def _get_news_service(self) -> "NewsService":
        if self._news_service is None:
            from services.news_service import NewsService
            self._news_service = NewsService()
        return self._news_service


    # ── Interest extraction ───────────────────────────────────

    def extract_interests(self) -> list[dict[str, str | float]]:
        candidates = []
        candidates.extend(self._extract_trait_interests())
        candidates.extend(self._extract_topic_interests())

        if not candidates:
            return []

        deduped = self._deduplicate_by_embedding(candidates)
        deduped.sort(key=lambda x: x["score"], reverse=True)
        return deduped[:MAX_INTERESTS]

    def _extract_trait_interests(self) -> list[dict[str, str | float]]:
        try:
            from models.fact import FactRow
            rows = [r.to_dict() for r in FactRow.traits().get()]
            traits = [
                r for r in rows
                if cast(float, r.get('retrieval_weight') or 0) >= TRAIT_MIN_CONFIDENCE
                and cast(int, r.get('evidence_count') or 0) >= TRAIT_MIN_REINFORCEMENTS
            ]
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to query traits: {e}")
            return []

        candidates: list[dict[str, str | float]] = []
        for row in traits:
            term = cast(str, row.get('value') or '').strip()
            if len(term) < 2 or len(term) > 100:
                continue
            score: float = cast(float, row.get('retrieval_weight') or 0) * cast(int, row.get('evidence_count') or 1)
            candidates.append({"term": term, "score": score, "source": "trait"})
        return candidates

    def _extract_topic_interests(self) -> list[dict[str, str | float]]:
        from services.transcript_service import Transcript  # noqa: PLC0415
        topics = Transcript.channel_activity(TOPIC_LOOKBACK_DAYS)
        candidates: list[dict[str, str | float]] = []
        max_freq = max((row[1] for row in topics), default=1)
        for row in topics:
            topic_name = row[0].strip()
            if len(topic_name) < 2:
                continue
            term = topic_name.replace("_", " ").replace("-", " ")
            candidates.append({"term": term, "score": row[1] / max_freq, "source": "topic"})
        return candidates

    def _deduplicate_by_embedding(self, candidates: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
        if len(candidates) <= 1:
            return candidates

        emb_svc = self._get_embedding_service()
        terms = [cast(str, c["term"]) for c in candidates]

        try:
            embeddings = emb_svc.generate_embeddings_batch(terms)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Embedding failed, skipping dedup: {e}")
            return candidates

        # Greedy dedup: keep highest-scored, drop anything too similar
        indexed = sorted(enumerate(candidates), key=lambda x: x[1]["score"], reverse=True)
        kept_indices: list[int] = []
        kept_embeddings: list[np.ndarray] = []

        for orig_idx, candidate in indexed:
            emb = embeddings[orig_idx]
            is_duplicate = False
            for kept_emb in kept_embeddings:
                sim = float(np.dot(emb, kept_emb))
                if sim >= SIMILARITY_DEDUP_THRESHOLD:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept_indices.append(orig_idx)
                kept_embeddings.append(emb)

        return [candidates[i] for i in kept_indices]

    # ── News scanning ─────────────────────────────────────────

    def scan(self) -> int:
        """Run one scan cycle. Returns number of signals written to world state."""
        interests = self.extract_interests()
        if not interests:
            logger.debug(f"{LOG_PREFIX} No interests extracted, skipping scan")
            return 0

        logger.info(f"{LOG_PREFIX} Scanning {len(interests)} interests: "
                     f"{[i['term'] for i in interests]}")

        from services.world_state import world_state as _world_state
        news_svc = self._get_news_service()
        signals_written = 0

        for interest in interests:
            try:
                articles = news_svc.search(
                    query=cast(str, interest["term"]),
                    limit=MAX_ARTICLES_PER_INTEREST,
                )
                if not articles:
                    continue

                best = articles[0]
                headline = f"{best.title} \u2014 {best.source}"
                _world_state.push_signal("news", headline, ttl=3600)
                signals_written += 1

            except Exception as e:
                logger.warning(f"{LOG_PREFIX} Failed to scan '{interest['term']}': {e}")
                continue

        logger.info(f"{LOG_PREFIX} Scan complete: {signals_written} signals written")
        return signals_written


# ── Worker entry point ────────────────────────────────────────

def world_awareness_worker() -> None:
    """Module-level entry point for run.py registration."""
    service = WorldAwarenessService()

    logger.info(f"{LOG_PREFIX} Service started (poll interval: {POLL_INTERVAL}s)")

    # Initial delay — let embedding model and DB initialize
    time.sleep(60)

    next_tick = time.monotonic() + POLL_INTERVAL
    while True:
        try:
            now = time.monotonic()
            sleep_secs = max(0, next_tick - now)
            time.sleep(sleep_secs)
            next_tick += POLL_INTERVAL
            service.scan()
        except KeyboardInterrupt:
            logger.info(f"{LOG_PREFIX} Shutting down")
            break
        except Exception as e:
            logger.exception(f"{LOG_PREFIX} Scan failed: {e}")
