"""
World Awareness Service — scheduled hourly interest-driven news scan.

Derives user interests from traits and topic frequency, queries the news tool,
and feeds results into world state. Zero LLM calls.
"""

import json
import logging
import time
from typing import Optional

import numpy as np

from services.time_utils import utc_now

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
    def __init__(self, database_service):
        self._db = database_service
        self._embedding_svc = None
        self._news_service = None
        self._world_state_svc = None

    # ── Lazy accessors ────────────────────────────────────────

    def _get_embedding_service(self):
        if self._embedding_svc is None:
            from services.embedding_service import get_embedding_service
            self._embedding_svc = get_embedding_service()
        return self._embedding_svc

    def _get_news_service(self):
        if self._news_service is None:
            from services.news_service import NewsService
            self._news_service = NewsService()
        return self._news_service

    def _get_world_state_service(self):
        if self._world_state_svc is None:
            from services.world_state_service import WorldStateService
            self._world_state_svc = WorldStateService(self._db)
        return self._world_state_svc

    # ── Interest extraction ───────────────────────────────────

    def extract_interests(self) -> list:
        """Extract ranked interest terms from user traits + topic frequency.

        Returns list of dicts: [{"term": str, "score": float, "source": "trait"|"topic"}, ...]
        """
        candidates = []
        candidates.extend(self._extract_trait_interests())
        candidates.extend(self._extract_topic_interests())

        if not candidates:
            return []

        deduped = self._deduplicate_by_embedding(candidates)
        deduped.sort(key=lambda x: x["score"], reverse=True)
        return deduped[:MAX_INTERESTS]

    def _extract_trait_interests(self) -> list:
        """Get interest terms from high-confidence user traits."""
        try:
            conn = self._db.get_connection()
            cursor = conn.execute(
                """SELECT key, value, confidence, evidence_count
                   FROM knowledge
                   WHERE kind IN ('trait', 'preference')
                     AND entity = 'user'
                     AND deleted_at IS NULL
                     AND confidence >= ?
                     AND evidence_count >= ?
                     AND reliability = 'reliable'
                     AND json_extract(data, '$.category') IN ('preference', 'core')
                   ORDER BY confidence DESC""",
                (TRAIT_MIN_CONFIDENCE, TRAIT_MIN_REINFORCEMENTS),
            )
            traits = cursor.fetchall()
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to query traits: {e}")
            return []

        candidates = []
        for row in traits:
            term = row[1].strip()
            if len(term) < 2 or len(term) > 100:
                continue
            score = row[2] * row[3]  # confidence * reinforcement_count
            candidates.append({"term": term, "score": score, "source": "trait"})
        return candidates

    def _extract_topic_interests(self) -> list:
        """Get interest terms from recent topic frequency."""
        try:
            conn = self._db.get_connection()
            cutoff = utc_now().isoformat()
            cursor = conn.execute(
                """SELECT topic, COUNT(*) as freq,
                          MAX(created_at) as last_seen
                   FROM topic_transcript
                   WHERE created_at >= datetime(?, '-' || ? || ' days')
                     AND role = 'user'
                     AND topic IS NOT NULL
                     AND topic != ''
                   GROUP BY topic
                   ORDER BY freq DESC
                   LIMIT 20""",
                (cutoff, TOPIC_LOOKBACK_DAYS),
            )
            topics = cursor.fetchall()
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to query topics: {e}")
            return []

        candidates = []
        max_freq = max((row[1] for row in topics), default=1)
        for row in topics:
            topic_name = row[0].strip()
            if len(topic_name) < 2:
                continue
            term = topic_name.replace("_", " ").replace("-", " ")
            freq_score = row[1] / max_freq
            candidates.append({"term": term, "score": freq_score, "source": "topic"})
        return candidates

    def _deduplicate_by_embedding(self, candidates: list) -> list:
        """Remove near-duplicate interests using embedding cosine similarity."""
        if len(candidates) <= 1:
            return candidates

        emb_svc = self._get_embedding_service()
        terms = [c["term"] for c in candidates]

        try:
            embeddings = emb_svc.generate_embeddings_batch(terms)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Embedding failed, skipping dedup: {e}")
            return candidates

        # Greedy dedup: keep highest-scored, drop anything too similar
        indexed = sorted(enumerate(candidates), key=lambda x: x[1]["score"], reverse=True)
        kept_indices = []
        kept_embeddings = []

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

        news_svc = self._get_news_service()
        world_svc = self._get_world_state_service()
        signals_written = 0

        for interest in interests:
            try:
                articles = news_svc.search(
                    query=interest["term"],
                    limit=MAX_ARTICLES_PER_INTEREST,
                )
                if not articles:
                    continue

                best = articles[0]
                world_svc.notify_external_signal(
                    signal_type="news",
                    source=SIGNAL_SOURCE,
                    content=f"{best.title} \u2014 {best.source}",
                    activation_energy=min(0.6, interest["score"] * 0.3),
                    metadata={
                        "interest": interest["term"],
                        "url": best.url,
                        "source_name": best.source,
                        "published_at": best.published_at,
                        "article_count": len(articles),
                    },
                )
                signals_written += 1

            except Exception as e:
                logger.warning(f"{LOG_PREFIX} Failed to scan '{interest['term']}': {e}")
                continue

        logger.info(f"{LOG_PREFIX} Scan complete: {signals_written} signals written")
        return signals_written


# ── Worker entry point ────────────────────────────────────────

def world_awareness_worker(shared_state=None):
    """Module-level entry point for run.py registration."""
    from services.database_service import DatabaseService

    db = DatabaseService()
    service = WorldAwarenessService(db)

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
            logger.error(f"{LOG_PREFIX} Scan failed: {e}", exc_info=True)
