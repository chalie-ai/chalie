"""
World Awareness Service — scheduled hourly interest-driven news scan.

Derives user interests from traits and topic frequency, queries the news tool,
and feeds results into world state. Zero LLM calls.
"""

import logging
import time

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
            from services.data_graph_service import get_data_graph_service
            rows = get_data_graph_service().fetch(
                kinds=['user_specific'],
                order_by='retrieval_weight DESC',
            )
            traits = [
                r for r in rows
                if (r.get('retrieval_weight') or 0) >= TRAIT_MIN_CONFIDENCE
                and (r.get('evidence_count') or 0) >= TRAIT_MIN_REINFORCEMENTS
            ]
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to query traits: {e}")
            return []

        candidates = []
        for row in traits:
            term = (row.get('value') or '').strip()
            if len(term) < 2 or len(term) > 100:
                continue
            score = (row.get('retrieval_weight') or 0) * (row.get('evidence_count') or 1)
            candidates.append({"term": term, "score": score, "source": "trait"})
        return candidates

    def _extract_topic_interests(self) -> list:
        """Get interest terms from recent topic frequency."""
        try:
            conn = self._db.get_connection()
            cutoff = utc_now().isoformat()
            cursor = conn.execute(
                """SELECT channel, COUNT(*) as freq,
                          MAX(created_at) as last_seen
                   FROM transcript
                   WHERE created_at >= datetime(?, '-' || ? || ' days')
                     AND role = 'user'
                     AND channel IS NOT NULL
                     AND channel != ''
                   GROUP BY channel
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

        from services.world_state import world_state as _world_state
        news_svc = self._get_news_service()
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
                headline = f"{best.title} \u2014 {best.source}"
                _world_state.push_signal("news", headline, ttl=3600)
                signals_written += 1

            except Exception as e:
                logger.warning(f"{LOG_PREFIX} Failed to scan '{interest['term']}': {e}")
                continue

        logger.info(f"{LOG_PREFIX} Scan complete: {signals_written} signals written")
        return signals_written


# ── Worker entry point ────────────────────────────────────────

def world_awareness_worker():
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
            logger.exception(f"{LOG_PREFIX} Scan failed: {e}")
