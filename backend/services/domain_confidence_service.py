"""
Domain Confidence Service — READ-ONLY memory-derived confidence for autonomous action.

Computes a 0.0–1.0 confidence score for Chalie acting autonomously in a given domain
by querying the existing memory hierarchy. No new storage — reads only.

Signal sources and weights:
    - Trait density       (0.30): user_traits matching the domain
    - Episodic success    (0.25): action success rate in domain episodes / interaction_log
    - Concept depth       (0.20): semantic_concepts related to the domain
    - Constraint penalty  (0.15): gate rejections in this domain (subtracted)
    - Recency weight      (0.10): exponential decay (14-day half-life) of recent activity

Calibration targets:
    Fresh instance:       ~0.1  (few traits, no episodes, no concepts)
    1-month instance:     ~0.3–0.5 (growing trait/concept density)
    6-month active domain: ~0.6–0.8

Caching: MemoryStore, 1-hour TTL per domain key.
"""

import json
import logging
import math
from typing import Optional

from services.time_utils import utc_now, parse_utc
from services.database_service import DatabaseService

logger = logging.getLogger(__name__)
LOG_PREFIX = "[DOMAIN CONFIDENCE]"

# ── Weights ──────────────────────────────────────────────────────────
W_TRAIT = 0.30
W_EPISODE = 0.25
W_CONCEPT = 0.20
W_CONSTRAINT = 0.15   # applied as (1.0 - penalty)
W_RECENCY = 0.10

# ── Calibration constants ─────────────────────────────────────────────
# Trait density: saturates at TRAIT_SATURATION traits
TRAIT_SATURATION = 10

# Concept depth: saturates at CONCEPT_SATURATION concepts
CONCEPT_SATURATION = 5

# Constraint penalty: saturates at PENALTY_SATURATION rejections
PENALTY_SATURATION = 5

# Recency: exponential decay, half-life 14 days
RECENCY_HALF_LIFE_DAYS = 14.0

# ── Cache constants ───────────────────────────────────────────────────
CACHE_KEY_PREFIX = "domain_confidence:"
CACHE_ALL_KEY = "domain_confidence:__keys__"
CACHE_TTL = 3600  # 1 hour


class DomainConfidenceService:
    """
    Reads existing memory tables to compute per-domain autonomous-action confidence.

    This service is READ-ONLY. It never writes to any table.

    Usage:
        svc = DomainConfidenceService(db_service, memory_store)
        score = svc.compute_domain_confidence("scheduling")
    """

    def __init__(self, db_service: DatabaseService, memory_store=None):
        """
        Initialize the service.

        Args:
            db_service: DatabaseService for SQLite access.
            memory_store: Optional MemoryStore for caching. When None, caching
                is skipped (useful for unit tests that don't need it).
        """
        self._db = db_service
        self._store = memory_store

    # ── Public API ────────────────────────────────────────────────────

    def compute_domain_confidence(self, domain: str, account_id: int = 1) -> float:
        """
        Return 0.0–1.0 confidence score for autonomous action in this domain.

        Checks the MemoryStore cache first (1h TTL). On cache miss, queries
        the memory hierarchy and stores the result.

        Args:
            domain: Domain string (e.g. "scheduling", "finance", "health").
                    Case-insensitive; normalised to lowercase for storage.
            account_id: Account to scope the query (default 1 for single-user).

        Returns:
            Float in [0.0, 1.0]. Low values indicate caution; high values
            indicate accumulated context in this domain.
        """
        domain = domain.strip().lower()
        if not domain:
            return 0.0

        # ── Cache read ──────────────────────────────────────────────
        cached = self._cache_get(domain)
        if cached is not None:
            logger.debug(f"{LOG_PREFIX} cache hit for domain='{domain}' score={cached:.3f}")
            return cached

        # ── Compute signals ─────────────────────────────────────────
        try:
            trait_score = self._trait_density(domain, account_id)
            episode_score = self._episodic_success(domain, account_id)
            concept_score = self._concept_depth(domain)
            constraint_penalty = self._constraint_penalty(domain, account_id)
            recency_score = self._recency_weight(domain, account_id)
        except Exception:
            logger.exception(f"{LOG_PREFIX} signal computation failed for domain='{domain}'")
            return 0.0

        raw = (
            W_TRAIT * trait_score
            + W_EPISODE * episode_score
            + W_CONCEPT * concept_score
            + W_CONSTRAINT * (1.0 - constraint_penalty)
            + W_RECENCY * recency_score
        )
        score = min(1.0, max(0.0, raw))

        logger.debug(
            f"{LOG_PREFIX} domain='{domain}' "
            f"trait={trait_score:.3f} episode={episode_score:.3f} "
            f"concept={concept_score:.3f} constraint_penalty={constraint_penalty:.3f} "
            f"recency={recency_score:.3f} -> score={score:.3f}"
        )

        # ── Cache write ─────────────────────────────────────────────
        self._cache_set(domain, score)
        return score

    def invalidate_domain(self, domain: str) -> None:
        """
        Evict the cached score for a single domain.

        Call this when a trait, episode, or constraint event fires for the domain
        so that the next call recomputes from fresh data.

        Args:
            domain: Domain string to invalidate (case-insensitive).
        """
        if self._store is None:
            return
        key = CACHE_KEY_PREFIX + domain.strip().lower()
        try:
            self._store.delete(key)
        except Exception:
            logger.debug(f"{LOG_PREFIX} cache invalidate failed for domain='{domain}'")

    def invalidate_all(self) -> None:
        """
        Evict all domain confidence cache entries.

        Iterates the tracked key set and deletes each entry. Safe to call
        after bulk memory operations (e.g., memory consolidation runs).
        """
        if self._store is None:
            return
        try:
            raw = self._store.get(CACHE_ALL_KEY)
            if not raw:
                return
            domains = json.loads(raw)
            for domain in domains:
                key = CACHE_KEY_PREFIX + domain
                try:
                    self._store.delete(key)
                except Exception:
                    pass
            self._store.delete(CACHE_ALL_KEY)
        except Exception:
            logger.debug(f"{LOG_PREFIX} invalidate_all failed")

    # ── Signal: Trait Density ─────────────────────────────────────────

    def _trait_density(self, domain: str, account_id: int) -> float:
        """
        Score based on how many user traits relate to this domain.

        Queries user_traits for rows whose category or trait_value contains the
        domain string. Score = min(1.0, count / TRAIT_SATURATION) * avg(confidence).

        Returns 0.0 when no matching traits exist.
        """
        sql = """
            SELECT COUNT(*) AS cnt,
                   AVG(confidence) AS avg_conf
            FROM user_traits
            WHERE (
                lower(category) LIKE lower(?)
                OR lower(trait_key) LIKE lower(?)
                OR lower(trait_value) LIKE lower(?)
            )
        """
        pattern = f"%{domain}%"
        try:
            with self._db.connection() as conn:
                conn.row_factory = __import__('sqlite3').Row
                row = conn.execute(sql, (pattern, pattern, pattern)).fetchone()
        except Exception:
            logger.debug(f"{LOG_PREFIX} trait_density query failed", exc_info=True)
            return 0.0

        if row is None:
            return 0.0

        count = row[0] or 0
        avg_conf = row[1] or 0.0

        if count == 0:
            return 0.0

        density = min(1.0, count / TRAIT_SATURATION)
        return density * avg_conf

    # ── Signal: Episodic Success ──────────────────────────────────────

    def _episodic_success(self, domain: str, account_id: int) -> float:
        """
        Score based on the success rate of past actions in this domain.

        Looks at interaction_log entries where event_type is 'tool_result' or
        'action_gate_rejected' and topic/payload contains the domain string.

        Success rate = tool_result count / (tool_result + action_gate_rejected) count.
        Returns 0.0 when no domain activity exists in the log.
        """
        sql = """
            SELECT
                SUM(CASE WHEN event_type = 'tool_result' THEN 1 ELSE 0 END) AS successes,
                SUM(CASE WHEN event_type = 'action_gate_rejected' THEN 1 ELSE 0 END) AS rejections,
                COUNT(*) AS total
            FROM interaction_log
            WHERE event_type IN ('tool_result', 'action_gate_rejected')
              AND (
                  lower(topic) LIKE lower(?)
                  OR lower(payload) LIKE lower(?)
              )
        """
        pattern = f"%{domain}%"
        try:
            with self._db.connection() as conn:
                conn.row_factory = __import__('sqlite3').Row
                row = conn.execute(sql, (pattern, pattern)).fetchone()
        except Exception:
            logger.debug(f"{LOG_PREFIX} episodic_success query failed", exc_info=True)
            return 0.0

        if row is None:
            return 0.0

        successes = row[0] or 0
        total = row[2] or 0

        if total == 0:
            return 0.0

        return successes / total

    # ── Signal: Concept Depth ─────────────────────────────────────────

    def _concept_depth(self, domain: str) -> float:
        """
        Score based on how many semantic concepts relate to this domain.

        Queries semantic_concepts for rows whose domain, concept_name, or
        definition contains the domain string.
        Score = min(1.0, count / CONCEPT_SATURATION) * avg(confidence).

        Returns 0.0 when no matching concepts exist.
        """
        sql = """
            SELECT COUNT(*) AS cnt,
                   AVG(confidence) AS avg_conf
            FROM semantic_concepts
            WHERE deleted_at IS NULL
              AND (
                  lower(domain) LIKE lower(?)
                  OR lower(concept_name) LIKE lower(?)
                  OR lower(definition) LIKE lower(?)
              )
        """
        pattern = f"%{domain}%"
        try:
            with self._db.connection() as conn:
                conn.row_factory = __import__('sqlite3').Row
                row = conn.execute(sql, (pattern, pattern, pattern)).fetchone()
        except Exception:
            logger.debug(f"{LOG_PREFIX} concept_depth query failed", exc_info=True)
            return 0.0

        if row is None:
            return 0.0

        count = row[0] or 0
        avg_conf = row[1] or 0.0

        if count == 0:
            return 0.0

        depth = min(1.0, count / CONCEPT_SATURATION)
        return depth * avg_conf

    # ── Signal: Constraint Penalty ────────────────────────────────────

    def _constraint_penalty(self, domain: str, account_id: int) -> float:
        """
        Penalty score based on gate rejection frequency in this domain.

        Queries interaction_log for 'action_gate_rejected' events where topic
        or payload contains the domain string.
        Penalty = min(1.0, rejection_count / PENALTY_SATURATION).

        Returns 0.0 when no rejections exist (no penalty).
        A high return value means the final confidence will be reduced.
        """
        sql = """
            SELECT COUNT(*) AS cnt
            FROM interaction_log
            WHERE event_type = 'action_gate_rejected'
              AND (
                  lower(topic) LIKE lower(?)
                  OR lower(payload) LIKE lower(?)
              )
        """
        pattern = f"%{domain}%"
        try:
            with self._db.connection() as conn:
                conn.row_factory = __import__('sqlite3').Row
                row = conn.execute(sql, (pattern, pattern)).fetchone()
        except Exception:
            logger.debug(f"{LOG_PREFIX} constraint_penalty query failed", exc_info=True)
            return 0.0

        if row is None:
            return 0.0

        count = row[0] or 0
        return min(1.0, count / PENALTY_SATURATION)

    # ── Signal: Recency Weight ────────────────────────────────────────

    def _recency_weight(self, domain: str, account_id: int) -> float:
        """
        Score based on how recently the domain was active.

        Finds the most recent interaction_log entry for this domain and applies
        exponential decay: score = 2^(-days_elapsed / HALF_LIFE).

        Returns 0.0 when no domain activity is recorded.
        Returns 1.0 when there is activity within the current day.
        """
        sql = """
            SELECT MAX(created_at) AS latest
            FROM interaction_log
            WHERE (
                lower(topic) LIKE lower(?)
                OR lower(payload) LIKE lower(?)
            )
        """
        pattern = f"%{domain}%"
        try:
            with self._db.connection() as conn:
                conn.row_factory = __import__('sqlite3').Row
                row = conn.execute(sql, (pattern, pattern)).fetchone()
        except Exception:
            logger.debug(f"{LOG_PREFIX} recency_weight query failed", exc_info=True)
            return 0.0

        if row is None or row[0] is None:
            return 0.0

        try:
            latest_dt = parse_utc(row[0])
        except Exception:
            return 0.0

        now = utc_now()
        elapsed_seconds = (now - latest_dt).total_seconds()
        elapsed_days = elapsed_seconds / 86400.0

        if elapsed_days < 0:
            elapsed_days = 0.0

        # Exponential decay: halves every RECENCY_HALF_LIFE_DAYS days
        score = math.pow(2.0, -elapsed_days / RECENCY_HALF_LIFE_DAYS)
        return min(1.0, max(0.0, score))

    # ── Cache helpers ─────────────────────────────────────────────────

    def _cache_get(self, domain: str) -> Optional[float]:
        """Return cached score for domain, or None on miss."""
        if self._store is None:
            return None
        try:
            raw = self._store.get(CACHE_KEY_PREFIX + domain)
            if raw is None:
                return None
            return float(raw)
        except Exception:
            return None

    def _cache_set(self, domain: str, score: float) -> None:
        """Store score for domain with TTL. Also tracks the key in CACHE_ALL_KEY."""
        if self._store is None:
            return
        try:
            self._store.set(CACHE_KEY_PREFIX + domain, str(score), ex=CACHE_TTL)
            # Track domain in the global key set for invalidate_all()
            raw = self._store.get(CACHE_ALL_KEY)
            domains: list = json.loads(raw) if raw else []
            if domain not in domains:
                domains.append(domain)
                self._store.set(CACHE_ALL_KEY, json.dumps(domains), ex=CACHE_TTL * 2)
        except Exception:
            logger.debug(f"{LOG_PREFIX} cache write failed for domain='{domain}'")
