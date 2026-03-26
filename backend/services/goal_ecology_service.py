"""
Goal Ecology Service — Persistent goal lifecycle management.

Goals emerge from accumulated evidence over time, strengthen with reinforcement,
and act when mature. Different goal types form at different speeds:

  stated      — User said it explicitly; instant high salience
  inferred    — 2-3 signals over hours/days
  emergent    — 5-10 signals over days/weeks
  developmental — 10+ signals over weeks/months

State machine:
  candidate → strengthening → actionable → active → completed
                                                  → decayed
                                                  → evolved
"""

import concurrent.futures
import json
import logging
import struct
import threading

from services.embedding_utils import pack_embedding
import uuid
from datetime import timedelta
from typing import Any, Dict, List, Optional

from services.database_service import get_lightweight_db_service
from services.time_utils import utc_now, parse_utc
from services.write_queue_service import get_write_queue

try:
    from services.telemetry_service import (
        get_telemetry_collector,
        GOAL_LIFECYCLE,
    )
    _TELEMETRY_AVAILABLE = True
except Exception as e:  # pragma: no cover
    _TELEMETRY_AVAILABLE = False

logger = logging.getLogger(__name__)
LOG_PREFIX = "[GOAL ECOLOGY]"

# ── Core motives (hardcoded constants, not DB) ──────────────────────────────

CORE_MOTIVES = {
    'coherence': 0.95,
    'truthfulness': 0.98,
    'competence_growth': 0.90,
    'relationship_preservation': 0.92,
    'initiative': 0.80,
    'continuity_of_self': 0.88,
    'impact': 0.82,
}

# ── Goal type thresholds ────────────────────────────────────────────────────

ACTIONABLE_THRESHOLDS = {
    'stated': 0.0,
    'inferred': 0.4,
    'emergent': 0.6,
    'developmental': 0.75,
}

# ── Timescale decay windows ─────────────────────────────────────────────────
# If a goal hasn't been reinforced within this window, it starts losing salience.

TIMESCALE_WINDOWS = {
    'immediate': timedelta(hours=1),
    'short_term': timedelta(days=2),
    'medium_term': timedelta(days=7),
    'long_term': timedelta(days=30),
}

# Salience decay rate per decay cycle for unreinforced goals
SALIENCE_DECAY_RATE = 0.05
# Below this salience threshold, unreinforced goals are marked decayed
DECAY_THRESHOLD = 0.05

# MemoryStore key prefix for unmatched signals
UNMATCHED_SIGNAL_PREFIX = "goal:unmatched:"
UNMATCHED_SIGNAL_TTL = 14 * 24 * 3600  # 14 days

# Bounded pool for background strategy generation.  max_workers=2 keeps
# concurrent LLM calls predictable; excess submissions queue internally
# and run when a slot frees up.
_strategy_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="strategy-gen"
)


def _infer_timescale_from_signals(signals: list) -> str:
    """
    Infer appropriate timescale for an emergent goal based on signal ambient context.

    Signals from deep_focus + routine contexts suggest longer-term developmental goals.
    Default is medium_term if no strong ambient signal.
    """
    deep_focus_count = sum(
        1 for s in signals
        if 'deep_focus' in str(s.get('ambient_context', ''))
    )
    routine_count = sum(
        1 for s in signals
        if 'routine' in str(s.get('ambient_context', ''))
    )

    total = len(signals)
    if total == 0:
        return 'medium_term'

    # If majority of signals came during deep focus + routine, likely long-term
    if deep_focus_count / total > 0.5 and routine_count / total > 0.3:
        return 'long_term'

    # If majority are from deep focus, suggest medium-to-long
    if deep_focus_count / total > 0.5:
        return 'medium_term'

    return 'medium_term'


def _trigger_strategy_generation(goal_id: str, goal_type: str) -> None:
    """
    Spawn a daemon thread to generate a strategy for a newly actionable goal.

    Completely non-blocking and fault-tolerant: if the thread fails for any
    reason the goal continues to function without a strategy.
    """
    def _run():
        try:
            from services.goal_strategy_service import generate_strategy
            from services.database_service import get_lightweight_db_service

            # Fetch description so the LLM prompt is meaningful
            db = get_lightweight_db_service()
            description = ''
            try:
                with db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT description FROM goals WHERE id = ?", (goal_id,)
                    )
                    row = cursor.fetchone()
                    cursor.close()
                    if row:
                        description = row[0] or ''
            except Exception as e:
                logger.debug(f"{LOG_PREFIX} Failed to fetch goal description for strategy generation: {e}")

            generate_strategy({'id': goal_id, 'type': goal_type, 'description': description})
        except Exception as e:
            logger.debug(
                f"{LOG_PREFIX} Strategy generation thread error for "
                f"goal {goal_id[:8]}: {e}"
            )

    try:
        _strategy_pool.submit(_run)
        logger.debug(f"{LOG_PREFIX} Strategy generation submitted to pool for goal {goal_id[:8]}")
    except concurrent.futures.BrokenExecutor:
        logger.warning(f"{LOG_PREFIX} Strategy pool broken, skipping generation for goal {goal_id[:8]}")


from services.embedding_utils import pack_embedding as _pack_embedding  # noqa: E402


class GoalEcologyService:
    """Manages persistent goal lifecycle, evidence accumulation, and salience scoring."""

    def __init__(self, db_service=None):
        """Initialise the service, wiring in the shared DB service and write queue.

        Args:
            db_service: Optional pre-constructed
                :class:`~services.database_service.DatabaseService` instance;
                when ``None`` the lightweight singleton is used.
        """
        self.db = db_service or get_lightweight_db_service()
        self._write_queue = get_write_queue()

    # ── CRUD ────────────────────────────────────────────────────────────────

    def _find_similar_goal(
        self, description: str, goal_type: str, threshold: float = 0.80
    ) -> Optional[Dict[str, Any]]:
        """
        Find an existing goal with a similar description and same type.

        Uses KNN embedding search over goals_vec. Returns the first match
        above threshold, or None.
        """
        try:
            from services.embedding_service import EmbeddingService
            embedding_svc = EmbeddingService()
            embedding = embedding_svc.generate_embedding(description)
            if not embedding:
                return None

            packed = pack_embedding(embedding)

            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT g.id, g.description, g.type, g.status,
                           g.salience, g.confidence, v.distance
                    FROM goals_vec v
                    JOIN goals g ON g.rowid = v.rowid
                    WHERE v.embedding MATCH ? AND k = 5
                      AND g.type = ?
                      AND g.status NOT IN ('completed', 'decayed', 'evolved')
                    ORDER BY v.distance
                """, (packed, goal_type))
                rows = cursor.fetchall()
                cursor.close()

            for row in rows:
                similarity = max(0.0, 1.0 - (row[6] or 1.0))
                if similarity >= threshold:
                    return {
                        'id': row[0],
                        'description': row[1],
                        'type': row[2],
                        'status': row[3],
                        'salience': row[4],
                        'confidence': row[5],
                        'similarity': similarity,
                    }

            return None

        except Exception as e:
            logger.debug(f"{LOG_PREFIX} _find_similar_goal failed (non-fatal): {e}")
            return None

    def create_goal(
        self,
        description: str,
        type: str = 'emergent',
        urgency: float = 0.0,
        parent_motives: Optional[List[str]] = None,
        identity_links: Optional[List[str]] = None,
        timescale: str = 'medium_term',
    ) -> Dict[str, Any]:
        """
        Create a new goal.

        Stated goals get instant high salience. Other types start as candidates.
        Deduplicates: if a similar goal of the same type already exists,
        reinforces it (bumps salience) and returns it instead of creating a duplicate.
        """
        # -- Dedup: reinforce existing similar goal instead of duplicating --
        existing = self._find_similar_goal(description, type)
        if existing:
            new_salience = min(1.0, existing['salience'] + 0.1)
            now_iso = utc_now().isoformat()

            def _reinforce(
                sal=new_salience, ts=now_iso, gid=existing['id'], db=self.db
            ):
                """Asynchronously update salience when dedup finds an existing goal."""
                with db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """UPDATE goals SET salience = ?, last_reinforced_at = ?,
                           updated_at = ? WHERE id = ?""",
                        (sal, ts, ts, gid),
                    )
                    cursor.close()

            self._write_queue.submit_sync(_reinforce)
            existing['salience'] = new_salience
            logger.info(
                f"{LOG_PREFIX} Reinforced existing goal "
                f"(id={existing['id'][:8]}, new salience={new_salience:.2f}) "
                f"instead of duplicating"
            )
            return existing

        goal_id = str(uuid.uuid4())
        now = utc_now().isoformat()

        # Stated goals start strong — user already declared intent
        if type == 'stated':
            confidence = 0.8
            salience = 0.7
            status = 'actionable'
            urgency = max(urgency, 0.5)
        else:
            confidence = 0.1
            salience = 0.0
            status = 'candidate'

        motives_json = json.dumps(parent_motives or [])
        links_json = json.dumps(identity_links or [])

        _insert_params = (
            goal_id, type, status, description, motives_json, links_json,
            confidence, salience, 0.0, urgency, timescale,
            None, None, 0,
            now, '[]', now, now,
        )

        def _insert_goal(params=_insert_params, db=self.db):
            """Insert the new goal row inside the write-queue worker thread."""
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO goals (
                           id, type, status, description, parent_motives, identity_links,
                           confidence, salience, commitment, urgency, timescale,
                           strategy, lineage_parent_id, evidence_count,
                           last_reinforced_at, outcome_feedback, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    params,
                )
                cursor.close()

        self._write_queue.submit_sync(_insert_goal)

        logger.info(
            f"{LOG_PREFIX} Created goal '{description[:60]}' "
            f"(type={type}, status={status}, salience={salience:.2f})"
        )

        # Post-creation: compute autobiography alignment (non-blocking, non-fatal)
        # Merge with any caller-provided motives/links rather than replacing.
        try:
            from services.goal_autobiography_bridge import compute_goal_alignment
            alignment = compute_goal_alignment(description)
            merged_motives = list(dict.fromkeys((parent_motives or []) + alignment['parent_motives']))
            merged_links = list(dict.fromkeys((identity_links or []) + alignment['identity_links']))
            if merged_motives or merged_links:
                _align_params = (
                    json.dumps(merged_motives),
                    json.dumps(merged_links),
                    utc_now().isoformat(),
                    goal_id,
                )

                def _update_alignment(params=_align_params, db=self.db):
                    """Persist autobiography alignment data on the write-queue thread."""
                    with db.connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            """UPDATE goals SET parent_motives = ?, identity_links = ?,
                               updated_at = ? WHERE id = ?""",
                            params,
                        )
                        cursor.close()

                self._write_queue.submit_sync(_update_alignment)
                self.recalculate_salience(goal_id)
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Alignment computation skipped: {e}")

        # Post-creation: find parent goal (non-blocking, non-fatal)
        try:
            parent_id = self._find_parent_goal(description, timescale, exclude_id=goal_id)
            if parent_id:
                _link_params = (parent_id, utc_now().isoformat(), goal_id)

                def _update_parent(params=_link_params, db=self.db):
                    """Set lineage_parent_id for the newly created goal."""
                    with db.connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            "UPDATE goals SET lineage_parent_id = ?, updated_at = ? WHERE id = ?",
                            params,
                        )
                        cursor.close()

                self._write_queue.submit_sync(_update_parent)
                logger.info(f"{LOG_PREFIX} Linked goal {goal_id[:8]} to parent {parent_id[:8]}")
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Parent detection skipped: {e}")

        # Post-creation: store embedding for KNN search (non-blocking, non-fatal)
        self._store_goal_embedding(goal_id, description)

        # Emit GOAL_LIFECYCLE telemetry for goal creation
        if _TELEMETRY_AVAILABLE:
            try:
                get_telemetry_collector().record(
                    GOAL_LIFECYCLE,
                    {
                        "action": "create",
                        "goal_id": goal_id,
                        "type": type,
                        "status": status,
                        "salience": salience,
                        "timescale": timescale,
                    },
                )
            except Exception as _tel_err:
                logger.debug(
                    "%s GOAL_LIFECYCLE 'create' telemetry emit failed (non-fatal): %s",
                    LOG_PREFIX,
                    _tel_err,
                )

        return {
            'id': goal_id,
            'type': type,
            'status': status,
            'description': description,
            'confidence': confidence,
            'salience': salience,
            'urgency': urgency,
            'timescale': timescale,
        }

    def _store_goal_embedding(self, goal_id: str, description: str) -> None:
        """Store or replace embedding for a goal in goals_vec. Non-fatal.

        The rowid lookup (SELECT) is performed on the calling thread.  The
        ``INSERT OR REPLACE`` into ``goals_vec`` is dispatched to the shared
        write-queue to avoid lock contention.

        Args:
            goal_id: UUID of the goal whose embedding should be stored.
            description: Text used to generate the embedding vector.
        """
        try:
            from services.embedding_service import EmbeddingService
            embedding_svc = EmbeddingService()
            embedding = embedding_svc.generate_embedding(description)
            if not embedding:
                return

            packed = pack_embedding(embedding)

            # SELECT rowid on calling thread (read-only, fine to run directly)
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT rowid FROM goals WHERE id = ?", (goal_id,))
                row = cursor.fetchone()
                cursor.close()

            if not row:
                return

            rowid = row[0]

            def _upsert_vec(rid=rowid, emb=packed, db=self.db):
                """Insert or replace the embedding blob in goals_vec."""
                with db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT OR REPLACE INTO goals_vec(rowid, embedding) VALUES (?, ?)",
                        (rid, emb),
                    )
                    cursor.close()

            self._write_queue.submit_sync(_upsert_vec)

        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Goal embedding storage non-fatal: {e}")

    def add_evidence(
        self,
        goal_id: str,
        signal_type: str,
        content: str,
        source: Optional[str] = None,
        strength: float = 1.0,
    ) -> None:
        """Add evidence to a goal, increment evidence_count, recalculate confidence/salience.

        The INSERT into ``goal_evidence`` and the UPDATE of ``goals.evidence_count``
        are dispatched together as a single :meth:`submit_sync` call so they commit
        atomically before :meth:`recalculate_salience` reads the updated count.

        Args:
            goal_id: UUID of the goal receiving new evidence.
            signal_type: Category label for the evidence (e.g. ``'topic_recurrence'``).
            content: Raw text content of the evidence signal.
            source: Optional origin identifier (conversation ID, URL, etc.).
            strength: Numeric weight in the range 0–1 (default ``1.0``).
        """
        now = utc_now().isoformat()

        def _write_evidence(
            gid=goal_id, stype=signal_type,
            cont=content, src=source, stren=strength, ts=now, db=self.db,
        ):
            """Atomically insert evidence and bump evidence_count on the write-queue thread."""
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO goal_evidence
                           (goal_id, signal_type, content, source, strength, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (gid, stype, cont, src, stren, ts),
                )
                cursor.execute(
                    """UPDATE goals
                       SET evidence_count = evidence_count + 1,
                           last_reinforced_at = ?,
                           updated_at = ?
                       WHERE id = ?""",
                    (ts, ts, gid),
                )
                cursor.close()

        # submit_sync ensures the writes are committed before recalculate_salience
        # reads the updated evidence_count from the same table.
        self._write_queue.submit_sync(_write_evidence)

        # Recalculate after evidence is committed
        self.recalculate_salience(goal_id)

        logger.debug(
            f"{LOG_PREFIX} Evidence added to goal {goal_id[:8]}: "
            f"type={signal_type}, strength={strength:.2f}"
        )

        # Emit GOAL_LIFECYCLE telemetry for evidence addition
        if _TELEMETRY_AVAILABLE:
            try:
                get_telemetry_collector().record(
                    GOAL_LIFECYCLE,
                    {
                        "action": "evidence_added",
                        "goal_id": goal_id,
                        "signal_type": signal_type,
                        "strength": strength,
                    },
                )
            except Exception as _tel_err:
                logger.debug(
                    "%s GOAL_LIFECYCLE 'evidence_added' telemetry emit failed (non-fatal): %s",
                    LOG_PREFIX,
                    _tel_err,
                )

    def find_matching_goals(self, text: str, threshold: float = 0.6, k: int = 10) -> List[Dict[str, Any]]:
        """Find active goals matching text using KNN embedding search over goals_vec."""
        try:
            from services.embedding_service import EmbeddingService
            embedding_svc = EmbeddingService()
            embedding = embedding_svc.generate_embedding(text)
            if not embedding:
                return []

            packed = pack_embedding(embedding)

            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT g.id, g.description, g.type, g.status,
                           g.salience, g.confidence, v.distance
                    FROM goals_vec v
                    JOIN goals g ON g.rowid = v.rowid
                    WHERE v.embedding MATCH ? AND k = ?
                      AND g.status NOT IN ('completed', 'decayed')
                    ORDER BY v.distance
                """, (packed, k))
                rows = cursor.fetchall()
                cursor.close()

            matches = []
            for row in rows:
                similarity = max(0.0, 1.0 - (row[6] or 1.0))
                if similarity >= threshold:
                    matches.append({
                        'id': row[0],
                        'description': row[1],
                        'type': row[2],
                        'status': row[3],
                        'salience': row[4],
                        'confidence': row[5],
                        'similarity': similarity,
                    })

            return matches

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} find_matching_goals failed: {e}")
            return []

    def recalculate_salience(self, goal_id: str) -> float:
        """Recalculate goal salience based on evidence and context.

        All SELECT queries run directly on the calling thread.  Only the final
        UPDATE (persisting the computed salience, confidence, and status) is
        dispatched to the write queue to avoid ``SQLITE_BUSY`` under concurrency.

        Formula::

            salience = (evidence_recency * 0.25 + evidence_density * 0.20 +
                        cross_context * 0.15 + motive_alignment * 0.15 +
                        identity_fit * 0.10 + urgency * 0.15)

        Args:
            goal_id: UUID of the goal to recompute.

        Returns:
            The newly computed salience value (0.0–1.0), or ``0.0`` if the goal
            does not exist.
        """
        # ── Read phase (direct, no lock contention) ──────────────────────────
        with self.db.connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """SELECT id, type, status, evidence_count, urgency,
                          parent_motives, identity_links, created_at, last_reinforced_at,
                          lineage_parent_id
                   FROM goals WHERE id = ?""",
                (goal_id,),
            )
            row = cursor.fetchone()
            if not row:
                cursor.close()
                return 0.0

            _, gtype, status, evidence_count, urgency, \
                motives_json, links_json, created_at, last_reinforced_at, \
                lineage_parent_id = row

            # Cross-context: diversity of evidence sources (0-1)
            cursor.execute(
                "SELECT COUNT(DISTINCT signal_type) FROM goal_evidence WHERE goal_id = ?",
                (goal_id,),
            )
            distinct_types = cursor.fetchone()[0]

            # Lineage boost: read parent salience if linked
            parent_salience = 0.0
            try:
                if lineage_parent_id:
                    cursor.execute(
                        "SELECT salience FROM goals WHERE id = ?", (lineage_parent_id,)
                    )
                    parent_row = cursor.fetchone()
                    if parent_row:
                        parent_salience = parent_row[0] or 0.0
            except Exception as e:
                logger.debug(f"{LOG_PREFIX} Parent salience lookup failed for goal {goal_id[:8]}: {e}")

            cursor.close()

        # ── Computation phase ────────────────────────────────────────────────
        now = utc_now()

        # Evidence recency: how recently was evidence added? (0-1)
        if last_reinforced_at:
            reinforced = parse_utc(last_reinforced_at)
            hours_since = max(0, (now - reinforced).total_seconds() / 3600)
            evidence_recency = max(0.0, 1.0 - (hours_since / 168.0))  # 7 day decay
        else:
            evidence_recency = 0.0

        # Evidence density: how much evidence per time? (0-1)
        created = parse_utc(created_at)
        age_hours = max(1.0, (now - created).total_seconds() / 3600)
        evidence_density = min(1.0, evidence_count / max(1, age_hours / 24))

        # Motive alignment: how many core motives are linked? (0-1)
        motives = json.loads(motives_json) if motives_json else []
        motive_alignment = 0.0
        for m in motives:
            if m in CORE_MOTIVES:
                motive_alignment = max(motive_alignment, CORE_MOTIVES[m])

        # Identity fit: how many identity links? (0-1)
        links = json.loads(links_json) if links_json else []
        identity_fit = min(1.0, len(links) * 0.3)

        cross_context = min(1.0, distinct_types / 3.0)
        urgency_score = min(1.0, max(0.0, urgency))

        salience = (
            evidence_recency * 0.25 +
            evidence_density * 0.20 +
            cross_context * 0.15 +
            motive_alignment * 0.15 +
            identity_fit * 0.10 +
            urgency_score * 0.15
        )
        salience = round(min(1.0, max(0.0, salience)), 4)

        # Lineage boost: child goals get up to 10% boost from high-salience parents
        if parent_salience > 0.5:
            salience = min(1.0, salience + parent_salience * 0.1)

        confidence = self._calculate_confidence(gtype, evidence_count)
        new_status = self._determine_status(gtype, status, salience, confidence)

        # ── Write phase (dispatched to write queue) ──────────────────────────
        def _update_salience(
            sal=salience, conf=confidence, ns=new_status,
            ts=now.isoformat(), gid=goal_id, db=self.db,
        ):
            """Persist recomputed salience, confidence, and status for this goal."""
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """UPDATE goals SET salience = ?, confidence = ?, status = ?,
                       updated_at = ? WHERE id = ?""",
                    (sal, conf, ns, ts, gid),
                )
                cursor.close()

        self._write_queue.submit_sync(_update_salience)

        # Trigger strategy generation when a goal first becomes actionable
        old_status = status  # captured from the SELECT above
        if new_status == 'actionable' and old_status != 'actionable':
            _trigger_strategy_generation(goal_id, gtype)

        return salience

    def _calculate_confidence(self, goal_type: str, evidence_count: int) -> float:
        """Calculate confidence based on goal type and evidence count."""
        if goal_type == 'stated':
            return min(1.0, 0.8 + evidence_count * 0.05)
        elif goal_type == 'inferred':
            return min(1.0, 0.1 + evidence_count * 0.15)
        elif goal_type == 'emergent':
            return min(1.0, 0.05 + evidence_count * 0.08)
        else:  # developmental
            return min(1.0, 0.02 + evidence_count * 0.05)

    def _determine_status(
        self, goal_type: str, current_status: str,
        salience: float, confidence: float,
    ) -> str:
        """Determine the goal's status based on salience and confidence."""
        # Don't transition out of terminal states
        if current_status in ('completed', 'decayed', 'evolved'):
            return current_status

        # Fast-track: stated goals with high confidence bypass salience ramp
        if goal_type == 'stated' and confidence >= 0.8 and current_status not in ('completed', 'decayed'):
            return 'actionable'

        threshold = ACTIONABLE_THRESHOLDS.get(goal_type, 0.6)

        if salience >= threshold and confidence >= 0.6:
            return 'actionable'
        elif confidence >= 0.3:
            return 'strengthening'
        else:
            return 'candidate'

    def decay_unreinforced(self) -> int:
        """Decay goals not reinforced within their timescale window.

        All candidates are identified via a single SELECT on the calling thread.
        Each individual UPDATE is then dispatched to the write queue using
        loop-safe default-argument binding to avoid closure capture bugs.

        Returns:
            Number of goals whose status was set to ``'decayed'`` in this cycle.
        """
        now = utc_now()
        decayed_count = 0
        now_iso = now.isoformat()

        # ── Read phase: identify candidates directly ──────────────────────────
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT id, timescale, salience, last_reinforced_at, created_at
                   FROM goals
                   WHERE status NOT IN ('completed', 'decayed', 'evolved')"""
            )
            rows = cursor.fetchall()
            cursor.close()

        # ── Write phase: queue each UPDATE individually ───────────────────────
        for row in rows:
            goal_id, timescale, salience, last_reinforced, created_at = row

            ref_time = parse_utc(last_reinforced) if last_reinforced else parse_utc(created_at)
            window = TIMESCALE_WINDOWS.get(timescale, TIMESCALE_WINDOWS['medium_term'])

            if (now - ref_time) > window:
                new_salience = max(0.0, salience - SALIENCE_DECAY_RATE)

                if new_salience < DECAY_THRESHOLD:
                    def _decay_goal(gid=goal_id, ts=now_iso, db=self.db):
                        """Mark goal as decayed (salience → 0, status → decayed)."""
                        with db.connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute(
                                """UPDATE goals SET salience = 0.0, status = 'decayed',
                                   updated_at = ? WHERE id = ?""",
                                (ts, gid),
                            )
                            cursor.close()

                    self._write_queue.submit_sync(_decay_goal)
                    decayed_count += 1
                    logger.info(
                        f"{LOG_PREFIX} Goal {goal_id[:8]} decayed "
                        f"(salience was {salience:.3f})"
                    )
                else:
                    def _reduce_salience(gid=goal_id, sal=new_salience, ts=now_iso, db=self.db):
                        """Reduce goal salience one step toward the decay threshold."""
                        with db.connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute(
                                "UPDATE goals SET salience = ?, updated_at = ? WHERE id = ?",
                                (sal, ts, gid),
                            )
                            cursor.close()

                    self._write_queue.submit_sync(_reduce_salience)

        if decayed_count:
            logger.info(f"{LOG_PREFIX} Decay cycle: {decayed_count} goals decayed")

        return decayed_count

    def get_active_stack(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Get top goals by salience, excluding terminal states."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, type, status, description, confidence, salience,
                       urgency, timescale, strategy, evidence_count,
                       last_reinforced_at, created_at, lineage_parent_id
                FROM goals
                WHERE status NOT IN ('completed', 'decayed', 'evolved')
                ORDER BY salience DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            cursor.close()

        return [
            {
                'id': r[0], 'type': r[1], 'status': r[2], 'description': r[3],
                'confidence': r[4], 'salience': r[5], 'urgency': r[6],
                'timescale': r[7], 'strategy': r[8], 'evidence_count': r[9],
                'last_reinforced_at': r[10], 'created_at': r[11],
                'lineage_parent_id': r[12],
            }
            for r in rows
        ]

    def get_actionable_goals(self) -> List[Dict[str, Any]]:
        """Get goals where salience >= threshold for their type AND strategy exists.

        Muted goals are excluded via SQL (is_muted column) with _is_muted() as
        fallback for pre-migration goals. Goals stay alive for context injection
        but should not trigger proactive actions when muted.
        """
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, type, status, description, confidence, salience,
                       urgency, timescale, strategy, evidence_count, outcome_feedback
                FROM goals
                WHERE status NOT IN ('completed', 'decayed', 'evolved')
                  AND strategy IS NOT NULL
                  AND (is_muted IS NULL OR is_muted = 0)
                ORDER BY salience DESC
            """)
            rows = cursor.fetchall()
            cursor.close()

        actionable = []
        for r in rows:
            goal_type = r[1]
            salience = r[5]
            outcome_feedback_json = r[10]
            threshold = ACTIONABLE_THRESHOLDS.get(goal_type, 0.6)

            # Legacy fallback: check JSON for pre-migration goals
            if self._is_muted(outcome_feedback_json):
                continue

            if salience >= threshold:
                actionable.append({
                    'id': r[0], 'type': r[1], 'status': r[2], 'description': r[3],
                    'confidence': r[4], 'salience': r[5], 'urgency': r[6],
                    'timescale': r[7], 'strategy': r[8], 'evidence_count': r[9],
                })

        return actionable

    def record_outcome(self, goal_id: str, response_type: str) -> None:
        """Record user feedback on a proactive goal action.

        Reads the current goal state on the calling thread, computes the new
        values, then dispatches the UPDATE to the write queue.

        Args:
            goal_id: UUID of the goal being responded to.
            response_type: One of ``'engaged'``, ``'acknowledged'``,
                ``'ignored'``, or ``'rejected'``.

        Strategy evolution side effects:

        - 3+ engaged outcomes: ``strategy_confirmed`` event appended to feedback.
        - 2+ rejected outcomes for the current strategy: strategy nulled out;
          the next drift cycle regenerates it with rejection history in the prompt.
        """
        now = utc_now()

        effects = {
            'engaged': {'confidence_delta': 0.15, 'salience_delta': 0.2},
            'acknowledged': {'confidence_delta': 0.05, 'salience_delta': 0.0},
            'ignored': {'confidence_delta': 0.0, 'salience_delta': -0.15},
            'rejected': {'confidence_delta': -0.3, 'salience_delta': -0.3},
        }
        effect = effects.get(response_type, {'confidence_delta': 0.0, 'salience_delta': 0.0})

        # ── Read phase ───────────────────────────────────────────────────────
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT confidence, salience, outcome_feedback, strategy FROM goals WHERE id = ?",
                (goal_id,),
            )
            row = cursor.fetchone()
            cursor.close()

        if not row:
            return

        confidence, salience, feedback_json, current_strategy = row
        feedback = json.loads(feedback_json) if feedback_json else []

        # ── Computation phase ────────────────────────────────────────────────
        new_confidence = min(1.0, max(0.0, confidence + effect['confidence_delta']))
        new_salience = min(1.0, max(0.0, salience + effect['salience_delta']))

        feedback_entry: dict = {'response': response_type, 'timestamp': now.isoformat()}
        if current_strategy:
            feedback_entry['strategy_hash'] = hash(current_strategy) % 10000
        feedback.append(feedback_entry)

        # ── Outcome feedback compaction ──────────────────────────────────────
        # Keep the 15 most-recent entries verbatim; compact everything older
        # into a single summary object so the JSON column stays small.
        _KEEP = 15
        _COMPACT_THRESHOLD = 20
        if len(feedback) > _COMPACT_THRESHOLD:
            # Separate any existing compaction summary from raw entries.
            existing_summary = next(
                (f for f in feedback if isinstance(f, dict) and f.get('_compacted')),
                None,
            )
            raw_entries = [f for f in feedback if not (isinstance(f, dict) and f.get('_compacted'))]

            # Oldest entries to compact (everything beyond the tail we keep).
            to_compact = raw_entries[:-_KEEP]
            recent = raw_entries[-_KEEP:]

            # Response-type counters for entries being compacted now.
            _response_types = ('engaged', 'rejected', 'acknowledged', 'ignored')
            new_counts: dict = {rt: 0 for rt in _response_types}
            for entry in to_compact:
                rt = entry.get('response', '')
                if rt in new_counts:
                    new_counts[rt] += 1

            # Merge with any prior compacted summary.
            if existing_summary:
                merged_total = existing_summary.get('total', 0) + len(to_compact)
                merged_counts = {
                    rt: existing_summary.get(rt, 0) + new_counts[rt]
                    for rt in _response_types
                }
                # Extend the period span.
                prior_period_start = existing_summary.get('period', '').split(' to ')[0]
                new_period_end = now.strftime('%Y-%m')
                period_str = f"{prior_period_start} to {new_period_end}"
            else:
                merged_total = len(to_compact)
                merged_counts = new_counts
                # Derive period from oldest/newest of the compacted entries.
                timestamps = [
                    e.get('timestamp', '') for e in to_compact if e.get('timestamp')
                ]
                if timestamps:
                    period_start = parse_utc(min(timestamps)).strftime('%Y-%m')
                    period_end = parse_utc(max(timestamps)).strftime('%Y-%m')
                else:
                    period_start = period_end = now.strftime('%Y-%m')
                period_str = f"{period_start} to {period_end}"

            compacted_entry: dict = {'_compacted': True, 'total': merged_total, 'period': period_str}
            compacted_entry.update(merged_counts)
            feedback = [compacted_entry] + recent

        strategy_update = ""
        if current_strategy:
            current_hash = hash(current_strategy) % 10000
            rejections_for_current = sum(
                1 for f in feedback
                if f.get('response') == 'rejected' and f.get('strategy_hash') == current_hash
            )
            if rejections_for_current >= 2:
                feedback.append({
                    'event': 'strategy_failed',
                    'strategy': current_strategy[:200],
                    'rejection_count': rejections_for_current,
                    'timestamp': now.isoformat(),
                })
                strategy_update = ", strategy = NULL"
                logger.info(
                    f"{LOG_PREFIX} Strategy invalidated for goal {goal_id[:8]} "
                    f"after {rejections_for_current} rejections"
                )

            engagements_for_current = sum(
                1 for f in feedback
                if f.get('response') == 'engaged' and f.get('strategy_hash') == current_hash
            )
            if engagements_for_current >= 3:
                feedback.append({
                    'event': 'strategy_confirmed',
                    'strategy': current_strategy[:200],
                    'engagement_count': engagements_for_current,
                    'timestamp': now.isoformat(),
                })

        sql = (
            f"UPDATE goals SET confidence = ?, salience = ?, outcome_feedback = ?,"
            f" updated_at = ?{strategy_update} WHERE id = ?"
        )
        write_params = [new_confidence, new_salience, json.dumps(feedback), now.isoformat(), goal_id]

        # ── Write phase ──────────────────────────────────────────────────────
        def _record(stmt=sql, params=write_params, db=self.db):
            """Persist outcome deltas and feedback history for this goal."""
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(stmt, params)
                cursor.close()

        self._write_queue.submit_sync(_record)

        logger.info(
            f"{LOG_PREFIX} Outcome recorded for goal {goal_id[:8]}: "
            f"{response_type} (conf={new_confidence:.2f}, sal={new_salience:.2f})"
        )

    def evolve_goal(self, goal_id: str, new_description: str) -> Dict[str, Any]:
        """Evolve a goal: create a new goal with lineage, mark old as ``'evolved'``.

        The SELECT is performed directly on the calling thread.  The UPDATE that
        marks the parent as evolved and the INSERT of the new goal are both
        dispatched via :meth:`submit_sync` (in order) so the returned dict
        reflects committed state.

        Args:
            goal_id: UUID of the goal to evolve.
            new_description: Description text for the successor goal.

        Returns:
            Dict with ``id``, ``type``, ``status``, ``description``, and
            ``lineage_parent_id`` of the newly created successor goal.

        Raises:
            ValueError: When *goal_id* does not exist in the database.
        """
        # ── Read phase ───────────────────────────────────────────────────────
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT type, parent_motives, identity_links, urgency, timescale,
                          confidence, evidence_count
                   FROM goals WHERE id = ?""",
                (goal_id,),
            )
            row = cursor.fetchone()
            cursor.close()

        if not row:
            raise ValueError(f"Goal {goal_id} not found")

        gtype, motives_json, links_json, urgency, timescale, confidence, evidence_count = row
        now = utc_now().isoformat()

        # ── Write 1: mark parent as evolved (must commit before INSERT) ──────
        def _mark_evolved(gid=goal_id, ts=now, db=self.db):
            """Set status = 'evolved' on the parent goal."""
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE goals SET status = 'evolved', updated_at = ? WHERE id = ?",
                    (ts, gid),
                )
                cursor.close()

        self._write_queue.submit_sync(_mark_evolved)

        # ── Write 2: insert successor goal ───────────────────────────────────
        new_goal_id = str(uuid.uuid4())
        motives = json.loads(motives_json) if motives_json else []
        links = json.loads(links_json) if links_json else []

        _insert_params = (
            new_goal_id, gtype, 'strengthening', new_description,
            json.dumps(motives), json.dumps(links),
            min(1.0, confidence * 0.8), 0.5, 0.0, urgency, timescale,
            goal_id, 0,
            now, '[]', now, now,
        )

        def _insert_evolved(params=_insert_params, db=self.db):
            """Insert the evolved successor goal row."""
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO goals (
                           id, type, status, description, parent_motives, identity_links,
                           confidence, salience, commitment, urgency, timescale,
                           lineage_parent_id, evidence_count,
                           last_reinforced_at, outcome_feedback, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    params,
                )
                cursor.close()

        self._write_queue.submit_sync(_insert_evolved)

        # Store embedding for KNN search
        self._store_goal_embedding(new_goal_id, new_description)

        logger.info(
            f"{LOG_PREFIX} Goal evolved: {goal_id[:8]} → {new_goal_id[:8]} "
            f"('{new_description[:60]}')"
        )

        return {
            'id': new_goal_id,
            'type': gtype,
            'status': 'strengthening',
            'description': new_description,
            'lineage_parent_id': goal_id,
        }

    def detect_patterns_from_unmatched(self) -> List[Dict[str, Any]]:
        """
        Cluster unmatched signals from MemoryStore into candidate goals.

        Looks for 3+ signals with similar content (embedding cosine > 0.7)
        and creates a new emergent goal from the cluster.
        """
        try:
            from services.memory_store import get_shared_store
            store = get_shared_store()

            # Get all unmatched signal keys
            keys = store.keys(f"{UNMATCHED_SIGNAL_PREFIX}*")
            if not keys or len(keys) < 3:
                return []

            # Load signal data
            signals = []
            for key in keys:
                raw = store.get(key)
                if raw:
                    try:
                        data = json.loads(raw)
                        signals.append({'key': key, **data})
                    except (json.JSONDecodeError, TypeError):
                        continue

            if len(signals) < 3:
                return []

            # Simple clustering: group by embedding similarity
            from services.embedding_service import get_embedding_service
            import numpy as np

            embedding_service = get_embedding_service()
            embeddings = []
            for s in signals:
                emb = embedding_service.generate_embedding_np(s.get('content', ''))
                embeddings.append(emb)

            # Find clusters (greedy: pick first signal, find all similar)
            used = set()
            clusters = []

            for i in range(len(signals)):
                if i in used:
                    continue

                cluster = [i]
                used.add(i)

                for j in range(i + 1, len(signals)):
                    if j in used:
                        continue
                    sim = float(np.dot(embeddings[i], embeddings[j]))
                    if sim > 0.7:
                        cluster.append(j)
                        used.add(j)

                if len(cluster) >= 3:
                    clusters.append(cluster)

            # Create goals from clusters
            created = []
            for cluster_indices in clusters:
                cluster_signals = [signals[i] for i in cluster_indices]
                # Use the most common content as description
                contents = [s.get('content', '') for s in cluster_signals]
                description = max(contents, key=len) if contents else 'Unnamed goal'

                # Infer timescale from ambient context in signals
                timescale = _infer_timescale_from_signals(cluster_signals)

                goal = self.create_goal(
                    description=description,
                    type='emergent',
                    timescale=timescale,
                )

                # Add all cluster signals as evidence
                for s in cluster_signals:
                    self.add_evidence(
                        goal_id=goal['id'],
                        signal_type=s.get('signal_type', 'topic_recurrence'),
                        content=s.get('content', ''),
                        source=s.get('source'),
                        strength=s.get('strength', 1.0),
                    )

                # Clean up consumed signals from MemoryStore
                for s in cluster_signals:
                    try:
                        store.delete(s['key'])
                    except Exception as e:
                        logger.debug(f"{LOG_PREFIX} Failed to delete consumed signal key {s['key']}: {e}")

                created.append(goal)

            if created:
                logger.info(f"{LOG_PREFIX} Pattern detection created {len(created)} emergent goals")

            return created

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Pattern detection failed: {e}")
            return []

    def complete_goal(self, goal_id: str) -> bool:
        """Mark goal as completed.

        The UPDATE is dispatched via :meth:`submit_sync` so the return value
        reflects the write result.

        Args:
            goal_id: UUID of the goal to complete.

        Returns:
            ``True`` if a row was updated, ``False`` if the goal did not exist
            or was already in a terminal state.
        """
        now = utc_now().isoformat()

        def _complete(gid=goal_id, ts=now, db=self.db):
            """Set goal status to 'completed' and return the affected-row count."""
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """UPDATE goals SET status = 'completed', updated_at = ?
                       WHERE id = ? AND status NOT IN ('completed', 'decayed')""",
                    (ts, gid),
                )
                affected = cursor.rowcount
                cursor.close()
            return affected

        return self._write_queue.submit_sync(_complete) > 0

    def dismiss_goal(self, goal_id: str) -> bool:
        """User-initiated dismissal — mark goal as decayed.

        Args:
            goal_id: UUID of the goal to dismiss.

        Returns:
            ``True`` if a row was updated, ``False`` otherwise.
        """
        now = utc_now().isoformat()

        def _dismiss(gid=goal_id, ts=now, db=self.db):
            """Set goal status to 'decayed' and salience to 0."""
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """UPDATE goals SET status = 'decayed', salience = 0.0, updated_at = ?
                       WHERE id = ? AND status NOT IN ('completed', 'decayed')""",
                    (ts, gid),
                )
                affected = cursor.rowcount
                cursor.close()
            return affected

        return self._write_queue.submit_sync(_dismiss) > 0

    def confirm_goal(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """User confirms a goal — elevate to ``'stated'`` type.

        The UPDATE is dispatched via :meth:`submit_sync` so the subsequent
        :meth:`get_goal` SELECT sees the committed state.

        Args:
            goal_id: UUID of the goal to confirm.

        Returns:
            The updated goal dict, or ``None`` if the goal does not exist.
        """
        now = utc_now().isoformat()

        def _confirm(gid=goal_id, ts=now, db=self.db):
            """Elevate goal type and status to stated/actionable."""
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """UPDATE goals
                       SET type = 'stated', confidence = MAX(confidence, 0.8),
                           salience = MAX(salience, 0.9), status = 'actionable',
                           updated_at = ?
                       WHERE id = ?""",
                    (ts, gid),
                )
                cursor.close()

        self._write_queue.submit_sync(_confirm)
        return self.get_goal(goal_id)

    def update_goal(self, goal_id: str, updates: dict) -> bool:
        """Update specific allowed fields on a goal.

        Args:
            goal_id: UUID of the goal to modify.
            updates: Dict of field→value pairs.  Only ``'urgency'``,
                ``'timescale'``, ``'description'``, and ``'strategy'`` are
                accepted; others are silently ignored.

        Returns:
            ``True`` if at least one row was updated, ``False`` otherwise.
        """
        if not updates:
            return True
        allowed = {'urgency', 'timescale', 'description', 'strategy'}
        filtered = {k: v for k, v in updates.items() if k in allowed}
        if not filtered:
            return False
        filtered['updated_at'] = utc_now().isoformat()
        set_clause = ', '.join(f"{k} = ?" for k in filtered)
        values = list(filtered.values()) + [goal_id]

        def _update(stmt=f"UPDATE goals SET {set_clause} WHERE id = ?", params=values, db=self.db):
            """Apply field updates to the goal row."""
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(stmt, params)
                affected = cursor.rowcount
                cursor.close()
            return affected

        return self._write_queue.submit_sync(_update) > 0

    def get_goal(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """Get a single goal by ID."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, type, status, description, confidence, salience,
                       urgency, timescale, strategy, evidence_count,
                       parent_motives, identity_links, lineage_parent_id,
                       last_reinforced_at, last_acted_at, outcome_feedback,
                       created_at, updated_at
                FROM goals WHERE id = ?
            """, (goal_id,))
            row = cursor.fetchone()
            cursor.close()

        if not row:
            return None

        return {
            'id': row[0], 'type': row[1], 'status': row[2], 'description': row[3],
            'confidence': row[4], 'salience': row[5], 'urgency': row[6],
            'timescale': row[7], 'strategy': row[8], 'evidence_count': row[9],
            'parent_motives': json.loads(row[10]) if row[10] else [],
            'identity_links': json.loads(row[11]) if row[11] else [],
            'lineage_parent_id': row[12],
            'last_reinforced_at': row[13], 'last_acted_at': row[14],
            'outcome_feedback': json.loads(row[15]) if row[15] else [],
            'created_at': row[16], 'updated_at': row[17],
        }

    # ── Goal Hierarchy ───────────────────────────────────────────────────────

    TIMESCALE_ORDER = {'immediate': 0, 'short_term': 1, 'medium_term': 2, 'long_term': 3}

    def _find_parent_goal(
        self, description: str, timescale: str, exclude_id: str = None
    ) -> Optional[str]:
        """
        Find a potential parent goal using KNN search over goals_vec.

        Only links shorter-timescale goals to longer-timescale ones.
        Returns the parent goal_id if found above 0.6 threshold, else None.
        """
        try:
            current_order = self.TIMESCALE_ORDER.get(timescale, -1)

            # Use KNN search to find similar goals efficiently
            from services.embedding_service import EmbeddingService
            embedding_svc = EmbeddingService()
            embedding = embedding_svc.generate_embedding(description)
            if not embedding:
                return None

            packed = pack_embedding(embedding)

            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT g.id, g.timescale, v.distance
                    FROM goals_vec v
                    JOIN goals g ON g.rowid = v.rowid
                    WHERE v.embedding MATCH ? AND k = 10
                      AND g.status NOT IN ('completed', 'decayed', 'evolved')
                    ORDER BY v.distance
                """, (packed,))
                rows = cursor.fetchall()
                cursor.close()

            best_id = None
            best_sim = 0.6  # minimum threshold

            for goal_id, goal_timescale, distance in rows:
                if goal_id == exclude_id:
                    continue
                # Only link to goals with strictly longer timescale
                if self.TIMESCALE_ORDER.get(goal_timescale, -1) <= current_order:
                    continue
                similarity = max(0.0, 1.0 - (distance or 1.0))
                if similarity > best_sim:
                    best_sim = similarity
                    best_id = goal_id

            return best_id

        except Exception as e:
            logger.debug(f"{LOG_PREFIX} _find_parent_goal skipped: {e}")
            return None

    def get_children(self, goal_id: str) -> List[Dict[str, Any]]:
        """Get child goals linked via lineage_parent_id."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, type, status, description, salience, confidence
                    FROM goals
                    WHERE lineage_parent_id = ? AND status NOT IN ('completed', 'decayed')
                    ORDER BY salience DESC
                """, (goal_id,))
                rows = cursor.fetchall()
                cursor.close()
            return [
                {'id': r[0], 'type': r[1], 'status': r[2], 'description': r[3],
                 'salience': r[4], 'confidence': r[5]}
                for r in rows
            ]
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} get_children failed for goal {goal_id[:8]}: {e}")
            return []

    # ── Mute / Unmute ────────────────────────────────────────────────────────

    @staticmethod
    def _is_muted(outcome_feedback_json: Optional[str]) -> bool:
        """Return True if the goal has an active mute marker in outcome_feedback."""
        if not outcome_feedback_json:
            return False
        try:
            entries = json.loads(outcome_feedback_json)
            if not isinstance(entries, list):
                return False
            # A goal is muted if the most recent mute marker exists and is not
            # followed by an unmute marker.
            muted = False
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if entry.get('muted') is True:
                    muted = True
                elif entry.get('muted') is False:
                    muted = False
            return muted
        except (json.JSONDecodeError, TypeError):
            return False

    def mute_goal(self, goal_id: str) -> bool:
        """Mute a goal to prevent proactive triggering while keeping it alive.

        The mute marker is stored as a special entry in ``outcome_feedback``:
        ``{"muted": true, "timestamp": "..."}``.  The goal stays in the active
        stack and context injection — only :meth:`get_actionable_goals` skips it.

        The SELECT for current feedback runs on the calling thread; the UPDATE
        is dispatched via :meth:`submit_sync` so ``True`` is returned only after
        the write is confirmed committed.

        Args:
            goal_id: UUID of the goal to mute.

        Returns:
            ``True`` if the goal was found and muted, ``False`` otherwise.
        """
        now = utc_now()

        # ── Read phase ───────────────────────────────────────────────────────
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT outcome_feedback FROM goals WHERE id = ?", (goal_id,)
            )
            row = cursor.fetchone()
            cursor.close()

        if not row:
            return False

        feedback = json.loads(row[0]) if row[0] else []
        if not isinstance(feedback, list):
            feedback = []
        feedback.append({'muted': True, 'timestamp': now.isoformat()})

        # ── Write phase ──────────────────────────────────────────────────────
        def _mute(fb_json=json.dumps(feedback), ts=now.isoformat(), gid=goal_id, db=self.db):
            """Persist mute marker and set is_muted = 1."""
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """UPDATE goals SET outcome_feedback = ?, is_muted = 1, updated_at = ?
                       WHERE id = ?""",
                    (fb_json, ts, gid),
                )
                cursor.close()

        self._write_queue.submit_sync(_mute)
        logger.info(f"{LOG_PREFIX} Goal {goal_id[:8]} muted")
        return True

    def unmute_goal(self, goal_id: str) -> bool:
        """Unmute a previously muted goal, re-enabling proactive triggering.

        Appends ``{"muted": false, "timestamp": "..."}`` to ``outcome_feedback``
        so the mute state is fully auditable.

        The SELECT for current feedback runs on the calling thread; the UPDATE
        is dispatched via :meth:`submit_sync`.

        Args:
            goal_id: UUID of the goal to unmute.

        Returns:
            ``True`` if the goal was found and unmuted, ``False`` otherwise.
        """
        now = utc_now()

        # ── Read phase ───────────────────────────────────────────────────────
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT outcome_feedback FROM goals WHERE id = ?", (goal_id,)
            )
            row = cursor.fetchone()
            cursor.close()

        if not row:
            return False

        feedback = json.loads(row[0]) if row[0] else []
        if not isinstance(feedback, list):
            feedback = []
        feedback.append({'muted': False, 'timestamp': now.isoformat()})

        # ── Write phase ──────────────────────────────────────────────────────
        def _unmute(fb_json=json.dumps(feedback), ts=now.isoformat(), gid=goal_id, db=self.db):
            """Persist unmute marker and set is_muted = 0."""
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """UPDATE goals SET outcome_feedback = ?, is_muted = 0, updated_at = ?
                       WHERE id = ?""",
                    (fb_json, ts, gid),
                )
                cursor.close()

        self._write_queue.submit_sync(_unmute)
        logger.info(f"{LOG_PREFIX} Goal {goal_id[:8]} unmuted")
        return True

    # ── Cross-Goal Inference ─────────────────────────────────────────────────

    def detect_goal_clusters(self) -> List[Dict[str, Any]]:
        """
        Detect clusters of thematically related goals.

        Fetches embeddings from goals_vec (no on-the-fly generation), computes
        pairwise similarity. If 3+ goals have mutual similarity > 0.55, returns
        them as a cluster with a suggested consolidated description.

        Returns:
            List of cluster dicts: [{'goals': [...], 'suggested_description': '...'}]
        """
        try:
            import numpy as np

            # Get all active goals with their embeddings from goals_vec
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT g.id, g.description, g.type, g.salience, g.confidence,
                           g.timescale, g.rowid
                    FROM goals g
                    WHERE g.status NOT IN ('completed', 'decayed', 'evolved')
                    ORDER BY g.salience DESC
                    LIMIT 20
                """)
                rows = cursor.fetchall()

                if len(rows) < 3:
                    cursor.close()
                    return []

                goals = []
                embeddings = []
                for r in rows:
                    rowid = r[6]
                    cursor.execute(
                        "SELECT embedding FROM goals_vec WHERE rowid = ?",
                        (rowid,),
                    )
                    vec_row = cursor.fetchone()
                    if vec_row and vec_row[0]:
                        blob = vec_row[0]
                        n_floats = len(blob) // 4
                        emb = np.array(struct.unpack(f'{n_floats}f', blob), dtype=np.float32)
                        embeddings.append(emb)
                        goals.append({
                            'id': r[0], 'description': r[1], 'type': r[2],
                            'salience': r[3], 'confidence': r[4], 'timescale': r[5]
                        })

                cursor.close()

            if len(goals) < 3:
                return []

            # Build similarity matrix
            n = len(goals)
            sim_matrix = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    sim = float(np.dot(embeddings[i], embeddings[j]))
                    sim_matrix[i][j] = sim
                    sim_matrix[j][i] = sim

            # Find clusters: greedy, require 3+ goals with pairwise > 0.55
            used = set()
            clusters = []

            for i in range(n):
                if i in used:
                    continue

                cluster = [i]
                for j in range(i + 1, n):
                    if j in used:
                        continue
                    # Check similarity with ALL current cluster members
                    if all(sim_matrix[j][k] > 0.55 for k in cluster):
                        cluster.append(j)

                if len(cluster) >= 3:
                    clusters.append(cluster)
                    used.update(cluster)

            if not clusters:
                return []

            # Check cooldown: don't re-detect clusters too frequently
            from services.memory_store import get_shared_store
            store = get_shared_store()

            results = []
            for cluster_indices in clusters:
                cluster_goals = [goals[i] for i in cluster_indices]
                cluster_ids = sorted([g['id'] for g in cluster_goals])
                cluster_key = f"goal_cluster:{'|'.join(cluster_ids[:5])}"

                # Skip if this cluster was recently detected or dismissed
                if store.get(f"{cluster_key}:cooldown"):
                    continue

                # Generate consolidated description via LLM
                suggested = _generate_cluster_description(cluster_goals)

                results.append({
                    'goals': cluster_goals,
                    'suggested_description': suggested,
                    'cluster_key': cluster_key,
                })

            return results

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Cross-goal detection failed: {e}")
            return []


def _generate_cluster_description(goals: list) -> str:
    """Generate a consolidated description for a goal cluster via LLM."""
    try:
        from services.background_llm_queue import create_background_llm_proxy
        proxy = create_background_llm_proxy("goal-strategy")

        system_prompt = (
            "Given these related goals, propose a single higher-order goal that "
            "encompasses all of them. Output ONLY the goal description in one sentence. "
            "Be specific and concrete, not vague."
        )

        goal_lines = []
        for i, g in enumerate(goals, 1):
            goal_lines.append(f"{i}. [{g['type']}] {g['description']}")
        user_message = "Related goals:\n" + "\n".join(goal_lines)

        response = proxy.send_message(system_prompt, user_message)
        if response and response.text.strip():
            return response.text.strip()[:300]

        # Fallback: longest description
        return max((g['description'] for g in goals), key=len)

    except Exception as e:
        logging.debug(f"[GOAL ECOLOGY] Cluster description generation failed: {e}")
        return max((g['description'] for g in goals), key=len)
