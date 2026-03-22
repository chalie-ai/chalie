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

import json
import logging
import struct
import threading
import uuid
from datetime import timedelta
from typing import Any, Dict, List, Optional

from services.database_service import get_lightweight_db_service
from services.time_utils import utc_now, parse_utc

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
            except Exception:
                pass

            generate_strategy({'id': goal_id, 'type': goal_type, 'description': description})
        except Exception as e:
            logger.debug(
                f"{LOG_PREFIX} Strategy generation thread error for "
                f"goal {goal_id[:8]}: {e}"
            )

    thread = threading.Thread(target=_run, daemon=True, name=f"goal-strategy-{goal_id[:8]}")
    thread.start()
    logger.debug(f"{LOG_PREFIX} Strategy generation thread spawned for goal {goal_id[:8]}")


def _pack_embedding(embedding) -> Optional[bytes]:
    """Pack a list/tuple of floats into a binary blob for sqlite-vec."""
    if embedding is None:
        return None
    if isinstance(embedding, bytes):
        return embedding
    if isinstance(embedding, (list, tuple)):
        return struct.pack(f'{len(embedding)}f', *embedding)
    if hasattr(embedding, 'tolist'):
        flat = embedding.tolist()
        return struct.pack(f'{len(flat)}f', *flat)
    return embedding


class GoalEcologyService:
    """Manages persistent goal lifecycle, evidence accumulation, and salience scoring."""

    def __init__(self, db_service=None):
        self.db = db_service or get_lightweight_db_service()

    # ── CRUD ────────────────────────────────────────────────────────────────

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
        """
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

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO goals (
                    id, type, status, description, parent_motives, identity_links,
                    confidence, salience, commitment, urgency, timescale,
                    strategy, lineage_parent_id, evidence_count,
                    last_reinforced_at, outcome_feedback, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                goal_id, type, status, description, motives_json, links_json,
                confidence, salience, 0.0, urgency, timescale,
                None, None, 0,
                now, '[]', now, now,
            ))
            cursor.close()

        logger.info(
            f"{LOG_PREFIX} Created goal '{description[:60]}' "
            f"(type={type}, status={status}, salience={salience:.2f})"
        )

        # Post-creation: compute autobiography alignment (non-blocking, non-fatal)
        try:
            from services.goal_autobiography_bridge import compute_goal_alignment
            alignment = compute_goal_alignment(description)
            if alignment['parent_motives'] or alignment['identity_links']:
                with self.db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE goals SET parent_motives = ?, identity_links = ?, updated_at = ?
                        WHERE id = ?
                    """, (
                        json.dumps(alignment['parent_motives']),
                        json.dumps(alignment['identity_links']),
                        utc_now().isoformat(),
                        goal_id,
                    ))
                    cursor.close()
                self.recalculate_salience(goal_id)
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Alignment computation skipped: {e}")

        # Post-creation: find parent goal (non-blocking, non-fatal)
        try:
            parent_id = self._find_parent_goal(description, timescale, exclude_id=goal_id)
            if parent_id:
                with self.db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE goals SET lineage_parent_id = ?, updated_at = ? WHERE id = ?",
                        (parent_id, utc_now().isoformat(), goal_id)
                    )
                    cursor.close()
                logger.info(f"{LOG_PREFIX} Linked goal {goal_id[:8]} to parent {parent_id[:8]}")
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Parent detection skipped: {e}")

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

    def add_evidence(
        self,
        goal_id: str,
        signal_type: str,
        content: str,
        source: Optional[str] = None,
        strength: float = 1.0,
    ) -> None:
        """Add evidence to a goal, increment evidence_count, recalculate confidence/salience."""
        evidence_id = str(uuid.uuid4())
        now = utc_now().isoformat()

        with self.db.connection() as conn:
            cursor = conn.cursor()

            # Insert evidence row
            cursor.execute("""
                INSERT INTO goal_evidence (id, goal_id, signal_type, content, source, strength, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (evidence_id, goal_id, signal_type, content, source, strength, now))

            # Increment evidence count and update reinforcement time
            cursor.execute("""
                UPDATE goals
                SET evidence_count = evidence_count + 1,
                    last_reinforced_at = ?,
                    updated_at = ?
                WHERE id = ?
            """, (now, now, goal_id))

            cursor.close()

        # Recalculate after evidence is committed
        self.recalculate_salience(goal_id)

        logger.debug(
            f"{LOG_PREFIX} Evidence added to goal {goal_id[:8]}: "
            f"type={signal_type}, strength={strength:.2f}"
        )

    def find_matching_goals(self, text: str, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Find active goals whose descriptions are semantically similar to text."""
        try:
            from services.embedding_service import get_embedding_service
            import numpy as np

            embedding_service = get_embedding_service()
            query_emb = embedding_service.generate_embedding_np(text)

            # Get all non-terminal goals
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, description, type, status, salience, confidence
                    FROM goals
                    WHERE status NOT IN ('completed', 'decayed')
                """)
                rows = cursor.fetchall()
                cursor.close()

            if not rows:
                return []

            matches = []
            for row in rows:
                goal_id, desc, gtype, status, salience, confidence = row
                desc_emb = embedding_service.generate_embedding_np(desc)
                similarity = float(np.dot(query_emb, desc_emb))

                if similarity >= threshold:
                    matches.append({
                        'id': goal_id,
                        'description': desc,
                        'type': gtype,
                        'status': status,
                        'salience': salience,
                        'confidence': confidence,
                        'similarity': similarity,
                    })

            # Sort by similarity descending
            matches.sort(key=lambda m: m['similarity'], reverse=True)
            return matches

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} find_matching_goals failed: {e}")
            return []

    def recalculate_salience(self, goal_id: str) -> float:
        """
        Recalculate goal salience based on evidence and context.

        Formula:
          salience = (evidence_recency * 0.25 + evidence_density * 0.20 +
                      motive_alignment * 0.15 + urgency * 0.15 +
                      cross_context * 0.15 + identity_fit * 0.10)
        """
        with self.db.connection() as conn:
            cursor = conn.cursor()

            # Get goal data
            cursor.execute("""
                SELECT id, type, status, evidence_count, urgency,
                       parent_motives, identity_links, created_at, last_reinforced_at,
                       lineage_parent_id
                FROM goals WHERE id = ?
            """, (goal_id,))
            row = cursor.fetchone()
            if not row:
                cursor.close()
                return 0.0

            _, gtype, status, evidence_count, urgency, \
                motives_json, links_json, created_at, last_reinforced_at, \
                lineage_parent_id = row

            # Evidence recency: how recently was evidence added? (0-1)
            now = utc_now()
            if last_reinforced_at:
                reinforced = parse_utc(last_reinforced_at)
                hours_since = max(0, (now - reinforced).total_seconds() / 3600)
                evidence_recency = max(0.0, 1.0 - (hours_since / 168.0))  # 7 day decay
            else:
                evidence_recency = 0.0

            # Evidence density: how much evidence per time? (0-1)
            created = parse_utc(created_at)
            age_hours = max(1.0, (now - created).total_seconds() / 3600)
            evidence_density = min(1.0, evidence_count / max(1, age_hours / 24))  # 1 evidence/day = 1.0

            # Motive alignment: how many core motives are linked? (0-1)
            motives = json.loads(motives_json) if motives_json else []
            motive_alignment = 0.0
            for m in motives:
                if m in CORE_MOTIVES:
                    motive_alignment = max(motive_alignment, CORE_MOTIVES[m])

            # Identity fit: how many identity links? (0-1)
            links = json.loads(links_json) if links_json else []
            identity_fit = min(1.0, len(links) * 0.3)

            # Cross-context: diversity of evidence sources (0-1)
            cursor.execute("""
                SELECT COUNT(DISTINCT signal_type) FROM goal_evidence
                WHERE goal_id = ?
            """, (goal_id,))
            distinct_types = cursor.fetchone()[0]
            cross_context = min(1.0, distinct_types / 3.0)

            # Urgency (already 0-1)
            urgency_score = min(1.0, max(0.0, urgency))

            # Weighted sum
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
            try:
                if lineage_parent_id:
                    cursor2 = conn.cursor()
                    cursor2.execute("SELECT salience FROM goals WHERE id = ?", (lineage_parent_id,))
                    parent_row = cursor2.fetchone()
                    cursor2.close()
                    if parent_row and parent_row[0] > 0.5:
                        lineage_boost = parent_row[0] * 0.1  # Up to 10% of parent's salience
                        salience = min(1.0, salience + lineage_boost)
            except Exception:
                pass

            # Update confidence based on evidence count and type
            confidence = self._calculate_confidence(gtype, evidence_count)

            # Determine status transition
            new_status = self._determine_status(gtype, status, salience, confidence)

            cursor.execute("""
                UPDATE goals SET salience = ?, confidence = ?, status = ?, updated_at = ?
                WHERE id = ?
            """, (salience, confidence, new_status, now.isoformat(), goal_id))
            cursor.close()

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
        """
        Decay goals not reinforced within their timescale window.

        Goals below DECAY_THRESHOLD salience are marked 'decayed'.
        Returns count of goals decayed.
        """
        now = utc_now()
        decayed_count = 0

        with self.db.connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, timescale, salience, last_reinforced_at, created_at
                FROM goals
                WHERE status NOT IN ('completed', 'decayed', 'evolved')
            """)
            rows = cursor.fetchall()

            for row in rows:
                goal_id, timescale, salience, last_reinforced, created_at = row

                # Determine the reference time (last reinforced or created)
                ref_time = parse_utc(last_reinforced) if last_reinforced else parse_utc(created_at)
                window = TIMESCALE_WINDOWS.get(timescale, TIMESCALE_WINDOWS['medium_term'])

                # Check if goal is past its reinforcement window
                if (now - ref_time) > window:
                    new_salience = max(0.0, salience - SALIENCE_DECAY_RATE)

                    if new_salience < DECAY_THRESHOLD:
                        cursor.execute("""
                            UPDATE goals SET salience = 0.0, status = 'decayed', updated_at = ?
                            WHERE id = ?
                        """, (now.isoformat(), goal_id))
                        decayed_count += 1
                        logger.info(f"{LOG_PREFIX} Goal {goal_id[:8]} decayed (salience was {salience:.3f})")
                    else:
                        cursor.execute("""
                            UPDATE goals SET salience = ?, updated_at = ?
                            WHERE id = ?
                        """, (new_salience, now.isoformat(), goal_id))

            cursor.close()

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

        Muted goals are excluded — they stay alive for context injection but
        should not trigger proactive actions.
        """
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, type, status, description, confidence, salience,
                       urgency, timescale, strategy, evidence_count, outcome_feedback
                FROM goals
                WHERE status NOT IN ('completed', 'decayed', 'evolved')
                  AND strategy IS NOT NULL
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

            # Skip muted goals
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
        """
        Record user feedback on a proactive goal action.

        response_type: engaged|acknowledged|ignored|rejected

        Strategy evolution:
        - 3+ engaged outcomes: strategy marked as confirmed (add strategy_confirmed flag)
        - 2+ rejected outcomes for current strategy: archive old strategy, null it out
          (next drift cycle will regenerate with rejection history in prompt)
        """
        now = utc_now()

        # Feedback effects
        effects = {
            'engaged': {'confidence_delta': 0.15, 'salience_delta': 0.2},
            'acknowledged': {'confidence_delta': 0.05, 'salience_delta': 0.0},
            'ignored': {'confidence_delta': 0.0, 'salience_delta': -0.15},
            'rejected': {'confidence_delta': -0.3, 'salience_delta': -0.3},
        }
        effect = effects.get(response_type, {'confidence_delta': 0.0, 'salience_delta': 0.0})

        with self.db.connection() as conn:
            cursor = conn.cursor()

            # Get current values
            cursor.execute("""
                SELECT confidence, salience, outcome_feedback, strategy FROM goals WHERE id = ?
            """, (goal_id,))
            row = cursor.fetchone()
            if not row:
                cursor.close()
                return

            confidence, salience, feedback_json, current_strategy = row
            feedback = json.loads(feedback_json) if feedback_json else []

            # Apply deltas
            new_confidence = min(1.0, max(0.0, confidence + effect['confidence_delta']))
            new_salience = min(1.0, max(0.0, salience + effect['salience_delta']))

            # Append feedback entry with strategy version tracking
            feedback_entry = {
                'response': response_type,
                'timestamp': now.isoformat(),
            }
            if current_strategy:
                feedback_entry['strategy_hash'] = hash(current_strategy) % 10000
            feedback.append(feedback_entry)

            # Strategy evolution logic
            strategy_update = ""
            if current_strategy:
                current_hash = hash(current_strategy) % 10000
                # Count rejections against current strategy
                rejections_for_current = sum(
                    1 for f in feedback
                    if f.get('response') == 'rejected'
                    and f.get('strategy_hash') == current_hash
                )

                if rejections_for_current >= 2:
                    # Archive failed strategy in feedback history
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

                # Track engagement confirmations
                engagements_for_current = sum(
                    1 for f in feedback
                    if f.get('response') == 'engaged'
                    and f.get('strategy_hash') == current_hash
                )
                if engagements_for_current >= 3:
                    feedback.append({
                        'event': 'strategy_confirmed',
                        'strategy': current_strategy[:200],
                        'engagement_count': engagements_for_current,
                        'timestamp': now.isoformat(),
                    })

            params = [new_confidence, new_salience, json.dumps(feedback), now.isoformat(), goal_id]
            cursor.execute(f"""
                UPDATE goals
                SET confidence = ?, salience = ?, outcome_feedback = ?,
                    updated_at = ?{strategy_update}
                WHERE id = ?
            """, params)
            cursor.close()

        logger.info(
            f"{LOG_PREFIX} Outcome recorded for goal {goal_id[:8]}: "
            f"{response_type} (conf={new_confidence:.2f}, sal={new_salience:.2f})"
        )

    def evolve_goal(self, goal_id: str, new_description: str) -> Dict[str, Any]:
        """
        Evolve a goal: create a new goal with lineage, mark old as 'evolved'.
        """
        with self.db.connection() as conn:
            cursor = conn.cursor()

            # Get parent goal data
            cursor.execute("""
                SELECT type, parent_motives, identity_links, urgency, timescale,
                       confidence, evidence_count
                FROM goals WHERE id = ?
            """, (goal_id,))
            row = cursor.fetchone()
            if not row:
                cursor.close()
                raise ValueError(f"Goal {goal_id} not found")

            gtype, motives_json, links_json, urgency, timescale, confidence, evidence_count = row

            # Mark parent as evolved
            now = utc_now().isoformat()
            cursor.execute("""
                UPDATE goals SET status = 'evolved', updated_at = ? WHERE id = ?
            """, (now, goal_id))
            cursor.close()

        # Create evolved goal with inherited properties
        new_goal_id = str(uuid.uuid4())
        motives = json.loads(motives_json) if motives_json else []
        links = json.loads(links_json) if links_json else []

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO goals (
                    id, type, status, description, parent_motives, identity_links,
                    confidence, salience, commitment, urgency, timescale,
                    lineage_parent_id, evidence_count,
                    last_reinforced_at, outcome_feedback, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                new_goal_id, gtype, 'strengthening', new_description,
                json.dumps(motives), json.dumps(links),
                min(1.0, confidence * 0.8), 0.5, 0.0, urgency, timescale,
                goal_id, 0,
                now, '[]', now, now,
            ))
            cursor.close()

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
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()

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
                    except Exception:
                        pass

                created.append(goal)

            if created:
                logger.info(f"{LOG_PREFIX} Pattern detection created {len(created)} emergent goals")

            return created

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Pattern detection failed: {e}")
            return []

    def complete_goal(self, goal_id: str) -> bool:
        """Mark goal as completed."""
        now = utc_now().isoformat()
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE goals SET status = 'completed', updated_at = ?
                WHERE id = ? AND status NOT IN ('completed', 'decayed')
            """, (now, goal_id))
            affected = cursor.rowcount
            cursor.close()
        return affected > 0

    def dismiss_goal(self, goal_id: str) -> bool:
        """User-initiated dismissal -- mark as decayed."""
        now = utc_now().isoformat()
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE goals SET status = 'decayed', salience = 0.0, updated_at = ?
                WHERE id = ? AND status NOT IN ('completed', 'decayed')
            """, (now, goal_id))
            affected = cursor.rowcount
            cursor.close()
        return affected > 0

    def confirm_goal(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """User confirms a goal -- elevate to stated type."""
        now = utc_now().isoformat()
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE goals
                SET type = 'stated', confidence = MAX(confidence, 0.8),
                    salience = MAX(salience, 0.9), status = 'actionable',
                    updated_at = ?
                WHERE id = ?
            """, (now, goal_id))
            cursor.close()
        return self.get_goal(goal_id)

    def update_goal(self, goal_id: str, updates: dict) -> bool:
        """Update specific fields on a goal."""
        if not updates:
            return True
        allowed = {'urgency', 'timescale', 'description', 'strategy'}
        filtered = {k: v for k, v in updates.items() if k in allowed}
        if not filtered:
            return False
        filtered['updated_at'] = utc_now().isoformat()
        set_clause = ', '.join(f"{k} = ?" for k in filtered)
        values = list(filtered.values()) + [goal_id]
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE goals SET {set_clause} WHERE id = ?",
                values,
            )
            affected = cursor.rowcount
            cursor.close()
        return affected > 0

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
        Find a potential parent goal for a new goal based on embedding similarity.

        Only links shorter-timescale goals to longer-timescale ones.
        Returns the parent goal_id if found above 0.6 threshold, else None.
        """
        try:
            current_order = self.TIMESCALE_ORDER.get(timescale, -1)

            # Fetch all active goals with a strictly longer timescale
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, description, timescale
                    FROM goals
                    WHERE status NOT IN ('completed', 'decayed', 'evolved')
                """)
                rows = cursor.fetchall()
                cursor.close()

            candidates = [
                (r[0], r[1])
                for r in rows
                if self.TIMESCALE_ORDER.get(r[2], -1) > current_order
                and r[0] != exclude_id
            ]

            if not candidates:
                return None

            from services.embedding_service import get_embedding_service
            import numpy as np

            emb_service = get_embedding_service()
            desc_emb = emb_service.generate_embedding_np(description)

            best_id = None
            best_sim = 0.6  # minimum threshold

            for cand_id, cand_desc in candidates:
                cand_emb = emb_service.generate_embedding_np(cand_desc)
                # L2-normalized embeddings from generate_embedding_np — dot product == cosine sim
                norm_a = float(np.linalg.norm(desc_emb))
                norm_b = float(np.linalg.norm(cand_emb))
                if norm_a < 1e-9 or norm_b < 1e-9:
                    continue
                sim = float(np.dot(desc_emb, cand_emb) / (norm_a * norm_b + 1e-9))
                if sim > best_sim:
                    best_sim = sim
                    best_id = cand_id

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
        except Exception:
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
        """
        Mute a goal to prevent proactive triggering while keeping it alive.

        The mute marker is stored as a special entry in outcome_feedback:
        {"muted": true, "timestamp": "..."}.  The goal stays in the active stack
        and context injection — only get_actionable_goals() skips it.

        Returns True if the goal was found and muted, False otherwise.
        """
        now = utc_now()
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT outcome_feedback FROM goals WHERE id = ?", (goal_id,)
            )
            row = cursor.fetchone()
            if not row:
                cursor.close()
                return False

            feedback_json = row[0]
            feedback = json.loads(feedback_json) if feedback_json else []
            if not isinstance(feedback, list):
                feedback = []

            feedback.append({'muted': True, 'timestamp': now.isoformat()})

            cursor.execute("""
                UPDATE goals SET outcome_feedback = ?, updated_at = ?
                WHERE id = ?
            """, (json.dumps(feedback), now.isoformat(), goal_id))
            cursor.close()

        logger.info(f"{LOG_PREFIX} Goal {goal_id[:8]} muted")
        return True

    def unmute_goal(self, goal_id: str) -> bool:
        """
        Unmute a previously muted goal, re-enabling proactive triggering.

        Appends {"muted": false, "timestamp": "..."} to outcome_feedback so the
        mute state is fully auditable.

        Returns True if the goal was found and unmuted, False otherwise.
        """
        now = utc_now()
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT outcome_feedback FROM goals WHERE id = ?", (goal_id,)
            )
            row = cursor.fetchone()
            if not row:
                cursor.close()
                return False

            feedback_json = row[0]
            feedback = json.loads(feedback_json) if feedback_json else []
            if not isinstance(feedback, list):
                feedback = []

            feedback.append({'muted': False, 'timestamp': now.isoformat()})

            cursor.execute("""
                UPDATE goals SET outcome_feedback = ?, updated_at = ?
                WHERE id = ?
            """, (json.dumps(feedback), now.isoformat(), goal_id))
            cursor.close()

        logger.info(f"{LOG_PREFIX} Goal {goal_id[:8]} unmuted")
        return True
