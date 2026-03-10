"""
Uncertainty Service — epistemic confidence tracking across memory systems.

Detects contradictions and ambiguities between memories, tracks their state,
and drives reliability tagging on the source tables (user_traits, episodes,
semantic_concepts). This is the single authority for epistemic confidence.
"""

import logging
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

# Severity matrix: frozenset of memory types → severity level.
# frozenset gives pair-order independence (trait+episode == episode+trait).
_SEVERITY_MAP = {
    frozenset({'trait', 'trait'}):     'critical',
    frozenset({'concept', 'trait'}):   'high',
    frozenset({'concept', 'concept'}): 'high',
    frozenset({'episode', 'trait'}):   'medium',
    frozenset({'episode', 'concept'}): 'medium',
    frozenset({'episode', 'episode'}): 'low',
}

# Reliability values ranked high→low for comparisons.
_RELIABILITY_RANK = {
    'reliable': 4,
    'uncertain': 3,
    'contradicted': 2,
    'superseded': 1,
}

# Table targeted by each memory type for reliability writes.
_RELIABILITY_TABLE = {
    'trait':   'user_traits',
    'episode': 'episodes',
    'concept': 'semantic_concepts',
}


class UncertaintyService:
    """
    Manages epistemic uncertainty records and reliability tagging.

    Constructor follows the UserTraitService pattern: takes a database_service
    instance, uses self.db.connection() context manager throughout.
    """

    def __init__(self, database_service):
        self.db = database_service

    # ── Public API ──────────────────────────────────────────────────────────

    def create_uncertainty(
        self,
        memory_a_type: str,
        memory_a_id: str,
        memory_b_type: Optional[str] = None,
        memory_b_id: Optional[str] = None,
        uncertainty_type: str = 'contradiction',
        detection_context: str = 'drift',
        reasoning: Optional[str] = None,
        temporal_signal: bool = False,
        surface_context: Optional[str] = None,
    ) -> str:
        """
        Record a new uncertainty and tag the implicated memories.

        Returns the UUID of the created uncertainty record.
        """
        uncertainty_id = str(uuid.uuid4())
        severity = self._classify_severity(memory_a_type, memory_b_type)

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO uncertainties (
                        id, memory_a_type, memory_a_id,
                        memory_b_type, memory_b_id,
                        uncertainty_type, severity, detection_context,
                        reasoning, temporal_signal, surface_context
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    uncertainty_id,
                    memory_a_type, memory_a_id,
                    memory_b_type, memory_b_id,
                    uncertainty_type, severity, detection_context,
                    reasoning, int(temporal_signal), surface_context,
                ))
                cursor.close()

            # Tag both memories with an appropriate reliability label.
            if uncertainty_type == 'contradiction':
                self._set_reliability(memory_a_type, memory_a_id, 'contradicted')
                if memory_b_type and memory_b_id:
                    self._set_reliability(memory_b_type, memory_b_id, 'contradicted')
            else:
                # unverified / stale / ambiguous — softer signal
                self._set_reliability(memory_a_type, memory_a_id, 'uncertain')
                if memory_b_type and memory_b_id:
                    self._set_reliability(memory_b_type, memory_b_id, 'uncertain')

            logger.info(
                f"[UNCERTAINTY] Created {uncertainty_type} ({severity}) "
                f"{memory_a_type}:{memory_a_id} ↔ {memory_b_type}:{memory_b_id}"
            )
            return uncertainty_id

        except Exception as e:
            logger.error(f"[UNCERTAINTY] Failed to create uncertainty: {e}")
            return uncertainty_id

    def resolve_uncertainty(
        self,
        uncertainty_id: str,
        strategy: str,
        detail: Optional[str] = None,
        winner_type: Optional[str] = None,
        winner_id: Optional[str] = None,
        loser_type: Optional[str] = None,
        loser_id: Optional[str] = None,
    ) -> bool:
        """
        Resolve an open uncertainty.

        When winner/loser are provided, sets winner reliability to 'reliable'
        and loser to 'superseded'. Returns False if already resolved.
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE uncertainties
                    SET state = 'resolved',
                        resolution_strategy = ?,
                        resolution_detail = ?,
                        resolved_at = datetime('now')
                    WHERE id = ? AND state != 'resolved'
                """, (strategy, detail, uncertainty_id))
                affected = cursor.rowcount
                cursor.close()

            if affected == 0:
                return False  # already resolved or not found

            if winner_type and winner_id:
                self._set_reliability(winner_type, winner_id, 'reliable', force=True)
            if loser_type and loser_id:
                self._set_reliability(loser_type, loser_id, 'superseded', force=True)

            logger.info(
                f"[UNCERTAINTY] Resolved {uncertainty_id} via '{strategy}'"
            )
            return True

        except Exception as e:
            logger.error(f"[UNCERTAINTY] Failed to resolve {uncertainty_id}: {e}")
            return False

    def get_active_uncertainties(
        self,
        severity_filter: Optional[str] = None,
        limit: int = 20,
    ) -> list:
        """
        Return open or surfaced uncertainties, most severe first.
        """
        severity_rank = "CASE severity WHEN 'critical' THEN 1 WHEN 'high' THEN 2 WHEN 'medium' THEN 3 ELSE 4 END"
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                if severity_filter:
                    cursor.execute(f"""
                        SELECT * FROM uncertainties
                        WHERE state IN ('open', 'surfaced') AND severity = ?
                        ORDER BY {severity_rank}, created_at DESC
                        LIMIT ?
                    """, (severity_filter, limit))
                else:
                    cursor.execute(f"""
                        SELECT * FROM uncertainties
                        WHERE state IN ('open', 'surfaced')
                        ORDER BY {severity_rank}, created_at DESC
                        LIMIT ?
                    """, (limit,))
                rows = cursor.fetchall()
                cols = [d[0] for d in cursor.description]
                cursor.close()
                return [dict(zip(cols, r)) for r in rows]
        except Exception as e:
            logger.error(f"[UNCERTAINTY] get_active_uncertainties failed: {e}")
            return []

    def get_uncertainties_for_memory(self, memory_type: str, memory_id: str) -> list:
        """
        Return all open/surfaced uncertainties that reference this memory
        on either side of the pair.
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM uncertainties
                    WHERE state IN ('open', 'surfaced')
                      AND (
                          (memory_a_type = ? AND memory_a_id = ?)
                          OR (memory_b_type = ? AND memory_b_id = ?)
                      )
                    ORDER BY created_at DESC
                """, (memory_type, memory_id, memory_type, memory_id))
                rows = cursor.fetchall()
                cols = [d[0] for d in cursor.description]
                cursor.close()
                return [dict(zip(cols, r)) for r in rows]
        except Exception as e:
            logger.error(f"[UNCERTAINTY] get_uncertainties_for_memory failed: {e}")
            return []

    def check_memory_reliability(self, memory_type: str, memory_id: str) -> str:
        """
        Query the source table for the current reliability label.
        Returns 'reliable' if the record is not found.
        """
        table = _RELIABILITY_TABLE.get(memory_type)
        if not table:
            return 'reliable'
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT reliability FROM {table} WHERE id = ?",
                    (memory_id,)
                )
                row = cursor.fetchone()
                cursor.close()
                return row[0] if row and row[0] else 'reliable'
        except Exception as e:
            logger.error(f"[UNCERTAINTY] check_memory_reliability failed: {e}")
            return 'reliable'

    def mark_surfaced(self, uncertainty_id: str, anti_nag_threshold: int = 3) -> None:
        """
        Move an open uncertainty to 'surfaced' and increment the surface counter.
        After anti_nag_threshold surfacings, reduce severity (Phase 4 anti-nag).
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE uncertainties
                    SET state = 'surfaced',
                        surfaced_count = surfaced_count + 1,
                        last_surfaced_at = datetime('now')
                    WHERE id = ? AND state = 'open'
                """, (uncertainty_id,))
                cursor.close()
            # Anti-nag: downgrade severity if over-exposed
            self.downgrade_overexposed(uncertainty_id, threshold=anti_nag_threshold)
        except Exception as e:
            logger.error(f"[UNCERTAINTY] mark_surfaced failed: {e}")

    def downgrade_overexposed(self, uncertainty_id: str, threshold: int = 2) -> bool:
        """
        Phase 4 — Anti-nag: if surfaced_count exceeds threshold, reduce severity.

        critical → high → medium → low (stops at low).
        Returns True if severity was reduced.
        """
        _downgrade = {'critical': 'high', 'high': 'medium', 'medium': 'low'}
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT severity, surfaced_count FROM uncertainties WHERE id = ?",
                    (uncertainty_id,)
                )
                row = cursor.fetchone()
                if row is None or row[1] <= threshold or row[0] == 'low':
                    cursor.close()
                    return False
                new_severity = _downgrade.get(row[0], 'low')
                cursor.execute(
                    "UPDATE uncertainties SET severity = ? WHERE id = ?",
                    (new_severity, uncertainty_id)
                )
                cursor.close()
            logger.info(
                f"[UNCERTAINTY] Anti-nag: downgraded {uncertainty_id} "
                f"{row[0]} → {new_severity} after {row[1]} surfacings"
            )
            # Log to interaction_log for constraint learning
            try:
                from services.interaction_log_service import InteractionLogService
                InteractionLogService(self.db).log_event(
                    event_type='uncertainty_downgraded',
                    payload={
                        'uncertainty_id': uncertainty_id,
                        'old_severity': row[0],
                        'new_severity': new_severity,
                        'surfaced_count': row[1],
                    },
                    source='uncertainty_service',
                )
            except Exception:
                pass
            return True
        except Exception as e:
            logger.error(f"[UNCERTAINTY] downgrade_overexposed failed: {e}")
            return False

    def resolve_by_reinforcement(
        self,
        memory_type: str,
        memory_id: str,
        reinforced_confidence: float,
    ) -> int:
        """
        Phase 4 — Evidence-based resolution: when a memory is reinforced,
        auto-resolve open uncertainties if the opposing memory is significantly
        weaker (confidence < half the reinforced value).

        Returns count of uncertainties resolved.
        """
        resolved = 0
        try:
            uncertainties = self.get_uncertainties_for_memory(memory_type, memory_id)
            for unc in uncertainties:
                if unc.get('uncertainty_type') != 'contradiction':
                    continue
                # Identify the opposing memory
                if unc.get('memory_a_type') == memory_type and unc.get('memory_a_id') == memory_id:
                    opp_type = unc.get('memory_b_type')
                    opp_id = unc.get('memory_b_id')
                else:
                    opp_type = unc.get('memory_a_type')
                    opp_id = unc.get('memory_a_id')

                if not opp_type or not opp_id:
                    continue

                # Check opposing memory confidence
                opp_confidence = self._get_memory_confidence(opp_type, opp_id)
                if opp_confidence is None:
                    continue

                # Resolve if reinforced side is more than 2x stronger
                if reinforced_confidence > opp_confidence * 2:
                    success = self.resolve_uncertainty(
                        uncertainty_id=unc['id'],
                        strategy='evidence_resolved',
                        detail=(
                            f"Reinforcement evidence: {memory_type}:{memory_id} "
                            f"confidence={reinforced_confidence:.2f} > 2x "
                            f"{opp_type}:{opp_id} confidence={opp_confidence:.2f}"
                        ),
                        winner_type=memory_type,
                        winner_id=memory_id,
                        loser_type=opp_type,
                        loser_id=opp_id,
                    )
                    if success:
                        resolved += 1
        except Exception as e:
            logger.error(f"[UNCERTAINTY] resolve_by_reinforcement failed: {e}")
        return resolved

    def resolve_decayed(self, memory_type: str, memory_id: str) -> int:
        """
        Bulk-resolve all open/surfaced uncertainties linked to a memory that
        has been deleted by the decay engine. Returns count resolved.
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE uncertainties
                    SET state = 'resolved',
                        resolution_strategy = 'decayed',
                        resolved_at = datetime('now')
                    WHERE state IN ('open', 'surfaced')
                      AND (
                          (memory_a_type = ? AND memory_a_id = ?)
                          OR (memory_b_type = ? AND memory_b_id = ?)
                      )
                """, (memory_type, memory_id, memory_type, memory_id))
                count = cursor.rowcount
                cursor.close()
            if count > 0:
                logger.info(
                    f"[UNCERTAINTY] Resolved {count} uncertainties "
                    f"for decayed {memory_type}:{memory_id}"
                )
            return count
        except Exception as e:
            logger.error(f"[UNCERTAINTY] resolve_decayed failed: {e}")
            return 0

    # ── Private helpers ─────────────────────────────────────────────────────

    def _classify_severity(
        self,
        memory_a_type: str,
        memory_b_type: Optional[str],
    ) -> str:
        if memory_b_type is None:
            return 'low'
        key = frozenset({memory_a_type, memory_b_type})
        return _SEVERITY_MAP.get(key, 'low')

    def _get_memory_confidence(self, memory_type: str, memory_id: str) -> Optional[float]:
        """Fetch the current confidence value of a memory record."""
        _conf_col = {
            'trait':   ('user_traits', 'confidence'),
            'episode': ('episodes', 'activation_score'),
            'concept': ('semantic_concepts', 'confidence'),
        }
        info = _conf_col.get(memory_type)
        if not info:
            return None
        table, col = info
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT {col} FROM {table} WHERE id = ?", (memory_id,))
                row = cursor.fetchone()
                cursor.close()
                return float(row[0]) if row and row[0] is not None else None
        except Exception:
            return None

    def _set_reliability(
        self,
        memory_type: str,
        memory_id: str,
        reliability: str,
        force: bool = False,
    ) -> None:
        """
        Update the reliability column on the source table for a memory.

        By default only downgrades reliability (rank guard) — prevents a late
        'uncertain' from silently overwriting an earlier 'contradicted'. Pass
        force=True to bypass the guard (used during explicit resolution).
        """
        table = _RELIABILITY_TABLE.get(memory_type)
        if not table:
            logger.warning(f"[UNCERTAINTY] Unknown memory_type '{memory_type}' for reliability set")
            return
        try:
            new_rank = _RELIABILITY_RANK.get(reliability, 0)
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT reliability FROM {table} WHERE id = ?",
                    (memory_id,)
                )
                row = cursor.fetchone()
                if row is None:
                    cursor.close()
                    return
                current = row[0] or 'reliable'
                current_rank = _RELIABILITY_RANK.get(current, 4)
                if force or new_rank < current_rank:
                    cursor.execute(
                        f"UPDATE {table} SET reliability = ? WHERE id = ?",
                        (reliability, memory_id)
                    )
                cursor.close()
        except Exception as e:
            logger.error(
                f"[UNCERTAINTY] _set_reliability failed for "
                f"{memory_type}:{memory_id} → {reliability}: {e}"
            )
