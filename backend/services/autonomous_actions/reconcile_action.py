"""
ReconcileAction — Drift-time cross-store contradiction sweep.

Samples recent high-confidence traits and concepts, runs pairwise vector
comparison, and classifies candidate pairs via ContradictionClassifierService.
Auto-resolves temporal changes silently; creates uncertainty records for
genuine contradictions.

Priority: 4 — below REFLECT (5), well below COMMUNICATE (10).
Runs at most once per RECONCILE_COOLDOWN_MINUTES to cap DB load.
"""

import logging
import time
from typing import Optional, Dict

from services.memory_client import MemoryClientService
from .base import AutonomousAction, ActionResult, ThoughtContext

logger = logging.getLogger(__name__)

LOG_PREFIX = "[RECONCILE]"

# MemoryStore key for cooldown tracking
_COOLDOWN_KEY = "reconcile_action:last_run"

# Default: don't reconcile more than once per 30 minutes
_DEFAULT_COOLDOWN_MINUTES = 30

# Max uncertainty records to create per reconcile pass (safety limit)
_MAX_UNCERTAINTY_CREATES = 5


class ReconcileAction(AutonomousAction):
    """
    Cross-store contradiction sweep executed during cognitive drift.

    Process:
    1. Cooldown gate — skip if ran recently
    2. Sample N traits + N concepts from DB
    3. Pairwise vector similarity — find high-similarity pairs
    4. LLM classification — is this a real contradiction?
    5. Auto-resolve temporal changes; create uncertainty records for the rest
    """

    def __init__(self, config: dict = None, services: dict = None):
        """Initialize ReconcileAction with optional config overrides and injected services.

        Args:
            config: Optional configuration overrides. Supported keys:
                ``cooldown_minutes`` (int, default 30) — minimum minutes between runs.
                ``n_traits`` (int, default 5) — traits to sample per reconcile pass.
                ``n_concepts`` (int, default 5) — concepts to sample per reconcile pass.
                ``min_activation_energy`` (float, default 0.25) — minimum drift energy
                    required to trigger reconciliation.
                ``enabled`` (bool, default True) — set to False to disable the action.
            services: Optional injected service dependencies. Supported keys:
                ``db_service`` — a pre-built DatabaseService instance; if absent,
                    the shared DB service is resolved lazily at execute time.
        """
        super().__init__(name='RECONCILE', enabled=True, priority=4)

        config = config or {}
        services = services or {}

        self.store = MemoryClientService.create_connection()
        self._db_service = services.get('db_service')

        self.cooldown_minutes = config.get('cooldown_minutes', _DEFAULT_COOLDOWN_MINUTES)
        self.n_traits = config.get('n_traits', 5)
        self.n_concepts = config.get('n_concepts', 5)
        self.min_activation_energy = config.get('min_activation_energy', 0.25)

        if not config.get('enabled', True):
            self.enabled = False

    def should_execute(self, thought: ThoughtContext) -> tuple:
        """Determine eligibility and score for a reconcile sweep.

        Two gates must pass:

        1. **Activation energy gate** — drift must exceed ``min_activation_energy``
           (lower bar than other actions; reconcile is background maintenance).
        2. **Cooldown gate** — at least ``cooldown_minutes`` must have elapsed
           since the previous reconcile run.

        Score is intentionally low and predictable:
        ``0.3 + (activation_energy * 0.1)``

        Args:
            thought: The current drift thought context.

        Returns:
            tuple: ``(score: float, eligible: bool)``. Returns ``(0.0, False)``
                when any gate fails; otherwise returns the maintenance score
                with ``eligible=True``.
        """
        self.last_gate_result = None

        # Gate 1: activation energy floor (lower than other actions — reconcile
        # works even on low-energy drifts since it's background maintenance)
        if thought.activation_energy < self.min_activation_energy:
            self.last_gate_result = {'gate': 'activation_energy', 'reason': f"energy {thought.activation_energy:.2f} < {self.min_activation_energy}"}
            return (0.0, False)

        # Gate 2: cooldown
        last_run = self.store.get(_COOLDOWN_KEY)
        if last_run:
            try:
                last_ts = float(last_run)
                elapsed_min = (time.time() - last_ts) / 60
                if elapsed_min < self.cooldown_minutes:
                    logger.debug(
                        f"{LOG_PREFIX} Skipping — {elapsed_min:.1f}m since last run "
                        f"(cooldown={self.cooldown_minutes}m)"
                    )
                    self.last_gate_result = {'gate': 'cooldown', 'reason': f"{elapsed_min:.0f}m < {self.cooldown_minutes}m cooldown"}
                    return (0.0, False)
            except (ValueError, TypeError):
                pass

        # Score: low, predictable — this is maintenance, not creative output
        score = 0.3 + (thought.activation_energy * 0.1)
        logger.debug(f"{LOG_PREFIX} Eligible: score={score:.3f}")
        return (score, True)

    def execute(self, thought: ThoughtContext) -> ActionResult:
        """Run the cross-store contradiction sweep.

        Workflow:
        1. Resolve DB service (injected or shared fallback).
        2. Import ContradictionClassifierService and UncertaintyService.
        3. Sample ``n_traits`` traits + ``n_concepts`` concepts from the database.
        4. Run pairwise vector comparison to find high-similarity candidate pairs.
        5. For each untracked pair:

           - ``temporal_change`` with a temporal signal → auto-supersede via
             :meth:`_auto_supersede`.
           - All other contradictions → create an uncertainty record (capped at
             ``_MAX_UNCERTAINTY_CREATES`` per pass).
        6. Record the cooldown timestamp.

        Args:
            thought: The current drift thought context (not directly used in the
                sweep logic, but part of the base class contract).

        Returns:
            ActionResult: ``action_name='RECONCILE'``. ``success=True`` on normal
                completion — including when no memories were found. ``success=False``
                when a required service is unavailable or cannot be imported.
                ``details`` keys: ``memories_sampled``, ``candidate_pairs``,
                ``uncertainties_created``, ``auto_resolved``.

        Raises:
            No exceptions are raised; all errors are caught and returned as a
            failed ``ActionResult``.
        """
        db = self._db_service
        if db is None:
            # Try to get shared DB service
            try:
                from services.database_service import get_shared_db_service
                db = get_shared_db_service()
            except Exception as e:
                logger.debug(f"{LOG_PREFIX} No DB service available: {e}")
                return ActionResult(action_name='RECONCILE', success=False,
                                    details={'reason': 'no_db'})

        try:
            from services.contradiction_classifier_service import ContradictionClassifierService
            from services.uncertainty_service import UncertaintyService
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Failed to import services: {e}")
            return ActionResult(action_name='RECONCILE', success=False,
                                details={'reason': str(e)})

        classifier = ContradictionClassifierService(db_service=db)
        uncertainty_svc = UncertaintyService(db)

        # Sample memories
        memories = classifier.sample_memories_for_reconcile(
            n_traits=self.n_traits,
            n_concepts=self.n_concepts,
        )
        if not memories:
            self._record_run()
            return ActionResult(action_name='RECONCILE', success=True,
                                details={'reason': 'no_memories'})

        # Cross-store comparison
        results = classifier.reconcile_memory_batch(memories)

        created_count = 0
        auto_resolved_count = 0

        for result in results:
            if created_count >= _MAX_UNCERTAINTY_CREATES:
                break

            mem_a = result['memory_a']
            mem_b = result['memory_b']
            id_a = mem_a.get('id')
            id_b = mem_b.get('id')

            # Skip if already tracked
            if id_a and id_b and classifier.pair_already_tracked(id_a, id_b):
                continue

            classification = result['classification']

            if classification == 'temporal_change' and result.get('temporal_signal'):
                # Auto-resolve: supersede whichever memory appears older/weaker
                self._auto_supersede(uncertainty_svc, mem_a, mem_b, result, db)
                auto_resolved_count += 1
            else:
                # Create uncertainty record
                type_a = mem_a['type'] if mem_a['type'] != 'incoming' else 'trait'
                type_b = mem_b['type'] if mem_b['type'] != 'incoming' else 'trait'
                uncertainty_svc.create_uncertainty(
                    memory_a_type=type_a,
                    memory_a_id=id_a or 'unknown',
                    memory_b_type=type_b,
                    memory_b_id=id_b,
                    uncertainty_type='contradiction',
                    detection_context='drift',
                    reasoning=result.get('reasoning'),
                    temporal_signal=result.get('temporal_signal', False),
                    surface_context=result.get('surface_context'),
                )
                created_count += 1

        self._record_run()
        logger.info(
            f"{LOG_PREFIX} Completed: {len(memories)} memories sampled, "
            f"{len(results)} candidate pairs, {created_count} uncertainties created, "
            f"{auto_resolved_count} auto-resolved"
        )

        return ActionResult(
            action_name='RECONCILE',
            success=True,
            details={
                'memories_sampled': len(memories),
                'candidate_pairs': len(results),
                'uncertainties_created': created_count,
                'auto_resolved': auto_resolved_count,
            }
        )

    def _record_run(self):
        """Update cooldown timestamp."""
        self.store.set(_COOLDOWN_KEY, str(time.time()))
        # TTL slightly longer than cooldown so the key doesn't expire mid-cycle
        self.store.expire(_COOLDOWN_KEY, int(self.cooldown_minutes * 70))

    def _auto_supersede(
        self,
        uncertainty_svc,
        mem_a: dict,
        mem_b: dict,
        result: dict,
        db,
    ):
        """
        Auto-resolve a temporal change by superseding the older/weaker memory.

        Creates a resolved uncertainty record for the audit trail.
        """
        # Determine winner (higher confidence) and loser
        conf_a = mem_a.get('meta', {}).get('confidence', 0.5)
        conf_b = mem_b.get('meta', {}).get('confidence', 0.5)
        reinf_a = mem_a.get('meta', {}).get('reinforcement_count', 1)
        reinf_b = mem_b.get('meta', {}).get('reinforcement_count', 1)

        # Score based on confidence * reinforcement
        score_a = conf_a * (1 + 0.1 * reinf_a)
        score_b = conf_b * (1 + 0.1 * reinf_b)

        if score_a >= score_b:
            winner, loser = mem_a, mem_b
        else:
            winner, loser = mem_b, mem_a

        # Create and immediately resolve uncertainty
        unc_id = uncertainty_svc.create_uncertainty(
            memory_a_type=winner['type'] if winner['type'] != 'incoming' else 'trait',
            memory_a_id=winner.get('id') or 'unknown',
            memory_b_type=loser['type'] if loser['type'] != 'incoming' else 'trait',
            memory_b_id=loser.get('id'),
            uncertainty_type='contradiction',
            detection_context='drift',
            reasoning=result.get('reasoning'),
            temporal_signal=True,
        )

        uncertainty_svc.resolve_uncertainty(
            uncertainty_id=unc_id,
            strategy='temporal_supersede',
            detail=f"Auto-resolved during drift reconcile: {result.get('reasoning', '')}",
            winner_type=winner['type'] if winner['type'] != 'incoming' else 'trait',
            winner_id=winner.get('id'),
            loser_type=loser['type'] if loser['type'] != 'incoming' else 'trait',
            loser_id=loser.get('id'),
        )

        logger.info(
            f"{LOG_PREFIX} Auto-superseded: winner={winner.get('id')} "
            f"loser={loser.get('id')} reasoning={result.get('reasoning', '')[:80]}"
        )
