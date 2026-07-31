"""The decay/GC entry point invoked off-turn by the subconscious worker.

A thin, mp-less wrapper that wires a :class:`~orchestrators.decay_engine.DecayEngine`
once — registering the domain-service Decayables — and runs it each cycle. The
engine owns the sequencing, per-unit isolation, and the cross-kind ``data_graph``
maintenance; this service owns only the registration list and the cycle-complete
log. New per-vertical decayables plug in on the ``register()`` chain as they land.
"""

from __future__ import annotations

import logging

from orchestrators import DecayEngine
from services.discovery_service import DiscoveryService
from services.episodic_service import EpisodicService
from services.fact_service import FactService
from services.misc_service import MiscService

logger = logging.getLogger(__name__)


class DecayEngineService:
    """Off-turn decay cycle — constructs and runs the DecayEngine."""

    def __init__(self) -> None:
        self._engine = (
            DecayEngine()
            .register(EpisodicService())
            .register(FactService())
            .register(DiscoveryService())
            .register(MiscService())
            # tool_calls + transcript garbage collection are NOT registered
            # here: that sweep runs on its own hourly cron job
            # (cron.jobs.garbage_collection.GarbageCollectionJob via
            # services.garbage_collector_service.GarbageCollectorService), not
            # on this idle-gated decay cycle. Per-kind data_graph verticals
            # register here too (E1+).
        )
        logger.info("[DECAY ENGINE] Initialized")

    def run_once(self) -> None:
        """Subconscious-worker entry point (one decay cycle)."""
        self.run_decay_cycle()

    def run_decay_cycle(self) -> None:
        """Run one full decay cycle across every registered Decayable and the
        engine-owned cross-kind ``data_graph`` maintenance."""
        total = self._engine.run()
        logger.info("[DECAY ENGINE] Cycle complete: %d rows maintained", total)
