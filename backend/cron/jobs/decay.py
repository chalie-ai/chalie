"""Decay cron job — idle-gated decay cycle, ported verbatim from SubconsciousWorker._step_decay."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cron.base import IdleGatedJob

if TYPE_CHECKING:
    from services.decay_engine_service import DecayEngineService

logger = logging.getLogger(__name__)


class DecayJob(IdleGatedJob):
    """Idle-gated decay cycle.

    Ported from ``services.subconscious_worker.SubconsciousWorker._step_decay``.
    The idle-window and min-interval gates are inherited from ``IdleGatedJob``
    (30 min idle window, 5 min min-interval).
    """

    name = "decay"

    def __init__(self) -> None:
        super().__init__()
        # Single DecayEngineService instance — shared across ticks.
        # Lazy-built on first use so import failures surface as a step error.
        self._decay_engine: "DecayEngineService | None" = None

    def _run(self) -> str:
        """Run the unified decay cycle. Ported verbatim from
        ``SubconsciousWorker._step_decay``."""
        if self._decay_engine is None:
            from services.decay_engine_service import DecayEngineService
            self._decay_engine = DecayEngineService()
        self._decay_engine.run_once()
        return "ok"
