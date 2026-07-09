"""Synthesis cron job — idle-gated user synopsis refresh, ported verbatim from SubconsciousWorker._step_synthesis."""

from __future__ import annotations

import logging

from cron.base import IdleGatedJob
from services.user_synthesis import UserSynthesis

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SUBCONSCIOUS]"


class SynthesisJob(IdleGatedJob):
    """Idle-gated user synopsis refresh.

    Ported verbatim from ``services.subconscious_worker.SubconsciousWorker._step_synthesis``.
    The idle-window and min-interval gates are inherited from ``IdleGatedJob``
    (30 min idle window, 5 min min-interval).
    """

    name = "synthesis"

    def _run(self) -> str:
        """Run the user synopsis refresh. Ported verbatim from
        ``SubconsciousWorker._step_synthesis``."""
        from configs.channels import UserSummaryConfig
        from controllers.message_processor import MessageProcessor

        if not UserSynthesis.needs_refresh():
            logger.info(f"{LOG_PREFIX} No new traits since last synthesis; skipping")
            return "no new traits/patterns; skipped"

        config = UserSummaryConfig()
        MessageProcessor.process(config)
        return "ok"
