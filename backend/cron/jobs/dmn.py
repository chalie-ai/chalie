"""DMN cron job — idle-gated background reflection.

Ported verbatim from ``services.subconscious_worker.SubconsciousWorker._step_dmn``
so the step stays self-contained when the source worker is deleted. The
idle-window and min-interval gates are inherited from ``IdleGatedJob``.
"""

from __future__ import annotations

import logging

from cron.base import IdleGatedJob
from services.user_synthesis import UserSynthesis

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SUBCONSCIOUS]"


class DmnJob(IdleGatedJob):
    """Idle-gated background DMN reflection via DMNMessageProcessor.

    Self-heals via its "no user synthesis → skip" gate regardless of ordering
    relative to synthesis; the base ``IdleGatedJob`` applies a 30-min idle
    window and 5-min min-interval.
    """

    name = "dmn"

    def _run(self) -> str:
        """Step 6 — background DMN reflection via DMNMessageProcessor."""
        # DMN needs a user synthesis to reflect on; skip when none exists yet.
        if not UserSynthesis.get():
            logger.info(f"{LOG_PREFIX} Skipping DMN — no user synthesis available")
            return "skipped: no user synthesis"

        from configs.channels import DmnConfig  # noqa: PLC0415
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415
        MessageProcessor.process(DmnConfig())
        return "ok"
