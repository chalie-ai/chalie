"""DMN cron job — idle-gated background reflection.

Ported from ``services.subconscious_worker.SubconsciousWorker._step_dmn`` so the
step stays self-contained when the source worker is deleted. On top of the
idle-window and min-interval gates inherited from ``IdleGatedJob``, DMN adds a
*spine-advanced* gate: unlike the other cognition jobs — which are idempotent
against DB watermarks and simply no-op when there is no new work — DMN
manufactures its own inputs (each reflection becomes a memory the next
reflection reads). Left to fire every ``min_interval`` through one idle
stretch it would reflect on its own prior reflections, a self-feeding memory
loop. So it reflects at most once per idle session: only when a user message
has landed since the last reflection.
"""

from __future__ import annotations

import logging

from cron.base import IdleGatedJob
from services.time_utils import parse_utc
from services.user_synthesis import UserSynthesis

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SUBCONSCIOUS]"


class DmnJob(IdleGatedJob):
    """Idle-gated background DMN reflection via DMNMessageProcessor.

    ``should_run`` layers two DMN-specific preconditions onto the base
    idle/interval gates — a user synthesis must exist to reflect on, and the
    user spine must have advanced since the last reflection. Both live in the
    gate (never in ``_run``) so a skipped tick does not advance the base
    ``last_fired`` clock and thereby block the next legitimate reflection.
    """

    name = "dmn"

    def should_run(self) -> bool:
        return (
            super().should_run()
            and self._has_synthesis()
            and self._spine_advanced()
        )

    @staticmethod
    def _has_synthesis() -> bool:
        """DMN needs a user synthesis to reflect on; skip when none exists yet."""
        if UserSynthesis.get():
            return True
        logger.info(f"{LOG_PREFIX} Skipping DMN — no user synthesis available")
        return False

    def _spine_advanced(self) -> bool:
        """True only when a user message has landed since DMN last reflected.

        The base ``last_fired`` stamp doubles as "last reflected at" (DMN only
        ``execute``s when this gate passes, so nothing else advances it). Fire
        when never reflected; otherwise fire only if the newest user message is
        more recent than that stamp. No message, or a stamp newer than the last
        message, means we already reflected on this idle session — wait for the
        user to speak again.
        """
        from models.transcript import Transcript  # noqa: PLC0415

        last_fired = self._get_last_fired().load()
        if last_fired is None:
            return True

        last_msg_raw = Transcript.last_user_message_at()
        last_msg = parse_utc(last_msg_raw) if last_msg_raw else None
        return last_msg is not None and last_msg > last_fired

    def _run(self) -> str:
        """Step 6 — background DMN reflection via DMNMessageProcessor."""
        from configs.channels import DmnConfig  # noqa: PLC0415
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415
        MessageProcessor.process(DmnConfig())
        return "ok"
