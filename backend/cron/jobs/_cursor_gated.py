"""Shared base for cursor-driven, delta-gated idle cognition cron jobs.

One place for the read-cursor -> compute-delta -> fire-pass -> advance-cursor
body that PatternMatchJob and GeoPatternsJob both run. Subclasses declare
``cursor_key`` / ``min_delta`` and implement ``_latest_id`` / ``_make_config``.
Not registered directly (underscore-prefixed base).
"""

from __future__ import annotations

import logging

from cron.base import IdleGatedJob

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SUBCONSCIOUS]"


class CursorGatedJob(IdleGatedJob):
    cursor_key: str = ""
    min_delta: int = 0

    def _latest_id(self) -> int:
        """Newest transcript id in this job's window. Subclass hook."""
        raise NotImplementedError

    def _make_config(self, window_start: int, window_end: int):
        """Build the ProcessorConfig for this job's pass. Subclass hook."""
        raise NotImplementedError

    def _run(self) -> str:
        from models.machine_state import MachineStateRow  # noqa: PLC0415
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415

        # 1. Read cursor — newest active row wins. NOTE: reads are deliberately
        # NOT wrapped in try/except. If the DB is not yet ready (cold start) or
        # any error occurs, it must propagate — the cron runner logs it loudly
        # via logger.exception and keeps ticking. Never swallow it silently.
        cursor = 0
        row = MachineStateRow.newest_active_by_key(self.cursor_key)
        if row and row.value:
            try:
                cursor = int(row.value)
            except (TypeError, ValueError):
                cursor = 0

        # 2. Latest transcript id in this job's window (subclass-defined).
        latest = self._latest_id()

        delta = latest - cursor
        if delta < self.min_delta:
            logger.info(
                f"{LOG_PREFIX} {self.name}_skip cursor={cursor} "
                f"latest={latest} delta={delta}"
            )
            return f"skip cursor={cursor} latest={latest} delta={delta}"

        # 3. Fire the pass via the canonical entry point.
        MessageProcessor.process(self._make_config(cursor, latest))

        # 4. Advance cursor on success. Catch to add context, then re-raise so
        # the failure stays loud (a stuck cursor re-fires the same window next
        # tick — observable in operator logs).
        try:
            MachineStateRow.store(
                key=self.cursor_key,
                value=str(latest),
                source="subconscious_worker",
            )
        except Exception as exc:
            logger.warning(
                f"{LOG_PREFIX} {self.name} cursor write failed "
                f"cursor={cursor}->{latest} — next tick will re-fire same "
                f"window: {exc}"
            )
            raise
        return f"fired cursor={cursor}->{latest} delta={delta}"
