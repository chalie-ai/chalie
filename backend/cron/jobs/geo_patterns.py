"""Geo-patterns cron job — idle-gated geo-spatial pattern extractor, ported verbatim from SubconsciousWorker._step_geo_patterns."""

from __future__ import annotations

import logging

from cron.base import IdleGatedJob

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SUBCONSCIOUS]"


class GeoPatternsJob(IdleGatedJob):
    """Idle-gated geo-spatial pattern extractor.

    Ported verbatim from ``services.subconscious_worker.SubconsciousWorker._step_geo_patterns``.
    The idle-window and min-interval gates are inherited from ``IdleGatedJob``
    (30 min idle window, 5 min min-interval).
    """

    name = "geo_patterns"

    def _run(self) -> str:
        """Single-pass LLM geo-spatial pattern extractor. Ported verbatim from
        ``SubconsciousWorker._step_geo_patterns``."""
        from models.machine_state import MachineStateRow as _MachineStateRow  # noqa: PLC0415
        from models.transcript import Transcript as _Transcript  # noqa: PLC0415
        from configs.enums.channels import Channel as _Channel  # noqa: PLC0415
        from configs.channels import GeoConfig as _GeoConfig  # noqa: PLC0415
        from controllers.message_processor import MessageProcessor as _MessageProcessor  # noqa: PLC0415

        _DG_KEY_CURSOR = "geo_pattern_cursor"
        _MIN_DELTA = 30

        cursor = 0
        try:
            row = _MachineStateRow.newest_active_by_key(_DG_KEY_CURSOR)
            if row and row.value:
                try:
                    cursor = int(row.value)
                except (TypeError, ValueError):
                    cursor = 0

            # Same allowlist as the geo-pattern window (geo_pattern.py): only
            # user geo-activity channels advance the cursor, so a located row on
            # a muted channel can never fire the geo pass.
            latest = _Transcript.latest_id([_Channel.USER.value], require_location=True) or 0
        except Exception as exc:
            logger.debug(f"{LOG_PREFIX} geo_patterns no db: {exc}")
            return "skip: no db"

        delta = latest - cursor
        if delta < _MIN_DELTA:
            logger.info(
                f"{LOG_PREFIX} geo_patterns_skip cursor={cursor} "
                f"latest={latest} delta={delta}"
            )
            return f"skip cursor={cursor} latest={latest} delta={delta}"

        # Fire the geo pass via the canonical entry point.
        _MessageProcessor.process(_GeoConfig(cursor, latest))

        try:
            _MachineStateRow.store(
                key=_DG_KEY_CURSOR,
                value=str(latest),
                source="subconscious_worker",
            )
        except Exception as exc:
            logger.warning(
                f"{LOG_PREFIX} geo_patterns cursor write failed "
                f"cursor={cursor}->{latest} — next tick will re-fire same "
                f"window: {exc}"
            )
            raise
        return f"fired cursor={cursor}->{latest} delta={delta}"
