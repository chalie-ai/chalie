"""Geo-patterns cron job — idle-gated geo-spatial pattern extractor, ported verbatim from SubconsciousWorker._step_geo_patterns."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cron.jobs._cursor_gated import CursorGatedJob

if TYPE_CHECKING:
    from configs.channels import GeoConfig


class GeoPatternsJob(CursorGatedJob):
    """Idle-gated geo-spatial pattern extractor.

    Ported verbatim from ``services.subconscious_worker.SubconsciousWorker._step_geo_patterns``.
    The idle-window and min-interval gates are inherited from ``IdleGatedJob``
    (30 min idle window, 5 min min-interval).
    """

    name = "geo_patterns"
    cursor_key = "geo_pattern_cursor"
    min_delta = 30

    def _latest_id(self) -> int:
        from models.transcript import Transcript  # noqa: PLC0415
        from configs.enums.channels import Channel  # noqa: PLC0415

        # Only located user geo-activity rows advance the cursor.
        return Transcript.latest_id([Channel.USER.value], require_location=True) or 0

    def _make_config(self, window_start: int, window_end: int) -> GeoConfig:
        from configs.channels import GeoConfig  # noqa: PLC0415

        return GeoConfig(window_start, window_end)
