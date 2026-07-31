"""GarbageCollectionJob — hourly cron job for the transcript + tool_calls
retention sweep.

Mirrors :class:`~cron.jobs.scheduled_items.ScheduledItemsDispatcherJob`: NOT
idle-gated — garbage collection runs on schedule regardless of user activity,
never on the idle-gated cognition path. Its cron fields other than ``minute``
stay at :class:`~cron.base.ScheduledJob`'s ``"*"`` default, so it fires once
at the top of every hour.
"""

from __future__ import annotations

import logging

from cron.base import ScheduledJob
from services.garbage_collector_service import GarbageCollectorService

logger = logging.getLogger(__name__)


class GarbageCollectionJob(ScheduledJob):
    """Hourly cron job: sweep unlinked transcript rows and their anchored
    tool_calls together, then separately reap tool_calls orphans that
    pre-date the sweep."""

    name = "garbage_collection"
    minute = "0"

    def _run(self) -> str:
        """Run one garbage-collection pass. Implements the base ``_run`` work
        hook (the cron runner calls ``execute`` → logs this detail). A
        raising sweep propagates to the runner, which logs it — no local
        swallow."""
        result = GarbageCollectorService().run()
        return (
            f"transcripts_deleted={result.transcripts_deleted} "
            f"tool_calls_swept={result.tool_calls_swept} "
            f"tool_calls_orphans_reaped={result.tool_calls_orphans_reaped}"
        )
