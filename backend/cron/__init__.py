"""Cron package — minute-aligned runner over ScheduledJob classes.

The runner iterates ``JOBS`` every wall-clock minute. Each job declares its own
cron schedule via ``ScheduledJob.minute/hour/dom/month/dow`` (5-field crontab,
all defaulting to ``"*"``) and its own work via ``ScheduledJob.execute``.
Idle-gated cognition jobs subclass ``IdleGatedJob`` which adds user-idle-window
and min-interval gates on top of the cron match.

``JOBS`` holds one singleton instance per registered job (see below); the runner
fires each independently as its cron matches.
"""

from cron.base import IdleGatedJob, ScheduledJob, ScheduledJobProtocol
from cron.jobs.capability_sync import CapabilitySyncJob
from cron.jobs.discovery import DiscoveryJob
from cron.jobs.garbage_collection import GarbageCollectionJob
from cron.jobs.memory_hygiene import MemoryHygieneJob
from cron.jobs.scheduled_items import ScheduledItemsDispatcherJob

# Concrete singleton instances (all subclass ScheduledJob, which satisfies
# ScheduledJobProtocol and carries the runner's infra: ``name`` and ``_lock``).
# Built once at import so per-job cached state (last_fired)
# survives across ticks. Order is cosmetic — every job self-gates via
# ``should_run`` and shares no in-memory state, so the runner fires them
# independently regardless of position.
JOBS: tuple[ScheduledJob, ...] = (
    ScheduledItemsDispatcherJob(),
    CapabilitySyncJob(),
    DiscoveryJob(),
    GarbageCollectionJob(),
    MemoryHygieneJob(),
)

__all__ = [
    "JOBS",
    "IdleGatedJob",
    "ScheduledJob",
    "ScheduledJobProtocol",
]
