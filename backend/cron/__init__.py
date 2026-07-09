"""Cron package — minute-aligned runner over ScheduledJob classes.

The runner iterates ``JOBS`` every wall-clock minute. Each job declares its
own cron schedule via ``ScheduledJob.dom/hour/minute`` and its own work via
``ScheduledJob.execute``. Idle-gated cognition jobs subclass ``IdleGatedJob``
which adds user-idle-window and min-interval gates on top of the cron match.

``JOBS`` is intentionally empty in this skeleton — later tasks populate it
with singleton job instances. An empty tuple means the runner does nothing.
"""

from cron.base import IdleGatedJob, ScheduledJob, ScheduledJobProtocol
from cron.jobs.capability_sync import CapabilitySyncJob
from cron.jobs.consolidate import ConsolidateJob
from cron.jobs.decay import DecayJob
from cron.jobs.discovery import DiscoveryJob
from cron.jobs.dmn import DmnJob
from cron.jobs.fact_extraction import FactExtractionJob
from cron.jobs.geo_patterns import GeoPatternsJob
from cron.jobs.pattern_match import PatternMatchJob
from cron.jobs.scheduled_items import ScheduledItemsDispatcherJob
from cron.jobs.synthesis import SynthesisJob

# Concrete singleton instances (all subclass ScheduledJob, which satisfies
# ScheduledJobProtocol and carries the runner's infra: ``name`` and ``_lock``).
# Built once at import so per-job cached state (decay engine, last_fired)
# survives across ticks. Order is cosmetic — every job self-gates via
# ``should_run`` and shares no in-memory state, so the runner fires them
# independently regardless of position.
JOBS: tuple[ScheduledJob, ...] = (
    ScheduledItemsDispatcherJob(),
    ConsolidateJob(),
    FactExtractionJob(),
    DecayJob(),
    PatternMatchJob(),
    SynthesisJob(),
    DmnJob(),
    CapabilitySyncJob(),
    GeoPatternsJob(),
    DiscoveryJob(),
)

__all__ = [
    "JOBS",
    "IdleGatedJob",
    "ScheduledJob",
    "ScheduledJobProtocol",
]
