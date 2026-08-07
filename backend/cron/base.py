"""ScheduledJob base contract, cron utilities, and the idle-gated cognition job.

The cron package unifies two hand-rolled crontabs (``scheduler_service`` and
``subconscious_worker``) into one minute-aligned runner over ``ScheduledJob``
classes. Concrete jobs subclass ``ScheduledJob`` (or ``IdleGatedJob`` for the
idle-gated cognition loop) and register themselves in ``cron.JOBS``.

``CronBase`` provides shared cron utilities (e.g. the LLM-provider precondition
gate). ``ScheduledJobProtocol`` is the duck-type the runner uses to iterate the
registry without pulling in concrete classes at import time.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from services.cron_schedule import CronSchedule
from services.durable_timestamp import DurableTimestamp
from services.time_utils import parse_utc, utc_now

logger = logging.getLogger(__name__)


class CronBase:
    """Base utilities for the cron system."""

    @staticmethod
    def llm_provider_configured() -> bool:
        """True when at least one LLM provider is available for CHAT turns.

        Mirrors the selection fallback chain the CHAT path uses in
        :meth:`provider_service.ProviderService._select`: ``get_selected_provider``
        first, then ``get_providers``. A fresh install has neither set for the
        first seconds/minutes of life, so this is the shared precondition for any
        cron job that drives an LLM turn — gate on this in ``should_run`` before
        the job's own work starts, so a skipped tick never stamps ``last_fired``.
        """
        from services.provider_cache_service import ProviderCacheService  # noqa: PLC0415

        return (
            bool(ProviderCacheService.get_selected_provider())
            or bool(ProviderCacheService.get_providers())
        )


@runtime_checkable
class ScheduledJobProtocol(Protocol):
    """The duck-type the runner uses to iterate the job registry.

    Declares exactly two methods: ``should_run`` (the cron gate) and
    ``execute`` (the work). Concrete jobs implement both; the runner never
    instantiates this protocol directly.
    """

    def should_run(self) -> bool: ...

    def execute(self) -> None: ...


class ScheduledJob:
    """Concrete base for every cron job. Never registered directly.

    Subclasses set ``name`` and the five cron fields (``minute`` / ``hour`` /
    ``dom`` / ``month`` / ``dow`` — ``"*"`` means "every"). The runner acquires
    ``_lock`` non-reentrantly so a slow job can never stall the minute tick.
    """

    name: str = ""
    minute: str = "*"
    hour: str = "*"
    dom: str = "*"
    month: str = "*"
    dow: str = "*"

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def should_run(self) -> bool:
        """True when the five cron fields match the current local minute."""
        return CronSchedule.matches(
            utc_now(), self.minute, self.hour, self.dom, self.month, self.dow
        )

    def execute(self) -> None:
        """Run the job's work and log its status detail.

        Template method: every job implements ``_run`` (the work, returning a
        short status string) — never ``execute``. The base owns the uniform
        ``[CRON]`` logging so all jobs report the same way; ``IdleGatedJob``
        extends this to also stamp its durable last-fired clock.
        """
        logger.info("[CRON] %s: %s", self.name, self._run())

    def _run(self) -> str:
        """Subclass hook — the actual work; returns a short status detail."""
        raise NotImplementedError


class IdleGatedJob(ScheduledJob):
    """A minute-aligned job that self-throttles via idle-window + min-interval.

    Subclasses implement ``_run`` (the actual work). The base provides
    ``execute`` which wraps ``_run`` and, in a ``finally`` block, stamps the
    durable last-fired timestamp — so a failing job still respects its
    interval. The cron fields stay at their ``"*"`` defaults (fire every
    minute); the gates live in ``should_run``.
    """

    idle_window: timedelta = timedelta(minutes=30)
    min_interval: timedelta = timedelta(minutes=5)

    def __init__(self) -> None:
        super().__init__()
        self._last_fired: DurableTimestamp | None = None

    def _get_last_fired(self) -> DurableTimestamp:
        if self._last_fired is None:
            self._last_fired = DurableTimestamp(
                memory_key=f"cron:{self.name}:last_fired",
                data_graph_key=f"cron_{self.name}_last_fired",
                source="cron_runner",
            )
        return self._last_fired

    def should_run(self) -> bool:
        return (
            super().should_run()
            and self._is_idle()
            and self._interval_elapsed()
        )

    def execute(self) -> None:
        """Run the work, log it, then stamp last_fired even on failure.

        Overrides ``ScheduledJob.execute`` to add the durable ``last_fired``
        stamp in a ``finally`` — so a raising ``_run`` still advances the
        interval gate. Subclasses implement ``_run`` (inherited hook), never
        this.
        """
        try:
            detail = self._run()
            logger.info("[CRON] %s: %s", self.name, detail)
        finally:
            self._persist_fired(utc_now())

    def _is_idle(self) -> bool:
        """True when the user has been idle at least ``idle_window``."""
        from models.transcript import Transcript  # noqa: PLC0415

        last_msg_raw = Transcript.last_user_message_at()
        last_msg = parse_utc(last_msg_raw) if last_msg_raw else None
        now = utc_now()

        if last_msg is not None and now - last_msg < self.idle_window:
            return False
        return True

    def _interval_elapsed(self) -> bool:
        """True when ``min_interval`` has passed since the last durable stamp."""
        now = utc_now()
        last = self._get_last_fired().load()
        if last is None:
            return True
        return now - last >= self.min_interval

    def _persist_fired(self, when: datetime) -> None:
        self._get_last_fired().persist(when)
