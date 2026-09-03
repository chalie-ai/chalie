"""Daily memory-consolidation cron job.

Fires ``MemoryHygieneService.tick`` once per day, gated by the service's
own per-minute ``due()`` check rather than a fixed ``hour`` field. A
boundary missed while the app was down fires at the next tick after boot
(catch-up), and the service's pending-window math is the single source of
"due". The service carries an in-memory ``_empty_until`` short-circuit so
consecutive no-work ticks are cheap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cron.base import ScheduledJob, llm_provider_configured

if TYPE_CHECKING:
    from services.memory_hygiene_service import MemoryHygieneService


class MemoryHygieneJob(ScheduledJob):
    """Minute-aligned cron job for daily memory consolidation.

    Uses ``MemoryHygieneService.due()`` as the per-minute gate instead of a
    fixed hour, so a missed boundary while the app was down fires at the
    next tick after boot. The service instance is cached on the job so its
    in-memory ``_empty_until`` short-circuit survives across ticks.
    """

    name = "memory_hygiene"

    def __init__(self) -> None:
        super().__init__()
        self._service: MemoryHygieneService | None = None

    def _get_service(self) -> "MemoryHygieneService":
        """Lazily create and cache the service instance."""
        from services.memory_hygiene_service import MemoryHygieneService  # noqa: PLC0415

        if self._service is None:
            self._service = MemoryHygieneService()
        return self._service

    def should_run(self) -> bool:
        """Base cron match, plus LLM provider and the service's due() gate.

        When ``llm_provider_configured`` is False the job returns False
        silently — the job evaluates every minute and would spam if it
        logged on every skip. The service's ``due()`` is the single source
        of truth for whether consolidation work is pending.
        """
        if not super().should_run():
            return False
        if not llm_provider_configured():
            return False
        return self._get_service().due()

    def _run(self) -> str:
        """Run one consolidation tick and return its status string."""
        return self._get_service().tick()
