"""ScheduledItemsDispatcherJob — minute-aligned cron job for ``scheduled_items``.

Folds the entire ``scheduler_service._poll_and_fire`` loop (and its two
callees) into the new ``ScheduledJob`` contract. Self-contained: every
constant, import, and helper that the poll-and-fire path relies on is pulled
into this module so ``scheduler_service`` can be deleted without breaking it.

The poller wakes once per minute (``dom/hour/minute = None`` → base
``should_run()`` returns True every minute) and asks each enabled, already-
started row a stateless yes/no question via ``services.cron_schedule.matches``
— there is no materialized ``due_at`` to walk toward, no status machine, no
successor row to insert. A match fires the prompt through the chat chokepoint
for LLM execution with full tool access, keyed to the schedule's own thread:
the schedule's integer ``id`` IS the turn_id on the ``schedule`` channel
(``INTEGER PRIMARY KEY AUTOINCREMENT`` guarantees a cancelled schedule's id is
never reissued, so a dead thread can never be re-entered).
"""

from __future__ import annotations

import logging
import threading
from typing import cast

from cron.base import ScheduledJob
from models.scheduled_item import ScheduledItem
from services.cron_schedule import matches
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SCHEDULER]"

# Daemon-thread name prefix for the asynchronous fire. The poll loop must never
# block on the LLM work, so a fired prompt runs on its own thread named
# ``scheduled-work-<item_id>``.
_SCHEDULED_WORK_THREAD_PREFIX = "scheduled-work"


class ScheduledItemsDispatcherJob(ScheduledJob):
    """Minute-aligned cron job that polls ``scheduled_items`` and fires matches.

    Unlike :class:`~cron.base.IdleGatedJob`, this job is NOT idle-gated — it
    fires user schedules regardless of user activity. The cron triple stays
    ``None`` (inherited from :class:`ScheduledJob`), so ``should_run()`` returns
    True every minute; the minute-alignment sleep loop is owned by the cron
    runner, not by ``execute()``.
    """

    name = "scheduled_items"

    def _run(self) -> str:
        """Poll due ``scheduled_items`` and fire every row whose cron matches.

        Implements the base ``_run`` work hook (the cron runner calls
        ``execute`` → logs this detail). Not idle-gated: fires user schedules
        regardless of activity. A raising poll propagates to the runner, which
        logs it — no local swallow.
        """
        now = utc_now()

        # ScheduledItem.due_at is a LOCKLESS read (it must never take the write
        # lock, or a contended minute is skipped and a fixed-time schedule
        # missed until its next occurrence — see the model method's docstring).
        rows = ScheduledItem.due_at(now.isoformat())

        fired = 0
        checked = 0
        for row in rows:
            checked += 1
            if matches(
                now,
                cast("int | None", row["cron_dom"]),
                cast("int | None", row["cron_hour"]),
                cast("int | None", row["cron_minute"]),
            ):
                self._fire_item(cast(int, row["id"]), cast(str, row["message"]))
                fired += 1

        return f"fired {fired} of {checked} due row(s)"

    def _fire_item(self, item_id: int, message: str) -> None:
        """Fire a due prompt item via the LLM pipeline, on its own daemon thread.

        The poll loop must NOT execute the full LLM ACT loop inline — it fires
        asynchronously so the poller returns to sleep immediately.
        """
        if not message or not message.strip():
            logger.warning(f"{LOG_PREFIX} Skipping item '{item_id}' — empty message")
            return

        threading.Thread(
            target=self._fire_scheduled_prompt,
            args=(item_id, message),
            daemon=True,
            name=f"{_SCHEDULED_WORK_THREAD_PREFIX}-{item_id}",
        ).start()

    def _fire_scheduled_prompt(self, item_id: int, message: str) -> None:
        """Run a fired scheduled prompt on the ``schedule`` channel.

        turn_id = id: the schedule's integer id IS the turn_id on the ``schedule``
        channel. The first fire opens a MAIN turn (the id has never been used);
        every later fire supplies the same id and appends as a FORK — one
        recurring schedule is one growing thread. ``ScheduledConfig`` declares
        ``external_turn_id=True``, so the MP derives forked-ness from whether the
        turn already exists rather than rejecting the id as an invalid fork.

        Runs on its own daemon thread so the scheduler poll never blocks on the
        LLM loop; this is fire-and-forget — no result() join, no turn_id
        backfill, no cancellation check.
        """
        from configs.channels import ScheduledConfig  # noqa: PLC0415
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415

        MessageProcessor.process(ScheduledConfig(), raw_input=message, turn_id=item_id)
        logger.info(f"{LOG_PREFIX} Fired scheduled prompt on turn {item_id}: {message[:80]}")
