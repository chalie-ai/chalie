"""Scheduler turns action — nested resource under /api/scheduler/turns.

Covers:
- GET /api/scheduler/turns  → get (prompt-schedule threads, one row per schedule)

Only the sentinel/id-less form is meaningful — an id-addressed call 404s,
matching the discoverable-tools precedent (api/actions/mcp_clients/discoverable.py).
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.response.scheduler_turn import SchedulerTurn
from models.scheduled_item import ScheduledItem
from models.thread import Thread


class SchedulerTurns(Action):
    """Action listing prompt-schedule threads, one row per schedule (§13.5)."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"get"})
    response_dto = {"get": DocumentedResponse(SchedulerTurn, listing=True)}

    def get(self, id: int | str) -> ResponseReturnValue:
        """List prompt-schedule threads, one row per schedule (§13.5).

        Every live schedule is its own thread — ``turn_id`` is simply the
        schedule's own ``id`` (no series ``group_id`` to collapse occurrences
        into). The gist is the schedule thread's own label — read separately
        and merged here.
        """
        self.require_create_sentinel(id)
        schedules = ScheduledItem.recent()
        gists = Thread.summaries(ScheduledItem.SCHEDULE_CHANNEL, [cast(int, s.id) for s in schedules])
        items = [
            SchedulerTurn(
                turn_id=cast(int, s.id),
                gist=gists.get(cast(int, s.id)),
                preview=s.message,
                minute=s.cron_minute,
                hour=s.cron_hour,
                day=s.cron_dom,
                month=s.cron_month,
                weekday=s.cron_dow,
            )
            for s in schedules
        ]
        return SchedulerTurn.listing(items, page=1, limit=max(len(items), 1), total=len(items))
