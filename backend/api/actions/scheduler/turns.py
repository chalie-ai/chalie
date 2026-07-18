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
from exceptions import NotFoundError
from api.response.scheduler_turn import SchedulerTurn
from models.scheduled_item import ScheduledItem
from models.thread_gist import ThreadGist


class SchedulerTurns(Action):
    """Action listing prompt-schedule threads, one row per schedule (§13.5)."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"get"})
    response_dto = {"get": DocumentedResponse(SchedulerTurn, listing=True)}

    def slug(self) -> str:
        return "scheduler"

    def verb(self) -> str:
        return "turns"

    def get(self, id: int | str) -> ResponseReturnValue:
        """List prompt-schedule threads, one row per schedule (§13.5).

        Every live schedule is its own thread — ``turn_id`` is simply the
        schedule's own ``id`` (no series ``group_id`` to collapse occurrences
        into). The gist is sourced from ``thread_gist`` (keyed by
        channel=schedule, turn_id) — read separately and merged here, each
        model owning its own table's SQL rather than a cross-table JOIN.
        """
        if not self.is_create(id):
            raise NotFoundError("Not found")
        schedules = ScheduledItem.recent()
        gists = ThreadGist.bulk_get(ScheduledItem.SCHEDULE_CHANNEL, [cast(int, s.id) for s in schedules])
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
