"""Scheduler turns action — nested resource under /api/scheduler/turns.

Covers:
- GET /api/scheduler/turns  → get (prompt-schedule threads, one row per schedule)

Only the sentinel/id-less form is meaningful — an id-addressed call 404s,
matching the discoverable-tools precedent (api/actions/mcp_clients/discoverable.py).
"""

from __future__ import annotations

import logging
import threading
from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from exceptions import NotFoundError
from api.response.scheduler_turn import SchedulerTurn
from models.scheduled_item import ScheduledItem

logger = logging.getLogger(__name__)

# How many labels one read may generate. Each is an LLM call, and the first read
# after a fresh install finds every schedule labelless at once — uncapped that is
# a schedule-count-wide burst at a single local model. The dock re-reads every
# 60s, so a small cap drip-fills the backlog instead of stampeding it.
_GISTS_PER_READ = 3

# Schedules with a generation running right now. The cap above bounds one read;
# this bounds them across reads, tabs and dock re-opens — a generation slower
# than the 60s poll would otherwise be re-fired by every poll that sees the
# column still empty, and live LLM calls would grow without bound.
_in_flight: set[int] = set()
_in_flight_lock = threading.Lock()


def _generate_gist(turn_id: int, message: str) -> None:
    """Label one schedule from its own prompt and store it on its own row
    (daemon-thread body).

    A delegate that answered with nothing usable settles the row on the prompt
    itself — what the dock renders anyway — so absence stays a *fresh* signal:
    an unlabellable schedule can never be re-fired every poll forever, nor
    starve the older ones behind it under the per-read cap. A delegate that
    never answered leaves the column empty and is retried, the one case where a
    retry is right.

    The write is conditioned on the prompt still being the one that was
    labelled, so an edit landing mid-flight (which clears the column to ask for
    a fresh label) is never overwritten with a label describing the text it
    replaced."""
    try:
        from services.thread_gist_message_processor import generate_gist
        label = generate_gist(ScheduledItem.SCHEDULE_CHANNEL, turn_id, message)
        if label is not None:
            ScheduledItem.filter("id", turn_id).filter("message", message).update(
                gist=label or message
            )
    except Exception as exc:
        logger.warning(
            "[SCHEDULER TURNS] gist generation failed for %s turn=%s: %s",
            ScheduledItem.SCHEDULE_CHANNEL, turn_id, exc,
        )
    finally:
        with _in_flight_lock:
            _in_flight.discard(turn_id)


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
        into), and it carries its own label in its own ``gist`` column, read
        with the row.

        This read is also where labels are born: the ones missing are exactly
        the ones this read just found empty, so up to ``_GISTS_PER_READ`` of
        them are generated in the background. Schedules therefore self-heal on
        listing — no backfill job — and an edit that clears a stale label gets a
        fresh one on the next poll.
        """
        if not self.is_create(id):
            raise NotFoundError("Not found")
        schedules = ScheduledItem.recent()
        with _in_flight_lock:
            pending = [
                s for s in schedules
                if not s.gist and cast(int, s.id) not in _in_flight
            ][:_GISTS_PER_READ]
            _in_flight.update(cast(int, s.id) for s in pending)
        for s in pending:
            threading.Thread(
                target=_generate_gist, args=(cast(int, s.id), s.message),
                daemon=True, name="schedule-gist",
            ).start()
        items = [
            SchedulerTurn(
                turn_id=cast(int, s.id),
                gist=s.gist,
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
