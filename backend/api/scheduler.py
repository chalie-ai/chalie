"""Scheduler namespace — CRUD over scheduled_items via Pydantic DTOs.

All routes require session auth. Persistence is owned by the
:class:`~models.scheduled_item.ScheduledItem` model (the sole home of
``scheduled_items`` + ``scheduled_items_vec`` SQL); the ``/turns`` gist column is
read through :class:`~models.thread_gist.ThreadGist`. The only service touchpoint
is the fire-and-forget ``embed_scheduled_item`` thread on create. DTO-typed
through the foundation boundary decorators (``@expects``/``@responds``),
following the lists reference.

``scheduled_items`` is a prompt-only, dumb-cron table: one row per schedule,
forever. There is no ``item_type``/``due_at``/``status``/``group_id`` — a
stateless poller (``services.scheduler_service``) fires any enabled row whose
``start_at`` floor has passed and whose cron fields match the current
wall-clock minute. ``id`` is the SQLite auto-increment PK and doubles as the
schedule's ``turn_id`` on the ``'schedule'`` channel. Delete is a hard
``DELETE`` (no soft-cancel state to set).
"""

from __future__ import annotations

import logging
import threading
from typing import cast

from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from models.scheduled_item import ScheduledItem
from models.thread_gist import ThreadGist
from services.database import Database
from services.locale_service import parse_local
from services.time_utils import utc_now

from .auth import require_session
from .dto import Error, expects, register_dto, responds
from .dto.boundary import error
from .dto.scheduler_item import (
    SchedulerItem,
    SchedulerItemCreate,
    SchedulerItemUpdate,
    SchedulerTurn,
)
from .dto.scheduler_query import SchedulerListQuery

logger = logging.getLogger(__name__)

_ERR_INTERNAL = "Internal server error"
_ERR_NOT_FOUND = "Not found"

scheduler_ns = Namespace("scheduler", description="Scheduled item management", path="/api/scheduler")

register_dto(
    scheduler_ns,
    SchedulerItem,
    SchedulerItemCreate,
    SchedulerItemUpdate,
    SchedulerTurn,
    SchedulerListQuery,
    Error,
)

_S = scheduler_ns.models


def _item_dto(item: ScheduledItem) -> SchedulerItem:
    """Project a :class:`~models.scheduled_item.ScheduledItem` into the
    :class:`SchedulerItem` read DTO.

    Renames the model's ``cron_dom``/``cron_hour``/``cron_minute`` fields to the
    read DTO's ``day``/``hour``/``minute`` names.
    """
    return SchedulerItem.model_validate({
        "id": item.id,
        "message": item.message,
        "start_at": item.start_at,
        "day": item.cron_dom,
        "hour": item.cron_hour,
        "minute": item.cron_minute,
        "enabled": item.enabled,
        "channel": item.channel,
        "created_by_session": item.created_by_session,
        "created_at": item.created_at,
    })


def _embed(item_id: int, message: str) -> None:
    """Fire-and-forget embedding of a freshly created item (non-fatal on failure)."""
    try:
        from services.scheduler_service import embed_scheduled_item
        embed_scheduled_item(item_id, message)
    except Exception as exc:
        logger.warning("[SCHEDULER API] Embedding failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Collection + turns
# ---------------------------------------------------------------------------

@scheduler_ns.route("")
class SchedulerListResource(Resource):
    @require_session
    @scheduler_ns.response(200, "Scheduled items", model=_S["SchedulerItem"])
    @scheduler_ns.response(422, "Validation failed", model=_S["Error"])
    @scheduler_ns.response(500, _ERR_INTERNAL, model=_S["Error"])
    @responds(SchedulerItem, code=200)
    @expects(SchedulerListQuery, source="args")
    def get(self, dto: SchedulerListQuery) -> list[SchedulerItem] | ResponseReturnValue:
        """List the live schedules, newest first.

        The table is prompt-only now — every row already is one live schedule
        (no ``group_id`` series or ``status`` history to collapse), labelled
        active/disabled by ``enabled``.
        """
        try:
            items = ScheduledItem.recent(limit=dto.limit, offset=dto.offset)
            return [_item_dto(item) for item in items]
        except Exception:
            logger.exception("[SCHEDULER API] list error")
            return error(_ERR_INTERNAL, 500)

    @require_session
    @scheduler_ns.expect(_S["SchedulerItemCreate"])
    @scheduler_ns.response(201, "Item created", model=_S["SchedulerItem"])
    @scheduler_ns.response(422, "Validation failed", model=_S["Error"])
    @scheduler_ns.response(500, _ERR_INTERNAL, model=_S["Error"])
    @responds(SchedulerItem, code=201)
    @expects(SchedulerItemCreate)
    def post(self, dto: SchedulerItemCreate) -> SchedulerItem | ResponseReturnValue:
        """Create a new scheduled item and embed it asynchronously."""
        try:
            now = utc_now()
            start_at_utc = parse_local(dto.start_at) if dto.start_at else now
            item = ScheduledItem.create(
                message=dto.message,
                start_at=start_at_utc.isoformat(),
                cron_dom=dto.day,
                cron_hour=dto.hour,
                cron_minute=dto.minute,
                enabled=1 if dto.enabled else 0,
                channel=dto.channel,
                created_by_session=None,
            )  # INSERT + 1-1 gist seed, atomic; id autoincrements, created_at defaults in SQL
            item_id = cast(int, item.id)
            # Re-read so created_at (SQL default) is populated for the response.
            saved = ScheduledItem.get(item_id)

            threading.Thread(
                target=_embed, args=(item_id, dto.message),
                daemon=True, name="scheduler-embed",
            ).start()
            return _item_dto(cast(ScheduledItem, saved))
        except Exception:
            logger.exception("[SCHEDULER API] create error")
            return error(_ERR_INTERNAL, 500)


@scheduler_ns.route("/turns")
class SchedulerTurnsResource(Resource):
    @require_session
    @scheduler_ns.response(200, "Active schedule threads", model=_S["SchedulerTurn"])
    @scheduler_ns.response(500, _ERR_INTERNAL, model=_S["Error"])
    @responds(SchedulerTurn, code=200)
    def get(self) -> list[SchedulerTurn] | ResponseReturnValue:
        """List prompt-schedule threads, one row per schedule (§13.5).

        Every live schedule is its own thread now — ``turn_id`` is simply the
        schedule's own ``id`` (no series ``group_id`` to collapse occurrences
        into). The gist is sourced from ``thread_gist`` (keyed by
        channel=schedule, turn_id) — read separately and merged here, each model
        owning its own table's SQL rather than a cross-table JOIN.
        """
        try:
            schedules = ScheduledItem.recent()
            gists = ThreadGist.bulk_get(ScheduledItem.SCHEDULE_CHANNEL, [cast(int, s.id) for s in schedules])
            return [
                SchedulerTurn.model_validate({
                    "turn_id": s.id,
                    "gist": gists.get(cast(int, s.id)),
                    "preview": s.message,
                    "day": s.cron_dom,
                    "hour": s.cron_hour,
                    "minute": s.cron_minute,
                })
                for s in schedules
            ]
        except Exception:
            logger.exception("[SCHEDULER API] turns error")
            return error(_ERR_INTERNAL, 500)


# ---------------------------------------------------------------------------
# Item resource (id-addressed CRUD)
# ---------------------------------------------------------------------------

@scheduler_ns.route("/<int:item_id>")
@scheduler_ns.param("item_id", "The item identifier")
class SchedulerItemResource(Resource):
    @require_session
    @scheduler_ns.response(200, "The item", model=_S["SchedulerItem"])
    @scheduler_ns.response(404, _ERR_NOT_FOUND, model=_S["Error"])
    @scheduler_ns.response(500, _ERR_INTERNAL, model=_S["Error"])
    @responds(SchedulerItem, code=200)
    def get(self, item_id: int) -> SchedulerItem | ResponseReturnValue:
        """Fetch a single scheduled item by ID."""
        try:
            item = ScheduledItem.get(item_id)
            if item is None:
                return error(_ERR_NOT_FOUND, 404)
            return _item_dto(item)
        except Exception:
            logger.exception("[SCHEDULER API] get item error")
            return error(_ERR_INTERNAL, 500)

    @require_session
    @scheduler_ns.expect(_S["SchedulerItemUpdate"])
    @scheduler_ns.response(200, "Updated item", model=_S["SchedulerItem"])
    @scheduler_ns.response(404, _ERR_NOT_FOUND, model=_S["Error"])
    @scheduler_ns.response(422, "Validation failed", model=_S["Error"])
    @scheduler_ns.response(500, _ERR_INTERNAL, model=_S["Error"])
    @responds(SchedulerItem, code=200)
    @expects(SchedulerItemUpdate)
    def put(self, item_id: int, dto: SchedulerItemUpdate) -> SchedulerItem | ResponseReturnValue:
        """Replace a scheduled item's message/timing/enabled flag.

        There is no ``due_at`` to recompute any more: re-enabling a schedule
        via update simply resumes matching against the current wall-clock
        minute on its stored (or newly supplied) ``start_at`` floor — the
        poller derives everything else at fire time.
        """
        try:
            item = ScheduledItem.get(item_id)
            if item is None:
                return error(_ERR_NOT_FOUND, 404)
            # Anchor: the caller's new start_at when supplied, else the row's
            # existing one (a plain enabled-toggle omits it) — a future-dated
            # start_at is never re-floored to now.
            item.start_at = parse_local(dto.start_at).isoformat() if dto.start_at else item.start_at
            item.message = dto.message
            item.cron_dom = dto.day
            item.cron_hour = dto.hour
            item.cron_minute = dto.minute
            item.enabled = 1 if dto.enabled else 0
            item.save()  # UPDATE (id set); channel/created_by_session/created_at unchanged
            return _item_dto(item)
        except Exception as exc:
            logger.error("[SCHEDULER API] update error: %s", exc)
            return error(_ERR_INTERNAL, 500)

    @require_session
    @scheduler_ns.response(204, "Item cancelled")
    @scheduler_ns.response(404, _ERR_NOT_FOUND, model=_S["Error"])
    @scheduler_ns.response(500, _ERR_INTERNAL, model=_S["Error"])
    @responds(code=204)
    def delete(self, item_id: int) -> None | ResponseReturnValue:
        """Cancel a scheduled item — a hard delete (no soft-cancel state)."""
        try:
            item = ScheduledItem.get(item_id)
            if item is None:
                return error(_ERR_NOT_FOUND, 404)
            # Drop the main row and its vec embedding in one transaction so a
            # cancelled schedule never orphans its embedding (delete_embedding
            # rides in this transaction — rowid == id under INTEGER PRIMARY KEY).
            with Database.transaction():
                item.delete()
                ScheduledItem.delete_embedding(item_id)
            return None
        except Exception:
            logger.exception("[SCHEDULER API] cancel error")
            return error(_ERR_INTERNAL, 500)
