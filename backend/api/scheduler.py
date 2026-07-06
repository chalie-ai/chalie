"""Scheduler namespace — CRUD over scheduled_items via Pydantic DTOs.

All routes require session auth. The API owns its SQL (raw inline against the
``scheduled_items`` table); the only service touchpoint is the fire-and-forget
``embed_scheduled_item`` thread on create. DTO-typed through the foundation
boundary decorators (``@expects``/``@responds``), following the lists reference.

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
from typing import TYPE_CHECKING, cast

from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

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

if TYPE_CHECKING:
    from collections.abc import Iterable

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

_COLS = [
    "id", "message", "start_at",
    "cron_dom", "cron_hour", "cron_minute",
    "enabled", "channel", "created_by_session", "created_at",
]


def _item_dto(row: "Iterable[object]", cols: "list[str]") -> SchedulerItem:
    """Zip a select row into a :class:`SchedulerItem` read DTO.

    Renames the DB's ``cron_dom``/``cron_hour``/``cron_minute`` columns to the
    read DTO's ``day``/``hour``/``minute`` field names.
    """
    data = dict(zip(cols, row))
    data["day"] = data.pop("cron_dom", None)
    data["hour"] = data.pop("cron_hour", None)
    data["minute"] = data.pop("cron_minute", None)
    return SchedulerItem.model_validate(data)


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
            cols = ", ".join(_COLS)
            cursor = Database.conn().cursor()
            cursor.execute(
                f"SELECT {cols} FROM scheduled_items "
                "ORDER BY created_at DESC "
                "LIMIT ? OFFSET ?",
                [dto.limit, dto.offset],
            )
            rows = cursor.fetchall()
            return [_item_dto(r, _COLS) for r in rows]
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
            with Database.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO scheduled_items
                      (message, start_at, cron_dom, cron_hour, cron_minute,
                       enabled, channel, created_by_session)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dto.message, start_at_utc.isoformat(),
                        dto.day, dto.hour, dto.minute,
                        1 if dto.enabled else 0,
                        dto.channel, None,
                    ),
                )
                item_id = cursor.lastrowid
                cursor.execute(
                    f"SELECT {', '.join(_COLS)} FROM scheduled_items WHERE id = ?",
                    (item_id,),
                )
                row = cursor.fetchone()

            threading.Thread(
                target=_embed, args=(item_id, dto.message),
                daemon=True, name="scheduler-embed",
            ).start()
            return _item_dto(cast("Iterable[object]", row), _COLS)
        except Exception:
            logger.exception("[SCHEDULER API] create error")
            return error(_ERR_INTERNAL, 500)


@scheduler_ns.route("/turns")
class SchedulerTurnsResource(Resource):
    _TURN_COLS = ["turn_id", "gist", "preview", "day", "hour", "minute"]

    @require_session
    @scheduler_ns.response(200, "Active schedule threads", model=_S["SchedulerTurn"])
    @scheduler_ns.response(500, _ERR_INTERNAL, model=_S["Error"])
    @responds(SchedulerTurn, code=200)
    def get(self) -> list[SchedulerTurn] | ResponseReturnValue:
        """List prompt-schedule threads, one row per schedule (§13.5).

        Every live schedule is its own thread now — ``turn_id`` is simply the
        schedule's own ``id`` (no series ``group_id`` to collapse occurrences
        into). The gist is sourced from ``thread_gist`` (keyed by
        channel=schedule, turn_id).
        """
        try:
            rows = Database.conn().execute(
                "SELECT si.id AS turn_id, tg.gist, si.message, "
                "       si.cron_dom, si.cron_hour, si.cron_minute "
                "FROM scheduled_items si "
                "LEFT JOIN thread_gist tg ON tg.channel = 'schedule' AND tg.turn_id = si.id "
                "ORDER BY si.created_at DESC"
            ).fetchall()
            return [SchedulerTurn.model_validate(dict(zip(self._TURN_COLS, r))) for r in rows]
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
            row = Database.conn().execute(
                f"SELECT {', '.join(_COLS)} FROM scheduled_items WHERE id = ?",
                (item_id,),
            ).fetchone()
            if row is None:
                return error(_ERR_NOT_FOUND, 404)
            return _item_dto(row, _COLS)
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
            with Database.transaction() as conn:
                cursor = conn.cursor()
                existing = cursor.execute(
                    "SELECT start_at FROM scheduled_items WHERE id = ?",
                    (item_id,),
                ).fetchone()
                if existing is None:
                    return error(_ERR_NOT_FOUND, 404)
                # Anchor: the caller's new start_at when supplied, else the
                # row's existing one (a plain enabled-toggle omits it) — a
                # future-dated start_at is never re-floored to now.
                start_at_stored = parse_local(dto.start_at).isoformat() if dto.start_at else existing[0]
                cursor.execute(
                    """
                    UPDATE scheduled_items
                    SET message = ?, start_at = ?,
                        cron_dom = ?, cron_hour = ?, cron_minute = ?, enabled = ?
                    WHERE id = ?
                    """,
                    (
                        dto.message, start_at_stored,
                        dto.day, dto.hour, dto.minute, 1 if dto.enabled else 0, item_id,
                    ),
                )
                row = cursor.execute(
                    f"SELECT {', '.join(_COLS)} FROM scheduled_items WHERE id = ?",
                    (item_id,),
                ).fetchone()
            if row is None:
                return error(_ERR_NOT_FOUND, 404)
            return _item_dto(row, _COLS)
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
            with Database.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM scheduled_items WHERE id = ?",
                    (item_id,),
                )
                affected = cursor.rowcount
                # rowid == id under INTEGER PRIMARY KEY — drop the matching vec
                # row in the same transaction so the embedding isn't orphaned.
                cursor.execute(
                    "DELETE FROM scheduled_items_vec WHERE rowid = ?",
                    (item_id,),
                )
            if affected == 0:
                return error(_ERR_NOT_FOUND, 404)
            return None
        except Exception:
            logger.exception("[SCHEDULER API] cancel error")
            return error(_ERR_INTERNAL, 500)
