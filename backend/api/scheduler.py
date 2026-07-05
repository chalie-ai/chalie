"""Scheduler namespace — CRUD over scheduled_items via Pydantic DTOs.

All routes require session auth. The API owns its SQL (raw inline against the
``scheduled_items`` table); the only service touchpoint is the fire-and-forget
``embed_scheduled_item`` thread on create. DTO-typed through the foundation
boundary decorators (``@expects``/``@responds``), following the lists reference.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import TYPE_CHECKING, cast

from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from services.cron_schedule import next_due_from
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
from .dto.scheduler_query import SchedulerGroupQuery, SchedulerHistoryQuery, SchedulerListQuery

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

_ERR_INTERNAL = "Internal server error"
_ERR_NOT_FOUND = "Not found"
_ERR_NOT_PENDING = "Not found or item is not pending"

scheduler_ns = Namespace("scheduler", description="Scheduled item management", path="/api/scheduler")

register_dto(
    scheduler_ns,
    SchedulerItem,
    SchedulerItemCreate,
    SchedulerItemUpdate,
    SchedulerTurn,
    SchedulerListQuery,
    SchedulerHistoryQuery,
    SchedulerGroupQuery,
    Error,
)

_S = scheduler_ns.models

_COLS = [
    "id", "message", "start_at", "due_at",
    "cron_dom", "cron_hour", "cron_minute",
    "status", "enabled", "channel",
    "created_by_session", "created_at", "group_id",
]
_COLS_FULL = _COLS + ["source", "external_uid"]


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


def _embed(item_id: str, message: str) -> None:
    """Fire-and-forget embedding of a freshly created item (non-fatal on failure)."""
    try:
        from services.scheduler_service import embed_scheduled_item
        embed_scheduled_item(item_id, message)
    except Exception as exc:
        logger.warning("[SCHEDULER API] Embedding failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Collection + history + group
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
        """List scheduled items, one row per recurring series (``group_id``).

        A recurring series accumulates one fired row per occurrence plus one
        pending successor; the admin table wants the series, not its firing
        history. Each ``group_id`` collapses to a single representative — its
        active pending occurrence if one exists, else the latest by ``due_at``
        — so ``enabled``/cron/status reflect the live series. An optional
        ``status`` filter applies to that representative.
        """
        try:
            inner_conditions: list[str] = []
            inner_params: list[object] = []
            if not dto.include_hidden:
                inner_conditions.append("COALESCE(hidden, 0) = 0")
            inner_where = ("WHERE " + " AND ".join(inner_conditions)) if inner_conditions else ""

            outer_conditions: list[str] = ["rn = 1"]
            outer_params: list[object] = []
            if dto.status != "all":
                outer_conditions.append("status = ?")
                outer_params.append(dto.status)

            cols = ", ".join(_COLS_FULL)
            cursor = Database.conn().cursor()
            cursor.execute(
                f"SELECT {cols} FROM ("
                f"  SELECT {cols}, ROW_NUMBER() OVER ("
                "    PARTITION BY COALESCE(group_id, id) "
                "    ORDER BY CASE WHEN status = 'pending' THEN 0 ELSE 1 END, due_at DESC"
                "  ) AS rn "
                f"  FROM scheduled_items {inner_where}"
                ") "
                f"WHERE {' AND '.join(outer_conditions)} "
                "ORDER BY CASE WHEN status = 'pending' THEN 0 ELSE 1 END, due_at DESC "
                "LIMIT ? OFFSET ?",
                inner_params + outer_params + [dto.limit, dto.offset],
            )
            rows = cursor.fetchall()
            return [_item_dto(r, _COLS_FULL) for r in rows]
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
            item_id = uuid.uuid4().hex[:8]
            now = utc_now()
            start_at_utc = parse_local(dto.start_at) if dto.start_at else now
            due_utc, _ = next_due_from(start_at_utc, dto.day, dto.hour, dto.minute)
            with Database.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO scheduled_items
                      (id, item_type, message, start_at, due_at,
                       cron_dom, cron_hour, cron_minute,
                       status, enabled, channel,
                       created_by_session, created_at, group_id)
                    VALUES (?, 'prompt', ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
                    """,
                    (
                        item_id, dto.message, start_at_utc.isoformat(), due_utc.isoformat(),
                        dto.day, dto.hour, dto.minute,
                        1 if dto.enabled else 0,
                        dto.channel, None, now.isoformat(), item_id,
                    ),
                )
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


@scheduler_ns.route("/history")
class SchedulerHistoryResource(Resource):
    @require_session
    @scheduler_ns.response(204, "History pruned")
    @scheduler_ns.response(422, "Validation failed", model=_S["Error"])
    @scheduler_ns.response(500, _ERR_INTERNAL, model=_S["Error"])
    @responds(code=204)
    @expects(SchedulerHistoryQuery, source="args")
    def delete(self, dto: SchedulerHistoryQuery) -> None | ResponseReturnValue:
        """Delete fired/failed/cancelled items older than N days."""
        try:
            with Database.transaction() as conn:
                conn.execute(
                    """
                    DELETE FROM scheduled_items
                    WHERE status IN ('fired', 'failed', 'cancelled')
                      AND created_at < datetime('now', ? || ' days')
                    """,
                    (str(-dto.older_than_days),),
                )
            return None
        except Exception:
            logger.exception("[SCHEDULER API] prune history error")
            return error(_ERR_INTERNAL, 500)


@scheduler_ns.route("/group/<group_id>")
@scheduler_ns.param("group_id", "The recurring-group identifier")
class SchedulerGroupResource(Resource):
    @require_session
    @scheduler_ns.response(200, "Group fire history", model=_S["SchedulerItem"])
    @scheduler_ns.response(500, _ERR_INTERNAL, model=_S["Error"])
    @responds(SchedulerItem, code=200)
    @expects(SchedulerGroupQuery, source="args")
    def get(self, group_id: str, dto: SchedulerGroupQuery) -> list[SchedulerItem] | ResponseReturnValue:
        """Return fire history for a recurring schedule group (newest first)."""
        try:
            rows = Database.conn().execute(
                f"SELECT {', '.join(_COLS)} FROM scheduled_items "
                "WHERE group_id = ? ORDER BY due_at DESC LIMIT ?",
                (group_id, dto.limit),
            ).fetchall()
            return [_item_dto(r, _COLS) for r in rows]
        except Exception:
            logger.exception("[SCHEDULER API] group fires error")
            return error(_ERR_INTERNAL, 500)


@scheduler_ns.route("/turns")
class SchedulerTurnsResource(Resource):
    _TURN_COLS = ["turn_id", "gist", "preview", "day", "hour", "minute"]

    @require_session
    @scheduler_ns.response(200, "Active schedule threads", model=_S["SchedulerTurn"])
    @scheduler_ns.response(500, _ERR_INTERNAL, model=_S["Error"])
    @responds(SchedulerTurn, code=200)
    def get(self) -> list[SchedulerTurn] | ResponseReturnValue:
        """List active prompt-schedule threads, one row per ``turn_id`` (§13.5).

        A series' occurrences share a ``turn_id``, so they collapse to one growing
        thread (``preview``/``gist``/cron fields are uniform across a series).
        Fully-cancelled series drop out (the HAVING); a fired one-shot stays — its
        thread is still inspectable and replyable. The gist is sourced from
        ``thread_gist`` (keyed by channel=schedule, turn_id)."""
        try:
            rows = Database.conn().execute(
                "SELECT si.turn_id, tg.gist, si.message, "
                "       si.cron_dom, si.cron_hour, si.cron_minute "
                "FROM scheduled_items si "
                "LEFT JOIN thread_gist tg ON tg.channel = 'schedule' AND tg.turn_id = si.turn_id "
                "WHERE si.item_type = 'prompt' AND si.turn_id IS NOT NULL AND COALESCE(si.hidden, 0) = 0 "
                "GROUP BY si.turn_id "
                "HAVING COUNT(CASE WHEN si.status != 'cancelled' THEN 1 END) > 0 "
                "ORDER BY MAX(si.created_at) DESC"
            ).fetchall()
            return [SchedulerTurn.model_validate(dict(zip(self._TURN_COLS, r))) for r in rows]
        except Exception:
            logger.exception("[SCHEDULER API] turns error")
            return error(_ERR_INTERNAL, 500)


# ---------------------------------------------------------------------------
# Item resource (id-addressed CRUD)
# ---------------------------------------------------------------------------

@scheduler_ns.route("/<item_id>")
@scheduler_ns.param("item_id", "The item identifier")
class SchedulerItemResource(Resource):
    @require_session
    @scheduler_ns.response(200, "The item", model=_S["SchedulerItem"])
    @scheduler_ns.response(404, _ERR_NOT_FOUND, model=_S["Error"])
    @scheduler_ns.response(500, _ERR_INTERNAL, model=_S["Error"])
    @responds(SchedulerItem, code=200)
    def get(self, item_id: str) -> SchedulerItem | ResponseReturnValue:
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
    @scheduler_ns.response(404, _ERR_NOT_PENDING, model=_S["Error"])
    @scheduler_ns.response(422, "Validation failed", model=_S["Error"])
    @scheduler_ns.response(500, _ERR_INTERNAL, model=_S["Error"])
    @responds(SchedulerItem, code=200)
    @expects(SchedulerItemUpdate)
    def put(self, item_id: str, dto: SchedulerItemUpdate) -> SchedulerItem | ResponseReturnValue:
        """Update a pending scheduled item, including its ``enabled`` flag.

        ``due_at`` is always recomputed forward via ``next_due_from`` (which
        floors its anchor to now), so re-enabling a series resumes at the next
        occurrence rather than firing the missed one — the approved skip-the-
        catch-up semantics.
        """
        try:
            start_at_utc = parse_local(dto.start_at) if dto.start_at else utc_now()
            due_utc, _ = next_due_from(start_at_utc, dto.day, dto.hour, dto.minute)
            with Database.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM scheduled_items WHERE id = ? AND status = 'pending'",
                    (item_id,),
                )
                if cursor.fetchone() is None:
                    return error(_ERR_NOT_PENDING, 404)
                cursor.execute(
                    """
                    UPDATE scheduled_items
                    SET message = ?, start_at = ?, due_at = ?,
                        cron_dom = ?, cron_hour = ?, cron_minute = ?, enabled = ?
                    WHERE id = ? AND status = 'pending'
                    """,
                    (
                        dto.message, start_at_utc.isoformat(), due_utc.isoformat(),
                        dto.day, dto.hour, dto.minute, 1 if dto.enabled else 0, item_id,
                    ),
                )
                if cursor.rowcount == 0:
                    return error(_ERR_NOT_PENDING, 404)
                cursor.execute(
                    f"SELECT {', '.join(_COLS)} FROM scheduled_items WHERE id = ?",
                    (item_id,),
                )
                row = cursor.fetchone()
            if row is None:
                return error(_ERR_NOT_PENDING, 404)
            return _item_dto(row, _COLS)
        except Exception as exc:
            logger.error("[SCHEDULER API] update error: %s", exc)
            return error(_ERR_INTERNAL, 500)

    @require_session
    @scheduler_ns.response(204, "Item cancelled")
    @scheduler_ns.response(404, _ERR_NOT_PENDING, model=_S["Error"])
    @scheduler_ns.response(500, _ERR_INTERNAL, model=_S["Error"])
    @responds(code=204)
    def delete(self, item_id: str) -> None | ResponseReturnValue:
        """Cancel a pending scheduled item (soft set status to 'cancelled')."""
        try:
            with Database.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE scheduled_items SET status = 'cancelled' "
                    "WHERE id = ? AND status = 'pending'",
                    (item_id,),
                )
                affected = cursor.rowcount
            if affected == 0:
                return error(_ERR_NOT_PENDING, 404)
            return None
        except Exception:
            logger.exception("[SCHEDULER API] cancel error")
            return error(_ERR_INTERNAL, 500)
