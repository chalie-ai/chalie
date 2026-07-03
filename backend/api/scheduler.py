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

from .auth import require_session
from .dto import Error, expects, register_dto, responds
from .dto.boundary import error
from .dto.scheduler_item import SchedulerItem, SchedulerItemCreate, SchedulerItemUpdate, SchedulerTurn
from .dto.scheduler_query import SchedulerGroupQuery, SchedulerHistoryQuery, SchedulerListQuery

if TYPE_CHECKING:
    from collections.abc import Iterable

    from services.database_service import DatabaseService

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
    "id", "item_type", "message", "due_at", "recurrence",
    "status", "channel",
    "created_by_session", "created_at", "last_fired_at", "group_id", "is_prompt",
]
_COLS_FULL = _COLS + ["source", "external_uid"]


def _get_db() -> "DatabaseService":
    from services.database_service import get_shared_db_service
    return get_shared_db_service()


def _item_dto(row: "Iterable[object]", cols: "list[str]") -> SchedulerItem:
    """Zip a select row into a :class:`SchedulerItem` read DTO."""
    return SchedulerItem.model_validate(dict(zip(cols, row)))


def _embed(item_id: str, message: str, db: "DatabaseService") -> None:
    """Fire-and-forget embedding of a freshly created item (non-fatal on failure)."""
    try:
        from services.scheduler_service import embed_scheduled_item
        embed_scheduled_item(item_id, message, db)
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
        """List scheduled items, optionally filtered by status, newest pending first."""
        try:
            db = _get_db()
            conditions: list[str] = []
            params: list[object] = []
            if dto.status != "all":
                conditions.append("status = ?")
                params.append(dto.status)
            if not dto.include_hidden:
                conditions.append("COALESCE(hidden, 0) = 0")
            where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT {', '.join(_COLS_FULL)} FROM scheduled_items {where_clause} "
                    "ORDER BY CASE WHEN status = 'pending' THEN 0 ELSE 1 END, due_at DESC "
                    "LIMIT ? OFFSET ?",
                    params + [dto.limit, dto.offset],
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
            from services.time_utils import utc_now

            item_id = uuid.uuid4().hex[:8]
            now = utc_now()
            db = _get_db()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO scheduled_items
                      (id, item_type, message, due_at, recurrence,
                       status, channel,
                       created_by_session, created_at, group_id, is_prompt)
                    VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
                    """,
                    (
                        item_id, dto.item_type, dto.message, dto.due_at.isoformat(),
                        dto.recurrence, dto.channel,
                        None, now.isoformat(), item_id, dto.item_type == "prompt",
                    ),
                )
                cursor.execute(
                    f"SELECT {', '.join(_COLS)} FROM scheduled_items WHERE id = ?",
                    (item_id,),
                )
                row = cursor.fetchone()
                conn.commit()

            threading.Thread(
                target=_embed, args=(item_id, dto.message, db),
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
            db = _get_db()
            with db.connection() as conn:
                conn.execute(
                    """
                    DELETE FROM scheduled_items
                    WHERE status IN ('fired', 'failed', 'cancelled')
                      AND created_at < datetime('now', ? || ' days')
                    """,
                    (str(-dto.older_than_days),),
                )
                conn.commit()
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
            db = _get_db()
            with db.connection() as conn:
                rows = conn.execute(
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
    _TURN_COLS = ["turn_id", "gist", "preview", "recurrence", "last_fired_at", "next_due_at"]

    @require_session
    @scheduler_ns.response(200, "Active schedule threads", model=_S["SchedulerTurn"])
    @scheduler_ns.response(500, _ERR_INTERNAL, model=_S["Error"])
    @responds(SchedulerTurn, code=200)
    def get(self) -> list[SchedulerTurn] | ResponseReturnValue:
        """List active prompt-schedule threads, one row per ``turn_id`` (§13.5).

        A series' occurrences share a ``turn_id``, so they collapse to one growing
        thread (``preview``/``gist``/``recurrence`` are uniform across a series).
        Fully-cancelled series drop out (the HAVING); a fired one-shot stays — its
        thread is still inspectable and replyable. The gist is sourced from
        ``thread_gist`` (keyed by channel=schedule, turn_id)."""
        try:
            db = _get_db()
            with db.connection() as conn:
                rows = conn.execute(
                    "SELECT si.turn_id, tg.gist, si.message, si.recurrence, "
                    "       MAX(si.last_fired_at), "
                    "       MIN(CASE WHEN si.status = 'pending' THEN si.due_at END) "
                    "FROM scheduled_items si "
                    "LEFT JOIN thread_gist tg ON tg.channel = 'schedule' AND tg.turn_id = si.turn_id "
                    "WHERE si.is_prompt = 1 AND si.turn_id IS NOT NULL AND COALESCE(si.hidden, 0) = 0 "
                    "GROUP BY si.turn_id "
                    "HAVING COUNT(CASE WHEN si.status != 'cancelled' THEN 1 END) > 0 "
                    "ORDER BY MAX(si.last_fired_at) DESC"
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
            db = _get_db()
            with db.connection() as conn:
                row = conn.execute(
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
        """Update a pending scheduled item."""
        try:
            db = _get_db()
            with db.connection() as conn:
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
                    SET item_type = ?, message = ?, due_at = ?, recurrence = ?,
                        is_prompt = ?
                    WHERE id = ? AND status = 'pending'
                    """,
                    (
                        dto.item_type, dto.message, dto.due_at.isoformat(), dto.recurrence,
                        dto.item_type == "prompt", item_id,
                    ),
                )
                if cursor.rowcount == 0:
                    conn.commit()
                    return error(_ERR_NOT_PENDING, 404)
                cursor.execute(
                    f"SELECT {', '.join(_COLS)} FROM scheduled_items WHERE id = ?",
                    (item_id,),
                )
                row = cursor.fetchone()
                conn.commit()
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
            db = _get_db()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE scheduled_items SET status = 'cancelled' "
                    "WHERE id = ? AND status = 'pending'",
                    (item_id,),
                )
                affected = cursor.rowcount
                conn.commit()
            if affected == 0:
                return error(_ERR_NOT_PENDING, 404)
            return None
        except Exception:
            logger.exception("[SCHEDULER API] cancel error")
            return error(_ERR_INTERNAL, 500)
