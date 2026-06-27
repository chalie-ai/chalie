"""
Scheduler API — CRUD endpoints for scheduled_items.

Routes (all require session auth):
  GET    /scheduler              — list with status filter and pagination
  POST   /scheduler              — create a new scheduled item
  GET    /scheduler/<id>         — fetch a single item
  PUT    /scheduler/<id>         — update a pending item
  DELETE /scheduler/<id>         — cancel a pending item
  DELETE /scheduler/history      — prune fired/failed/cancelled items
"""

import uuid
import logging
from datetime import datetime
from typing import TYPE_CHECKING, cast

from flask import request
from flask.typing import ResponseReturnValue

from flask_restx import Namespace, Resource
from .auth import require_session

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

_ERR_INTERNAL = "Internal server error"
_ERR_NOT_PENDING = "Not found or item is not pending"

scheduler_bp = Namespace("scheduler", description="Scheduled item management", path="/scheduler")

_VALID_STATUSES = {"pending", "fired", "failed", "cancelled"}
_VALID_TYPES = {"notification", "prompt"}
_VALID_RECURRENCES = {"daily", "weekly", "monthly", "weekdays", "hourly"}
_INTERVAL_PREFIX = "interval:"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_hhmm(s: str) -> "str | None":
    if not s:
        return None
    parts = s.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        h, m = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return None
    return f"{h:02d}:{m:02d}"


def _validate_recurrence(recurrence: "str | None") -> "tuple[str | None, str | None]":
    """Returns (normalised_recurrence, None) on success, or (None, error_str)."""
    if recurrence is None:
        return None, None
    recurrence = recurrence.strip()
    if recurrence in _VALID_RECURRENCES:
        return recurrence, None
    if recurrence.startswith(_INTERVAL_PREFIX):
        try:
            mins = int(recurrence[len(_INTERVAL_PREFIX):])
            if not (1 <= mins <= 1440):
                return None, "interval must be between 1 and 1440 minutes"
            return f"interval:{mins}", None
        except (ValueError, TypeError):
            return None, "interval recurrence must be 'interval:N' where N is 1–1440"
    return None, f"recurrence must be one of: {', '.join(sorted(_VALID_RECURRENCES))}, or 'interval:N'"


def _validate_window(window_start: "str | None", window_end: "str | None", recurrence: "str | None") -> "str | None":
    """Returns error string or None."""
    if (window_start or window_end) and recurrence != "hourly":
        return "window_start/window_end are only valid for 'hourly' recurrence"
    if window_start and not window_end:
        return "window_end is required when window_start is set"
    if window_end and not window_start:
        return "window_start is required when window_end is set"
    return None


def _validate_item(data: "dict[str, object]", require_future: bool = True) -> "tuple[dict[str, object] | None, str | None]":
    """Returns (clean_dict, None) on success, or (None, error_str) on failure."""
    message = (cast(str, data.get("message")) or "").strip()
    if not message:
        return None, "message is required"
    if len(message) > 1000:
        return None, "message must be 1000 characters or fewer"

    due_at_raw = data.get("due_at")
    if not due_at_raw:
        return None, "due_at is required"
    try:
        from services.time_utils import parse_utc, utc_now
        due_at = parse_utc(str(due_at_raw))
    except (ValueError, TypeError):
        return None, "due_at must be a valid ISO 8601 datetime"

    if require_future and due_at <= utc_now():
        return None, "due_at must be in the future"

    item_type = (cast(str, data.get("item_type")) or "notification").strip()
    if item_type in ("event", "system"):
        return None, "item_type 'event' and 'system' are reserved for internal use"
    if item_type not in _VALID_TYPES:
        return None, f"item_type must be one of: {', '.join(sorted(_VALID_TYPES))}"

    recurrence, rec_err = _validate_recurrence(cast("str | None", data.get("recurrence")) or None)
    if rec_err:
        return None, rec_err

    window_start = _normalize_hhmm(cast(str, data.get("window_start")) or "")
    window_end = _normalize_hhmm(cast(str, data.get("window_end")) or "")
    win_err = _validate_window(window_start, window_end, recurrence)
    if win_err:
        return None, win_err

    return {
        "message": message,
        "due_at": due_at,
        "item_type": item_type,
        "recurrence": recurrence,
        "window_start": window_start,
        "window_end": window_end,
        "is_prompt": (item_type == "prompt"),
    }, None


def _serialize_item(row: "dict[str, object]") -> "dict[str, object]":
    """Convert datetime fields to ISO strings for JSON serialisation."""
    out = dict(row)
    for field in ("due_at", "created_at", "last_fired_at"):
        val = out.get(field)
        if isinstance(val, datetime):
            out[field] = val.isoformat()
        elif val is None:
            out[field] = None
    return out


def _row_to_dict(row: "Iterable[object]", cols: "list[str]") -> "dict[str, object]":
    return dict(zip(cols, row))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@scheduler_bp.route("")
@scheduler_bp.response(200, "List of scheduled items")
@scheduler_bp.response(201, "Item created")
class SchedulerListResource(Resource):
    @require_session
    def get(self) -> ResponseReturnValue:
        """List scheduled items with optional status filter and pagination."""
        status_filter = request.args.get("status", "all").strip()
        include_hidden = request.args.get("include_hidden", "").lower() == "true"
        try:
            limit = min(int(request.args.get("limit", 50)), 200)
            offset = max(int(request.args.get("offset", 0)), 0)
        except (ValueError, TypeError):
            return {"error": "limit and offset must be integers"}, 400

        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()

            cols = ["id", "item_type", "message", "due_at", "recurrence",
                    "window_start", "window_end", "status", "channel",
                    "created_by_session", "created_at", "last_fired_at", "group_id", "is_prompt",
                    "source", "external_uid"]

            conditions: list[str] = []
            params: list[object] = []

            if status_filter != "all":
                if status_filter not in _VALID_STATUSES:
                    return {"error": f"status must be one of: all, {', '.join(sorted(_VALID_STATUSES))}"}, 400
                conditions.append("status = ?")
                params.append(status_filter)

            if not include_hidden:
                conditions.append("COALESCE(hidden, 0) = 0")

            where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

            with db.connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    f"SELECT COUNT(*) FROM scheduled_items {where_clause}",
                    params
                )
                total = cursor.fetchone()[0]

                cursor.execute(
                    f"""
                    SELECT {', '.join(cols)}
                    FROM scheduled_items
                    {where_clause}
                    ORDER BY
                        CASE WHEN status = 'pending' THEN 0 ELSE 1 END,
                        due_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    params + [limit, offset]
                )
                rows = cursor.fetchall()

            items = [_serialize_item(_row_to_dict(r, cols)) for r in rows]
            return {"items": items, "total": total, "limit": limit, "offset": offset}

        except Exception as e:
            logger.error(f"[SCHEDULER API] list error: {e}")
            return {"error": _ERR_INTERNAL}, 500

    @require_session
    def post(self) -> ResponseReturnValue:
        """Create a new scheduled item."""
        data = request.get_json(silent=True) or {}
        clean, err = _validate_item(data, require_future=True)
        if err:
            return {"error": err}, 400

        item_id = uuid.uuid4().hex[:8]
        from services.time_utils import utc_now
        now = utc_now()

        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()

            cols = ["id", "item_type", "message", "due_at", "recurrence",
                    "window_start", "window_end", "status", "channel",
                    "created_by_session", "created_at", "last_fired_at", "group_id", "is_prompt"]

            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO scheduled_items
                      (id, item_type, message, due_at, recurrence,
                       window_start, window_end, status, channel,
                       created_by_session, created_at, group_id, is_prompt)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
                    """,
                    (
                        item_id,
                        cast("dict[str, object]", clean)["item_type"],
                        cast("dict[str, object]", clean)["message"],
                        cast(datetime, cast("dict[str, object]", clean)["due_at"]).isoformat(),
                        cast("dict[str, object]", clean)["recurrence"],
                        cast("dict[str, object]", clean)["window_start"],
                        cast("dict[str, object]", clean)["window_end"],
                        data.get("channel", "general"),
                        None,
                        now.isoformat(),
                        item_id,
                        cast("dict[str, object]", clean)["is_prompt"],
                    )
                )

                cursor.execute(
                    f"SELECT {', '.join(cols)} FROM scheduled_items WHERE id = ?",
                    (item_id,)
                )
                row = cursor.fetchone()
                conn.commit()

            def _embed() -> None:
                try:
                    from services.scheduler_service import embed_scheduled_item
                    embed_scheduled_item(item_id, cast(str, cast("dict[str, object]", clean)["message"]), db)
                except Exception as emb_err:
                    logger.warning(f"[SCHEDULER API] Embedding failed (non-fatal): {emb_err}")
            import threading
            threading.Thread(target=_embed, daemon=True, name="scheduler-embed").start()

            item = _serialize_item(_row_to_dict(row, cols))
            return {"item": item}, 201

        except Exception as e:
            logger.error(f"[SCHEDULER API] create error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@scheduler_bp.route("/history")
@scheduler_bp.response(200, "History pruned")
class SchedulerHistoryResource(Resource):
    @require_session
    def delete(self) -> ResponseReturnValue:
        """Delete fired/failed/cancelled items older than N days (default 30)."""
        try:
            older_than_days = max(int(request.args.get("older_than_days", 30)), 1)
        except (ValueError, TypeError):
            return {"error": "older_than_days must be a positive integer"}, 400

        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()

            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    DELETE FROM scheduled_items
                    WHERE status IN ('fired', 'failed', 'cancelled')
                      AND created_at < datetime('now', ? || ' days')
                    """,
                    (str(-older_than_days),)
                )
                deleted = cursor.rowcount
                conn.commit()

            return {"deleted": deleted}

        except Exception as e:
            logger.error(f"[SCHEDULER API] prune history error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@scheduler_bp.route("/group/<group_id>")
@scheduler_bp.response(200, "Group items")
@scheduler_bp.param("group_id", "str", "Group identifier")
class SchedulerGroupResource(Resource):
    @require_session
    def get(self, group_id: str) -> ResponseReturnValue:
        """Return fire history for a recurring schedule group (newest first, max 50)."""
        try:
            limit = min(int(request.args.get("limit", 10)), 50)
        except (ValueError, TypeError):
            limit = 10

        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()

            cols = ["id", "item_type", "message", "due_at", "recurrence",
                    "window_start", "window_end", "status", "channel",
                    "created_by_session", "created_at", "last_fired_at", "group_id", "is_prompt"]

            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT {', '.join(cols)}
                    FROM scheduled_items
                    WHERE group_id = ?
                    ORDER BY due_at DESC
                    LIMIT ?
                    """,
                    (group_id, limit)
                )
                rows = cursor.fetchall()

            items = [_serialize_item(_row_to_dict(r, cols)) for r in rows]
            return {"items": items}

        except Exception as e:
            logger.error(f"[SCHEDULER API] group fires error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@scheduler_bp.route("/<item_id>")
@scheduler_bp.response(200, "Item details")
@scheduler_bp.response(404, "Not found")
@scheduler_bp.param("item_id", "str", "Item identifier")
class SchedulerItemResource(Resource):
    @require_session
    def get(self, item_id: str) -> ResponseReturnValue:
        """Fetch a single scheduled item by ID."""
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()

            cols = ["id", "item_type", "message", "due_at", "recurrence",
                    "window_start", "window_end", "status", "channel",
                    "created_by_session", "created_at", "last_fired_at", "group_id", "is_prompt"]

            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT {', '.join(cols)} FROM scheduled_items WHERE id = ?",
                    (item_id,)
                )
                row = cursor.fetchone()

            if not row:
                return {"error": "Not found"}, 404

            return {"item": _serialize_item(_row_to_dict(row, cols))}

        except Exception as e:
            logger.error(f"[SCHEDULER API] get item error: {e}")
            return {"error": _ERR_INTERNAL}, 500

    @require_session
    def put(self, item_id: str) -> ResponseReturnValue:
        """Update a pending scheduled item."""
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()

            cols = ["id", "item_type", "message", "due_at", "recurrence",
                    "window_start", "window_end", "status", "channel",
                    "created_by_session", "created_at", "last_fired_at", "group_id", "is_prompt"]

            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM scheduled_items WHERE id = ? AND status = 'pending'",
                    (item_id,)
                )
                if not cursor.fetchone():
                    return {"error": _ERR_NOT_PENDING}, 404

            data = request.get_json(silent=True) or {}
            clean, err = _validate_item(data, require_future=True)
            if err:
                return {"error": err}, 400

            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE scheduled_items
                    SET item_type = ?,
                        message = ?,
                        due_at = ?,
                        recurrence = ?,
                        window_start = ?,
                        window_end = ?,
                        is_prompt = ?
                    WHERE id = ? AND status = 'pending'
                    """,
                    (
                        cast("dict[str, object]", clean)["item_type"],
                        cast("dict[str, object]", clean)["message"],
                        cast(datetime, cast("dict[str, object]", clean)["due_at"]).isoformat(),
                        cast("dict[str, object]", clean)["recurrence"],
                        cast("dict[str, object]", clean)["window_start"],
                        cast("dict[str, object]", clean)["window_end"],
                        cast("dict[str, object]", clean)["is_prompt"],
                        item_id,
                    )
                )

                if cursor.rowcount == 0:
                    conn.commit()
                    return {"error": _ERR_NOT_PENDING}, 404

                cursor.execute(
                    f"SELECT {', '.join(cols)} FROM scheduled_items WHERE id = ?",
                    (item_id,)
                )
                row = cursor.fetchone()
                conn.commit()

            if not row:
                return {"error": _ERR_NOT_PENDING}, 404

            return {"item": _serialize_item(_row_to_dict(row, cols))}

        except Exception as e:
            logger.error(f"[SCHEDULER API] update error: {e}")
            return {"error": _ERR_INTERNAL}, 500

    @require_session
    def delete(self, item_id: str) -> ResponseReturnValue:
        """Cancel a pending scheduled item (sets status to 'cancelled')."""
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()

            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE scheduled_items
                    SET status = 'cancelled'
                    WHERE id = ? AND status = 'pending'
                    """,
                    (item_id,)
                )
                affected = cursor.rowcount
                conn.commit()

            if affected == 0:
                return {"error": _ERR_NOT_PENDING}, 404

            return {"status": "cancelled", "id": item_id}

        except Exception as e:
            logger.error(f"[SCHEDULER API] cancel error: {e}")
            return {"error": _ERR_INTERNAL}, 500
