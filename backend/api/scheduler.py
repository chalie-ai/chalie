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
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

from .auth import require_session

logger = logging.getLogger(__name__)

scheduler_bp = Blueprint("scheduler", __name__)

_VALID_STATUSES = {"pending", "fired", "failed", "cancelled"}
_VALID_TYPES = {"reminder", "task"}
_VALID_RECURRENCES = {"daily", "weekly", "monthly", "weekdays", "hourly"}
_INTERVAL_PREFIX = "interval:"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_hhmm(s: str) -> str | None:
    """Return HH:MM string if valid, else None."""
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


def _validate_item(data: dict, require_future: bool = True) -> tuple:
    """
    Validate item fields.  Returns (clean_dict, None) on success,
    or (None, error_str) on failure.
    """
    message = (data.get("message") or "").strip()
    if not message:
        return None, "message is required"
    if len(message) > 1000:
        return None, "message must be 1000 characters or fewer"

    due_at_raw = data.get("due_at")
    if not due_at_raw:
        return None, "due_at is required"
    try:
        due_at = datetime.fromisoformat(str(due_at_raw).replace("Z", "+00:00"))
        if due_at.tzinfo is None:
            due_at = due_at.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None, "due_at must be a valid ISO 8601 datetime"

    if require_future and due_at <= datetime.now(timezone.utc):
        return None, "due_at must be in the future"

    item_type = (data.get("item_type") or "reminder").strip()
    if item_type not in _VALID_TYPES:
        return None, f"item_type must be one of: {', '.join(sorted(_VALID_TYPES))}"

    recurrence = data.get("recurrence") or None
    if recurrence is not None:
        recurrence = recurrence.strip()
        if recurrence not in _VALID_RECURRENCES:
            # Allow interval:N (1–1440 minutes)
            if recurrence.startswith(_INTERVAL_PREFIX):
                try:
                    mins = int(recurrence[len(_INTERVAL_PREFIX):])
                    if not (1 <= mins <= 1440):
                        return None, "interval must be between 1 and 1440 minutes"
                    recurrence = f"interval:{mins}"
                except (ValueError, TypeError):
                    return None, "interval recurrence must be 'interval:N' where N is 1–1440"
            else:
                return None, f"recurrence must be one of: {', '.join(sorted(_VALID_RECURRENCES))}, or 'interval:N'"

    window_start = _normalize_hhmm(data.get("window_start") or "")
    window_end = _normalize_hhmm(data.get("window_end") or "")

    if (window_start or window_end) and recurrence != "hourly":
        return None, "window_start/window_end are only valid for 'hourly' recurrence"

    if window_start and not window_end:
        return None, "window_end is required when window_start is set"
    if window_end and not window_start:
        return None, "window_start is required when window_end is set"

    is_prompt = bool(data.get("is_prompt", False))

    return {
        "message": message,
        "due_at": due_at,
        "item_type": item_type,
        "recurrence": recurrence,
        "window_start": window_start,
        "window_end": window_end,
        "is_prompt": is_prompt,
    }, None


def _serialize_item(row: dict) -> dict:
    """Convert datetime fields to ISO strings for JSON serialisation."""
    out = dict(row)
    for field in ("due_at", "created_at", "last_fired_at"):
        val = out.get(field)
        if isinstance(val, datetime):
            out[field] = val.isoformat()
        elif val is None:
            out[field] = None
    return out


def _row_to_dict(row, cols) -> dict:
    return dict(zip(cols, row))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@scheduler_bp.route("/scheduler", methods=["GET"])
@require_session
def list_scheduler():
    """List scheduled items with optional status filter and pagination."""
    status_filter = request.args.get("status", "all").strip()
    try:
        limit = min(int(request.args.get("limit", 50)), 200)
        offset = max(int(request.args.get("offset", 0)), 0)
    except (ValueError, TypeError):
        return jsonify({"error": "limit and offset must be integers"}), 400

    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        cols = ["id", "item_type", "message", "due_at", "recurrence",
                "window_start", "window_end", "status", "topic",
                "created_by_session", "created_at", "last_fired_at", "group_id", "is_prompt"]

        where_clause = ""
        params: list = []

        if status_filter != "all":
            if status_filter not in _VALID_STATUSES:
                return jsonify({"error": f"status must be one of: all, {', '.join(sorted(_VALID_STATUSES))}"}), 400
            where_clause = "WHERE status = %s"
            params.append(status_filter)

        with db.connection() as conn:
            cursor = conn.cursor()

            # Total count
            cursor.execute(
                f"SELECT COUNT(*) FROM scheduled_items {where_clause}",
                params
            )
            total = cursor.fetchone()[0]

            # Ordered results: pending first, then by due_at DESC
            cursor.execute(
                f"""
                SELECT {', '.join(cols)}
                FROM scheduled_items
                {where_clause}
                ORDER BY
                    CASE WHEN status = 'pending' THEN 0 ELSE 1 END,
                    due_at DESC
                LIMIT %s OFFSET %s
                """,
                params + [limit, offset]
            )
            rows = cursor.fetchall()

        items = [_serialize_item(_row_to_dict(r, cols)) for r in rows]
        return jsonify({"items": items, "total": total, "limit": limit, "offset": offset})

    except Exception as e:
        logger.error(f"[SCHEDULER API] list error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@scheduler_bp.route("/scheduler", methods=["POST"])
@require_session
def create_scheduler():
    """Create a new scheduled item."""
    data = request.get_json(silent=True) or {}
    clean, err = _validate_item(data, require_future=True)
    if err:
        return jsonify({"error": err}), 400

    item_id = uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc)

    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        cols = ["id", "item_type", "message", "due_at", "recurrence",
                "window_start", "window_end", "status", "topic",
                "created_by_session", "created_at", "last_fired_at", "group_id", "is_prompt"]

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO scheduled_items
                  (id, item_type, message, due_at, recurrence,
                   window_start, window_end, status, topic,
                   created_by_session, created_at, group_id, is_prompt)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', %s, %s, %s, %s, %s)
                RETURNING id, item_type, message, due_at, recurrence,
                          window_start, window_end, status, topic,
                          created_by_session, created_at, last_fired_at, group_id, is_prompt
                """,
                (
                    item_id,
                    clean["item_type"],
                    clean["message"],
                    clean["due_at"],
                    clean["recurrence"],
                    clean["window_start"],
                    clean["window_end"],
                    data.get("topic", "general"),
                    None,  # created_by_session — not available in dashboard context
                    now,
                    item_id,  # group_id = own id (root of a new series)
                    clean["is_prompt"],
                )
            )
            row = cursor.fetchone()
            conn.commit()

        item = _serialize_item(_row_to_dict(row, cols))
        return jsonify({"item": item}), 201

    except Exception as e:
        logger.error(f"[SCHEDULER API] create error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@scheduler_bp.route("/scheduler/history", methods=["DELETE"])
@require_session
def prune_history():
    """Delete fired/failed/cancelled items older than N days (default 30)."""
    try:
        older_than_days = max(int(request.args.get("older_than_days", 30)), 1)
    except (ValueError, TypeError):
        return jsonify({"error": "older_than_days must be a positive integer"}), 400

    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM scheduled_items
                WHERE status IN ('fired', 'failed', 'cancelled')
                  AND created_at < NOW() - INTERVAL '%s days'
                """,
                (older_than_days,)
            )
            deleted = cursor.rowcount
            conn.commit()

        return jsonify({"deleted": deleted})

    except Exception as e:
        logger.error(f"[SCHEDULER API] prune history error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@scheduler_bp.route("/scheduler/group/<group_id>", methods=["GET"])
@require_session
def get_scheduler_group(group_id):
    """Return fire history for a recurring schedule group (newest first, max 50)."""
    try:
        limit = min(int(request.args.get("limit", 10)), 50)
    except (ValueError, TypeError):
        limit = 10

    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        cols = ["id", "item_type", "message", "due_at", "recurrence",
                "window_start", "window_end", "status", "topic",
                "created_by_session", "created_at", "last_fired_at", "group_id", "is_prompt"]

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT {', '.join(cols)}
                FROM scheduled_items
                WHERE group_id = %s
                ORDER BY due_at DESC
                LIMIT %s
                """,
                (group_id, limit)
            )
            rows = cursor.fetchall()

        items = [_serialize_item(_row_to_dict(r, cols)) for r in rows]
        return jsonify({"items": items})

    except Exception as e:
        logger.error(f"[SCHEDULER API] group fires error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@scheduler_bp.route("/scheduler/<item_id>", methods=["GET"])
@require_session
def get_scheduler_item(item_id):
    """Fetch a single scheduled item by ID."""
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        cols = ["id", "item_type", "message", "due_at", "recurrence",
                "window_start", "window_end", "status", "topic",
                "created_by_session", "created_at", "last_fired_at", "group_id", "is_prompt"]

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT {', '.join(cols)} FROM scheduled_items WHERE id = %s",
                (item_id,)
            )
            row = cursor.fetchone()

        if not row:
            return jsonify({"error": "Not found"}), 404

        return jsonify({"item": _serialize_item(_row_to_dict(row, cols))})

    except Exception as e:
        logger.error(f"[SCHEDULER API] get item error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@scheduler_bp.route("/scheduler/<item_id>", methods=["PUT"])
@require_session
def update_scheduler_item(item_id):
    """Update a pending scheduled item."""
    data = request.get_json(silent=True) or {}
    clean, err = _validate_item(data, require_future=True)
    if err:
        return jsonify({"error": err}), 400

    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        cols = ["id", "item_type", "message", "due_at", "recurrence",
                "window_start", "window_end", "status", "topic",
                "created_by_session", "created_at", "last_fired_at", "group_id", "is_prompt"]

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE scheduled_items
                SET item_type = %s,
                    message = %s,
                    due_at = %s,
                    recurrence = %s,
                    window_start = %s,
                    window_end = %s,
                    is_prompt = %s
                WHERE id = %s AND status = 'pending'
                RETURNING id, item_type, message, due_at, recurrence,
                          window_start, window_end, status, topic,
                          created_by_session, created_at, last_fired_at, group_id, is_prompt
                """,
                (
                    clean["item_type"],
                    clean["message"],
                    clean["due_at"],
                    clean["recurrence"],
                    clean["window_start"],
                    clean["window_end"],
                    clean["is_prompt"],
                    item_id,
                )
            )
            row = cursor.fetchone()
            conn.commit()

        if not row:
            return jsonify({"error": "Not found or item is not pending"}), 404

        return jsonify({"item": _serialize_item(_row_to_dict(row, cols))})

    except Exception as e:
        logger.error(f"[SCHEDULER API] update error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@scheduler_bp.route("/scheduler/<item_id>", methods=["DELETE"])
@require_session
def cancel_scheduler_item(item_id):
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
                WHERE id = %s AND status = 'pending'
                """,
                (item_id,)
            )
            affected = cursor.rowcount
            conn.commit()

        if affected == 0:
            return jsonify({"error": "Not found or item is not pending"}), 404

        return jsonify({"status": "cancelled", "id": item_id})

    except Exception as e:
        logger.error(f"[SCHEDULER API] cancel error: {e}")
        return jsonify({"error": "Internal server error"}), 500
