"""
ScheduleAbility — Native innate skill for reminders and scheduled tasks.

Backed by SQLite (scheduled_items table). Provides create, list, and cancel actions.
All DB access via get_shared_db_service() (lazy import inside function).
"""

import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import ClassVar

from abilities._base import Ability
from services.innate_skills._tag import tag as _skill_tag
from services.time_utils import utc_now

logger = logging.getLogger(__name__)
LOG_PREFIX = "[SCHEDULER SKILL]"


class ScheduleAbility(Ability):
    NAME = "schedule"
    SUMMARY = "Create, list, or cancel scheduled reminders and recurring prompts by natural language time expression."
    EXAMPLES = [
        "remind me to call the dentist tomorrow at 3pm",
        "set a daily reminder at 8am to take my vitamins",
        "what reminders do I have this week",
        "cancel my dentist reminder",
        "schedule a weekly check-in every Monday at 9am",
        "remind me in 2 hours to check the oven",
        "show me everything coming up in the next hour",
        "set an hourly reminder between 9am and 5pm to drink water",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "list", "cancel"],
                "description": "The scheduler action to perform.",
            },
            "message": {
                "type": "string",
                "description": (
                    "Required for create: what to remind or prompt (max 1000 chars). "
                    "Optional for cancel: fuzzy content match when item_id is unknown."
                ),
            },
            "due_at": {
                "type": "string",
                "description": (
                    "Required for create: ISO 8601 timestamp with timezone offset "
                    "(e.g. '2026-03-20T09:00:00+02:00'). Use the user's timezone offset. "
                    "Must be in the future. If off by a few seconds, the scheduler will auto-correct."
                ),
            },
            "item_type": {
                "type": "string",
                "enum": ["notification", "prompt"],
                "description": (
                    "Optional for create (default 'notification'). "
                    "'notification' = timed reminder shown as-is. "
                    "'prompt' = fed to Chalie as a conversational turn at the scheduled time."
                ),
            },
            "recurrence": {
                "type": "string",
                "description": (
                    "Optional for create: 'daily', 'weekly', 'monthly', 'weekdays', "
                    "'hourly', or 'interval:N' (every N minutes, 1-1440). Omit for one-time."
                ),
            },
            "window_start": {
                "type": "string",
                "description": (
                    "Optional for create with recurrence='hourly': HH:MM start of active "
                    "window (e.g. '09:00'). Requires window_end."
                ),
            },
            "window_end": {
                "type": "string",
                "description": (
                    "Optional for create with recurrence='hourly': HH:MM end of active "
                    "window (e.g. '17:00'). Requires window_start."
                ),
            },
            "item_id": {
                "type": "string",
                "description": "Optional for cancel: exact ID returned at create time. Prefer this when known.",
            },
            "time_range": {
                "type": "string",
                "enum": ["today", "tomorrow", "this_week", "next_hour", "soon", "all"],
                "description": (
                    "Optional for list: filter results by time window. "
                    "'soon' = next 6 hours. Default 'all'."
                ),
            },
        },
        "required": ["action"],
    }
    TIMEOUT = 10

    _PAST_DUE_GRACE_SECONDS: ClassVar[int] = 120

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        action = params.get("action", "list").lower()

        if action == "create":
            result = _create(channel, params, self._PAST_DUE_GRACE_SECONDS)
        elif action == "list":
            result = _list(params)
        elif action == "cancel":
            result = _cancel(params)
        else:
            result = {"status": "error", "error": f"Unknown scheduler action: {action}"}

        body = json.dumps(result)
        return {"text": _skill_tag("schedule", body, action=action)}


def _create(channel: str, params: dict, past_due_grace: int) -> dict:
    try:
        from services.database_service import get_shared_db_service

        logger.debug(
            f"{LOG_PREFIX} _create called — message={params.get('message', '')!r:.80}, "
            f"due_at={params.get('due_at', '')!r}, item_type={params.get('item_type', 'notification')!r}, "
            f"recurrence={params.get('recurrence')!r}"
        )

        message = params.get("message", "").strip()
        if not message:
            return {"status": "error", "error": "message is required"}

        if len(message) > 1000:
            return {"status": "error", "error": "message exceeds 1000 characters"}

        due_at_str = params.get("due_at", "").strip()
        if not due_at_str:
            return {"status": "error", "error": "due_at (ISO 8601 with timezone) is required"}

        from services.time_utils import parse_utc
        _SENTINEL = datetime.min.replace(tzinfo=timezone.utc)
        due_at = parse_utc(due_at_str)
        if due_at == _SENTINEL:
            return {"status": "error", "error": f"invalid ISO 8601 due_at: '{due_at_str}'"}

        now = utc_now()
        if due_at <= now:
            past_seconds = (now - due_at).total_seconds()
            if past_seconds <= past_due_grace:
                original_due_at_iso = due_at.isoformat()
                due_at = now + timedelta(seconds=5)
                logger.info(
                    f"{LOG_PREFIX} _create: past-due {original_due_at_iso} bumped to "
                    f"{due_at.isoformat()} (grace)"
                )
            else:
                from services.time_formatter_service import TimeFormatterService
                logger.warning(
                    f"{LOG_PREFIX} _create rejected — due_at {due_at.isoformat()} is not in the future "
                    f"(now={now.isoformat()})"
                )
                local_now = TimeFormatterService.local(now, fmt="%Y-%m-%dT%H:%M:%S%z") or now.isoformat()
                return {"status": "error", "error": f"due_at must be in the future (current time: {local_now})"}

        item_type = params.get("item_type", "notification").lower()
        if item_type not in ("notification", "prompt"):
            return {"status": "error", "error": f"item_type must be 'notification' or 'prompt', got {item_type}"}

        recurrence = params.get("recurrence")
        if recurrence:
            recurrence = recurrence.lower()
            valid_recurrences = ("daily", "weekly", "monthly", "weekdays", "hourly")
            if recurrence not in valid_recurrences:
                if recurrence.startswith("interval:"):
                    try:
                        mins = int(recurrence.split(":", 1)[1])
                        if not (1 <= mins <= 1440):
                            return {"status": "error", "error": f"interval minutes must be 1–1440, got {mins}"}
                        recurrence = f"interval:{mins}"
                    except (ValueError, IndexError):
                        return {"status": "error", "error": "interval recurrence must be 'interval:N' where N is 1–1440"}
                else:
                    return {"status": "error", "error": f"recurrence must be one of {valid_recurrences} or 'interval:N', got {recurrence}"}

        window_start = params.get("window_start")
        window_end = params.get("window_end")

        if (window_start or window_end) and recurrence != "hourly":
            return {"status": "error", "error": "window_start/window_end only valid with recurrence='hourly'"}

        if window_start or window_end:
            if not (window_start and window_end):
                return {"status": "error", "error": "both window_start and window_end required if using hourly windows"}

            window_start = _normalize_hhmm(window_start)
            if not window_start:
                return {"status": "error", "error": "window_start must be HH:MM format (e.g., '09:00')"}

            window_end = _normalize_hhmm(window_end)
            if not window_end:
                return {"status": "error", "error": "window_end must be HH:MM format (e.g., '17:00')"}

            if window_start >= window_end:
                return {"status": "error", "error": "window_start must be before window_end"}

        db = get_shared_db_service()
        item_id = uuid.uuid4().hex[:8]

        is_prompt = (item_type == "prompt")
        with db.connection() as conn:
            cursor = conn.cursor()

            # Atomic dedup: INSERT only when no pending row with the same message
            # was created in the last 60 seconds.  A single statement eliminates
            # the SELECT-then-INSERT race window that caused duplicate rows when
            # the model invoked schedule(...) twice in the same turn.
            cursor.execute("""
                INSERT INTO scheduled_items
                  (id, item_type, message, due_at, recurrence, window_start, window_end,
                   status, channel, created_by_session, created_at, group_id, is_prompt)
                SELECT ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, datetime('now'), ?, ?
                WHERE NOT EXISTS (
                    SELECT 1 FROM scheduled_items
                    WHERE status = 'pending'
                      AND message = ?
                      AND created_at > datetime('now', '-60 seconds')
                )
            """, (
                item_id, item_type, message, due_at,
                recurrence, window_start, window_end,
                channel, None,
                item_id,  # group_id = own id (root of new series)
                is_prompt,
                message,  # WHERE NOT EXISTS param
            ))
            conn.commit()

            if cursor.rowcount == 0:
                cursor.execute("""
                    SELECT id FROM scheduled_items
                    WHERE status = 'pending'
                      AND message = ?
                      AND created_at > datetime('now', '-60 seconds')
                    LIMIT 1
                """, (message,))
                existing_row = cursor.fetchone()
                existing_id = existing_row[0] if existing_row else item_id
                logger.info(f"{LOG_PREFIX} Dedup: '{message[:60]}' already exists as {existing_id}")
                from services.time_formatter_service import TimeFormatterService
                local_due = TimeFormatterService.local(due_at, fmt="%Y-%m-%dT%H:%M:%S%z") or due_at.isoformat()
                return {
                    "status": "success",
                    "action_performed": "create",
                    "record": {"id": existing_id, "message": message, "due_at": local_due},
                    "note": "already_existed",
                }

        try:
            from services.scheduler_service import embed_scheduled_item
            embed_scheduled_item(item_id, message, db)
        except Exception as emb_err:
            logger.warning(f"{LOG_PREFIX} Embedding failed (non-fatal): {emb_err}")

        logger.info(f"{LOG_PREFIX} Created {item_type}: {item_id}")
        from services.time_formatter_service import TimeFormatterService
        local_due = TimeFormatterService.local(due_at, fmt="%Y-%m-%dT%H:%M:%S%z") or due_at.isoformat()
        return {
            "status": "success",
            "action_performed": "create",
            "record": {
                "id": item_id,
                "message": message,
                "due_at": local_due,
                "item_type": item_type,
                "recurrence": recurrence,
            },
        }

    except Exception as e:
        logger.error(f"{LOG_PREFIX} Create failed: {e}", exc_info=True)
        return {"status": "error", "error": f"Create failed: {e}"}


def _list(params: dict) -> dict:
    try:
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        time_range = params.get("time_range", "all").lower()

        start_dt, end_dt, _ = _resolve_time_range(time_range)

        with db.connection() as conn:
            cursor = conn.cursor()
            if start_dt and end_dt:
                cursor.execute("""
                    SELECT id, item_type, message, due_at, recurrence, status
                    FROM scheduled_items
                    WHERE status IN ('pending', 'fired') AND due_at BETWEEN ? AND ?
                      AND hidden = 0
                    ORDER BY due_at ASC
                """, (start_dt, end_dt))
            else:
                cursor.execute("""
                    SELECT id, item_type, message, due_at, recurrence, status
                    FROM scheduled_items
                    WHERE status = 'pending' AND hidden = 0
                    ORDER BY due_at ASC
                """)
            rows = cursor.fetchall()

        from services.time_formatter_service import TimeFormatterService
        records = []
        for row in rows:
            item_id, item_type, message, due_at, recurrence, status = row
            local_due = TimeFormatterService.local(due_at, fmt="%Y-%m-%dT%H:%M:%S%z") or str(due_at)
            records.append({
                "id": item_id,
                "message": message,
                "due_at": local_due,
                "item_type": item_type,
                "recurrence": recurrence,
                "status": status,
            })

        return {
            "status": "success",
            "action_performed": "list",
            "records": records,
        }

    except Exception as e:
        logger.error(f"{LOG_PREFIX} List failed: {e}", exc_info=True)
        return {"status": "error", "error": f"List failed: {e}"}


def _resolve_time_range(time_range: str):
    now_utc = utc_now()

    try:
        from services.client_context_service import ClientContextService
        from zoneinfo import ZoneInfo
        ctx = ClientContextService().get()
        tz_name = ctx.get("timezone", "UTC")
        tz = ZoneInfo(tz_name)
        client_now = now_utc.astimezone(tz)
    except Exception:
        tz = timezone.utc
        client_now = now_utc

    def _start_of_day(dt):
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    if time_range == "today":
        start = _start_of_day(client_now).astimezone(timezone.utc)
        end = start + timedelta(days=1)
        return start, end, "Today's Schedule"

    elif time_range == "tomorrow":
        start = (_start_of_day(client_now) + timedelta(days=1)).astimezone(timezone.utc)
        end = start + timedelta(days=1)
        return start, end, "Tomorrow's Schedule"

    elif time_range == "this_week":
        start = _start_of_day(client_now).astimezone(timezone.utc)
        end = start + timedelta(days=7)
        return start, end, "This Week's Schedule"

    elif time_range == "next_hour":
        start = now_utc
        end = now_utc + timedelta(hours=1)
        return start, end, "Coming Up Next Hour"

    elif time_range == "soon":
        start = now_utc
        end = now_utc + timedelta(hours=6)
        return start, end, "Coming Up Soon"

    else:
        return None, None, "All Scheduled Items"


def _cancel(params: dict) -> dict:
    try:
        from services.database_service import get_shared_db_service

        item_id = params.get("item_id", "").strip()
        message_query = params.get("message", "").strip()

        if not item_id and not message_query:
            return {"status": "error", "error": "item_id or message is required to cancel"}

        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()

            if item_id:
                cursor.execute(
                    "UPDATE scheduled_items SET status='cancelled' WHERE id=? AND status='pending'",
                    (item_id,)
                )
                affected = cursor.rowcount
            else:
                pattern = f"%{message_query}%"
                cursor.execute(
                    """SELECT id, item_type, message, due_at, recurrence
                       FROM scheduled_items
                       WHERE status='pending' AND hidden = 0 AND message LIKE ?
                       ORDER BY due_at ASC""",
                    (pattern,)
                )
                matches = cursor.fetchall()

                if not matches:
                    conn.commit()
                    return {"status": "error", "error": f"no pending reminder matching '{message_query}' found"}

                if len(matches) > 1:
                    descriptions = ", ".join(
                        f"'{r[2][:40]}' (id:{r[0]})" for r in matches
                    )
                    conn.commit()
                    return {
                        "status": "error",
                        "error": (
                            f"multiple pending reminders match '{message_query}': {descriptions}. "
                            f"Use item_id to cancel the specific one."
                        ),
                    }

                item_id = matches[0][0]
                cursor.execute(
                    "UPDATE scheduled_items SET status='cancelled' WHERE id=? AND status='pending'",
                    (item_id,)
                )
                affected = cursor.rowcount

            conn.commit()

        if affected == 0:
            return {"status": "error", "error": f"item {item_id} not found or already fired/cancelled"}

        logger.info(f"{LOG_PREFIX} Cancelled {item_id}")

        record = {"id": item_id}
        try:
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, item_type, message, due_at, recurrence, status FROM scheduled_items WHERE id = ?",
                    (item_id,)
                )
                row = cursor.fetchone()
                if row:
                    record = {
                        "id": row[0],
                        "item_type": row[1],
                        "message": row[2],
                        "due_at": str(row[3]),
                        "recurrence": row[4],
                        "status": row[5],
                    }
        except Exception as fetch_err:
            logger.debug(f"{LOG_PREFIX} Could not fetch cancelled row (non-fatal): {fetch_err}")

        return {
            "status": "success",
            "action_performed": "delete",
            "record": record,
        }

    except Exception as e:
        logger.error(f"{LOG_PREFIX} Cancel failed: {e}", exc_info=True)
        return {"status": "error", "error": f"Cancel failed: {e}"}


def _normalize_hhmm(time_str: str) -> str | None:
    try:
        parts = time_str.split(":")
        if len(parts) != 2:
            return None

        hour = int(parts[0].strip())
        minute = int(parts[1].strip())

        if not (0 <= hour < 24 and 0 <= minute < 60):
            return None

        return f"{hour:02d}:{minute:02d}"
    except Exception:
        return None
