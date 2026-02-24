"""
Scheduler Skill — Native innate skill for reminders and scheduled tasks.

Backed by PostgreSQL (scheduled_items table). Provides create, list, and cancel actions.
All DB access via get_shared_db_service() (lazy import inside function).
"""

import uuid
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SCHEDULER SKILL]"


def handle_scheduler(topic: str, params: dict) -> str:
    """
    Dispatch scheduler actions based on params['action'].

    Args:
        topic: Conversation topic for context
        params: Dict with action and parameters

    Returns:
        Human-readable result string
    """
    action = params.get("action", "list").lower()

    if action == "create":
        return _create(topic, params)
    elif action == "list":
        return _list(topic, params)
    elif action == "cancel":
        return _cancel(topic, params)
    else:
        return f"Unknown scheduler action: {action}"


def _create(topic: str, params: dict) -> str:
    """Create a new scheduled item."""
    try:
        from services.database_service import get_shared_db_service

        logger.debug(
            f"{LOG_PREFIX} _create called — message={params.get('message', '')!r:.80}, "
            f"due_at={params.get('due_at', '')!r}, item_type={params.get('item_type', 'notification')!r}, "
            f"recurrence={params.get('recurrence')!r}"
        )

        message = params.get("message", "").strip()
        if not message:
            return "Error: message is required"

        if len(message) > 1000:
            return "Error: message exceeds 1000 characters"

        due_at_str = params.get("due_at", "").strip()
        if not due_at_str:
            return "Error: due_at (ISO 8601 with timezone) is required"

        # Parse due_at
        try:
            due_at = datetime.fromisoformat(due_at_str)
        except Exception as e:
            return f"Error: invalid ISO 8601 due_at: {e}"

        # Ensure due_at has timezone info (convert naive to UTC if needed)
        if due_at.tzinfo is None:
            due_at = due_at.replace(tzinfo=timezone.utc)

        # Validate in future
        now = datetime.now(timezone.utc)
        if due_at <= now:
            logger.warning(
                f"{LOG_PREFIX} _create rejected — due_at {due_at.isoformat()} is not in the future "
                f"(now={now.isoformat()})"
            )
            return f"Error: due_at must be in the future (current time: {now.isoformat()})"

        item_type = params.get("item_type", "notification").lower()
        if item_type not in ("notification", "prompt"):
            return f"Error: item_type must be 'notification' or 'prompt', got {item_type}"

        recurrence = params.get("recurrence")
        if recurrence:
            recurrence = recurrence.lower()
            valid_recurrences = ("daily", "weekly", "monthly", "weekdays", "hourly")
            if recurrence not in valid_recurrences:
                if recurrence.startswith("interval:"):
                    try:
                        mins = int(recurrence.split(":", 1)[1])
                        if not (1 <= mins <= 1440):
                            return f"Error: interval minutes must be 1–1440, got {mins}"
                        recurrence = f"interval:{mins}"
                    except (ValueError, IndexError):
                        return "Error: interval recurrence must be 'interval:N' where N is 1–1440"
                else:
                    return f"Error: recurrence must be one of {valid_recurrences} or 'interval:N', got {recurrence}"

        # Validate and normalize window_start/window_end
        window_start = params.get("window_start")
        window_end = params.get("window_end")

        if (window_start or window_end) and recurrence != "hourly":
            return "Error: window_start/window_end only valid with recurrence='hourly'"

        if window_start or window_end:
            if not (window_start and window_end):
                return "Error: both window_start and window_end required if using hourly windows"

            # Validate HH:MM format
            window_start = _normalize_hhmm(window_start)
            if not window_start:
                return "Error: window_start must be HH:MM format (e.g., '09:00')"

            window_end = _normalize_hhmm(window_end)
            if not window_end:
                return "Error: window_end must be HH:MM format (e.g., '17:00')"

            if window_start >= window_end:
                return "Error: window_start must be before window_end"

        # Insert into database
        db = get_shared_db_service()
        item_id = uuid.uuid4().hex[:8]

        is_prompt = (item_type == "prompt")
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO scheduled_items
                  (id, item_type, message, due_at, recurrence, window_start, window_end,
                   status, topic, created_by_session, created_at, group_id, is_prompt)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', %s, %s, NOW(), %s, %s)
            """, (
                item_id, item_type, message, due_at,
                recurrence, window_start, window_end,
                topic, None,
                item_id,  # group_id = own id (root of new series)
                is_prompt,
            ))
            conn.commit()

        # Emit card (non-fatal if it fails)
        try:
            from services.scheduler_card_service import SchedulerCardService
            SchedulerCardService().emit_create_card(topic, {
                "id": item_id,
                "item_type": item_type,
                "message": message,
                "due_at": due_at,
                "recurrence": recurrence,
                "status": "pending",
            })
        except Exception as card_err:
            logger.warning(f"{LOG_PREFIX} Card emit failed (non-fatal): {card_err}")

        # Format due_at for response
        due_at_fmt = due_at.strftime("%Y-%m-%d %H:%M:%S %Z")

        result = f"Scheduled: '{message}' on {due_at_fmt} (id: {item_id})"
        logger.info(f"{LOG_PREFIX} Created {item_type}: {item_id}")
        return result

    except Exception as e:
        logger.error(f"{LOG_PREFIX} Create failed: {e}", exc_info=True)
        return f"Error creating scheduled item: {e}"


def _list(topic: str, params: dict) -> str:
    """List pending scheduled items, optionally filtered by time_range."""
    try:
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        time_range = params.get("time_range", "all").lower()

        # Build time bounds and card label
        start_dt, end_dt, card_label = _resolve_time_range(time_range)

        with db.connection() as conn:
            cursor = conn.cursor()
            if start_dt and end_dt:
                # Include fired items so "today" shows what already fired today
                cursor.execute("""
                    SELECT id, item_type, message, due_at, recurrence, status
                    FROM scheduled_items
                    WHERE status IN ('pending', 'fired') AND due_at BETWEEN %s AND %s
                    ORDER BY due_at ASC
                """, (start_dt, end_dt))
            else:
                cursor.execute("""
                    SELECT id, item_type, message, due_at, recurrence, status
                    FROM scheduled_items
                    WHERE status = 'pending'
                    ORDER BY due_at ASC
                """)
            rows = cursor.fetchall()

        items = []
        lines = []
        for row in rows:
            item_id, item_type, message, due_at, recurrence, status = row
            items.append({
                "id": item_id,
                "item_type": item_type,
                "message": message,
                "due_at": due_at,
                "recurrence": recurrence,
                "status": status,
            })
            due_str = due_at.strftime("%Y-%m-%d %H:%M:%S")
            recur_str = f" ({recurrence})" if recurrence else ""
            lines.append(f"  • [{item_id}] {message} — {due_str}{recur_str}")

        # Emit query card (non-fatal if it fails)
        try:
            from services.scheduler_card_service import SchedulerCardService
            SchedulerCardService().emit_query_card(topic, items, card_label)
        except Exception as card_err:
            logger.warning(f"{LOG_PREFIX} Card emit failed (non-fatal): {card_err}")

        if not lines:
            return f"No scheduled items ({card_label.lower()})."

        return f"Scheduled items ({card_label.lower()}):\n" + "\n".join(lines)

    except Exception as e:
        logger.error(f"{LOG_PREFIX} List failed: {e}", exc_info=True)
        return f"Error listing scheduled items: {e}"


def _resolve_time_range(time_range: str):
    """
    Return (start_dt, end_dt, label) for the given time_range string.
    Uses ClientContextService for the user's timezone when available.
    Falls back to UTC midnight boundaries.
    """
    from datetime import timezone, timedelta

    now_utc = datetime.now(timezone.utc)

    # Try to get user's local time for accurate day boundaries
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
        end = (start + timedelta(days=1))
        return start, end, "Today's Schedule"

    elif time_range == "tomorrow":
        start = (_start_of_day(client_now) + timedelta(days=1)).astimezone(timezone.utc)
        end = (start + timedelta(days=1))
        return start, end, "Tomorrow's Schedule"

    elif time_range == "this_week":
        start = _start_of_day(client_now).astimezone(timezone.utc)
        end = (start + timedelta(days=7))
        return start, end, "This Week's Schedule"

    elif time_range == "next_hour":
        start = now_utc
        end = now_utc + timedelta(hours=1)
        return start, end, "Coming Up Next Hour"

    elif time_range == "soon":
        start = now_utc
        end = now_utc + timedelta(hours=6)
        return start, end, "Coming Up Soon"

    else:  # "all" or unrecognised
        return None, None, "All Scheduled Items"


def _cancel(topic: str, params: dict) -> str:
    """Cancel a scheduled item by item_id or by fuzzy message match."""
    try:
        from services.database_service import get_shared_db_service

        item_id = params.get("item_id", "").strip()
        message_query = params.get("message", "").strip()

        if not item_id and not message_query:
            return "Error: item_id or message is required to cancel"

        db = get_shared_db_service()
        item_data = None

        with db.connection() as conn:
            cursor = conn.cursor()

            if item_id:
                # Exact lookup by ID
                cursor.execute(
                    "SELECT id, item_type, message, due_at, recurrence FROM scheduled_items WHERE id=%s",
                    (item_id,)
                )
                row = cursor.fetchone()
                if row:
                    item_data = {
                        "id": row[0], "item_type": row[1], "message": row[2],
                        "due_at": row[3], "recurrence": row[4], "status": "cancelled",
                    }
                cursor.execute(
                    "UPDATE scheduled_items SET status='cancelled' WHERE id=%s AND status='pending'",
                    (item_id,)
                )
                affected = cursor.rowcount
            else:
                # Fuzzy match by message content (pending items only)
                pattern = f"%{message_query}%"
                cursor.execute(
                    """SELECT id, item_type, message, due_at, recurrence
                       FROM scheduled_items
                       WHERE status='pending' AND message ILIKE %s
                       ORDER BY due_at ASC""",
                    (pattern,)
                )
                matches = cursor.fetchall()

                if not matches:
                    conn.commit()
                    return f"Error: no pending reminder matching '{message_query}' found"

                if len(matches) > 1:
                    descriptions = ", ".join(
                        f"'{r[2][:40]}' (id:{r[0]})" for r in matches
                    )
                    conn.commit()
                    return (
                        f"Error: multiple pending reminders match '{message_query}': {descriptions}. "
                        f"Use item_id to cancel the specific one."
                    )

                row = matches[0]
                item_id = row[0]
                item_data = {
                    "id": row[0], "item_type": row[1], "message": row[2],
                    "due_at": row[3], "recurrence": row[4], "status": "cancelled",
                }
                cursor.execute(
                    "UPDATE scheduled_items SET status='cancelled' WHERE id=%s AND status='pending'",
                    (item_id,)
                )
                affected = cursor.rowcount

            conn.commit()

        if affected == 0:
            return f"Error: item {item_id} not found or already fired/cancelled"

        # Emit cancel card (non-fatal if it fails)
        if item_data:
            try:
                from services.scheduler_card_service import SchedulerCardService
                SchedulerCardService().emit_cancel_card(topic, item_data)
            except Exception as card_err:
                logger.warning(f"{LOG_PREFIX} Card emit failed (non-fatal): {card_err}")

        logger.info(f"{LOG_PREFIX} Cancelled {item_id}")
        return f"Cancelled scheduled item {item_id}"

    except Exception as e:
        logger.error(f"{LOG_PREFIX} Cancel failed: {e}", exc_info=True)
        return f"Error cancelling scheduled item: {e}"


def _normalize_hhmm(time_str: str) -> str:
    """
    Normalize a time string to HH:MM format.
    Accepts "9:00", "09:00", "9:0", etc.
    Returns normalized "HH:MM" or None if invalid.
    """
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
