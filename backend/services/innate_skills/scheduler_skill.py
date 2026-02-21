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
            return f"Error: due_at must be in the future (current time: {now.isoformat()})"

        item_type = params.get("item_type", "reminder").lower()
        if item_type not in ("reminder", "task"):
            return f"Error: item_type must be 'reminder' or 'task', got {item_type}"

        recurrence = params.get("recurrence")
        if recurrence:
            recurrence = recurrence.lower()
            valid_recurrences = ("daily", "weekly", "monthly", "weekdays", "hourly")
            if recurrence not in valid_recurrences:
                return f"Error: recurrence must be one of {valid_recurrences}, got {recurrence}"

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

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO scheduled_items
                  (id, item_type, message, due_at, recurrence, window_start, window_end, status, topic, created_by_session, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', %s, %s, NOW())
            """, (
                item_id, item_type, message, due_at,
                recurrence, window_start, window_end,
                topic, ""
            ))
            conn.commit()

        # Format due_at for response
        due_at_fmt = due_at.strftime("%Y-%m-%d %H:%M:%S %Z")

        result = f"Scheduled: '{message}' on {due_at_fmt} (id: {item_id})"
        logger.info(f"{LOG_PREFIX} Created {item_type}: {item_id}")
        return result

    except Exception as e:
        logger.error(f"{LOG_PREFIX} Create failed: {e}", exc_info=True)
        return f"Error creating scheduled item: {e}"


def _list(topic: str, params: dict) -> str:
    """List all pending scheduled items."""
    try:
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, item_type, message, due_at, recurrence
                FROM scheduled_items
                WHERE status = 'pending'
                ORDER BY due_at ASC
            """)
            rows = cursor.fetchall()

        if not rows:
            return "No scheduled items."

        lines = []
        for row in rows:
            item_id, item_type, message, due_at, recurrence = row
            due_str = due_at.strftime("%Y-%m-%d %H:%M:%S")
            recur_str = f" ({recurrence})" if recurrence else ""
            lines.append(f"  • [{item_id}] {message} — {due_str}{recur_str}")

        return "Scheduled items:\n" + "\n".join(lines)

    except Exception as e:
        logger.error(f"{LOG_PREFIX} List failed: {e}", exc_info=True)
        return f"Error listing scheduled items: {e}"


def _cancel(topic: str, params: dict) -> str:
    """Cancel a scheduled item."""
    try:
        from services.database_service import get_shared_db_service

        item_id = params.get("item_id", "").strip()
        if not item_id:
            return "Error: item_id is required"

        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE scheduled_items SET status='cancelled' WHERE id=%s AND status='pending'",
                (item_id,)
            )
            affected = cursor.rowcount
            conn.commit()

        if affected == 0:
            return f"Error: item {item_id} not found or already fired/cancelled"

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
