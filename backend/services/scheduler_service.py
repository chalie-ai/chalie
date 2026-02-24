"""
Scheduler Service — Background poller for scheduled items in PostgreSQL.

Polls scheduled_items table every 60 seconds. Fires due items through
the prompt queue — frontal cortex handles tone and framing naturally.

Uses FOR UPDATE SKIP LOCKED to prevent double-firing with multiple workers.
Entry point: scheduler_worker(shared_state=None) registered in consumer.py.
"""

import logging
import time
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SCHEDULER]"
_POLL_INTERVAL = 60  # seconds


def scheduler_worker(shared_state=None):
    """Module-level entry point for consumer.py."""
    logging.basicConfig(level=logging.INFO)
    logger.info(f"{LOG_PREFIX} Service started (poll interval: {_POLL_INTERVAL}s)")

    next_tick = time.monotonic() + _POLL_INTERVAL
    while True:
        try:
            now = time.monotonic()
            sleep_secs = max(0, next_tick - now)
            time.sleep(sleep_secs)
            next_tick += _POLL_INTERVAL
            _poll_and_fire()
        except KeyboardInterrupt:
            logger.info(f"{LOG_PREFIX} Shutting down")
            break
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Poll cycle error: {e}")
            next_tick = time.monotonic() + _POLL_INTERVAL


def _poll_and_fire():
    """Poll for due items and fire them through the prompt queue."""
    try:
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        now = datetime.now(timezone.utc)

        # Check for overdue items (potential stall warning)
        with db.connection() as conn:
            cursor = conn.cursor()
            overdue_threshold = now - timedelta(minutes=5)
            cursor.execute(
                "SELECT COUNT(*) FROM scheduled_items WHERE status='pending' AND due_at < %s",
                (overdue_threshold,)
            )
            overdue_count = cursor.fetchone()[0]
            if overdue_count > 0:
                logger.warning(
                    f"{LOG_PREFIX} {overdue_count} item(s) overdue by >5min — possible stall"
                )

        # Atomic claim: lock rows (SKIP LOCKED = safe with multiple workers)
        # LIMIT 100 prevents long transaction locks / prompt queue floods
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, item_type, message, due_at, recurrence,
                       window_start, window_end, topic, created_by_session, group_id,
                       is_prompt
                FROM scheduled_items
                WHERE status = 'pending' AND due_at <= %s
                ORDER BY due_at
                LIMIT 100
                FOR UPDATE SKIP LOCKED
            """, (now,))
            rows = cursor.fetchall()

            cols = [
                "id", "item_type", "message", "due_at", "recurrence",
                "window_start", "window_end", "topic", "created_by_session", "group_id",
                "is_prompt"
            ]

            for row in rows:
                item = dict(zip(cols, row))
                try:
                    # Cancel-wins guard: re-check status before firing.
                    # FOR UPDATE held the lock, but a cancel may have committed
                    # just before we got here.  If status is no longer 'pending'
                    # we skip without firing.
                    cursor.execute(
                        "SELECT status FROM scheduled_items WHERE id=%s",
                        (item["id"],)
                    )
                    current = cursor.fetchone()
                    if not current or current[0] != "pending":
                        continue

                    _fire_item(item)
                    cursor.execute(
                        "UPDATE scheduled_items SET status='fired', last_fired_at=%s WHERE id=%s",
                        (now, item["id"])
                    )

                    if item["recurrence"]:
                        next_item = _build_recurrence(item, now)
                        if next_item:
                            cursor.execute("""
                                INSERT INTO scheduled_items
                                  (id, item_type, message, due_at, recurrence,
                                   window_start, window_end, status, topic,
                                   created_by_session, created_at, group_id, is_prompt)
                                VALUES (%s,%s,%s,%s,%s,%s,%s,'pending',%s,%s,%s,%s,%s)
                            """, (
                                next_item["id"],
                                next_item["item_type"],
                                next_item["message"],
                                next_item["due_at"],
                                next_item["recurrence"],
                                next_item.get("window_start"),
                                next_item.get("window_end"),
                                next_item["topic"],
                                next_item["created_by_session"],
                                now,
                                next_item.get("group_id"),
                                next_item.get("is_prompt", False),
                            ))

                except Exception as e:
                    logger.error(f"{LOG_PREFIX} Failed to fire {item['id']}: {e}")
                    cursor.execute(
                        "UPDATE scheduled_items SET status='failed' WHERE id=%s",
                        (item["id"],)
                    )

            conn.commit()

    except Exception as e:
        logger.error(f"{LOG_PREFIX} Poll and fire error: {e}")


def _fire_item(item: dict):
    """Fire a due item — directly or via LLM pipeline depending on item_type."""
    message = item.get("message", "")
    source = item.get("item_type", "notification")
    is_prompt = (source == "prompt")

    if is_prompt:
        # Route through digest_worker → cognitive triage → LLM (original behaviour)
        from services.prompt_queue import PromptQueue
        from services.client_context_service import ClientContextService
        from workers.digest_worker import digest_worker

        client_context_text = ClientContextService().format_for_prompt()
        queue = PromptQueue(queue_name="prompt-queue", worker_func=digest_worker)
        queue.enqueue(message, {
            "source": source,
            "destination": "web",
            "scheduled_at": item.get("due_at", datetime.now(timezone.utc)).isoformat(),
            "scheduled_message": message,
            "topic": item.get("topic", "general"),
            "client_context": client_context_text,
        })
        logger.info(f"{LOG_PREFIX} Fired {source} (via LLM) '{item.get('id')}': {message[:80]}")
    else:
        # Direct delivery — bypass LLM, publish straight to output events
        from services.output_service import OutputService

        OutputService().enqueue_text(
            topic=item.get("topic", "general"),
            response=message,
            mode=source.upper(),
            confidence=1.0,
            generation_time=0.0,
            original_metadata={"source": source},
        )
        logger.info(f"{LOG_PREFIX} Fired {source} (direct) '{item.get('id')}': {message[:80]}")


def _build_recurrence(item: dict, fired_at: datetime) -> dict:
    """
    Build the next occurrence for a recurring schedule.
    Returns new row dict with fresh 8-char hex id, or None if one-time.
    """
    import uuid

    recurrence = item.get("recurrence")
    if not recurrence:
        return None

    try:
        due_at = item["due_at"]
        if isinstance(due_at, str):
            due_at = datetime.fromisoformat(due_at)

        # Ensure timezone-aware (convert naive to UTC if needed)
        if due_at.tzinfo is None:
            due_at = due_at.replace(tzinfo=timezone.utc)

    except Exception:
        return None

    next_due = _calculate_next_due(due_at, recurrence, item.get("window_start"), item.get("window_end"), fired_at)
    if next_due is None:
        return None

    return {
        "id": uuid.uuid4().hex[:8],
        "item_type": item.get("item_type", "notification"),
        "message": item.get("message", ""),
        "due_at": next_due,
        "recurrence": recurrence,
        "window_start": item.get("window_start"),
        "window_end": item.get("window_end"),
        "topic": item.get("topic"),
        "created_by_session": item.get("created_by_session"),
        "group_id": item.get("group_id") or item.get("id"),  # inherit group, fall back to own id
        "is_prompt": item.get("item_type", "notification") == "prompt",
    }


def _calculate_next_due(
    due_at: datetime,
    recurrence: str,
    window_start: str = None,
    window_end: str = None,
    fired_at: datetime = None
) -> datetime:
    """
    Calculate next due datetime for a recurrence rule.
    Handles hourly windows and drift prevention.
    """
    import calendar

    if recurrence == "daily":
        return due_at + timedelta(days=1)

    elif recurrence == "weekly":
        return due_at + timedelta(weeks=1)

    elif recurrence == "monthly":
        month = due_at.month + 1
        year = due_at.year
        if month > 12:
            month = 1
            year += 1
        last_day = calendar.monthrange(year, month)[1]
        day = min(due_at.day, last_day)
        return due_at.replace(year=year, month=month, day=day)

    elif recurrence == "weekdays":
        next_dt = due_at + timedelta(days=1)
        while next_dt.weekday() >= 5:  # Skip Sat (5) and Sun (6)
            next_dt += timedelta(days=1)
        return next_dt

    elif recurrence == "hourly":
        next_dt = due_at + timedelta(hours=1)

        # If window_start/window_end set, enforce the window
        if window_start and window_end:
            start_hour, start_min = map(int, window_start.split(":"))
            end_hour, end_min = map(int, window_end.split(":"))

            # Cascade prevention: loop up to 48 times to find next valid hour within window
            for _ in range(48):
                next_hour = next_dt.hour
                next_min = next_dt.minute

                # Check if within window
                current_time = (next_hour, next_min)
                window_start_time = (start_hour, start_min)
                window_end_time = (end_hour, end_min)

                if window_start_time <= current_time < window_end_time:
                    # Within window, return this time
                    return next_dt

                # Outside window: advance to next day's window_start
                if current_time >= window_end_time:
                    # Past end of window, advance to tomorrow's start
                    next_dt = (next_dt + timedelta(days=1)).replace(
                        hour=start_hour, minute=start_min
                    )
                else:
                    # Before window_start, jump to window_start today
                    next_dt = next_dt.replace(hour=start_hour, minute=start_min)
            # Fallback if loop completes without finding a valid slot
            return None

        return next_dt

    elif recurrence.startswith("interval:"):
        try:
            mins = int(recurrence.split(":", 1)[1])
            return due_at + timedelta(minutes=mins)
        except (ValueError, IndexError):
            return None

    return None
