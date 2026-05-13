"""
Scheduler Service — Background poller for scheduled items in SQLite.

Polls scheduled_items table every 60 seconds. Fires due items either as
direct notifications (OutputService) or through ScheduledMessageProcessor
for prompt-type items that need LLM execution with full tool access.

SQLite's WAL mode provides implicit locking — no explicit row locks needed.
Entry point: scheduler_worker() registered in run.py.
"""

import logging
import threading
import time

from services.embedding_utils import pack_embedding
from utils.logger import Logger
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SCHEDULER]"
_POLL_INTERVAL = 60  # seconds

# Cap concurrent scheduled prompt executions to prevent resource exhaustion
_PROMPT_SEMAPHORE = threading.Semaphore(3)

# System handler registry — capabilities register callbacks for item_type='system'
_SYSTEM_HANDLERS: dict[str, callable] = {}


def register_system_handler(source: str, callback: callable):
    """Register a callback for system scheduled items with the given topic."""
    _SYSTEM_HANDLERS[source] = callback
    logger.info(f"{LOG_PREFIX} Registered system handler: {source}")


def embed_scheduled_item(item_id: str, message: str, db=None) -> None:
    """
    Generate and store an embedding for a scheduled item.

    Inserts into scheduled_items_vec keyed by the item's rowid.
    Non-fatal: logs a warning and returns silently on any failure.

    Args:
        item_id: The text primary key of the scheduled item (e.g. '3f8a2b1c').
        message: The message field to embed.
        db: Optional DatabaseService instance; fetches shared db if not provided.
    """
    try:
        from services.embedding_service import get_embedding_service
        if db is None:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()

        emb_service = get_embedding_service()
        embedding = emb_service.generate_embedding(message)
        if not embedding:
            return

        packed = pack_embedding(embedding)

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT rowid FROM scheduled_items WHERE id = ?", (item_id,)
            )
            row = cursor.fetchone()
            if not row:
                return

            item_rowid = row[0]
            cursor.execute(
                "INSERT OR REPLACE INTO scheduled_items_vec (rowid, embedding) VALUES (?, ?)",
                (item_rowid, packed),
            )

        logger.debug(f"{LOG_PREFIX} Embedded scheduled item {item_id}")
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Failed to embed scheduled item {item_id}: {e}")


def scheduler_worker():
    """Module-level entry point for run.py."""
    Logger.start()
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
    """Poll for due items and fire them — direct delivery or ScheduledMessageProcessor."""
    try:
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        # Check for overdue items (potential stall warning)
        with db.connection() as conn:
            cursor = conn.cursor()
            overdue_threshold = (now - timedelta(minutes=5)).isoformat()
            cursor.execute(
                "SELECT COUNT(*) FROM scheduled_items WHERE status='pending' AND due_at < ? AND item_type NOT IN ('event') AND hidden=0",
                (overdue_threshold,)
            )
            overdue_count = cursor.fetchone()[0]
            if overdue_count > 0:
                logger.warning(
                    f"{LOG_PREFIX} {overdue_count} item(s) overdue by >5min — possible stall"
                )

        # Atomic claim: SQLite WAL mode provides implicit locking.
        # LIMIT 100 prevents long transaction locks / prompt queue floods.
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, item_type, message, due_at, recurrence,
                       window_start, window_end, channel, created_by_session, group_id,
                       is_prompt
                FROM scheduled_items
                WHERE status = 'pending' AND due_at <= ? AND item_type NOT IN ('event') AND COALESCE(hidden, 0) = 0
                ORDER BY due_at
                LIMIT 100
            """, (now_iso,))
            rows = cursor.fetchall()

            cols = [
                "id", "item_type", "message", "due_at", "recurrence",
                "window_start", "window_end", "channel", "created_by_session", "group_id",
                "is_prompt"
            ]

            for row in rows:
                item = dict(zip(cols, row))
                try:
                    # Cancel-wins guard: re-check status before firing.
                    # If status is no longer 'pending' we skip without firing.
                    cursor.execute(
                        "SELECT status FROM scheduled_items WHERE id=?",
                        (item["id"],)
                    )
                    current = cursor.fetchone()
                    if not current or current[0] != "pending":
                        continue

                    _fire_item(item)
                    cursor.execute(
                        "UPDATE scheduled_items SET status='fired', last_fired_at=? WHERE id=?",
                        (now_iso, item["id"])
                    )

                    if item["recurrence"]:
                        next_item = _build_recurrence(item)
                        if next_item:
                            cursor.execute("""
                                INSERT INTO scheduled_items
                                  (id, item_type, message, due_at, recurrence,
                                   window_start, window_end, status, channel,
                                   created_by_session, created_at, group_id, is_prompt)
                                VALUES (?,?,?,?,?,?,?,'pending',?,?,?,?,?)
                            """, (
                                next_item["id"],
                                next_item["item_type"],
                                next_item["message"],
                                next_item["due_at"].isoformat() if isinstance(next_item["due_at"], datetime) else next_item["due_at"],
                                next_item["recurrence"],
                                next_item.get("window_start"),
                                next_item.get("window_end"),
                                next_item["channel"],
                                next_item["created_by_session"],
                                now_iso,
                                next_item.get("group_id"),
                                1 if next_item.get("is_prompt", False) else 0,
                            ))
                            embed_scheduled_item(next_item["id"], next_item["message"])

                except Exception as e:
                    logger.error(f"{LOG_PREFIX} Failed to fire {item['id']}: {e}")
                    cursor.execute(
                        "UPDATE scheduled_items SET status='failed' WHERE id=?",
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

    if source == "system":
        # System handler dispatch — capabilities register callbacks
        handler_key = item.get("channel", item.get("topic", ""))
        handler = _SYSTEM_HANDLERS.get(handler_key)
        if handler:
            try:
                handler()
                logger.info(f"{LOG_PREFIX} Fired system handler '{handler_key}' for '{item.get('id')}'")
            except Exception as exc:
                logger.error(f"{LOG_PREFIX} System handler '{handler_key}' failed: {exc}")
        else:
            logger.warning(f"{LOG_PREFIX} No system handler for topic '{handler_key}'")
        return

    if is_prompt:
        # Guard: empty/whitespace prompts are not actionable
        if not message or not message.strip():
            logger.warning(f"{LOG_PREFIX} Skipping prompt item '{item.get('id', '?')}' — empty message")
            return

        # Dispatch via ScheduledMessageProcessor in a daemon thread
        item_id = item.get('id', 'unknown')

        def _run():
            _PROMPT_SEMAPHORE.acquire()
            try:
                from services.scheduled_message_processor import ScheduledMessageProcessor
                from services.output_service import OutputService

                response_text = ScheduledMessageProcessor(
                    raw_input=message,
                    metadata={'item_id': item_id},
                ).send()
                response_text = (response_text or '').strip()
                if not response_text:
                    response_text = "Scheduled task completed but produced no output."

                OutputService().enqueue_proactive(
                    topic='user',
                    response=response_text,
                    source='scheduled_prompt',
                )
                logger.info(f"{LOG_PREFIX} Scheduled prompt {item_id} complete")
            except Exception as exc:
                logger.error(f"{LOG_PREFIX} Scheduled prompt {item_id} failed: {exc}", exc_info=True)
                try:
                    from services.output_service import OutputService
                    OutputService().enqueue_proactive(
                        topic='user',
                        response="A scheduled task could not be completed.",
                        source='scheduled_prompt',
                    )
                except Exception:
                    pass
            finally:
                _PROMPT_SEMAPHORE.release()

        t = threading.Thread(target=_run, daemon=True, name=f"scheduled-prompt-{item_id}")
        t.start()
        logger.info(f"{LOG_PREFIX} Dispatched scheduled prompt '{item_id}': {message[:80]}")
    else:
        # Direct delivery — bypass LLM, publish straight to output events
        from services.output_service import OutputService

        OutputService().enqueue_text(
            topic=item.get("channel", item.get("topic", "general")),
            response=message,
            mode=source.upper(),
            confidence=1.0,
            generation_time=0.0,
            original_metadata={"source": source},
        )
        logger.info(f"{LOG_PREFIX} Fired {source} (direct) '{item.get('id')}': {message[:80]}")


def _build_recurrence(item: dict, _fired_at: object = None) -> dict:
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

    next_due = _calculate_next_due(due_at, recurrence, item.get("window_start"), item.get("window_end"))
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
        "channel": item.get("channel", item.get("topic")),
        "created_by_session": item.get("created_by_session"),
        "group_id": item.get("group_id") or item.get("id"),  # inherit group, fall back to own id
        "is_prompt": item.get("item_type", "notification") == "prompt",
    }


def _calculate_next_due(
    due_at: datetime,
    recurrence: str,
    window_start: str = None,
    window_end: str = None,
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
        # Windows are in user-local time — convert for comparison
        if window_start and window_end:
            start_hour, start_min = map(int, window_start.split(":"))
            end_hour, end_min = map(int, window_end.split(":"))

            from services.time_utils import get_user_tz
            tz = get_user_tz()

            # Cascade prevention: loop up to 48 times to find next valid hour within window
            for _ in range(48):
                local_dt = next_dt.astimezone(tz)
                current_time = (local_dt.hour, local_dt.minute)
                window_start_time = (start_hour, start_min)
                window_end_time = (end_hour, end_min)

                if window_start_time <= current_time < window_end_time:
                    return next_dt

                # Outside window: advance in local time, convert back to UTC
                if current_time >= window_end_time:
                    local_next = (local_dt + timedelta(days=1)).replace(
                        hour=start_hour, minute=start_min
                    )
                else:
                    local_next = local_dt.replace(hour=start_hour, minute=start_min)
                next_dt = local_next.astimezone(timezone.utc)
            return None

        return next_dt

    elif recurrence.startswith("interval:"):
        try:
            mins = int(recurrence.split(":", 1)[1])
            return due_at + timedelta(minutes=mins)
        except (ValueError, IndexError):
            return None

    return None
