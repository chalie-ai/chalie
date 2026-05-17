"""
Scheduler Service — Background poller for scheduled items in SQLite.

Polls scheduled_items table every 60 seconds. Fires due items either as
direct notifications (OutputService) or through ScheduledPromptProcessor
for prompt-type items that need LLM execution with full tool access.

SQLite's WAL mode provides implicit locking — no explicit row locks needed.
Entry point: scheduler_worker() registered in run.py.
"""

import logging
import threading
import time
import uuid

from services.embedding_utils import pack_embedding
from utils.logger import Logger
from datetime import datetime, timedelta

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
    """Poll for due items and fire them — direct delivery or ScheduledPromptProcessor."""
    try:
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        from services.time_utils import utc_now
        now = utc_now()
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

                    # Skip execution during quiet hours or weekends (for weekday schedules)
                    skip = _in_quiet_hours(item)
                    if item["recurrence"] == "weekdays":
                        from services.locale_service import local_now
                        if local_now().weekday() >= 5:
                            skip = True

                    if skip:
                        logger.debug(f"{LOG_PREFIX} Skipping {item['id']} — quiet hours / weekend")
                    else:
                        _fire_item(item)
                    cursor.execute(
                        "UPDATE scheduled_items SET status='fired', last_fired_at=? WHERE id=?",
                        (now_iso, item["id"])
                    )

                    # Generate next occurrence
                    next_due = _next_due(item)
                    if next_due is not None:
                        next_id = uuid.uuid4().hex[:8]
                        cursor.execute("""
                            INSERT INTO scheduled_items
                              (id, item_type, message, due_at, recurrence,
                               window_start, window_end, status, channel,
                               created_by_session, created_at, group_id, is_prompt)
                            VALUES (?,?,?,?,?,?,?,'pending',?,?,?,?,?)
                        """, (
                            next_id, item["item_type"], item["message"],
                            next_due.isoformat(),
                            item["recurrence"], item.get("window_start"), item.get("window_end"),
                            item.get("channel"), item.get("created_by_session"),
                            now_iso, item.get("group_id") or item["id"],
                            1 if item.get("is_prompt") else 0,
                        ))
                        embed_scheduled_item(next_id, item["message"])

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

        item_id = item.get('id', 'unknown')

        def _run():
            _PROMPT_SEMAPHORE.acquire()
            try:
                from services.user_message_processor import ScheduledPromptProcessor
                from services.output_service import OutputService

                response_text = ScheduledPromptProcessor(
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


_RECURRENCE_MAP = {
    "daily": ("day", 1),
    "weekly": ("day", 7),
    "weekdays": ("day", 1),
    "hourly": ("hour", 1),
}


def _in_quiet_hours(item: dict) -> bool:
    """Check if now is outside the item's active window (quiet hours)."""
    ws, we = item.get("window_start"), item.get("window_end")
    if not ws or not we:
        return False
    from services.locale_service import local_now
    now_local = local_now()
    current = (now_local.hour, now_local.minute)
    return not (tuple(map(int, ws.split(":"))) <= current < tuple(map(int, we.split(":"))))


def _next_due(item: dict) -> datetime | None:
    """Return the next UTC due datetime for a recurring item, or None."""
    import calendar
    from services.locale_service import calculate_interval
    from services.time_utils import parse_utc

    recurrence = item.get("recurrence")
    if not recurrence:
        return None

    try:
        due_at = parse_utc(item["due_at"])
    except Exception:
        return None

    if recurrence in _RECURRENCE_MAP:
        next_utc, _ = calculate_interval(due_at, *_RECURRENCE_MAP[recurrence])
        return next_utc

    if recurrence == "monthly":
        year, month = due_at.year, due_at.month + 1
        if month > 12:
            month, year = 1, year + 1
        day = min(due_at.day, calendar.monthrange(year, month)[1])
        return due_at.replace(year=year, month=month, day=day)

    if recurrence.startswith("interval:"):
        try:
            mins = int(recurrence.split(":", 1)[1])
            next_utc, _ = calculate_interval(due_at, "minute", mins)
            return next_utc
        except (ValueError, IndexError):
            return None

    return None
