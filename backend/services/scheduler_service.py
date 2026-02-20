"""
Scheduler Service — Background poller for scheduled items.

Polls SQLite database (src/data/scheduler.db) every 60 seconds. Fires due items through
the digest pipeline — frontal cortex handles tone and framing naturally.

Uses system local time (not UTC) to stay in sync with Chalie regardless of system clock offset.
Pattern: follows cognitive_drift_engine.py (while True + time.sleep).
Entry point: scheduler_worker(shared_state=None) registered in consumer.py.
"""

import calendar
import sqlite3
import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SCHEDULER]"

_DATA_DIR = Path(__file__).parent.parent / "data"
_DB_PATH = _DATA_DIR / "scheduler.db"
_POLL_INTERVAL = 60  # seconds


def scheduler_worker(shared_state=None):
    """Module-level entry point for consumer.py."""
    logging.basicConfig(level=logging.INFO)
    logger.info(f"{LOG_PREFIX} Service started (poll interval: {_POLL_INTERVAL}s)")

    while True:
        try:
            time.sleep(_POLL_INTERVAL)
            _poll_and_fire()
        except KeyboardInterrupt:
            logger.info(f"{LOG_PREFIX} Shutting down")
            break
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Poll cycle error: {e}")
            time.sleep(10)


def _poll_and_fire():
    """Check for due items and fire them through the prompt queue."""
    try:
        conn = _get_db()
        now = datetime.now()

        # Find all pending items with due_at <= now
        cursor = conn.execute(
            """SELECT id, type, message, due_at, recurrence, status, created_at, last_fired_at
               FROM scheduled_items
               WHERE status = 'pending' AND due_at <= ?
               ORDER BY due_at""",
            (now.isoformat(),)
        )
        rows = cursor.fetchall()

        if not rows:
            conn.close()
            return

        logger.info(f"{LOG_PREFIX} {len(rows)} due item(s) to fire")

        # Fire each due item
        fired_ids = []
        for row in rows:
            item = {
                "id": row[0],
                "type": row[1],
                "message": row[2],
                "due_at": row[3],
                "recurrence": row[4],
                "status": row[5],
                "created_at": row[6],
                "last_fired_at": row[7],
            }
            try:
                _fire_item(item, now)
                fired_ids.append(item["id"])
            except Exception as e:
                logger.error(f"{LOG_PREFIX} Failed to fire item {item.get('id')}: {e}")

        # Update statuses and add recurrence items
        for fired_id in fired_ids:
            conn.execute(
                "UPDATE scheduled_items SET status = 'fired', last_fired_at = ? WHERE id = ?",
                (now.isoformat(), fired_id)
            )

        # Handle recurrences
        for fired_id in fired_ids:
            cursor = conn.execute(
                """SELECT id, type, message, due_at, recurrence, status, created_at, last_fired_at
                   FROM scheduled_items WHERE id = ?""",
                (fired_id,)
            )
            row = cursor.fetchone()
            if row:
                item = {
                    "id": row[0],
                    "type": row[1],
                    "message": row[2],
                    "due_at": row[3],
                    "recurrence": row[4],
                    "status": row[5],
                    "created_at": row[6],
                    "last_fired_at": row[7],
                }
                next_item = _create_recurrence(item, now)
                if next_item:
                    conn.execute(
                        """INSERT INTO scheduled_items (id, type, message, due_at, recurrence, status, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (next_item["id"], next_item["type"], next_item["message"],
                         next_item["due_at"], next_item["recurrence"], next_item["status"],
                         next_item["created_at"])
                    )
                    logger.info(
                        f"{LOG_PREFIX} Recurrence created for {item['id']}: "
                        f"next due {next_item['due_at']}"
                    )

        conn.commit()
        conn.close()

    except Exception as e:
        logger.error(f"{LOG_PREFIX} Poll and fire error: {e}")


def _fire_item(item: dict, now: datetime):
    """Enqueue a due item on the prompt-queue for frontal cortex processing."""
    from services.prompt_queue import PromptQueue
    from services.client_context_service import ClientContextService

    message = item.get("message", "")
    source = item.get("type", "reminder")  # 'reminder' or 'task'

    # Get client context (timezone, location) for accurate rendering
    client_context_text = ClientContextService().format_for_prompt()

    queue = PromptQueue(queue_name="prompt-queue")
    queue.enqueue({
        "prompt": message,
        "metadata": {
            "source": source,
            "destination": "web",
            "scheduled_at": item.get("due_at", now.isoformat()),
            "scheduled_message": message,
            "topic": item.get("topic", "general"),
            "client_context": client_context_text,
        }
    })

    logger.info(f"{LOG_PREFIX} Fired {source} '{item.get('id')}': {message[:80]}")


def _create_recurrence(item: dict, fired_at: datetime) -> dict:
    """Create a new pending item for a recurring schedule. Returns None if one-time."""
    recurrence = item.get("recurrence")
    if not recurrence:
        return None

    try:
        due_at = datetime.fromisoformat(item["due_at"])
        # Strip timezone info if present — use system local time
        if due_at.tzinfo is not None:
            due_at = due_at.replace(tzinfo=None)
    except Exception:
        return None

    next_due = _calculate_next_due(due_at, recurrence)
    if next_due is None:
        return None

    new_item = dict(item)
    new_item["id"] = uuid.uuid4().hex[:8]
    new_item["due_at"] = next_due.isoformat()
    new_item["status"] = "pending"
    new_item["last_fired_at"] = None
    new_item["created_at"] = fired_at.isoformat()
    return new_item


def _calculate_next_due(due_at: datetime, recurrence: str) -> datetime:
    """Calculate next due datetime for a recurrence rule."""
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
        # Clamp to last valid day (e.g. Jan 31 → Feb 28)
        last_day = calendar.monthrange(year, month)[1]
        day = min(due_at.day, last_day)
        return due_at.replace(year=year, month=month, day=day)

    elif recurrence == "weekdays":
        next_dt = due_at + timedelta(days=1)
        while next_dt.weekday() >= 5:  # Skip Sat (5) and Sun (6)
            next_dt += timedelta(days=1)
        return next_dt

    return None


# ── SQLite Database ────────────────────────────────────────────────

def _get_db():
    """Open database connection and initialize schema."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(_DB_PATH))
    # Enable WAL mode for safe concurrent access
    conn.execute("PRAGMA journal_mode=WAL")

    # Initialize schema
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scheduled_items (
            id            TEXT PRIMARY KEY,
            type          TEXT NOT NULL DEFAULT 'reminder',
            message       TEXT NOT NULL,
            due_at        TEXT NOT NULL,
            recurrence    TEXT,
            status        TEXT NOT NULL DEFAULT 'pending',
            created_at    TEXT DEFAULT (datetime('now')),
            last_fired_at TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_status_due ON scheduled_items(status, due_at)
    """)
    conn.commit()

    return conn
