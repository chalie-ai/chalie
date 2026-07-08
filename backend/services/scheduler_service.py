"""
Scheduler Service — Background poller for scheduled items in SQLite.

The dumb-cron model: ``scheduled_items`` holds prompt-only rows (id, message,
start_at, cron_dom/hour/minute, enabled). The poller wakes on every wall-clock
minute boundary and asks each enabled, already-started row a stateless
yes/no question via ``services.cron_schedule.matches`` — there is no
materialized ``due_at`` to walk toward, no status machine, no successor row to
insert. A match fires the prompt through the chat chokepoint for LLM
execution with full tool access, keyed to the schedule's own thread: the
schedule's integer ``id`` IS the turn_id on the ``schedule`` channel
(``INTEGER PRIMARY KEY AUTOINCREMENT`` guarantees a cancelled schedule's id is
never reissued, so a dead thread can never be re-entered).

SQLite's WAL mode provides implicit locking — no explicit row locks needed.
Entry point: scheduler_worker() registered in run.py.
"""

import logging
import threading
import time
from typing import cast

from models.scheduled_item import ScheduledItem
from services.cron_schedule import matches
from services.embedding_utils import pack_embedding
from services.time_utils import utc_now
from utils.logger import Logger

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SCHEDULER]"

# Daemon-thread name prefix for the asynchronous fire. The poll loop must never
# block on the LLM work, so a fired prompt runs on its own thread named
# ``scheduled-work-<item_id>``.
_SCHEDULED_WORK_THREAD_PREFIX = "scheduled-work"


def embed_scheduled_item(item_id: int, message: str) -> None:
    """Non-fatal: logs a warning and returns silently on any failure."""
    try:
        from services.embedding_service import get_embedding_service

        emb_service = get_embedding_service()
        embedding = emb_service.generate_embedding(message)
        if not embedding:
            return

        packed = pack_embedding(embedding)
        if packed is None:
            return
        ScheduledItem.write_embedding(item_id, packed)

        logger.debug(f"{LOG_PREFIX} Embedded scheduled item {item_id}")
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Failed to embed scheduled item {item_id}: {e}")


def scheduler_worker() -> None:
    Logger.start()
    logger.info(f"{LOG_PREFIX} Service started (minute-aligned poll)")

    while True:
        try:
            now = utc_now()
            sleep_secs = 60 - now.second - now.microsecond / 1e6
            time.sleep(sleep_secs)
            _poll_and_fire()
        except KeyboardInterrupt:
            logger.info(f"{LOG_PREFIX} Shutting down")
            break
        except Exception as e:
            logger.exception(f"{LOG_PREFIX} Poll cycle error: {e}")


def _poll_and_fire() -> None:
    try:
        now = utc_now()

        # ScheduledItem.due_at is a LOCKLESS read (it must never take the write
        # lock, or a contended minute is skipped and a fixed-time schedule
        # missed until its next occurrence — see the model method's docstring).
        rows = ScheduledItem.due_at(now.isoformat())

        for row in rows:
            if matches(
                now,
                cast("int | None", row["cron_dom"]),
                cast("int | None", row["cron_hour"]),
                cast("int | None", row["cron_minute"]),
            ):
                _fire_item(cast(int, row["id"]), cast(str, row["message"]))

    except Exception as e:
        logger.error(f"{LOG_PREFIX} Poll and fire error: {e}")


def _fire_item(item_id: int, message: str) -> None:
    """Fire a due prompt item via the LLM pipeline, on its own daemon thread.

    The poll loop must NOT execute the full LLM ACT loop inline — it fires
    asynchronously so the poller returns to sleep immediately.
    """
    if not message or not message.strip():
        logger.warning(f"{LOG_PREFIX} Skipping item '{item_id}' — empty message")
        return

    threading.Thread(
        target=_fire_scheduled_prompt,
        args=(item_id, message),
        daemon=True,
        name=f"{_SCHEDULED_WORK_THREAD_PREFIX}-{item_id}",
    ).start()


def _fire_scheduled_prompt(item_id: int, message: str) -> None:
    """Run a fired scheduled prompt on the ``schedule`` channel.

    turn_id = id: the schedule's integer id IS the turn_id on the ``schedule``
    channel. The first fire opens a MAIN turn (the id has never been used);
    every later fire supplies the same id and appends as a FORK — one
    recurring schedule is one growing thread. ``ScheduledConfig`` declares
    ``external_turn_id=True``, so the MP derives forked-ness from whether the
    turn already exists rather than rejecting the id as an invalid fork.

    Runs on its own daemon thread so the scheduler poll never blocks on the
    LLM loop; this is fire-and-forget — no result() join, no turn_id
    backfill, no cancellation check.
    """
    from configs.channels import ScheduledConfig  # noqa: PLC0415
    from controllers.message_processor import MessageProcessor  # noqa: PLC0415

    MessageProcessor.process(ScheduledConfig(), raw_input=message, turn_id=item_id)
    logger.info(f"{LOG_PREFIX} Fired scheduled prompt on turn {item_id}: {message[:80]}")
