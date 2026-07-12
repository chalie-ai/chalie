"""
Scheduler Service — Background poller for scheduled items in SQLite.

Polls scheduled_items table every 60 seconds. Fires due items either as
direct WebSocket broadcasts (WebSocketBroker) or through the chat chokepoint
for prompt-type items that need LLM execution with full tool access.

SQLite's WAL mode provides implicit locking — no explicit row locks needed.
Entry point: scheduler_worker() registered in run.py.
"""

import logging
import threading
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from services.database_service import DatabaseService

from services.embedding_utils import pack_embedding
from utils.logger import Logger
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SCHEDULER]"
_POLL_INTERVAL = 60  # seconds

# Stage-2 origin label for the return hop onto the user channel.
_SCHEDULED_SOURCE = "scheduled"

# Daemon-thread name prefix for the asynchronous two-stage fire. The poll loop
# must never block on the LLM work, so a fired prompt runs on its own thread
# named ``scheduled-work-<item_id>``.
_SCHEDULED_WORK_THREAD_PREFIX = "scheduled-work"

# How a fired scheduled prompt's worked result is framed when it is handed to
# the user-facing turn that actually surfaces (Stage 2). The placeholder is
# filled with the work loop's synthesis.
_SCHEDULED_DELIVERY_TEMPLATE = (
    "A scheduled task just ran in the background. Relay its result to me now, "
    "in your own voice:\n\n{result}"
)

# System handler registry — capabilities register callbacks for item_type='system'
_SYSTEM_HANDLERS: dict[str, Callable[..., object]] = {}


def register_system_handler(source: str, callback: Callable[..., object]) -> None:
    """Register a callback for system scheduled items with the given topic."""
    _SYSTEM_HANDLERS[source] = callback
    logger.info(f"{LOG_PREFIX} Registered system handler: {source}")


def embed_scheduled_item(item_id: str, message: str, db: "DatabaseService | None" = None) -> None:
    """Non-fatal: logs a warning and returns silently on any failure."""
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


def scheduler_worker() -> None:
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
            logger.exception(f"{LOG_PREFIX} Poll cycle error: {e}")
            next_tick = time.monotonic() + _POLL_INTERVAL


def _poll_and_fire() -> None:
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
                       is_prompt, metadata
                FROM scheduled_items
                WHERE status = 'pending' AND due_at <= ? AND item_type NOT IN ('event') AND COALESCE(hidden, 0) = 0
                ORDER BY due_at
                LIMIT 100
            """, (now_iso,))
            rows = cursor.fetchall()

            cols = [
                "id", "item_type", "message", "due_at", "recurrence",
                "window_start", "window_end", "channel", "created_by_session", "group_id",
                "is_prompt", "metadata"
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
                    elif not _fire_item(item):
                        # Deferred (e.g. vault locked on a prompt): leave the row
                        # 'pending' and do NOT generate its next occurrence, or the
                        # schedule would duplicate every poll until it fires. It is
                        # retried in place on the next cycle — see _fire_item.
                        continue
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


def _load_speed_from_history() -> float | None:
    """Read the location-history ring buffer and estimate current speed."""
    try:
        import json
        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()
        raw_entries = store.lrange("client_context:history", 0, 11)
        if not raw_entries:
            return None
        entries = []
        for raw in raw_entries:
            try:
                entry = json.loads(raw if isinstance(raw, str) else raw.decode('utf-8'))
                entries.append(entry)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        if not entries:
            return None
        from services.geo_utils import estimate_speed_from_history
        return estimate_speed_from_history(entries)
    except Exception:
        return None


_MIN_ADVISORY_BUFFER_MINUTES = 10  # minimum warning before departure time


def _build_departure_advisory(item: dict[str, object]) -> str | None:
    """Return a departure advisory string when location data is available."""
    try:
        import json
        meta_raw = item.get("metadata") or "{}"
        meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
    except (TypeError, ValueError):
        return None

    dest_lat = cast(dict[str, object], meta).get("destination_lat")
    dest_lon = cast(dict[str, object], meta).get("destination_lon")
    destination_name = cast(dict[str, object], meta).get("destination")

    if dest_lat is None or dest_lon is None:
        return None

    try:
        from services.locale_service import get_location
        from services.time_utils import utc_now, parse_utc
        from services.geo_utils import distance_km, estimate_travel_minutes, DEFAULT_SPEED_KMH

        user_loc = get_location()
        user_lat = user_loc.get("lat")
        user_lon = user_loc.get("lon")

        if user_lat is None or user_lon is None:
            return None

        dist = distance_km(float(cast(float, user_lat)), float(cast(float, user_lon)), float(cast(float, dest_lat)), float(cast(float, dest_lon)))

        speed = _load_speed_from_history() or DEFAULT_SPEED_KMH
        travel_minutes = estimate_travel_minutes(dist, speed)

        due_at_raw = item.get("due_at")
        if not due_at_raw:
            return None
        from datetime import timezone as _tz
        due_dt = parse_utc(cast(str, due_at_raw))
        if due_dt == datetime.min.replace(tzinfo=_tz.utc):
            return None
        now = utc_now()
        time_to_event_minutes = (due_dt - now).total_seconds() / 60.0

        latest_depart_minutes = time_to_event_minutes - travel_minutes
        label = destination_name or "your destination"
        buffer = max(travel_minutes * 0.25, _MIN_ADVISORY_BUFFER_MINUTES)

        if latest_depart_minutes <= 0:
            return (
                f"You should already be on your way to {label}! "
                f"Estimated travel time: {travel_minutes:.0f} min ({dist:.1f} km)."
            )
        if latest_depart_minutes <= buffer:
            return (
                f"Leave for {label} in ~{latest_depart_minutes:.0f} min. "
                f"Travel time: {travel_minutes:.0f} min ({dist:.1f} km)."
            )
    except Exception as exc:
        logger.debug(f"[SCHEDULER] _build_departure_advisory failed: {exc}")

    return None


def _fire_item(item: dict[str, object]) -> bool:
    """Fire a due item — directly or via LLM pipeline depending on item_type.

    Returns ``True`` if the item ran (or was dispatched), ``False`` if it was
    deferred and should stay ``pending`` for the next poll cycle. A prompt is
    deferred when the vault is locked: its daemon thread cannot resolve the
    provider API key and would crash silently after the item is stamped fired.
    """
    advisory = _build_departure_advisory(item)
    message = cast(str, item.get("message", ""))
    if advisory:
        message = f"{advisory}\n\n{message}"
    source = cast(str, item.get("item_type", "notification"))
    is_prompt = (source == "prompt")

    if source == "system":
        # System handler dispatch — capabilities register callbacks
        handler_key = cast(str, item.get("channel", item.get("topic", "")))
        handler = _SYSTEM_HANDLERS.get(handler_key)
        if handler:
            try:
                handler()
                logger.info(f"{LOG_PREFIX} Fired system handler '{handler_key}' for '{item.get('id')}'")
            except Exception as exc:
                logger.error(f"{LOG_PREFIX} System handler '{handler_key}' failed: {exc}")
        else:
            logger.warning(f"{LOG_PREFIX} No system handler for topic '{handler_key}'")
        return True

    if is_prompt:
        # Guard: empty/whitespace prompts are a permanent data defect (not a
        # transient condition like a locked vault). Stamp fired and move on —
        # returning False would retry them every 60s forever.
        if not message or not message.strip():
            logger.warning(f"{LOG_PREFIX} Skipping prompt item '{item.get('id', '?')}' — empty message")
            return True
        # Defer when the vault is locked: the work loop calls build_client(),
        # which calls _unseal_api_key() → returns None when the DEK is not in
        # memory → ValueError "provider requires 'api_key'", killing the daemon
        # thread after the item was already stamped fired. Staying 'pending'
        # lets the next 60s poll retry once the vault is unlocked.
        from services.vault_service import get_vault_service
        if not get_vault_service().is_unlocked():
            logger.warning(
                f"{LOG_PREFIX} Vault locked — deferring prompt item '{item.get('id', '?')}'"
            )
            return False
        # Fire asynchronously: the two-stage work runs the full LLM ACT loop and
        # must NOT execute on the scheduler poll thread, which is mid-transaction
        # claiming/marking items. A nested write commit on the shared thread-local
        # connection would flush the in-progress 'fired' UPDATE and break the
        # claim atomicity, and the poll lock would be held for the whole loop.
        item_id = cast(str, item.get('id', 'unknown'))
        threading.Thread(
            target=_fire_scheduled_prompt,
            args=(item_id, message),
            daemon=True,
            name=f"{_SCHEDULED_WORK_THREAD_PREFIX}-{item_id}",
        ).start()
    else:
        # Direct delivery — bypass LLM, broadcast straight to WebSocket
        from services.markup import sanitize
        from services.websocket_broker import WebSocketBroker

        WebSocketBroker().broadcast({
            'type': source,
            'content': sanitize(message),
        })
        logger.info(f"{LOG_PREFIX} Fired {source} (direct) '{item.get('id')}': {message[:80]}")
    return True


def _run_scheduled_work_loop(message: str) -> str:
    """Stage 1 — run the scheduled instruction on its own muted channel.

    An independent ``MessageProcessor.process`` carries out the task with the
    full tool surface on channel ``scheduled``. That channel resolves to the
    MUTED source profile, so the work loop produces no episodes, facts, geo or
    pattern signal — only the return hop (Stage 2) is encoded. Returns the work
    loop's synthesis, or '' if it exhausted its budget without an answer.
    """
    from configs.channels import ScheduledConfig  # noqa: PLC0415
    from services.message_processor import MessageProcessor  # noqa: PLC0415
    from services.processor_config import ProcessorConfig  # noqa: PLC0415

    config = ScheduledConfig(ProcessorConfig.PolicyChannel.SUBCONSCIOUS)
    return MessageProcessor.process(message, config)


def _fire_scheduled_prompt(item_id: str, message: str) -> None:
    """Fire a scheduled prompt in two stages (modelled on the web delegates).

    Stage 1 runs the task on the muted ``scheduled`` channel and returns its
    worked result. Stage 2 hands that result to an ordinary user-channel turn via
    ``dispatch_message`` — that turn is what reaches the UI and is episodically
    encoded as a normal user episode. If the work loop produced nothing usable,
    the original instruction is delivered so the user is never left silent.

    Runs on its own daemon thread (``scheduled-work-<item_id>``) so the scheduler
    poll transaction is never held during the LLM loop. A Stage-1 failure is
    logged with context and re-raised — Stage 2 is never fired with a fabricated
    success — so the failure surfaces rather than masquerading as a delivered task.
    """
    from api.chat import dispatch_message  # noqa: PLC0415

    try:
        result = _run_scheduled_work_loop(message)
    except Exception:
        logger.exception(
            f"{LOG_PREFIX} Scheduled work loop raised for '{item_id}' — "
            f"not firing return hop"
        )
        raise

    if result.strip():
        delivery = _SCHEDULED_DELIVERY_TEMPLATE.format(result=result)
        logger.info(
            f"{LOG_PREFIX} Scheduled work loop produced result for '{item_id}': "
            f"{result[:80]}"
        )
    else:
        # Work loop exhausted its budget or was cancelled — fall back to the raw
        # instruction so the scheduled prompt still surfaces to the user.
        delivery = message
        logger.warning(
            f"{LOG_PREFIX} Scheduled work loop empty for '{item_id}' — "
            f"delivering raw instruction"
        )

    dispatch_message(delivery, source=_SCHEDULED_SOURCE, hidden_input=True)
    logger.info(f"{LOG_PREFIX} Dispatched scheduled prompt '{item_id}': {message[:80]}")


_RECURRENCE_MAP = {
    "daily": ("day", 1),
    "weekly": ("day", 7),
    "weekdays": ("day", 1),
    "hourly": ("hour", 1),
}


def _in_quiet_hours(item: dict[str, object]) -> bool:
    """Check if now is outside the item's active window (quiet hours)."""
    ws, we = item.get("window_start"), item.get("window_end")
    if not ws or not we:
        return False
    from services.locale_service import local_now
    now_local = local_now()
    current = (now_local.hour, now_local.minute)
    return not (tuple(map(int, cast(str, ws).split(":"))) <= current < tuple(map(int, cast(str, we).split(":"))))


def _next_due(item: dict[str, object]) -> datetime | None:
    """Return the next UTC due datetime for a recurring item, or None."""
    import calendar
    from services.locale_service import calculate_interval
    from services.time_utils import parse_utc

    recurrence = item.get("recurrence")
    if not recurrence:
        return None

    try:
        due_at = parse_utc(cast(str, item["due_at"]))
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

    if cast(str, recurrence).startswith("interval:"):
        try:
            mins = int(cast(str, recurrence).split(":", 1)[1])
            next_utc, _ = calculate_interval(due_at, "minute", mins)
            return next_utc
        except (ValueError, IndexError):
            return None

    return None
