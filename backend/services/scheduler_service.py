"""
Scheduler Service — Background poller for scheduled items in SQLite.

Polls scheduled_items table every 60 seconds. Every row the poller selects is
``item_type='prompt'`` (user schedule) — 'event' rows are CalDAV calendar
entries and are excluded from the poll and never fired. Due prompts are fired
through the chat chokepoint for LLM execution with full tool access.

SQLite's WAL mode provides implicit locking — no explicit row locks needed.
Entry point: scheduler_worker() registered in run.py.
"""

import logging
import threading
import time
import uuid
from typing import cast

from services.cron_schedule import next_due_from
from services.database import Database
from services.embedding_utils import pack_embedding
from utils.logger import Logger
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SCHEDULER]"
_POLL_INTERVAL = 60  # seconds

# Daemon-thread name prefix for the asynchronous fire. The poll loop must never
# block on the LLM work, so a fired prompt runs on its own thread named
# ``scheduled-work-<item_id>``.
_SCHEDULED_WORK_THREAD_PREFIX = "scheduled-work"


def embed_scheduled_item(item_id: str, message: str) -> None:
    """Non-fatal: logs a warning and returns silently on any failure."""
    try:
        from services.embedding_service import get_embedding_service

        emb_service = get_embedding_service()
        embedding = emb_service.generate_embedding(message)
        if not embedding:
            return

        packed = pack_embedding(embedding)

        with Database.transaction() as conn:
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
        from services.time_utils import utc_now
        now = utc_now()
        now_iso = now.isoformat()

        # Check for overdue items (potential stall warning)
        with Database.transaction() as conn:
            cursor = conn.cursor()
            overdue_threshold = (now - timedelta(minutes=5)).isoformat()
            cursor.execute(
                "SELECT COUNT(*) FROM scheduled_items WHERE status='pending' AND enabled=1 AND due_at < ? AND item_type NOT IN ('event') AND hidden=0",
                (overdue_threshold,)
            )
            overdue_count = cursor.fetchone()[0]
            if overdue_count > 0:
                logger.warning(
                    f"{LOG_PREFIX} {overdue_count} item(s) overdue by >5min — possible stall"
                )

        # Atomic claim: SQLite WAL mode provides implicit locking.
        # LIMIT 100 prevents long transaction locks / prompt queue floods.
        with Database.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, item_type, message, due_at, start_at,
                       cron_dom, cron_hour, cron_minute,
                       turn_id, channel, created_by_session, group_id,
                       metadata
                FROM scheduled_items
                WHERE status = 'pending' AND enabled = 1 AND due_at <= ? AND item_type NOT IN ('event') AND COALESCE(hidden, 0) = 0
                ORDER BY due_at
                LIMIT 100
            """, (now_iso,))
            rows = cursor.fetchall()

            cols = [
                "id", "item_type", "message", "due_at", "start_at",
                "cron_dom", "cron_hour", "cron_minute",
                "turn_id", "channel", "created_by_session", "group_id",
                "metadata"
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
                        "UPDATE scheduled_items SET status='fired' WHERE id=?",
                        (item["id"],)
                    )

                    # Cron always has a next match — every prompt schedule
                    # recurs indefinitely ("cron is the schedule"; there is no
                    # one-time schedule). Advance from the just-fired due_at + 1
                    # minute so the same minute isn't re-matched.
                    from services.time_utils import parse_utc
                    fired_due_at = parse_utc(cast(str, item["due_at"]))
                    next_utc, _ = next_due_from(
                        fired_due_at + timedelta(minutes=1),
                        item["cron_dom"], item["cron_hour"], item["cron_minute"],
                    )
                    next_id = uuid.uuid4().hex[:8]
                    cursor.execute("""
                        INSERT INTO scheduled_items
                          (id, item_type, message, start_at, due_at,
                           cron_dom, cron_hour, cron_minute,
                           turn_id, status, channel,
                           created_by_session, created_at, group_id, metadata)
                        VALUES (?,?,?,?,?,?,?,?,?,'pending',?,?,?,?,?)
                    """, (
                        next_id, item["item_type"], item["message"],
                        item["start_at"], next_utc.isoformat(),
                        item["cron_dom"], item["cron_hour"], item["cron_minute"],
                        item.get("turn_id"),
                        item.get("channel"), item.get("created_by_session"),
                        now_iso, item.get("group_id") or item["id"],
                        item.get("metadata"),
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


def _fire_item(item: dict[str, object]) -> None:
    """Fire a due prompt item via the LLM pipeline.

    Every row the poller selects is ``item_type='prompt'`` ('event' rows are
    CalDAV calendar entries, excluded by the poll query and never fired), so
    this collapses to the prompt-firing path.
    """
    advisory = _build_departure_advisory(item)
    message = cast(str, item.get("message", ""))
    if advisory:
        message = f"{advisory}\n\n{message}"

    # Guard: empty/whitespace prompts are not actionable
    if not message or not message.strip():
        logger.warning(f"{LOG_PREFIX} Skipping prompt item '{item.get('id', '?')}' — empty message")
        return
    # Fire asynchronously: the full LLM ACT loop must NOT execute on the
    # scheduler poll thread, which is mid-transaction claiming/marking items.
    # A nested write commit on the shared thread-local connection would flush
    # the in-progress 'fired' UPDATE and break the claim atomicity, and the
    # poll lock would be held for the whole loop. The work opens (or reuses,
    # for a recurring series) the schedule's own thread on the ``schedule``
    # channel, keyed by ``group_key``.
    item_id = cast(str, item.get('id', 'unknown'))
    group_key = cast(str, item.get("group_id") or item["id"])
    threading.Thread(
        target=_fire_scheduled_prompt,
        args=(item_id, message, group_key, item.get("turn_id")),
        daemon=True,
        name=f"{_SCHEDULED_WORK_THREAD_PREFIX}-{item_id}",
    ).start()


def _fire_scheduled_prompt(
    item_id: str, message: str, group_key: str, turn_id: "int | None",
) -> None:
    """Run a fired scheduled prompt as its own thread on the ``schedule`` channel.

    One schedule = one ``turn_id``: the first fire of a series lets the MP allocate
    the thread (writing its own opening row), reads the fresh id back off it, and
    persists that id across every occurrence; later fires + user replies append to
    that same turn (§13.1). The turn runs the
    standard ``MessageProcessor`` under ``ScheduledConfig`` — full tool surface,
    channel+turn-scoped history, episodic encoding, the five WS turn-state
    signals — and self-surfaces in its own thread + the dock. There is no
    user-channel relay (§13.9).

    Runs on its own daemon thread so the scheduler poll transaction is never held
    during the LLM loop.
    """
    from configs.channels import ScheduledConfig  # noqa: PLC0415
    from controllers.message_processor import MessageProcessor  # noqa: PLC0415

    # Series continuity: reuse an earlier occurrence's turn_id if one was already
    # allocated, so a recurring schedule is ONE growing thread (not one per fire).
    if turn_id is None:
        with Database.transaction() as conn:
            row = conn.execute(
                "SELECT turn_id FROM scheduled_items "
                "WHERE COALESCE(group_id, id) = ? AND turn_id IS NOT NULL LIMIT 1",
                (group_key,),
            ).fetchone()
        turn_id = cast("int | None", row[0]) if row else None

    # forked-ness is the MessageProcessor's internal switch, derived purely from
    # whether a turn_id is supplied — never set here. First fire (no turn_id) → MAIN:
    # the MP writes its own opening row and allocates the thread, reporting the fresh
    # id back via ``meta["turn_id"]``, which we persist for series continuity. A
    # re-fire (or user reply) supplies that id → the MP appends past settle0 → FORK,
    # firing the §5.1 gist delegate like a user thread (§13.1 / §6.2).
    first_fire = turn_id is None
    meta: dict[str, object] = {"turn_id": turn_id}
    config = ScheduledConfig()
    mp = MessageProcessor.process(
        config, raw_input=message, metadata=meta,
        turn_id=turn_id if turn_id is not None else -1,
    )
    if first_fire:
        # process() is fire-and-forget: it returns as soon as the turn is opened,
        # before the LLM loop runs. Join the drive thread so the cancellation
        # check below reads the turn's real terminal state — this function
        # already runs on its own dedicated per-item thread (see module
        # docstring), so blocking here holds nothing else up.
        mp.result()
        # A cancel deletes nothing — a cancelled first fire's rows persist, so
        # persisting its turn_id would be safe. But a turn the user killed
        # mid-flight is a poor anchor for the whole series: skip the persist so
        # the next fire starts the thread fresh (turn_id still NULL — the
        # pre-existing first-fire footing), leaving the cancelled turn behind.
        from models.turn_execution import TurnExecution  # noqa: PLC0415
        fresh_turn_id = cast("int", meta["turn_id"])
        execution = MessageProcessor(config, fresh_turn_id).turn_execution_service.latest_for_turn()
        if execution is not None and execution.state != TurnExecution.CANCELLED:
            with Database.transaction() as conn:
                conn.execute(
                    "UPDATE scheduled_items SET turn_id = ? "
                    "WHERE COALESCE(group_id, id) = ? AND turn_id IS NULL",
                    (fresh_turn_id, group_key),
                )
        elif execution is None:
            # Not the same fact as a confirmed cancel — the tracker's own row
            # lookup failed (already logged loudly inside
            # TurnExecutionService.latest_for_turn), so this fire's true outcome
            # is unknown. Persistence is skipped either way (an unconfirmed
            # turn_id is no safer to write than a cancelled one), but the two
            # causes are distinct and must not share a log message.
            logger.warning(
                f"{LOG_PREFIX} first fire of '{item_id}' — execution row unavailable, "
                "skipping first-fire turn_id persistence"
            )
        else:
            logger.info(
                f"{LOG_PREFIX} first fire of '{item_id}' ended cancelled — "
                "leaving schedule on fresh-first-fire footing"
            )
    logger.info(f"{LOG_PREFIX} Fired scheduled prompt '{item_id}' on turn {meta['turn_id']}: {message[:80]}")
