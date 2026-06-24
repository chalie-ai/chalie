"""
ScheduleAbility — Native innate skill for reminders and scheduled tasks.

Backed by SQLite (scheduled_items table). Provides create, list, search, cancel, and update actions.
``update`` is a thin compose: it cancels the target ``item_id`` then creates a fresh item from the
remaining params — no bespoke UPDATE path to keep in sync with create's validation.
All DB access via get_shared_db_service() (lazy import inside function).

``due_at`` accepts natural language ("tomorrow 9am", "in 2 hours") OR ISO 8601.
Natural language is resolved in the user's timezone (read from the telemetry
heartbeat via ``locale_service``) — the highest-reasoning-load field in the
fleet no longer demands a hand-authored UTC offset. The resolved instant is
echoed back in the success body (``due_at_utc`` / ``due_at_local``) so the model
can confirm to the user; an unparseable string returns ``code=invalid-time``
with example forms — NEVER a guessed time and NEVER the ``parse_utc``
``datetime.min`` sentinel.

Every action returns a :class:`ToolResult` built only via ``ok`` / ``err``. The
rich scheduler card travels via ``ToolResult(rich=…)``; the dispatcher owns the
ordinal + the single span instruction and injects the card only when the
invoking channel broadcasts to the user. This ability never formats a wire
envelope.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone, timedelta
from typing import ClassVar, cast

from abilities._ability import Ability
from abilities._params import Keys
from abilities._result import ToolResult
from services.time_utils import utc_now

logger = logging.getLogger(__name__)
LOG_PREFIX = "[SCHEDULER SKILL]"

# ISO 8601 with offset, used by TimeFormatterService.local() to render due_at in the user's timezone.
_LOCAL_ISO_FMT = "%Y-%m-%dT%H:%M:%S%z"

# Example due_at forms surfaced in the invalid-time hint so a weak model can
# self-correct without re-reading the schema.
_DUE_AT_EXAMPLES = "'tomorrow 9am', 'in 2 hours', 'friday 5pm', or ISO 8601 like '2026-03-20T09:00:00+02:00'"


# ---------------------------------------------------------------------------
# Shared query engine — single SQL path for all scheduled_items readers
# ---------------------------------------------------------------------------

def query_items(
    *,
    hidden: int | None = 0,
    source: str | None = None,
    item_type: str | None = None,
    status: tuple[str, ...] = ("pending",),
    date_from: str | None = None,
    date_to: str | None = None,
    external_uid: str | None = None,
    limit: int | None = None,
    columns: tuple[str, ...] = (
        "id", "item_type", "message", "due_at", "recurrence", "status",
    ),
) -> list[dict[str, object]]:
    """This is the single query path for all callers — the schedule ability,
    the calendar ability, and anything else that reads scheduled_items.
    """
    from services.database_service import get_shared_db_service

    db = get_shared_db_service()
    col_str = ", ".join(columns)
    clauses: list[str] = []
    params: list[object] = []

    if status:
        placeholders = ", ".join("?" for _ in status)
        clauses.append(f"status IN ({placeholders})")
        params.extend(status)

    if hidden is not None:
        clauses.append("COALESCE(hidden, 0) = ?")
        params.append(hidden)

    if source:
        clauses.append("source = ?")
        params.append(source)

    if item_type:
        clauses.append("item_type = ?")
        params.append(item_type)

    if date_from:
        clauses.append("due_at >= ?")
        params.append(date_from)

    if date_to:
        clauses.append("due_at <= ?")
        params.append(date_to)

    if external_uid:
        clauses.append("external_uid = ?")
        params.append(external_uid)

    where = " AND ".join(clauses) if clauses else "1=1"
    query = f"SELECT {col_str} FROM scheduled_items WHERE {where} ORDER BY due_at ASC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    with db.connection() as conn:
        rows = conn.execute(query, params).fetchall()

    return [dict(zip(columns, row)) for row in rows]


_ACTIONS = ("create", "list", "search", "cancel", "update")


class ScheduleAbility(Ability):
    # Pre-gated by the dispatcher BEFORE run(): create requires a 'message';
    # search requires a 'query'; list/cancel require nothing at this layer
    # (cancel needs item_id OR message — an either/or the map cannot express, so
    # run() validates it). An unknown action → one unknown-action error whose
    # valid= names these keys; a known action missing a required param → one
    # missing-params error. run() never sees a malformed call.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "create": (Keys.message,),
        "list": (),
        "search": (Keys.query,),
        "cancel": (),
        "update": (Keys.item_id, Keys.message),
    }

    def get_name(self) -> str:
        return "schedule"

    def get_summary(self) -> str:
        return (
            "Create, list, or cancel persistent reminders and recurring prompts at a "
            "specific calendar date/time, with optional destination for departure reminders. "
            "For a short ephemeral countdown the user wants "
            "to watch tick down on screen (focus blocks, kitchen timers, breath holds), "
            "use the `timer` tool instead — not `schedule`."
        )

    def get_examples(self) -> list[str]:
        return [
            "remind me to call the dentist tomorrow at 3pm",
            "set a daily reminder at 8am to take my vitamins",
            "what reminders do I have this week",
            "cancel my dentist reminder",
            "schedule a weekly check-in every Monday at 9am",
            "remind me to email the quarterly report by Friday 5pm",
            "remind me to go to the gym at 7pm",
            "set a reminder to leave for work at 8am",
        ]

    def get_search_tooltip(self) -> str:
        return "reminders and scheduled tasks"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.action: {
                "type": "string",
                "enum": ["create", "list", "search", "cancel", "update"],
                "description": (
                    "The scheduler action to perform. If the user wants a short "
                    "ephemeral countdown (e.g. 'set a 5 minute timer', 'start a "
                    "25 minute focus block'), STOP — call the `timer` tool "
                    "instead of `schedule`. `schedule` is for calendar-anchored "
                    "reminders that persist across restarts."
                ),
            },
            Keys.message: {
                "type": "string",
                "description": (
                    "Required for create: what to remind or prompt (max 1000 chars). "
                    "Optional for cancel: fuzzy content match when item_id is unknown."
                ),
            },
            Keys.due_at: {
                "type": "string",
                "description": (
                    "Required for create UNLESS destination_location is provided "
                    "(location-triggered reminders do not need a time). "
                    "Natural language is accepted and resolved in the user's "
                    "timezone (e.g. 'tomorrow 9am', 'in 2 hours', 'friday 5pm', "
                    "'today at 5pm'), OR an ISO 8601 timestamp "
                    "(e.g. '2026-03-20T09:00:00+02:00'). You do NOT need to compute "
                    "the user's UTC offset — pass the time as the user said it. "
                    "Must resolve to a future instant."
                ),
            },
            Keys.item_type: {
                "type": "string",
                "enum": ["notification", "prompt"],
                "description": (
                    "Optional for create (default 'notification'). "
                    "'notification' = timed reminder shown as-is. "
                    "'prompt' = fed to Chalie as a conversational turn at the scheduled time."
                ),
            },
            Keys.recurrence: {
                "type": "string",
                "description": (
                    "Optional for create: 'daily', 'weekly', 'monthly', 'weekdays', "
                    "'hourly', or 'interval:N' (every N minutes, 1-1440). Omit for one-time."
                ),
            },
            Keys.window_start: {
                "type": "string",
                "description": (
                    "Optional for create with recurrence='hourly': HH:MM start of active "
                    "window (e.g. '09:00'). Requires window_end."
                ),
            },
            Keys.window_end: {
                "type": "string",
                "description": (
                    "Optional for create with recurrence='hourly': HH:MM end of active "
                    "window (e.g. '17:00'). Requires window_start."
                ),
            },
            Keys.item_id: {
                "type": "string",
                "description": (
                    "Optional for cancel: exact ID returned at create time. Prefer this when known. "
                    "Required for update: the existing item to replace — pass the new message/due_at/"
                    "recurrence alongside it and the old item is cancelled and recreated with the new values."
                ),
            },
            Keys.time_range: {
                "type": "string",
                "enum": ["today", "tomorrow", "this_week", "next_hour", "soon", "all"],
                "description": (
                    "Optional for list: filter results by time window. "
                    "'soon' = next 6 hours. Default 'all'."
                ),
            },
            Keys.query: {
                "type": "string",
                "description": (
                    "Required for search: semantic query to find matching scheduled items."
                ),
            },
            Keys.limit: {
                "type": "integer",
                "description": (
                    "Optional for search: maximum number of matching items to "
                    "return (default 5, clamped to 1–50)."
                ),
            },
            Keys.destination_location: {
                "type": "string",
                "description": (
                    "Name of a saved place or address for the destination. "
                    "Used for location-triggered reminders — both departure-time "
                    "and arrival-based (e.g. 'remind me when I get home'). "
                    "When provided without due_at, the reminder fires on arrival "
                    "at the destination."
                ),
            },
        },
        "required": [Keys.action],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    _PAST_DUE_GRACE_SECONDS: ClassVar[int] = 120

    def run(self, params: dict[str, object]) -> ToolResult:
        action = (cast(str, params.get(Keys.action)) or "list").lower()

        if action == "create":
            channel = getattr(getattr(self.mp, "config", None), "channel", "") or ""
            return _create(channel, params, self._PAST_DUE_GRACE_SECONDS)
        if action == "list":
            return _list(params)
        if action == "search":
            return _search(self, params, self.mp)
        if action == "cancel":
            return _cancel(params)
        if action == "update":
            # Compose, don't duplicate: retire the target then recreate from the
            # same params. Cancelling first clears create's pending-dedup guard so
            # a same-message time change isn't swallowed as a duplicate.
            _cancel(params)
            channel = getattr(getattr(self.mp, "config", None), "channel", "") or ""
            return _create(channel, params, self._PAST_DUE_GRACE_SECONDS)

        # Unreachable in practice (ACTION_REQUIRED pre-gates unknown actions);
        # kept as a self-correcting belt-and-braces error.
        return ToolResult.err(
            f"Unknown scheduler action: {action}",
            code="unknown-action",
            hint="choose one of the valid actions below.",
            valid=_ACTIONS,
        )



def _resolve_place_coords(name: str) -> tuple[float, float] | None:
    try:
        from services.data_graph_service import get_data_graph_service, KIND_PLACE
        dgs = get_data_graph_service()
        rows = dgs.fetch(kinds=[KIND_PLACE])
        matched = next((r for r in rows if r and cast(str, r.get("key", "")).lower() == name.lower()), None)
        if not matched:
            return None
        raw_value = cast(str, matched.get("value")) or "{}"
        import json as _json
        place_data: dict[str, object] = _json.loads(raw_value) if isinstance(raw_value, str) else cast(dict[str, object], raw_value)
        lat = place_data.get("lat")
        lon = place_data.get("lon")
        if lat is None or lon is None:
            return None
        return float(cast(float, lat)), float(cast(float, lon))
    except Exception as exc:
        import logging as _logging
        _logging.debug(f"[SCHEDULE] _resolve_place_coords failed for {name!r}: {exc}")
        return None


def _build_destination_metadata(destination: str) -> dict[str, object]:
    """When no matching saved place is found, stores the destination name as-is (no geocoding)."""
    coords = _resolve_place_coords(destination)
    meta: dict[str, object] = {"destination": destination}
    if coords:
        meta["destination_lat"] = coords[0]
        meta["destination_lon"] = coords[1]
    return meta


def _validate_message(params: dict[str, object]) -> tuple[str, ToolResult | None]:
    message = (cast(str, params.get(Keys.message)) or "").strip()
    if not message:
        return "", ToolResult.err(
            "message is required for create.",
            code="missing-message",
            hint="pass the reminder text in 'message'.",
        )
    if len(message) > 1000:
        return "", ToolResult.err(
            "message exceeds 1000 characters.",
            code="message-too-long",
            hint="shorten the reminder text to 1000 characters or fewer.",
        )
    return message, None


def _parse_due_at(due_at_str: str) -> datetime | None:
    """Natural language is resolved in the user's timezone (read from the telemetry
    heartbeat via ``locale_service``) so the model never has to compute a UTC offset.
    Returns ``None`` when the string is unparseable — the caller maps ``None`` to
    ``code=invalid-time`` and NEVER persists the ``parse_utc`` ``datetime.min`` sentinel.
    """
    import dateparser
    from services.locale_service import get_timezone_name, local_now

    tz_name = get_timezone_name()
    base_local = local_now()
    parsed = dateparser.parse(
        _normalise_due_at(due_at_str),
        settings={
            "TIMEZONE": tz_name,
            "TO_TIMEZONE": "UTC",
            "RETURN_AS_TIMEZONE_AWARE": True,
            "RELATIVE_BASE": base_local,
            "PREFER_DATES_FROM": "future",
        },
    )
    if parsed is None:
        return None
    return cast(datetime, parsed).astimezone(timezone.utc)


# Weekday tokens dateparser handles when bare but chokes on with a leading
# 'next/this/coming/on' qualifier — we strip the qualifier (dateparser already
# rolls a bare weekday forward) and drop the connective 'at' before a clock time.
_WEEKDAY = r"(?:mon|tue|wed|thu|fri|sat|sun)"
_QUALIFIED_WEEKDAY_RE = re.compile(rf"\b(?:next|this|coming|on)\s+({_WEEKDAY}[a-z]*)", re.IGNORECASE)
_AT_TIME_RE = re.compile(r"\bat\s+(?=\d|noon\b|midnight\b)", re.IGNORECASE)


def _normalise_due_at(text: str) -> str:
    """Light pre-normalisation so common phrasings dateparser otherwise misses
    ('next monday at 8am', 'this friday at 5pm') resolve cleanly."""
    text = _QUALIFIED_WEEKDAY_RE.sub(r"\1", text)
    text = _AT_TIME_RE.sub("", text)
    return text.strip()


def _resolve_due_at(params: dict[str, object], past_due_grace: int) -> tuple[datetime | None, ToolResult | None]:
    """When destination_location is given without due_at, defaults to now + 30 days.
    Past-due items within the grace window are bumped to now + 5 seconds.
    """
    due_at_str = (cast(str, params.get(Keys.due_at)) or "").strip()
    if not due_at_str:
        if (cast(str, params.get(Keys.destination_location)) or "").strip():
            return utc_now() + timedelta(days=30), None
        return None, ToolResult.err(
            "due_at is required for create (unless a destination_location is given).",
            code="invalid-time",
            hint=f"pass a time like {_DUE_AT_EXAMPLES}.",
        )

    due_at = _parse_due_at(due_at_str)
    if due_at is None:
        return None, ToolResult.err(
            f"Could not understand the time {due_at_str!r}.",
            code="invalid-time",
            hint=f"pass a time like {_DUE_AT_EXAMPLES}.",
        )

    now = utc_now()
    if due_at > now:
        return due_at, None

    past_seconds = (now - due_at).total_seconds()
    if past_seconds <= past_due_grace:
        original_due_at_iso = due_at.isoformat()
        due_at = now + timedelta(seconds=5)
        logger.info(f"{LOG_PREFIX} _create: past-due {original_due_at_iso} bumped to {due_at.isoformat()} (grace)")
        return due_at, None

    from services.time_formatter_service import TimeFormatterService
    logger.warning(f"{LOG_PREFIX} _create rejected — due_at {due_at.isoformat()} is not in the future (now={now.isoformat()})")
    local_now_str = TimeFormatterService.local(now, fmt=_LOCAL_ISO_FMT) or now.isoformat()
    return None, ToolResult.err(
        f"due_at resolved to a past time (current time: {local_now_str}).",
        code="due-in-past",
        hint="pass a future time, e.g. 'tomorrow 9am' or 'in 2 hours'.",
    )


_VALID_RECURRENCES = ("daily", "weekly", "monthly", "weekdays", "hourly")


def _validate_recurrence(params: dict[str, object]) -> tuple[str | None, ToolResult | None]:
    recurrence = cast(str | None, params.get(Keys.recurrence))
    if not recurrence:
        return None, None
    recurrence = recurrence.lower()
    if recurrence in _VALID_RECURRENCES:
        return recurrence, None
    if recurrence.startswith("interval:"):
        try:
            mins = int(recurrence.split(":", 1)[1])
        except (ValueError, IndexError):
            return None, ToolResult.err(
                "interval recurrence must be 'interval:N' where N is 1–1440.",
                code="invalid-recurrence",
                hint="use e.g. 'interval:30' for every 30 minutes.",
                valid=(*_VALID_RECURRENCES, "interval:N"),
            )
        if not (1 <= mins <= 1440):
            return None, ToolResult.err(
                f"interval minutes must be 1–1440, got {mins}.",
                code="invalid-recurrence",
                hint="use an interval between 1 and 1440 minutes.",
                valid=(*_VALID_RECURRENCES, "interval:N"),
            )
        return f"interval:{mins}", None
    return None, ToolResult.err(
        f"Unknown recurrence {recurrence!r}.",
        code="invalid-recurrence",
        hint="omit for one-time, or use one of the valid keywords below.",
        valid=(*_VALID_RECURRENCES, "interval:N"),
    )


def _validate_windows(params: dict[str, object], recurrence: str | None) -> tuple[str | None, str | None, ToolResult | None]:
    window_start = cast(str | None, params.get(Keys.window_start))
    window_end = cast(str | None, params.get(Keys.window_end))

    if not (window_start or window_end):
        return None, None, None
    if recurrence != "hourly":
        return None, None, ToolResult.err(
            "window_start/window_end are only valid with recurrence='hourly'.",
            code="invalid-window",
            hint="set recurrence='hourly' to use an active window, or drop the window params.",
        )
    if not (window_start and window_end):
        return None, None, ToolResult.err(
            "both window_start and window_end are required for an hourly window.",
            code="invalid-window",
            hint="pass both, e.g. window_start='09:00' and window_end='17:00'.",
        )
    window_start = _normalize_hhmm(window_start)
    if not window_start:
        return None, None, ToolResult.err(
            "window_start must be HH:MM (e.g. '09:00').",
            code="invalid-window",
            hint="pass a 24-hour time like '09:00'.",
        )
    window_end = _normalize_hhmm(window_end)
    if not window_end:
        return None, None, ToolResult.err(
            "window_end must be HH:MM (e.g. '17:00').",
            code="invalid-window",
            hint="pass a 24-hour time like '17:00'.",
        )
    if window_start >= window_end:
        return None, None, ToolResult.err(
            "window_start must be before window_end.",
            code="invalid-window",
            hint="set window_start earlier than window_end.",
        )
    return window_start, window_end, None


def _validate_item_type(params: dict[str, object]) -> tuple[str, ToolResult | None]:
    item_type = (cast(str, params.get(Keys.item_type)) or "notification").lower()
    if item_type not in ("notification", "prompt"):
        return "", ToolResult.err(
            f"item_type must be 'notification' or 'prompt', got {item_type!r}.",
            code="invalid-item-type",
            hint="use 'notification' (shown as-is) or 'prompt' (fed back to Chalie).",
            valid=("notification", "prompt"),
        )
    return item_type, None


def _due_at_strings(due_at: datetime) -> tuple[str, str]:
    from services.time_formatter_service import TimeFormatterService
    utc_iso = due_at.astimezone(timezone.utc).isoformat()
    local_iso = TimeFormatterService.local(due_at, fmt=_LOCAL_ISO_FMT) or utc_iso
    return utc_iso, local_iso


def _create(channel: str, params: dict[str, object], past_due_grace: int) -> ToolResult:
    try:
        from services.database_service import get_shared_db_service

        logger.debug(
            f"{LOG_PREFIX} _create called — message={params.get(Keys.message, '')!r:.80}, "
            f"due_at={params.get(Keys.due_at, '')!r}, item_type={params.get(Keys.item_type, 'notification')!r}, "
            f"recurrence={params.get(Keys.recurrence)!r}"
        )

        message, err = _validate_message(params)
        if err:
            return err
        due_at, err = _resolve_due_at(params, past_due_grace)
        if err:
            return err
        item_type, err = _validate_item_type(params)
        if err:
            return err
        recurrence, err = _validate_recurrence(params)
        if err:
            return err
        window_start, window_end, err = _validate_windows(params, recurrence)
        if err:
            return err

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
                utc_iso, local_iso = _due_at_strings(cast(datetime, due_at))
                record = {
                    "id": existing_id,
                    "message": message,
                    "due_at_utc": utc_iso,
                    "due_at_local": local_iso,
                    "note": "already_existed",
                }
                return _create_result(record)

        destination = (cast(str, params.get(Keys.destination_location)) or "").strip()
        if destination:
            dest_meta = _build_destination_metadata(destination)
            try:
                with db.connection() as conn:
                    conn.execute(
                        "UPDATE scheduled_items SET metadata=? WHERE id=?",
                        (json.dumps(dest_meta), item_id),
                    )
                    conn.commit()
            except Exception as meta_err:
                logger.warning(f"{LOG_PREFIX} Failed to write destination metadata (non-fatal): {meta_err}")

        try:
            from services.scheduler_service import embed_scheduled_item
            embed_scheduled_item(item_id, message, db)
        except Exception as emb_err:
            logger.warning(f"{LOG_PREFIX} Embedding failed (non-fatal): {emb_err}")

        logger.info(f"{LOG_PREFIX} Created {item_type}: {item_id}")
        utc_iso, local_iso = _due_at_strings(cast(datetime, due_at))
        record = {
            "id": item_id,
            "message": message,
            "due_at_utc": utc_iso,
            "due_at_local": local_iso,
            "item_type": item_type,
            "recurrence": recurrence,
        }
        return _create_result(record)

    except Exception as e:
        logger.exception(f"{LOG_PREFIX} Create failed: {e}")
        return ToolResult.err(f"Create failed: {e}", code="create-failed")


def _create_result(record: dict[str, object]) -> ToolResult:
    """The dispatcher injects the ordinal + span instruction only when the
    invoking channel broadcasts to the user."""
    body: dict[str, object] = {"status": "success", "action_performed": "create", "record": record}
    card: dict[str, object] = {"action_performed": "create", "record": record}
    same_day = _fetch_same_day_items(record)
    if same_day:
        card["same_day_items"] = same_day
    return ToolResult.ok(body, rich=card, action="create")


def _search(ability: "ScheduleAbility", params: dict[str, object], mp: object = None) -> ToolResult:
    query = (cast(str, params.get(Keys.query)) or "").strip()
    limit = ability.param(params, Keys.limit, default=5, clamp=(1, 50))

    try:
        from services.embedding_service import get_embedding_service
        from services.database_service import get_shared_db_service
        from services.embedding_utils import pack_embedding

        emb = get_embedding_service().generate_embedding(query, mp=mp)
        if not emb:
            return ToolResult.ok(
                {"status": "success", "action_performed": "search", "records": []},
                action="search", count=0,
            )

        blob = pack_embedding(emb)
        if blob is None:
            return ToolResult.ok(
                {"status": "success", "action_performed": "search", "records": []},
                action="search", count=0,
            )

        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.execute(
                """
                SELECT s.id, s.message, s.due_at, s.recurrence, s.status,
                       s.item_type, v.distance
                FROM scheduled_items_vec v
                JOIN scheduled_items s ON s.rowid = v.rowid
                WHERE v.embedding MATCH ?
                  AND k = ?
                  AND s.status = 'pending'
                  AND s.hidden = 0
                ORDER BY v.distance
                """,
                (blob, limit),
            )
            rows = cursor.fetchall()

        records = []
        for row in rows:
            rec = _serialise_item_row(dict(row))
            rec["distance"] = float(row["distance"])
            records.append(rec)

        return ToolResult.ok(
            {"status": "success", "action_performed": "search", "records": records},
            action="search", count=len(records),
        )

    except Exception as e:
        logger.exception(f"{LOG_PREFIX} Search failed: {e}")
        return ToolResult.err(f"Search failed: {e}", code="search-failed")


def _list(params: dict[str, object]) -> ToolResult:
    try:
        time_range = (cast(str, params.get(Keys.time_range)) or "all").lower()
        start_dt, end_dt, _ = _resolve_time_range(time_range)

        rows = query_items(
            hidden=0,
            status=("pending", "fired") if start_dt else ("pending",),
            date_from=cast(str | None, start_dt),
            date_to=cast(str | None, end_dt),
        )

        records = [_serialise_item_row(row) for row in rows]
        return ToolResult.ok(
            {"status": "success", "action_performed": "list", "records": records},
            action="list", count=len(records),
        )

    except Exception as e:
        logger.exception(f"{LOG_PREFIX} List failed: {e}")
        return ToolResult.err(f"List failed: {e}", code="list-failed")


def _serialise_item_row(row: dict[str, object]) -> dict[str, object]:
    from services.time_formatter_service import TimeFormatterService
    from services.time_utils import parse_utc
    due_raw = row.get("due_at")
    utc_dt = parse_utc(cast(datetime | str, due_raw))
    sentinel = datetime.min.replace(tzinfo=timezone.utc)
    utc_iso = utc_dt.isoformat() if utc_dt != sentinel else str(due_raw)
    local_iso = TimeFormatterService.local(cast(datetime | str | None, due_raw), fmt=_LOCAL_ISO_FMT) or utc_iso
    return {
        "id": row.get("id"),
        "message": row.get("message"),
        "due_at_utc": utc_iso,
        "due_at_local": local_iso,
        "item_type": row.get("item_type"),
        "recurrence": row.get("recurrence"),
        "status": row.get("status"),
    }


def _resolve_time_range(time_range: str) -> tuple[datetime | None, datetime | None, str]:
    now_utc = utc_now()

    from services.locale_service import local_now
    client_now = local_now()

    def _start_of_day(dt: datetime) -> datetime:
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


def _cancel(params: dict[str, object]) -> ToolResult:
    try:
        from services.database_service import get_shared_db_service

        item_id = (cast(str, params.get(Keys.item_id)) or "").strip()
        message_query = (cast(str, params.get(Keys.message)) or "").strip()

        if not item_id and not message_query:
            return ToolResult.err(
                "item_id or message is required to cancel.",
                code="cancel-target-required",
                hint="pass item_id (exact) or message (fuzzy match).",
            )

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
                    return ToolResult.err(
                        f"No pending reminder matching {message_query!r} found.",
                        code="not-found",
                        hint="call schedule with action=list to see what exists.",
                    )

                if len(matches) > 1:
                    descriptions = ", ".join(
                        f"'{r[2][:40]}' (id:{r[0]})" for r in matches
                    )
                    conn.commit()
                    return ToolResult.err(
                        f"Multiple pending reminders match {message_query!r}: {descriptions}.",
                        code="ambiguous-match",
                        hint="pass item_id to cancel the specific one.",
                    )

                item_id = matches[0][0]
                cursor.execute(
                    "UPDATE scheduled_items SET status='cancelled' WHERE id=? AND status='pending'",
                    (item_id,)
                )
                affected = cursor.rowcount

            conn.commit()

        if affected == 0:
            return ToolResult.err(
                f"Item {item_id} not found or already fired/cancelled.",
                code="not-found",
                hint="call schedule with action=list to see pending items.",
            )

        logger.info(f"{LOG_PREFIX} Cancelled {item_id}")

        record: dict[str, object] = {"id": item_id}
        try:
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, item_type, message, due_at, recurrence, status FROM scheduled_items WHERE id = ?",
                    (item_id,)
                )
                row = cursor.fetchone()
                if row:
                    record = _serialise_item_row({
                        "id": row[0],
                        "item_type": row[1],
                        "message": row[2],
                        "due_at": row[3],
                        "recurrence": row[4],
                        "status": row[5],
                    })
        except Exception as fetch_err:
            logger.debug(f"{LOG_PREFIX} Could not fetch cancelled row (non-fatal): {fetch_err}")

        return ToolResult.ok(
            {"status": "success", "action_performed": "cancel", "record": record},
            action="cancel",
        )

    except Exception as e:
        logger.exception(f"{LOG_PREFIX} Cancel failed: {e}")
        return ToolResult.err(f"Cancel failed: {e}", code="cancel-failed")


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


def _fetch_same_day_items(new_record: dict[str, object]) -> list[dict[str, object]]:
    """Returns an empty list on any failure — the rich-media card is the only
    consumer and the missing section degrades gracefully on the client.
    """
    new_id = new_record.get("id")
    new_due_utc_str = new_record.get("due_at_utc")
    if not new_id or not new_due_utc_str:
        return []

    try:
        from services.database_service import get_shared_db_service
        from services.time_formatter_service import TimeFormatterService
        from services.time_utils import parse_utc
        from services.locale_service import to_local

        new_due_utc = parse_utc(cast(datetime | str, new_due_utc_str))
        if new_due_utc == datetime.min.replace(tzinfo=timezone.utc):
            return []

        local_dt = to_local(new_due_utc)
        start_local = local_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        start_utc = start_local.astimezone(timezone.utc)
        end_utc = start_utc + timedelta(days=1)

        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, message, due_at, recurrence
                FROM scheduled_items
                WHERE status = 'pending'
                  AND hidden = 0
                  AND id != ?
                  AND due_at >= ?
                  AND due_at < ?
                ORDER BY due_at ASC
            """, (new_id, start_utc.strftime("%Y-%m-%d %H:%M:%S"), end_utc.strftime("%Y-%m-%d %H:%M:%S")))
            rows = cursor.fetchall()

        items = []
        for item_id, message, due_at, recurrence in rows:
            local_due = TimeFormatterService.local(due_at, fmt=_LOCAL_ISO_FMT) or str(due_at)
            items.append({
                "id": item_id,
                "message": message,
                "due_at": local_due,
                "recurrence": recurrence,
            })
        return items
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} same-day fetch failed: {e}")
        return []
