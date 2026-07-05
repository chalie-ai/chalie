"""
ScheduleAbility — Native innate skill for reminders and scheduled tasks.

Backed by SQLite (scheduled_items table). Provides create, list, search, cancel, and update actions.
``update`` is a thin compose: it cancels the target ``item_id`` then creates a fresh item from the
remaining params — no bespoke UPDATE path to keep in sync with create's validation.
All DB access via the ``Database`` gateway.

Every user schedule created here is ``item_type='prompt'`` — it fires as a
conversational turn. The scheduler no longer models a notification/prompt
choice or a keyword ``recurrence`` string: a schedule is a day-of-month / hour
/ minute crontab (``cron_dom`` / ``cron_hour`` / ``cron_minute``, NULL = "every")
anchored to a ``start_at`` activation floor. ``start_at`` is a plain LOCAL
wall-clock ISO string (optional — defaults to local now); the model is
instructed to copy it straight from the World State's ``local_time`` telemetry
and never compute a UTC offset. ``validate_cron``/``next_due_from``
(``services.cron_schedule``) own the crontab math; this ability only calls
them and surfaces ``ValueError`` as a clear user/LLM-facing error.

Every action returns a :class:`ToolResult` built only via ``ok`` / ``err``. The
rich scheduler card travels via ``ToolResult(rich=…)``; the dispatcher owns the
ordinal + the single span instruction and injects the card only when the
invoking channel broadcasts to the user. This ability never formats a wire
envelope.
"""

import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import ClassVar, cast

from abilities._ability import Ability
from abilities._params import Keys
from abilities._result import ToolResult
from services.cron_schedule import next_due_from, validate_cron
from services.database import Database
from services.locale_service import format_date, local_now, parse_local, to_local
from services.time_utils import utc_now

logger = logging.getLogger(__name__)
LOG_PREFIX = "[SCHEDULER SKILL]"

# ISO 8601 with offset, used by format_date() to render start_at/due_at in the user's timezone.
_LOCAL_ISO_FMT = "%Y-%m-%dT%H:%M:%S%z"


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
        "id", "item_type", "message", "start_at", "due_at",
        "cron_dom", "cron_hour", "cron_minute", "status",
    ),
) -> list[dict[str, object]]:
    """This is the single query path for all callers — the schedule ability,
    the calendar ability, and anything else that reads scheduled_items.
    """
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

    conn = Database.conn()
    rows = conn.execute(query, params).fetchall()

    return [dict(zip(columns, row)) for row in rows]


_ACTIONS = ("create", "list", "search", "cancel", "update", "enable", "disable")


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
        "enable": (),
        "disable": (),
    }

    def get_name(self) -> str:
        return "schedule"

    def get_summary(self) -> str:
        return (
            "Create, list, cancel, or enable/disable persistent reminders and recurring "
            "prompts at a specific calendar date/time, with optional destination for departure "
            "reminders. Disable pauses a recurring prompt without deleting it; enable resumes it. "
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
                "enum": ["create", "list", "search", "cancel", "update", "enable", "disable"],
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
                    "Optional for cancel/enable/disable: fuzzy content match when item_id is unknown."
                ),
            },
            Keys.start_at: {
                "type": "string",
                "description": (
                    "Optional for create — the Start Time: the earliest local wall-clock "
                    "moment this schedule may activate. Omit it to start from right now. "
                    "When you do pass it, COPY IT VERBATIM from the current 'local_time' "
                    "value in the World State block you are given each turn — never "
                    "compute a UTC offset or convert timezones yourself, always work in "
                    "the user's own local wall-clock time. Example: '2026-03-20T09:00:00'."
                ),
            },
            Keys.day: {
                "type": "integer",
                "description": (
                    "Optional for create: the day-of-month (1-31) this reminder fires "
                    "on, in the user's LOCAL wall-clock time. Omit it (or pass null) to "
                    "mean 'every day' — required only for a monthly schedule. "
                    "Constraint: you can only set 'every' (omit) on a field that is "
                    "COARSER than every field you DO fix — so if you fix 'day' you must "
                    "also fix hour and minute (you cannot have a fixed day with an "
                    "'every' hour or minute)."
                ),
            },
            Keys.hour: {
                "type": "integer",
                "description": (
                    "Optional for create: the hour (0-23) this reminder fires at, in "
                    "the user's LOCAL wall-clock time. Omit it (or pass null) to mean "
                    "'every hour'. Constraint: if you fix 'hour' you must also fix "
                    "minute (you cannot have a fixed hour with an 'every' minute) — "
                    "e.g. day: every, hour: 3, minute: 0 means 'every day at 03:00'."
                ),
            },
            Keys.minute: {
                "type": "integer",
                "description": (
                    "Optional for create: the minute (0-59) this reminder fires at, in "
                    "the user's LOCAL wall-clock time. Omit it (or pass null) to mean "
                    "'every minute' — only valid when hour and day are ALSO omitted. "
                    "You cannot set minute to 'every' while hour is fixed."
                ),
            },
            Keys.item_id: {
                "type": "string",
                "description": (
                    "Optional for cancel: exact ID returned at create time. Prefer this when known. "
                    "Required for update: the existing item to replace — pass the new message/"
                    "start_at/day/hour/minute alongside it and the old item is cancelled and "
                    "recreated with the new values."
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
                    "and arrival-based (e.g. 'remind me when I get home')."
                ),
            },
        },
        "required": [Keys.action],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: dict[str, object]) -> ToolResult:
        action = (cast(str, params.get(Keys.action)) or "list").lower()

        if action == "create":
            channel = getattr(getattr(self.mp, "config", None), "channel", "") or ""
            return _create(channel, params)
        if action == "list":
            return _list(params)
        if action == "search":
            return _search(self, params, self.mp)
        if action == "cancel":
            return _cancel(params)
        if action == "enable":
            return _set_enabled(params, enabled=True)
        if action == "disable":
            return _set_enabled(params, enabled=False)
        if action == "update":
            # Compose, don't duplicate: retire the target then recreate from the
            # same params. Cancelling first clears create's pending-dedup guard so
            # a same-message time change isn't swallowed as a duplicate.
            #
            # Order matters. Validate the replacement BEFORE cancelling so an
            # illegal cron can't destroy the existing schedule (validation is
            # pure — no DB writes). Then require the cancel to LAND before
            # recreating, so a bad item_id surfaces the error instead of silently
            # creating a duplicate under the guise of an update.
            _, err = _parse_create_params(params)
            if err is not None:
                return err
            cancel_result = _cancel(params)
            if cancel_result.status == "error":
                return cancel_result
            channel = getattr(getattr(self.mp, "config", None), "channel", "") or ""
            return _create(channel, params)

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
        from models.data_graph import DataGraph
        from services.data_graph_service import KIND_PLACE
        rows = [r.to_dict() for r in DataGraph.live(KIND_PLACE).get()]
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


def _resolve_start_at(params: dict[str, object]) -> tuple[datetime, ToolResult | None]:
    """``start_at`` is a plain LOCAL wall-clock ISO string — the model copies it
    verbatim from the World State's ``local_time`` telemetry and never converts
    timezones itself. Optional; defaults to local now when omitted.
    """
    start_at_str = (cast(str, params.get(Keys.start_at)) or "").strip()
    if not start_at_str:
        return utc_now(), None

    try:
        return parse_local(start_at_str), None
    except (ValueError, TypeError) as parse_err:
        return utc_now(), ToolResult.err(
            f"Could not understand start_at {start_at_str!r}: {parse_err}",
            code="invalid-time",
            hint=(
                "pass start_at as a local wall-clock ISO timestamp, e.g. "
                "'2026-03-20T09:00:00' — copy it straight from the World "
                "State's current local_time."
            ),
        )


def _coerce_cron_field(
    params: dict[str, object], key: str, label: str
) -> tuple[int | None, ToolResult | None]:
    """Read one of ``day``/``hour``/``minute`` as ``int | None`` (``None`` = every).

    Range/every-prefix validity is enforced afterwards by ``validate_cron`` —
    this only coerces the raw value to an int (or rejects a non-numeric one).
    """
    raw = params.get(key)
    if raw is None or raw == "":
        return None, None
    if isinstance(raw, float) and not raw.is_integer():
        # int(3.5) would silently truncate to 3 and reschedule; reject it loudly.
        return None, ToolResult.err(
            f"{label} must be a whole number, got {raw!r}.",
            code="invalid-cron",
            hint=f"omit {key} (or pass null) to mean 'every {label}'.",
        )
    try:
        return int(cast(int, raw)), None
    except (TypeError, ValueError):
        return None, ToolResult.err(
            f"{label} must be an integer, got {raw!r}.",
            code="invalid-cron",
            hint=f"omit {key} (or pass null) to mean 'every {label}'.",
        )


def _format_record(
    item_id: str,
    message: str,
    start_at: datetime,
    due_at: datetime,
    dom: int | None,
    hour: int | None,
    minute: int | None,
) -> dict[str, object]:
    """The user/LLM-facing shape for a scheduled item — ``item_type`` is internal
    only and never surfaces here."""
    return {
        "id": item_id,
        "message": message,
        "start_at": format_date(start_at, fmt=_LOCAL_ISO_FMT, for_ui=True),
        "due_at": format_date(due_at, fmt=_LOCAL_ISO_FMT, for_ui=True),
        "day": dom,
        "hour": hour,
        "minute": minute,
    }


def _parse_create_params(
    params: dict[str, object],
) -> tuple[tuple[str, datetime, int | None, int | None, int | None] | None, ToolResult | None]:
    """Validate + coerce the shared create/update inputs WITHOUT touching the DB.

    Returns ``((message, start_at, dom, hour, minute), None)`` when every field is
    legal, or ``(None, error)`` on the first invalid one. Pure — the ``update``
    path relies on this to reject an illegal replacement *before* cancelling the
    existing schedule (see :meth:`ScheduleAbility.run`).
    """
    message, err = _validate_message(params)
    if err:
        return None, err
    start_at, err = _resolve_start_at(params)
    if err:
        return None, err
    dom, err = _coerce_cron_field(params, Keys.day, "day")
    if err:
        return None, err
    hour, err = _coerce_cron_field(params, Keys.hour, "hour")
    if err:
        return None, err
    minute, err = _coerce_cron_field(params, Keys.minute, "minute")
    if err:
        return None, err

    try:
        validate_cron(dom, hour, minute)
    except ValueError as cron_err:
        return None, ToolResult.err(
            str(cron_err),
            code="invalid-cron",
            hint=(
                "'every' (omit/null) is only allowed on a field COARSER than "
                "every fixed field — e.g. day: every, hour: 3, minute: 0 means "
                "'every day at 03:00', but minute cannot be 'every' while hour "
                "is fixed."
            ),
        )

    return (message, start_at, dom, hour, minute), None


def _create(channel: str, params: dict[str, object]) -> ToolResult:
    try:
        logger.debug(
            f"{LOG_PREFIX} _create called — message={params.get(Keys.message, '')!r:.80}, "
            f"start_at={params.get(Keys.start_at, '')!r}, day={params.get(Keys.day)!r}, "
            f"hour={params.get(Keys.hour)!r}, minute={params.get(Keys.minute)!r}"
        )

        parsed, err = _parse_create_params(params)
        if err is not None:
            return err
        if parsed is None:
            # Unreachable: _parse_create_params returns a tuple whenever err is
            # None. Kept as a loud belt-and-braces guard rather than an assert.
            return ToolResult.err("Create failed: parameters vanished", code="create-failed")
        message, start_at, dom, hour, minute = parsed

        due_at = next_due_from(start_at, dom, hour, minute)[0]

        item_id = uuid.uuid4().hex[:8]

        conn = Database.conn()
        cursor = conn.cursor()

        # Atomic dedup: INSERT only when no pending row with the same message
        # was created in the last 60 seconds.  A single statement eliminates
        # the SELECT-then-INSERT race window that caused duplicate rows when
        # the model invoked schedule(...) twice in the same turn.
        cursor.execute("""
            INSERT INTO scheduled_items
              (id, item_type, message, start_at, due_at, cron_dom, cron_hour, cron_minute,
               status, channel, created_by_session, created_at, group_id)
            SELECT ?, 'prompt', ?, ?, ?, ?, ?, ?, 'pending', ?, ?, datetime('now'), ?
            WHERE NOT EXISTS (
                SELECT 1 FROM scheduled_items
                WHERE status = 'pending'
                  AND message = ?
                  AND created_at > datetime('now', '-60 seconds')
            )
        """, (
            item_id, message, start_at.isoformat(), due_at.isoformat(), dom, hour, minute,
            channel, None,
            item_id,  # group_id = own id (root of new series)
            message,  # WHERE NOT EXISTS param
        ))

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
            record = _format_record(existing_id, message, start_at, due_at, dom, hour, minute)
            record["note"] = "already_existed"
            return _create_result(record, item_id=existing_id, due_at_utc=due_at)

        destination = (cast(str, params.get(Keys.destination_location)) or "").strip()
        if destination:
            dest_meta = _build_destination_metadata(destination)
            try:
                Database.conn().execute(
                    "UPDATE scheduled_items SET metadata=? WHERE id=?",
                    (json.dumps(dest_meta), item_id),
                )
            except Exception as meta_err:
                logger.warning(f"{LOG_PREFIX} Failed to write destination metadata (non-fatal): {meta_err}")

        try:
            from services.scheduler_service import embed_scheduled_item
            embed_scheduled_item(item_id, message)
        except Exception as emb_err:
            logger.warning(f"{LOG_PREFIX} Embedding failed (non-fatal): {emb_err}")

        logger.info(f"{LOG_PREFIX} Created prompt: {item_id}")
        record = _format_record(item_id, message, start_at, due_at, dom, hour, minute)
        return _create_result(record, item_id=item_id, due_at_utc=due_at)

    except Exception as e:
        logger.exception(f"{LOG_PREFIX} Create failed: {e}")
        return ToolResult.err(f"Create failed: {e}", code="create-failed")


def _create_result(record: dict[str, object], *, item_id: str, due_at_utc: datetime) -> ToolResult:
    """The dispatcher injects the ordinal + span instruction only when the
    invoking channel broadcasts to the user."""
    body: dict[str, object] = {"status": "success", "action_performed": "create", "record": record}
    card: dict[str, object] = {"action_performed": "create", "record": record}
    same_day = _fetch_same_day_items(item_id, due_at_utc)
    if same_day:
        card["same_day_items"] = same_day
    return ToolResult.ok(body, rich=card, action="create")


def _search(ability: "ScheduleAbility", params: dict[str, object], mp: object = None) -> ToolResult:
    query = (cast(str, params.get(Keys.query)) or "").strip()
    limit = ability.param(params, Keys.limit, default=5, clamp=(1, 50))

    try:
        from services.embedding_service import get_embedding_service
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

        conn = Database.conn()
        cursor = conn.execute(
            """
            SELECT s.id, s.message, s.start_at, s.due_at,
                   s.cron_dom, s.cron_hour, s.cron_minute, s.status,
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
    """The user/LLM-facing shape for a scheduled item row — ``item_type`` is
    internal only and never surfaces here (drop it if present on *row*)."""
    return {
        "id": row.get("id"),
        "message": row.get("message"),
        "start_at": format_date(cast("datetime | str | None", row.get("start_at")), fmt=_LOCAL_ISO_FMT, for_ui=True),
        "due_at": format_date(cast("datetime | str | None", row.get("due_at")), fmt=_LOCAL_ISO_FMT, for_ui=True),
        "day": row.get("cron_dom"),
        "hour": row.get("cron_hour"),
        "minute": row.get("cron_minute"),
        "status": row.get("status"),
    }


def _resolve_time_range(time_range: str) -> tuple[datetime | None, datetime | None, str]:
    now_utc = utc_now()
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


def _resolve_pending_target(params: dict[str, object], *, verb: str) -> tuple[str | None, ToolResult | None]:
    """Resolve the pending item a mutating action targets.

    Accepts ``item_id`` (exact) or ``message`` (fuzzy ``LIKE`` match, hidden
    rows excluded). Returns ``(item_id, None)`` on a unique hit, or
    ``(None, error)`` when neither was given, nothing matched, or the fuzzy
    match was ambiguous. ``verb`` names the operation in the error text so
    cancel/enable/disable each read naturally. One resolver for every
    id-or-message action (Law 9)."""
    item_id = (cast(str, params.get(Keys.item_id)) or "").strip()
    message_query = (cast(str, params.get(Keys.message)) or "").strip()

    if not item_id and not message_query:
        return None, ToolResult.err(
            f"item_id or message is required to {verb}.",
            code=f"{verb}-target-required",
            hint="pass item_id (exact) or message (fuzzy match).",
        )
    if item_id:
        return item_id, None

    pattern = f"%{message_query}%"
    matches = Database.conn().execute(
        "SELECT id, message FROM scheduled_items "
        "WHERE status='pending' AND hidden = 0 AND message LIKE ? ORDER BY due_at ASC",
        (pattern,),
    ).fetchall()

    if not matches:
        return None, ToolResult.err(
            f"No pending reminder matching {message_query!r} found.",
            code="not-found",
            hint="call schedule with action=list to see what exists.",
        )
    if len(matches) > 1:
        descriptions = ", ".join(f"'{r[1][:40]}' (id:{r[0]})" for r in matches)
        return None, ToolResult.err(
            f"Multiple pending reminders match {message_query!r}: {descriptions}.",
            code="ambiguous-match",
            hint="pass item_id to target the specific one.",
        )
    return matches[0][0], None


def _fetch_item_record(item_id: str) -> dict[str, object]:
    """Serialise a single item row for a tool result; ``{"id": item_id}`` on any
    failure (the record is advisory — never fatal to the mutating action)."""
    try:
        row = Database.conn().execute(
            "SELECT id, message, start_at, due_at, cron_dom, cron_hour, cron_minute, status "
            "FROM scheduled_items WHERE id = ?",
            (item_id,),
        ).fetchone()
        if row:
            return _serialise_item_row({
                "id": row[0], "message": row[1], "start_at": row[2], "due_at": row[3],
                "cron_dom": row[4], "cron_hour": row[5], "cron_minute": row[6], "status": row[7],
            })
    except Exception as fetch_err:
        logger.debug(f"{LOG_PREFIX} Could not fetch row {item_id} (non-fatal): {fetch_err}")
    return {"id": item_id}


def _cancel(params: dict[str, object]) -> ToolResult:
    try:
        item_id, err = _resolve_pending_target(params, verb="cancel")
        if err is not None:
            return err

        with Database.transaction() as conn:
            affected = conn.execute(
                "UPDATE scheduled_items SET status='cancelled' WHERE id=? AND status='pending'",
                (item_id,),
            ).rowcount

        if affected == 0:
            return ToolResult.err(
                f"Item {item_id} not found or already fired/cancelled.",
                code="not-found",
                hint="call schedule with action=list to see pending items.",
            )

        logger.info(f"{LOG_PREFIX} Cancelled {item_id}")
        return ToolResult.ok(
            {"status": "success", "action_performed": "cancel", "record": _fetch_item_record(cast(str, item_id))},
            action="cancel",
        )

    except Exception as e:
        logger.exception(f"{LOG_PREFIX} Cancel failed: {e}")
        return ToolResult.err(f"Cancel failed: {e}", code="cancel-failed")


def _set_enabled(params: dict[str, object], *, enabled: bool) -> ToolResult:
    """Enable or disable a pending series. Disabling makes the poller skip it;
    enabling recomputes ``due_at`` forward from now (via ``next_due_from``, which
    floors its anchor) so the series resumes at the NEXT occurrence rather than
    firing the one missed while disabled — the approved skip-the-catch-up rule."""
    verb = "enable" if enabled else "disable"
    try:
        item_id, err = _resolve_pending_target(params, verb=verb)
        if err is not None:
            return err

        with Database.transaction() as conn:
            cursor = conn.cursor()
            if enabled:
                cron = cursor.execute(
                    "SELECT cron_dom, cron_hour, cron_minute FROM scheduled_items "
                    "WHERE id=? AND status='pending'",
                    (item_id,),
                ).fetchone()
                if cron is None:
                    affected = 0
                else:
                    due_at = next_due_from(utc_now(), cron[0], cron[1], cron[2])[0]
                    affected = cursor.execute(
                        "UPDATE scheduled_items SET enabled=1, due_at=? WHERE id=? AND status='pending'",
                        (due_at.isoformat(), item_id),
                    ).rowcount
            else:
                affected = cursor.execute(
                    "UPDATE scheduled_items SET enabled=0 WHERE id=? AND status='pending'",
                    (item_id,),
                ).rowcount

        if affected == 0:
            return ToolResult.err(
                f"Item {item_id} not found or not pending.",
                code="not-found",
                hint="call schedule with action=list to see pending items.",
            )

        logger.info(f"{LOG_PREFIX} {verb.capitalize()}d {item_id}")
        return ToolResult.ok(
            {"status": "success", "action_performed": verb, "record": _fetch_item_record(cast(str, item_id))},
            action=verb,
        )

    except Exception as e:
        logger.exception(f"{LOG_PREFIX} {verb.capitalize()} failed: {e}")
        return ToolResult.err(f"{verb.capitalize()} failed: {e}", code=f"{verb}-failed")


def _fetch_same_day_items(exclude_id: str, due_at_utc: datetime) -> list[dict[str, object]]:
    """Returns an empty list on any failure — the rich-media card is the only
    consumer and the missing section degrades gracefully on the client.
    """
    try:
        local_dt = to_local(due_at_utc)
        start_local = local_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        start_utc = start_local.astimezone(timezone.utc)
        end_utc = start_utc + timedelta(days=1)

        conn = Database.conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, message, due_at
            FROM scheduled_items
            WHERE status = 'pending'
              AND hidden = 0
              AND id != ?
              AND due_at >= ?
              AND due_at < ?
            ORDER BY due_at ASC
        """, (exclude_id, start_utc.strftime("%Y-%m-%d %H:%M:%S"), end_utc.strftime("%Y-%m-%d %H:%M:%S")))
        rows = cursor.fetchall()

        items = []
        for item_id, message, due_at in rows:
            items.append({
                "id": item_id,
                "message": message,
                "due_at": format_date(due_at, fmt=_LOCAL_ISO_FMT, for_ui=True) or str(due_at),
            })
        return items
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} same-day fetch failed: {e}")
        return []
