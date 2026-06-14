"""
CalendarAbility — List, view, create, update, and delete calendar events.

Read operations (``list_events``, ``get_event``) query ``scheduled_items`` via the
shared ``query_items`` engine in the schedule ability — the scheduler owns the
single SQL path for that table. Calendar events are the rows the CalDAV ingest
mirrors there (``source='mail'``, ``item_type='event'``, ``hidden=1``,
``external_uid='caldav:<uid>'``).

Write operations (``create_event``, ``update_event``, ``delete_event``) delegate
to MailCapability's CalDAV handler via the shared :class:`CapabilityAbility`
base, which owns capability loading, the not-connected / unknown-action /
handler-unavailable errors, handler dispatch, and result wrapping.

Two write-path safeguards live here, ahead of the base's connected gate so they
fire loudly regardless of CalDAV state:

* **datetime validation (data-corruption fix):** ``dtstart`` / ``dtend`` are
  resolved through ``_parse_dt`` (natural language in the user's timezone OR ISO
  8601) into a clean UTC ISO string written back onto ``params``. An unparseable
  value returns ``code=invalid-time`` with example forms and NOTHING is written —
  the old path ran the raw string through ``parse_utc``, which returns the
  ``datetime.min`` (year-0001) sentinel and persisted it to the CalDAV server.
* **fuzzy addressing:** ``update_event`` / ``delete_event`` / ``get_event`` accept
  a ``title`` that is resolved to a CalDAV uid by fuzzy match; >1 match returns
  ``code=ambiguous-match`` listing the candidate uids — never a silent first-hit.

Every action returns a :class:`ToolResult` built only via ``ok`` / ``err``. The
rich calendar card travels via ``ToolResult(rich=…)``; the dispatcher owns the
ordinal + the single span instruction and injects the card only when the invoking
channel broadcasts to the user. This ability never formats a wire envelope.
"""

import logging
from datetime import timedelta, timezone
from typing import ClassVar

from abilities._capability import CapabilityAbility
from abilities._params import Keys
from abilities._result import ToolResult
from services.time_utils import utc_now
from utils.data_utils import parse_json_column

logger = logging.getLogger(__name__)
LOG_PREFIX = "[CALENDAR ABILITY]"

# Example datetime forms surfaced in the invalid-time hint so a weak model can
# self-correct without re-reading the schema.
_DT_EXAMPLES = "'tomorrow 3pm', 'friday 9am', or ISO 8601 like '2026-03-20T15:00:00+02:00'"

# Default list window when the model passes no bounds — the schema advertises it,
# so the code now honours it instead of returning everything.
_DEFAULT_WINDOW_DAYS = 7

# Read actions answered directly from scheduled_items; write actions delegate to
# the CalDAV handler via the CapabilityAbility base.
_READ_ACTIONS = ("list_events", "get_event")
_WRITE_ACTIONS = ("create_event", "update_event", "delete_event")
_ALL_ACTIONS = (*_READ_ACTIONS, *_WRITE_ACTIONS)


class CalendarAbility(CapabilityAbility):
    CAPABILITY_KEY: ClassVar[str] = "mail"
    DEFAULT_ACTION: ClassVar[str] = "list_events"
    NOT_CONNECTED_HINT: ClassVar[str] = (
        "Configure the mail integration in the Brain dashboard."
    )
    # Maps the model-facing write action onto the CalDAV handler name (identical
    # here — the base uses it to look up the handler and to build the valid= ladder
    # for unknown-action errors).
    ACTION_HANDLERS: ClassVar[dict[str, str]] = {
        "create_event": "create_event",
        "update_event": "update_event",
        "delete_event": "delete_event",
    }

    # Pre-gated by the dispatcher BEFORE run(): create_event requires summary +
    # both endpoints. list_events / the uid-or-title reads validate their either/or
    # targets in run() (a constraint the map cannot express). An unknown action →
    # one unknown-action error whose valid= names all five real actions.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "list_events": (),
        "get_event": (),
        "create_event": (Keys.summary, Keys.dtstart, Keys.dtend),
        "update_event": (),
        "delete_event": (),
    }

    def get_name(self) -> str:
        return "calendar"

    def get_summary(self) -> str:
        return (
            "List, view, create, update, and delete calendar events on the connected "
            "CalDAV account. Available when the user asks about meetings, appointments, "
            "or wants to add, move, or cancel a calendar event."
        )

    def get_examples(self) -> list[str]:
        return [
            "what's on my calendar today",
            "show my meetings this week",
            "move the project meeting to 3pm tomorrow",
            "add a dentist appointment friday at 9am",
            "cancel my lunch with Alex",
            "what time is my next meeting",
            "get details for the standup",
            "delete the budget review event",
        ]

    def get_search_tooltip(self) -> str:
        return "calendar events"

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    _PARAMETERS: ClassVar[dict] = {
        "type": "object",
        "properties": {
            Keys.action: {
                "type": "string",
                "enum": list(_ALL_ACTIONS),
                "description": (
                    "The calendar action to perform. "
                    "list_events — list events within a date range. "
                    "get_event — fetch one event by uid or fuzzy title. "
                    "create_event — add a new event (needs summary, dtstart, dtend). "
                    "update_event — change fields of an event addressed by uid or title. "
                    "delete_event — remove an event addressed by uid or title."
                ),
            },
            Keys.date_from: {
                "type": "string",
                "description": (
                    "list_events: ISO date lower bound (YYYY-MM-DD). "
                    "Defaults to today when omitted."
                ),
            },
            Keys.date_to: {
                "type": "string",
                "description": (
                    "list_events: ISO date upper bound (YYYY-MM-DD). "
                    "Defaults to 7 days after date_from when omitted."
                ),
            },
            Keys.limit: {
                "type": "integer",
                "description": "list_events: maximum number of events to return (1–200, default 50).",
            },
            Keys.uid: {
                "type": "string",
                "description": (
                    "get_event / update_event / delete_event: the event's CalDAV UID. "
                    "Prefer this when known; otherwise pass title for a fuzzy match."
                ),
            },
            Keys.title: {
                "type": "string",
                "description": (
                    "get_event / update_event / delete_event: event title for a fuzzy "
                    "match when the uid is unknown. If more than one event matches you "
                    "get the candidate uids back to disambiguate."
                ),
            },
            Keys.summary: {
                "type": "string",
                "description": "create_event / update_event: the event title.",
            },
            Keys.dtstart: {
                "type": "string",
                "description": (
                    "create_event / update_event: start datetime. Natural language is "
                    "accepted and resolved in the user's timezone ('tomorrow 3pm', "
                    "'friday 9am'), OR ISO 8601 ('2026-03-20T15:00:00+02:00'). You do "
                    "NOT need to compute the UTC offset."
                ),
            },
            Keys.dtend: {
                "type": "string",
                "description": (
                    "create_event / update_event: end datetime, same formats as dtstart."
                ),
            },
        },
        "required": [Keys.action],
    }

    # ── Entry point — reads inline, writes through the base ────────────────────

    def run(self, params: dict) -> ToolResult:
        action = str(params.get(Keys.action, self.DEFAULT_ACTION)).lower()

        if action in _READ_ACTIONS:
            return _read_events(action, params)

        # Write actions: validate datetimes + resolve fuzzy title BEFORE the base's
        # connected gate, so a malformed dtstart errors loudly even when CalDAV is
        # not wired up — and a clean UTC ISO string is what ever reaches the server.
        err = _normalise_write_datetimes(params)
        if err is not None:
            return err

        if action in ("update_event", "delete_event"):
            err = _resolve_target_uid(params)
            if err is not None:
                return err

        return super().run(params)


# ---------------------------------------------------------------------------
# Datetime validation — the data-corruption fix
# ---------------------------------------------------------------------------

def _parse_dt(value: str):
    """Resolve a natural-language OR ISO 8601 datetime to a UTC datetime.

    Natural language ("tomorrow 3pm", "friday 9am") is resolved in the user's
    timezone (read from the telemetry heartbeat via ``locale_service``); ISO 8601
    with an offset is parsed too. Returns a tz-aware UTC ``datetime`` on success,
    or ``None`` when the string is unparseable — the caller maps ``None`` to
    ``code=invalid-time`` and NEVER persists the ``parse_utc`` ``datetime.min``
    sentinel.
    """
    import dateparser
    from services.locale_service import get_timezone_name, local_now

    parsed = dateparser.parse(
        value,
        settings={
            "TIMEZONE": get_timezone_name(),
            "TO_TIMEZONE": "UTC",
            "RETURN_AS_TIMEZONE_AWARE": True,
            "RELATIVE_BASE": local_now(),
            "PREFER_DATES_FROM": "future",
        },
    )
    if parsed is None:
        return None
    return parsed.astimezone(timezone.utc)


def _normalise_write_datetimes(params: dict) -> ToolResult | None:
    """Resolve any supplied ``dtstart`` / ``dtend`` to clean UTC ISO strings on
    ``params``. Returns an ``invalid-time`` ToolResult (and leaves params untouched)
    when a value is present but unparseable — so the CalDAV handler only ever sees
    a valid ISO string, never the datetime.min sentinel."""
    for field in (Keys.dtstart, Keys.dtend):
        raw = params.get(field)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        resolved = _parse_dt(text)
        if resolved is None:
            return ToolResult.err(
                f"Could not understand the {field} {text!r}.",
                code="invalid-time",
                hint=f"pass a time like {_DT_EXAMPLES}.",
            )
        params[field] = resolved.isoformat()
    return None


# ---------------------------------------------------------------------------
# Fuzzy addressing — uid OR title, with explicit disambiguation
# ---------------------------------------------------------------------------

def _resolve_target_uid(params: dict) -> ToolResult | None:
    """For uid-addressed write actions, fill ``params['uid']`` from a fuzzy
    ``title`` match when no uid was given. Returns an error ToolResult on a
    missing target, no match, or an ambiguous (>1) match; ``None`` on success."""
    if (params.get(Keys.uid) or "").strip():
        return None

    title = (params.get(Keys.title) or "").strip()
    if not title:
        return ToolResult.err(
            "uid or title is required to address the event.",
            code="missing-target",
            hint="pass the event's uid, or a title to fuzzy-match.",
        )

    matches = _match_events_by_title(title)
    if not matches:
        return ToolResult.err(
            f"No event matching title {title!r} was found.",
            code="not-found",
            hint="call calendar with action=list_events to see what exists.",
        )
    if len(matches) > 1:
        return _ambiguous_match(title, matches, "re-issue the action with the chosen uid.")

    params[Keys.uid] = matches[0]["uid"]
    return None


def _ambiguous_match(title: str, matches: list[dict], hint: str) -> ToolResult:
    """Build the disambiguation error: the candidate uids + titles go in the BODY
    (never a silent first-hit pick) so the model can re-issue with a chosen uid."""
    candidates = ", ".join(f"{m['title']!r} (uid:{m['uid']})" for m in matches)
    return ToolResult.err(
        f"Multiple events match title {title!r}: {candidates}. Pick one by uid.",
        code="ambiguous-match",
        hint=hint,
    )


def _match_events_by_title(title: str) -> list[dict]:
    """Return upcoming events whose title contains *title* (case-insensitive),
    each as the contract event dict. Reads through the shared scheduler query
    engine over the next default window so the match set is bounded."""
    from abilities.schedule import query_items

    rows = query_items(
        hidden=1, source="mail", item_type="event",
        date_from=utc_now().isoformat(),
        columns=_EVENT_COLUMNS,
    )
    needle = title.lower()
    return [ev for ev in (_format_event(r) for r in rows) if needle in ev["title"].lower()]


# ---------------------------------------------------------------------------
# Read helpers — query scheduled_items via schedule.query_items
# ---------------------------------------------------------------------------

_EVENT_COLUMNS = ("message", "due_at", "metadata", "external_uid")


def _read_events(action: str, params: dict) -> ToolResult:
    from abilities.schedule import query_items

    if action == "get_event":
        uid = (params.get(Keys.uid) or "").strip()
        title = (params.get(Keys.title) or "").strip()
        if not uid and not title:
            return ToolResult.err(
                "uid or title is required for get_event.",
                code="missing-target",
                hint="pass the event's uid, or a title to fuzzy-match.",
            )

        if uid:
            rows = query_items(
                hidden=1, source="mail", item_type="event",
                external_uid=f"caldav:{uid}",
                columns=_EVENT_COLUMNS, limit=1,
            )
            if not rows:
                return ToolResult.err(
                    f"Event not found (uid: {uid}).",
                    code="not-found",
                    hint="call calendar with action=list_events to see what exists.",
                )
            event = _format_event(rows[0])
        else:
            matches = _match_events_by_title(title)
            if not matches:
                return ToolResult.err(
                    f"No event matching title {title!r} was found.",
                    code="not-found",
                    hint="call calendar with action=list_events to see what exists.",
                )
            if len(matches) > 1:
                return _ambiguous_match(title, matches, "re-issue get_event with the chosen uid.")
            event = matches[0]

        body = {"action_performed": "get_event", "event": event}
        return ToolResult.ok(body, rich=dict(body), action="get_event")

    # list_events — honour the advertised default window (today → +7 days).
    limit = min(int(params.get(Keys.limit) or 50), 200)
    date_from, date_to = _resolve_window(params)

    rows = query_items(
        hidden=1, source="mail", item_type="event",
        date_from=date_from, date_to=date_to,
        limit=limit, columns=_EVENT_COLUMNS,
    )

    events = [_format_event(row) for row in rows]
    body = {"action_performed": "list_events", "events": events, "count": len(events)}
    return ToolResult.ok(body, rich=dict(body), action="list_events", count=len(events))


def _resolve_window(params: dict) -> tuple[str, str]:
    """Return the (date_from, date_to) ISO bounds for list_events, applying the
    advertised defaults (today → +7 days) when the model omits them."""
    date_from = (params.get(Keys.date_from) or "").strip()
    date_to = (params.get(Keys.date_to) or "").strip()

    if not date_from:
        start = utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
        date_from = start.isoformat()
    if not date_to:
        base = _parse_dt(date_from) or utc_now()
        date_to = (base + timedelta(days=_DEFAULT_WINDOW_DAYS)).isoformat()

    return date_from, date_to


def _format_event(row: dict) -> dict:
    meta = parse_json_column(row.get("metadata"))
    ext_uid = row.get("external_uid", "")
    uid = ext_uid.removeprefix("caldav:") if ext_uid else meta.get("uid", "")
    return {
        "uid": uid,
        "title": row.get("message", ""),
        "dtstart": row.get("due_at"),
        "dtend": meta.get("dtend"),
        "location": meta.get("location"),
        "attendees": meta.get("attendees", []),
        "all_day": meta.get("all_day", False),
        "calendar_name": meta.get("calendar_name"),
    }
