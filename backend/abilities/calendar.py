"""
CalendarAbility — List, view, create, update, and delete calendar events.

Every action — reads (``list_events``, ``get_event``) and writes
(``create_event``, ``update_event``, ``delete_event``) — is served LIVE from the
connected CalDAV server. There is no persisted calendar mirror: the CalDAV →
``scheduled_items`` mirror was decommissioned and calendar reads are now
re-homed onto direct CalDAV queries. Recurring events carry an RFC-5545
``recurrence`` (RRULE) so one-shot and recurring reminders are both stored as
real calendar events.

All five actions delegate to MailCapability's CalDAV handler via the shared
:class:`CapabilityAbility` base, which owns capability loading, the
not-connected / unknown-action / handler-unavailable errors, handler dispatch,
and result wrapping. uid-addressed calls reach the server directly; a
title-addressed call resolves to a unique uid by a live substring match over the
CalDAV window (handled server-side), so ``update_event`` / ``delete_event`` /
``get_event`` accept a ``title`` when the uid is unknown.

Two write-path safeguards live here, ahead of the base's connected gate so they
fire loudly regardless of CalDAV state:

* **datetime validation (data-corruption fix):** ``dtstart`` / ``dtend`` (writes)
  and ``date_from`` / ``date_to`` (list window) are resolved through ``_parse_dt``
  (natural language in the user's timezone OR ISO 8601) into a clean UTC ISO
  string written onto the outbound handler params. An unparseable value returns
  ``code=invalid-time`` with example forms and NOTHING is written — the old path
  ran the raw string through ``parse_utc``, which returns the ``datetime.min``
  (year-0001) sentinel and persisted it to the CalDAV server.
* **target presence:** ``update_event`` / ``delete_event`` require either a
  ``uid`` or a ``title`` before the request leaves this ability; the CalDAV
  handler does the live title→uid resolution (and reports ambiguity).

Every action returns a :class:`ToolResult` built only via ``ok`` / ``err``. This
ability never formats a wire envelope.
"""

import logging
from datetime import datetime, timezone
from typing import ClassVar, cast

from abilities._capability import CapabilityAbility
from configs.enums.param_key import Keys
from abilities._result import ToolResult
from contracts.params.capability_params_bag import CapabilityParamsBag

logger = logging.getLogger(__name__)
LOG_PREFIX = "[CALENDAR ABILITY]"

# Example datetime forms surfaced in the invalid-time hint so a weak model can
# self-correct without re-reading the schema.
_DT_EXAMPLES = "'tomorrow 3pm', 'friday 9am', or ISO 8601 like '2026-03-20T15:00:00+02:00'"

# Every action delegates to the live CalDAV handler via the CapabilityAbility
# base. Reads and writes are split only so run() can pick which pre-flight
# validation applies (writes normalise dtstart/dtend + require a target; reads
# normalise the date window).
_READ_ACTIONS = ("list_events", "get_event")
_WRITE_ACTIONS = ("create_event", "update_event", "delete_event")
_ALL_ACTIONS = (*_READ_ACTIONS, *_WRITE_ACTIONS)

# update_event / delete_event address an event by uid OR title; the CalDAV
# handler resolves a title to a unique uid live. This ability only enforces that
# a target is present before the request leaves the tool.
_TARGET_ADDRESSED_WRITES = ("update_event", "delete_event")


class CalendarAbility(CapabilityAbility):
    DISCOVERABLE: ClassVar[bool] = False  # pim-delegate-exclusive; pinned on PimConfig only
    CAPABILITY_KEY: ClassVar[str] = "mail"
    DEFAULT_ACTION: ClassVar[str] = "list_events"
    NOT_CONNECTED_HINT: ClassVar[str] = (
        "Configure the mail integration in the Brain dashboard."
    )
    # Maps the model-facing action onto the CalDAV handler name (identical here —
    # the base uses it to look up the handler and to build the valid= ladder for
    # unknown-action errors). All five are live-CalDAV handlers.
    ACTION_HANDLERS: ClassVar[dict[str, str]] = {
        "list_events": "list_events",
        "get_event": "get_event",
        "create_event": "create_event",
        "update_event": "update_event",
        "delete_event": "delete_event",
    }

    # Pre-gated by the dispatcher BEFORE run(): create_event requires summary +
    # both endpoints. list_events / get_event validate their either/or targets in
    # the handler. An unknown action → one unknown-action error whose valid= names
    # all five real actions.
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

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.action: {
                "type": "string",
                "enum": list(_ALL_ACTIONS),
                "description": (
                    "The calendar action to perform. "
                    "list_events — list events in a window (defaults to now..+7 days). "
                    "get_event — fetch one event by uid or title. "
                    "create_event — add a new event (needs summary, dtstart, dtend; pass "
                    "recurrence for a repeating event). "
                    "update_event — change fields of an event addressed by uid or title. "
                    "delete_event — remove an event addressed by uid or title."
                ),
            },
            Keys.date_from: {
                "type": "string",
                "description": (
                    "list_events: window lower bound. Natural language ('today', "
                    "'monday') or ISO date/datetime. Defaults to now."
                ),
            },
            Keys.date_to: {
                "type": "string",
                "description": (
                    "list_events: window upper bound. Natural language ('sunday', "
                    "'next week') or ISO date/datetime. Defaults to 7 days after the "
                    "lower bound."
                ),
            },
            Keys.limit: {
                "type": "integer",
                "description": (
                    "list_events: maximum number of events to return (1–200, default 50)."
                ),
            },
            Keys.uid: {
                "type": "string",
                "description": (
                    "get_event / update_event / delete_event: the event's CalDAV UID. "
                    "Preferred when known; otherwise pass title."
                ),
            },
            Keys.title_: {
                "type": "string",
                "description": (
                    "get_event / update_event / delete_event: event title to match "
                    "when the uid is unknown. Resolved live to a unique event; an "
                    "ambiguous match asks for a uid."
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
            Keys.recurrence: {
                "type": "string",
                "description": (
                    "create_event / update_event: RFC-5545 recurrence rule (RRULE) for a "
                    "repeating event, e.g. 'FREQ=DAILY', 'FREQ=WEEKLY;BYDAY=MO,WE,FR', "
                    "'FREQ=MONTHLY'. Omit for a one-off event."
                ),
            },
        },
        "required": [Keys.action],
    }

    # ── Entry point — pre-flight validation, then live dispatch via the base ───

    def run(self, params: CapabilityParamsBag) -> ToolResult:
        action = self._resolve_action(params)
        handler_params = dict(params.extra)

        # list_events: resolve the natural-language window to clean ISO before it
        # reaches the handler's parse_utc (which would sentinel an unparseable value).
        if action == "list_events":
            err = _normalise_datetimes(handler_params, (Keys.date_from, Keys.date_to))
            if err is not None:
                return err
            return self._dispatch(action, handler_params)

        # get_event validates its uid-or-title either/or in the handler.
        if action == "get_event":
            return self._dispatch(action, handler_params)

        # Write actions: validate datetimes + require an addressable target BEFORE
        # the base's connected gate, so a malformed dtstart errors loudly even when
        # CalDAV is not wired up — and a clean UTC ISO string is what reaches the
        # server. Live title→uid resolution happens in the handler.
        err = _normalise_datetimes(handler_params, (Keys.dtstart, Keys.dtend))
        if err is not None:
            return err

        if action in _TARGET_ADDRESSED_WRITES:
            err = _require_target(handler_params)
            if err is not None:
                return err

        return self._dispatch(action, handler_params)


# ---------------------------------------------------------------------------
# Datetime validation — the data-corruption fix
# ---------------------------------------------------------------------------

def _parse_dt(value: str) -> datetime | None:
    """Returns ``None`` when the string is unparseable — the caller maps ``None`` to
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
    return cast(datetime, parsed).astimezone(timezone.utc)


def _normalise_datetimes(
    params: dict[str, object], fields: tuple[str, ...]
) -> ToolResult | None:
    """Resolve any supplied ``fields`` (write endpoints or a list window) to clean
    UTC ISO strings on ``params``. Returns an ``invalid-time`` ToolResult (and
    leaves params untouched) when a value is present but unparseable — so the
    CalDAV handler only ever sees a valid ISO string, never the datetime.min
    sentinel."""
    for field in fields:
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
# Target addressing — uid OR title (title resolved live by the CalDAV handler)
# ---------------------------------------------------------------------------

def _require_target(params: dict[str, object]) -> ToolResult | None:
    """update_event / delete_event must name an event by ``uid`` or ``title``.
    Both pass through to the handler, which resolves a title to a unique uid over
    the live CalDAV window (and reports ambiguity). Returns ``None`` when a target
    is present."""
    if (cast(str, params.get(Keys.uid)) or "").strip():
        return None
    if (cast(str, params.get(Keys.title_)) or "").strip():
        return None
    return ToolResult.err(
        "uid or title is required to address the event.",
        code="missing-target",
        hint="pass the event's uid, or a title to match.",
    )


