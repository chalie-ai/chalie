"""
CalendarAbility — List, view, create, update, and delete calendar events.

Read operations (``list_events``, ``get_event``) used to query ``scheduled_items``
via the shared ``query_items`` engine in the schedule ability, over rows the
CalDAV ingest mirrored there. That mirror has been decommissioned (TKT-1434):
``scheduled_items`` is now a prompt-only cron table with zero CalDAV columns
(no ``item_type``/``source``/``external_uid``/``hidden``/``metadata``/``due_at``).
Both read actions now return a clean, deliberate ``err(...)`` instead of
querying dropped columns — calendar reads are being re-homed to a live CalDAV
integration by a separate effort.

Write operations (``create_event``, ``update_event``, ``delete_event``) delegate
to MailCapability's CalDAV handler via the shared :class:`CapabilityAbility`
base, which owns capability loading, the not-connected / unknown-action /
handler-unavailable errors, handler dispatch, and result wrapping. uid-addressed
writes reach the live CalDAV server directly with zero ``scheduled_items``
coupling; title-addressed writes used the same decommissioned mirror to
fuzzy-resolve a uid and now return the same clean error asking for the uid
directly.

Two write-path safeguards live here, ahead of the base's connected gate so they
fire loudly regardless of CalDAV state:

* **datetime validation (data-corruption fix):** ``dtstart`` / ``dtend`` are
  resolved through ``_parse_dt`` (natural language in the user's timezone OR ISO
  8601) into a clean UTC ISO string written back onto ``params``. An unparseable
  value returns ``code=invalid-time`` with example forms and NOTHING is written —
  the old path ran the raw string through ``parse_utc``, which returns the
  ``datetime.min`` (year-0001) sentinel and persisted it to the CalDAV server.
* **fuzzy addressing:** ``update_event`` / ``delete_event`` / ``get_event`` used to
  accept a ``title`` resolved to a CalDAV uid by fuzzy match over the (now
  decommissioned) ``scheduled_items`` mirror. With that mirror gone, a call that
  supplies ``title`` without a ``uid`` gets the same clean migrating-away error;
  pass ``uid`` directly for writes in the meantime.

Every action returns a :class:`ToolResult` built only via ``ok`` / ``err``. The
rich calendar card travels via ``ToolResult(rich=…)``; the dispatcher owns the
ordinal + the single span instruction and injects the card only when the invoking
channel broadcasts to the user. This ability never formats a wire envelope.
"""

import logging
from datetime import datetime, timezone
from typing import ClassVar, cast

from abilities._capability import CapabilityAbility
from abilities._params import Keys
from abilities._result import ToolResult

logger = logging.getLogger(__name__)
LOG_PREFIX = "[CALENDAR ABILITY]"

# Example datetime forms surfaced in the invalid-time hint so a weak model can
# self-correct without re-reading the schema.
_DT_EXAMPLES = "'tomorrow 3pm', 'friday 9am', or ISO 8601 like '2026-03-20T15:00:00+02:00'"

# Read actions used to be answered directly from scheduled_items; that mirror is
# decommissioned (TKT-1434) so both now return a clean error. Write actions still
# delegate to the CalDAV handler via the CapabilityAbility base.
_READ_ACTIONS = ("list_events", "get_event")
_WRITE_ACTIONS = ("create_event", "update_event", "delete_event")
_ALL_ACTIONS = (*_READ_ACTIONS, *_WRITE_ACTIONS)

# Calendar reads (list_events/get_event) and fuzzy title→uid resolution both read
# the CalDAV→scheduled_items mirror, decommissioned in TKT-1434 (scheduled_items is
# now a prompt-only cron table with zero CalDAV columns). Both surfaces return this
# clean, deliberate error instead of referencing dropped columns.
_ERR_CALENDAR_READS_MIGRATING = (
    "Calendar reads are being migrated to the live CalDAV integration and are "
    "temporarily unavailable."
)


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
                    "list_events / get_event — TEMPORARILY UNAVAILABLE while calendar "
                    "reads migrate to a live CalDAV integration; both return a clean error. "
                    "create_event — add a new event (needs summary, dtstart, dtend). "
                    "update_event — change fields of an event addressed by uid. "
                    "delete_event — remove an event addressed by uid."
                ),
            },
            Keys.date_from: {
                "type": "string",
                "description": (
                    "list_events: ISO date lower bound (YYYY-MM-DD). Unused while "
                    "list_events is temporarily unavailable."
                ),
            },
            Keys.date_to: {
                "type": "string",
                "description": (
                    "list_events: ISO date upper bound (YYYY-MM-DD). Unused while "
                    "list_events is temporarily unavailable."
                ),
            },
            Keys.limit: {
                "type": "integer",
                "description": (
                    "list_events: maximum number of events to return (1–200, default 50). "
                    "Unused while list_events is temporarily unavailable."
                ),
            },
            Keys.uid: {
                "type": "string",
                "description": (
                    "get_event / update_event / delete_event: the event's CalDAV UID. "
                    "Required for update_event / delete_event — title-based fuzzy match is "
                    "temporarily unavailable, so uid must be passed directly."
                ),
            },
            Keys.title: {
                "type": "string",
                "description": (
                    "get_event / update_event / delete_event: event title for a fuzzy "
                    "uid match. TEMPORARILY UNAVAILABLE (returns a clean error) while "
                    "calendar reads migrate to a live CalDAV integration — pass uid instead."
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

    def run(self, params: dict[str, object]) -> ToolResult:
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


def _normalise_write_datetimes(params: dict[str, object]) -> ToolResult | None:
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

def _resolve_target_uid(params: dict[str, object]) -> ToolResult | None:
    """For uid-addressed write actions, a supplied ``uid`` passes straight
    through to the CalDAV handler with zero ``scheduled_items`` coupling.
    Fuzzy title-to-uid resolution used to read the CalDAV→scheduled_items
    mirror; that mirror is decommissioned (TKT-1434), so a ``title`` given
    without a ``uid`` now returns the same clean migrating-away error. Returns
    ``None`` on success (uid already present)."""
    if (cast(str, params.get(Keys.uid)) or "").strip():
        return None

    title = (cast(str, params.get(Keys.title)) or "").strip()
    if not title:
        return ToolResult.err(
            "uid or title is required to address the event.",
            code="missing-target",
            hint="pass the event's uid, or a title to fuzzy-match.",
        )

    return ToolResult.err(
        _ERR_CALENDAR_READS_MIGRATING,
        code="calendar-read-unavailable",
        hint="pass the event's uid directly until fuzzy title lookup returns.",
    )


# ---------------------------------------------------------------------------
# Read helpers — DECOMMISSIONED (TKT-1434)
#
# list_events / get_event used to query scheduled_items via schedule.query_items
# over rows the CalDAV ingest mirrored there (source='mail', item_type='event',
# hidden=1, external_uid='caldav:<uid>'). That mirror has been removed outright
# — scheduled_items is now a prompt-only cron table with zero CalDAV columns.
# Calendar reads are being re-homed to a live CalDAV integration by a separate
# effort; until then both actions return a clean, deliberate error instead of
# referencing dropped columns.
# ---------------------------------------------------------------------------


def _read_events(action: str, params: dict[str, object]) -> ToolResult:  # noqa: ARG001
    return ToolResult.err(
        _ERR_CALENDAR_READS_MIGRATING,
        code="calendar-read-unavailable",
        hint="calendar writes (create_event/update_event/delete_event by uid) still work.",
        action=action,
    )


