"""
ScheduleAbility — Native innate skill for recurring scheduled tasks.

A schedule is a crontab-driven prompt that wakes Chalie to *act* on a recurrence
(produce a briefing, poll something, run a routine); it does NOT notify the user.
Reminding the user of an appointment or a one-off errand at a time is a calendar
event, owned by the ``pim`` delegate — not this ability.

Backed by SQLite (scheduled_items table). Provides create, list, search, cancel, and update actions.
``update`` is a thin compose: it cancels the target ``item_id`` then creates a fresh item from the
remaining params — no bespoke UPDATE path to keep in sync with create's validation.
All DB access via the :class:`~models.scheduled_item.ScheduledItem` model; this ability owns
orchestration, embedding generation, and response shaping only — input validation lives in the
:class:`~contracts.params.schedule_params_bag.ScheduleParamsBag` router, and no SQL of its own.

``scheduled_items`` is a prompt-only, crontab table now: one row per schedule, forever
(``id`` INTEGER PRIMARY KEY AUTOINCREMENT, also the thread's ``turn_id`` on the ``'schedule'``
channel). There is no ``item_type``/notification-vs-prompt choice, no stored ``due_at``/
``status``/``group_id`` — a stateless poller wakes every wall-clock minute and fires any enabled
row whose ``start_at`` floor has passed and whose five crontab fields
(``cron_minute``/``cron_hour``/``cron_dom``/``cron_month``/``cron_dow``, ``*`` = every) match the
current LOCAL minute. Each field is a standard crontab expression — ``*``, ``5``, ``*/5``,
``0,15,30,45``, ``9-17`` (numeric only, no ``mon``/``jan`` names). Cancel is a hard ``DELETE``; a
cancelled id is never reissued (AUTOINCREMENT), so its thread can never be re-entered. ``start_at``
is a plain LOCAL wall-clock ISO string (optional — defaults to local now); the model is instructed
to copy it straight from the World State's ``local_time`` telemetry and never compute a UTC offset.
``validate_cron`` (``services.cron_schedule``) owns the crontab shape rule; the input bag calls it
at the seam and surfaces ``ValueError`` as ``code=invalid-cron``.

Every action returns a :class:`ToolResult` built only via ``ok`` / ``err``. The
rich scheduler card travels via ``ToolResult(rich=…)``; the dispatcher owns the
ordinal + the single span instruction and injects the card only when the
invoking channel broadcasts to the user. This ability never formats a wire
envelope.
"""

import logging
from datetime import datetime
from typing import ClassVar, cast

from abilities._ability import Ability
from configs.enums.param_key import Keys
from abilities._result import ToolResult
from contracts.params.param_bag import ParamBag
from contracts.params.schedule_params_bag import (
    ScheduleCancelParams,
    ScheduleCreateParams,
    ScheduleDisableParams,
    ScheduleEnableParams,
    ScheduleListParams,
    ScheduleParamsBag,
    ScheduleSearchParams,
    ScheduleUpdateParams,
)
from models.scheduled_item import ScheduledItem
from services.database import Database
from services.locale_service import LocaleService
from configs.enums.ability_category import AbilityCategory

logger = logging.getLogger(__name__)
LOG_PREFIX = "[SCHEDULER SKILL]"

# ISO 8601 with offset, used by format_date() to render start_at in the user's timezone.
_LOCAL_ISO_FMT = "%Y-%m-%dT%H:%M:%S%z"

_COLS: tuple[str, ...] = (
    "id", "message", "start_at",
    "cron_minute", "cron_hour", "cron_dom", "cron_month", "cron_dow",
    "enabled", "channel", "created_by_session", "created_at",
)

_ACTIONS = ("create", "list", "search", "cancel", "update", "enable", "disable")


class ScheduleAbility(Ability[ScheduleParamsBag]):
    # Pre-gated by the dispatcher BEFORE run(): create requires a 'message';
    # search requires a 'query'; list/cancel require nothing at this layer
    # (cancel needs item_id OR message — an either/or the map cannot express,
    # so the bag validates it). An unknown action → one unknown-action error
    # whose valid= names these keys; a known action missing a required param →
    # one missing-params error. run() never sees a malformed call.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "create": (Keys.message,),
        "list": (),
        "search": (Keys.query,),
        "cancel": (),
        "update": (Keys.item_id, Keys.message),
        "enable": (),
        "disable": (),
    }
    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("scheduler", "recurring task", "scheduled tasks", "cron")

    # The typed input contract: the dispatch seam builds the bag via
    # ScheduleParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = ScheduleParamsBag
    NAME: ClassVar[str] = "schedule"
    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.PRODUCTIVITY

    def get_summary(self) -> str:
        from abilities.pim import PimAbility  # noqa: PLC0415
        from abilities.timer import TimerAbility  # noqa: PLC0415

        return (
            "Create, list, cancel, or enable/disable recurring scheduled tasks — a "
            "crontab-driven prompt that wakes Chalie to act on a schedule (produce a "
            "recurring briefing, poll something, run a routine). Supports full crontab-"
            "style recurrence — a fixed time, every N minutes/hours, specific weekdays, "
            "day-of-month, or any combination (e.g. 'every 5 minutes', 'every weekday "
            "at 9am', 'the 1st of each month'). Disable pauses a task without deleting "
            "it; enable resumes it. To remind the user of an appointment or a one-off "
            f"errand at a time, use the `{PimAbility.NAME}` tool (it creates a real calendar event) — "
            f"not `{self.NAME}`. For a short ephemeral countdown the user wants to watch "
            f"tick down on screen (focus blocks, kitchen timers), use the `{TimerAbility.NAME}` tool."
        )

    def get_examples(self) -> list[str]:
        return [
            "every weekday at 9am summarise my unread emails",
            "check the news every 2 hours during the day",
            "every Monday at 8am brief me on the week ahead",
            "on the 1st of each month generate a spending summary",
            "every evening at 9pm ask me how my day went",
            "run a mailbox check every 30 minutes",
            "what scheduled tasks do I have",
            "cancel the daily email summary",
        ]

    def get_search_tooltip(self) -> str:
        return "recurring scheduled tasks"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.action: {
                "type": "string",
                "enum": ["create", "list", "search", "cancel", "update", "enable", "disable"],
                "description": (
                    f"The scheduler action to perform. `{NAME}` runs a recurring "
                    "prompt that wakes Chalie to act — it does NOT notify the user. "
                    "To remind the user of an appointment or errand at a time, STOP "
                    "— use the `pim` tool (it creates a real calendar event) instead. "
                    "For a short ephemeral countdown (e.g. 'set a 5 minute timer', "
                    "'start a 25 minute focus block'), STOP — call the `timer` tool "
                    f"instead of `{NAME}`."
                ),
            },
            Keys.message: {
                "type": "string",
                "description": (
                    "Required for create: the prompt to run when the schedule fires "
                    "(max 1000 chars). Optional for cancel/enable/disable: fuzzy "
                    "content match when item_id is unknown."
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
            Keys.minute: {
                "type": "string",
                "description": (
                    "Optional for create: WHICH MINUTE of the hour this fires, as a "
                    "standard crontab field (0-59) in the user's LOCAL time. Default "
                    "'*' = every minute. Examples: '0' = on the hour, '30' = at :30, "
                    "'*/5' = every 5 minutes, '0,15,30,45' = every quarter hour, "
                    "'0-29' = the first half of each hour. Numeric only. For 'every 5 "
                    "minutes' set minute='*/5' and leave the rest '*'."
                ),
            },
            Keys.hour: {
                "type": "string",
                "description": (
                    "Optional for create: WHICH HOUR this fires, as a standard crontab "
                    "field (0-23, 24-hour clock) in the user's LOCAL time. Default '*' "
                    "= every hour. Examples: '9' = 9am, '17' = 5pm, '*/2' = every 2 "
                    "hours, '9-17' = every hour 9am-5pm. For a once-a-day task set "
                    "hour and minute (e.g. hour='8', minute='0' = 08:00 daily)."
                ),
            },
            Keys.day: {
                "type": "string",
                "description": (
                    "Optional for create: WHICH DAY-OF-MONTH this fires, as a standard "
                    "crontab field (1-31) in the user's LOCAL time. Default '*' = every "
                    "day. Examples: '1' = the 1st, '15' = the 15th, '1,15' = 1st and "
                    "15th. Note: a day number missing from a short month (e.g. '31') "
                    "simply never fires that month. Use 'weekday' instead for "
                    "'every Monday'-style schedules."
                ),
            },
            Keys.month: {
                "type": "string",
                "description": (
                    "Optional for create: WHICH MONTH this fires, as a standard crontab "
                    "field (1-12, 1=January) in the user's LOCAL time. Default '*' = "
                    "every month. Examples: '1' = January only, '*/3' = quarterly "
                    "(Jan/Apr/Jul/Oct), '6-8' = the summer months. Numeric only."
                ),
            },
            Keys.weekday: {
                "type": "string",
                "description": (
                    "Optional for create: WHICH DAY-OF-WEEK this fires, as a standard "
                    "crontab field (0-6, 0=Sunday, 1=Monday, … 6=Saturday; 7 also = "
                    "Sunday) in the user's LOCAL time. Default '*' = every day. "
                    "Examples: '1' = every Monday, '1-5' = weekdays, '0,6' = weekends, "
                    "'5' = every Friday. Combine with hour/minute: weekday='1', "
                    "hour='9', minute='0' = every Monday at 09:00. (If BOTH day and "
                    "weekday are set, the schedule fires on days matching EITHER — "
                    "standard crontab.)"
                ),
            },
            Keys.item_id: {
                "type": "string",
                "description": (
                    "Optional for cancel: exact ID returned at create time. Prefer this when known. "
                    "Required for update: the existing item to replace — pass the new message/"
                    "start_at/minute/hour/day/month/weekday alongside it and the old item is "
                    "cancelled and recreated with the new values."
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
        },
        "required": [Keys.action],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: ScheduleParamsBag) -> ToolResult:
        mp = self.mp
        if mp is None:
            raise RuntimeError("schedule.run() dispatched without a bound MessageProcessor")

        # Update SUBCLASSES Create (update = cancel + create over the same
        # validated fields), so it must be narrowed FIRST or the create branch
        # would swallow it.
        if isinstance(params, ScheduleUpdateParams):
            # Compose, don't duplicate: retire the target then recreate from
            # the same fields. The bag validated the replacement at the seam —
            # BEFORE anything here ran — so an illegal cron can never destroy
            # the existing schedule. The cancel must still LAND before
            # recreating, so a bad item_id surfaces the error instead of
            # silently creating a duplicate under the guise of an update.
            cancel_result = _cancel(params.item_id, "")
            if cancel_result.status == "error":
                return cancel_result
            return _create(mp.config.channel, params)
        if isinstance(params, ScheduleCreateParams):
            return _create(mp.config.channel, params)
        if isinstance(params, ScheduleListParams):
            return _list()
        if isinstance(params, ScheduleSearchParams):
            return _search(params, mp)
        if isinstance(params, ScheduleCancelParams):
            return _cancel(params.item_id, params.message)
        if isinstance(params, ScheduleEnableParams):
            return _set_enabled(params.item_id, params.message, enabled=True)
        if isinstance(params, ScheduleDisableParams):
            return _set_enabled(params.item_id, params.message, enabled=False)

        # Unreachable off the dispatch path (the router factory only yields the
        # leaves above); kept as a self-correcting belt-and-braces error for a
        # hand-built foreign subclass.
        return ToolResult.err(
            f"Unknown schedule params bag: {type(params).__name__}.",
            code="unknown-action",
            hint="choose one of the valid actions below.",
            valid=_ACTIONS,
        )


def _format_record(
    item_id: int,
    message: str,
    start_at: datetime,
    minute: str,
    hour: str,
    day: str,
    month: str,
    weekday: str,
) -> dict[str, object]:
    """The user/LLM-facing shape for a scheduled item — the five raw crontab
    field expressions verbatim (``day`` = day-of-month, ``weekday`` =
    day-of-week)."""
    return {
        "id": item_id,
        "message": message,
        "start_at": LocaleService.format_date(start_at, fmt=_LOCAL_ISO_FMT, for_ui=True),
        "minute": minute,
        "hour": hour,
        "day": day,
        "month": month,
        "weekday": weekday,
    }


def _create(channel: str, params: ScheduleCreateParams) -> ToolResult:
    try:
        logger.debug(
            f"{LOG_PREFIX} _create called — message={params.message!r:.80}, "
            f"start_at={params.start_at!r}, minute={params.minute!r}, "
            f"hour={params.hour!r}, day={params.day!r}, "
            f"month={params.month!r}, weekday={params.weekday!r}"
        )

        item = ScheduledItem.create(
            message=params.message,
            start_at=params.start_at.isoformat(),
            cron_minute=params.minute,
            cron_hour=params.hour,
            cron_dom=params.day,
            cron_month=params.month,
            cron_dow=params.weekday,
            enabled=1,
            channel=channel,
            created_by_session=None,
        )  # INSERT + 1-1 gist seed, atomic
        item_id = cast(int, item.id)

        try:
            from services.scheduler_service import embed_scheduled_item
            embed_scheduled_item(item_id, params.message)
        except Exception as emb_err:
            logger.warning(f"{LOG_PREFIX} Embedding failed (non-fatal): {emb_err}")

        logger.info(f"{LOG_PREFIX} Created prompt: {item_id}")
        record = _format_record(
            item_id, params.message, params.start_at,
            params.minute, params.hour, params.day, params.month, params.weekday,
        )
        return _create_result(record)

    except Exception as e:
        logger.exception(f"{LOG_PREFIX} Create failed: {e}")
        return ToolResult.err(f"Create failed: {e}", code="create-failed")


def _create_result(record: dict[str, object]) -> ToolResult:
    """The dispatcher injects the ordinal + span instruction only when the
    invoking channel broadcasts to the user."""
    body: dict[str, object] = {"status": "success", "action_performed": "create", "record": record}
    card: dict[str, object] = {"action_performed": "create", "record": record}
    return ToolResult.ok(body, rich=card, action="create")


def _search(params: ScheduleSearchParams, mp: object = None) -> ToolResult:
    try:
        from services.embedding_service import get_embedding_service
        from services.embedding_utils import pack_embedding

        emb = get_embedding_service().generate_embedding(params.query, mp=mp)
        if not emb:
            return ToolResult.no_results(action="search")

        blob = pack_embedding(emb)
        if blob is None:
            return ToolResult.no_results(action="search")

        rows = ScheduledItem.vector_search(blob, params.limit)

        records = []
        for row in rows:
            rec = _serialise_item_row(dict(row))
            rec["distance"] = float(row["distance"])
            records.append(rec)

        if not records:
            return ToolResult.no_results(action="search")

        return ToolResult.ok(
            {"status": "success", "action_performed": "search", "records": records},
            action="search", count=len(records),
        )

    except Exception as e:
        logger.exception(f"{LOG_PREFIX} Search failed: {e}")
        return ToolResult.err(f"Search failed: {e}", code="search-failed")


def _list() -> ToolResult:
    try:
        rows = ScheduledItem.by_start_at(_COLS)
        records = [_serialise_item_row(row) for row in rows]
        if not records:
            return ToolResult.no_results(action="list")
        return ToolResult.ok(
            {"status": "success", "action_performed": "list", "records": records},
            action="list", count=len(records),
        )

    except Exception as e:
        logger.exception(f"{LOG_PREFIX} List failed: {e}")
        return ToolResult.err(f"List failed: {e}", code="list-failed")


def _serialise_item_row(row: dict[str, object]) -> dict[str, object]:
    """The user/LLM-facing shape for a scheduled item row — the five raw crontab
    field expressions (``day`` = day-of-month, ``weekday`` = day-of-week)."""
    return {
        "id": row.get("id"),
        "message": row.get("message"),
        "start_at": LocaleService.format_date(cast("datetime | str | None", row.get("start_at")), fmt=_LOCAL_ISO_FMT, for_ui=True),
        "minute": row.get("cron_minute"),
        "hour": row.get("cron_hour"),
        "day": row.get("cron_dom"),
        "month": row.get("cron_month"),
        "weekday": row.get("cron_dow"),
    }


def _resolve_pending_target(item_id: str, message_query: str) -> tuple[object, ToolResult | None]:
    """Resolve the item a mutating action targets.

    ``item_id`` (exact) wins; otherwise ``message_query`` is a fuzzy ``LIKE``
    match. The bag guarantees at least one is non-blank — a call with neither
    was rejected at the seam as ``<verb>-target-required``. Returns
    ``(item_id, None)`` on a unique hit, or ``(None, error)`` when nothing
    matched or the fuzzy match was ambiguous. One resolver for every
    id-or-message action (Law 9)."""
    if item_id:
        return item_id, None

    pattern = f"%{message_query}%"
    matches = ScheduledItem.search_by_message(pattern)

    if not matches:
        return None, ToolResult.err(
            f"No scheduled task matching {message_query!r} found.",
            code="not-found",
            hint="call schedule with action=list to see what exists.",
        )
    if len(matches) > 1:
        descriptions = ", ".join(f"'{cast(str, m['message'])[:40]}' (id:{m['id']})" for m in matches)
        return None, ToolResult.err(
            f"Multiple scheduled tasks match {message_query!r}: {descriptions}.",
            code="ambiguous-match",
            hint="pass item_id to target the specific one.",
        )
    return matches[0]["id"], None


def _fetch_item_record(item_id: object) -> dict[str, object]:
    """Serialise a single item row for a tool result; ``{"id": item_id}`` on any
    failure (the record is advisory — never fatal to the mutating action)."""
    try:
        item = ScheduledItem.get(item_id)
        if item is not None:
            return _serialise_item_row(item.to_dict())
    except Exception as fetch_err:
        logger.debug(f"{LOG_PREFIX} Could not fetch row {item_id} (non-fatal): {fetch_err}")
    return {"id": item_id}


def _cancel(target_id: str, message_query: str) -> ToolResult:
    try:
        item_id, err = _resolve_pending_target(target_id, message_query)
        if err is not None:
            return err

        with Database.transaction():
            affected = ScheduledItem.filter("id", item_id).delete()
            # rowid == id under INTEGER PRIMARY KEY, so the vec row keyed on
            # rowid is the same id. Drop it in the same transaction to avoid
            # orphaning the embedding when the schedule is hard-deleted.
            ScheduledItem.delete_embedding(cast(int, item_id))

        if affected == 0:
            return ToolResult.err(
                f"Item {item_id} not found.",
                code="not-found",
                hint="call schedule with action=list to see items.",
            )

        logger.info(f"{LOG_PREFIX} Cancelled {item_id}")
        return ToolResult.ok(
            {"status": "success", "action_performed": "cancel", "record": {"id": item_id}},
            action="cancel",
        )

    except Exception as e:
        logger.exception(f"{LOG_PREFIX} Cancel failed: {e}")
        return ToolResult.err(f"Cancel failed: {e}", code="cancel-failed")


def _set_enabled(target_id: str, message_query: str, *, enabled: bool) -> ToolResult:
    """Enable or disable a schedule. Disabling sets ``enabled=0`` so the poller
    skips it; enabling flips it back to 1. There is no ``due_at`` to recompute —
    the poller matches purely off the row's ``start_at`` floor and cron fields
    against the current wall-clock minute, so a re-enabled series simply
    resumes matching from the next minute a cron field lines up. A past
    ``start_at`` never re-floors to now; it already satisfies the floor check."""
    verb = "enable" if enabled else "disable"
    try:
        item_id, err = _resolve_pending_target(target_id, message_query)
        if err is not None:
            return err

        with Database.transaction():
            affected = ScheduledItem.filter("id", item_id).update(enabled=1 if enabled else 0)

        if affected == 0:
            return ToolResult.err(
                f"Item {item_id} not found.",
                code="not-found",
                hint="call schedule with action=list to see items.",
            )

        logger.info(f"{LOG_PREFIX} {verb.capitalize()}d {item_id}")
        return ToolResult.ok(
            {"status": "success", "action_performed": verb, "record": _fetch_item_record(item_id)},
            action=verb,
        )

    except Exception as e:
        logger.exception(f"{LOG_PREFIX} {verb.capitalize()} failed: {e}")
        return ToolResult.err(f"{verb.capitalize()} failed: {e}", code=f"{verb}-failed")
