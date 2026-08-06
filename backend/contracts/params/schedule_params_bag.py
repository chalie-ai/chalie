"""ScheduleParamsBag — the typed input contract of the ``schedule`` ability.

The multi-action shape: the router reads ``action`` once and fans out to the
per-action leaf; ``run()`` narrows back to a leaf via ``isinstance``. ``ScheduleUpdateParams`` SUBCLASSES
``ScheduleCreateParams``: update is a compose (cancel + create), so its leaf
carries every create field plus the target ``item_id`` — one validation path
for both actions, and an update bag IS a create bag wherever the create
handler needs one.

The create-side validation lives here in full — message shape, ``start_at``
parsing, crontab-field normalisation, ``validate_cron`` — because an illegal
cron or an unparseable timestamp is a bad PARAMETER, not a runtime failure.
Building the bag at the seam therefore rejects an illegal replacement BEFORE
``run()`` can cancel the row being updated: the update path's
validate-before-cancel invariant, held structurally. This is the first bag
importing from ``services`` — ``validate_cron`` stays the one source of the
crontab shape rule, and ``parse_local`` / ``utc_now`` define what a legal
``start_at`` means; duplicating either here would drift. All three modules
are leaf utilities (stdlib + each other only), so no import cycle is possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from services.cron_schedule import validate_cron
from services.locale_service import parse_local
from services.time_utils import utc_now

_ACTIONS = ("create", "list", "search", "cancel", "update", "enable", "disable")


class ScheduleParamsBag(ParamBag):
    """Router and boundary type: ``ScheduleAbility.run()`` is annotated with
    this class, but every instance that reaches it is one of the leaves below."""

    __slots__ = ()

    @staticmethod
    def _message(params: dict[str, object]) -> str | ToolResult:
        """The create/update prompt text: required, non-blank, at most 1000 chars."""
        raw = params.get(Keys.message)
        message = raw.strip() if isinstance(raw, str) else ""
        if not message:
            return ToolResult.err(
                "message is required for create.",
                code="missing-message",
                hint="pass the prompt text in 'message'.",
            )
        if len(message) > 1000:
            return ToolResult.err(
                "message exceeds 1000 characters.",
                code="message-too-long",
                hint="shorten the prompt text to 1000 characters or fewer.",
            )
        return message

    @staticmethod
    def _start_at(params: dict[str, object]) -> datetime | ToolResult:
        """``start_at`` is a plain LOCAL wall-clock ISO string — the model copies it
        verbatim from the World State's ``local_time`` telemetry and never converts
        timezones itself. Optional; defaults to local now when omitted."""
        raw = ParamBag.str_default(params, Keys.start_at, default="")
        if isinstance(raw, ToolResult):
            return raw
        text = raw.strip()
        if not text:
            return utc_now()
        try:
            return parse_local(text)
        except (ValueError, TypeError) as parse_err:
            return ToolResult.err(
                f"Could not understand start_at {text!r}: {parse_err}",
                code="invalid-time",
                hint=(
                    "pass start_at as a local wall-clock ISO timestamp, e.g. "
                    "'2026-03-20T09:00:00' — copy it straight from the World "
                    "State's current local_time."
                ),
            )

    @staticmethod
    def _cron_field(params: dict[str, object], key: str) -> str:
        """Read one crontab field (``minute``/``hour``/``day``/``month``/``weekday``)
        as a string expression; an omitted, null, or blank field means ``*`` (every).

        A weak model may emit the field as a JSON number (``5``) or a string
        (``"*/5"``); both are normalised to the trimmed string form here — a
        whole-number float like ``5.0`` is folded to ``"5"`` so it never stringifies
        as ``"5.0"``. Crontab-shape validity (range, malformed token) is enforced
        afterwards by ``validate_cron``, the single source of truth for the shape.
        """
        raw = params.get(key)
        if raw is None:
            return "*"
        if isinstance(raw, float) and raw.is_integer():
            raw = int(raw)
        text = str(raw).strip()
        return text or "*"

    @staticmethod
    def _create_fields(
        params: dict[str, object],
    ) -> tuple[str, datetime, str, str, str, str, str] | ToolResult:
        """Validate + coerce the shared create/update inputs; returns the error
        ``ToolResult`` for the first illegal one. Pure — no DB access."""
        message = ScheduleParamsBag._message(params)
        if isinstance(message, ToolResult):
            return message
        start_at = ScheduleParamsBag._start_at(params)
        if isinstance(start_at, ToolResult):
            return start_at
        minute = ScheduleParamsBag._cron_field(params, Keys.minute)
        hour = ScheduleParamsBag._cron_field(params, Keys.hour)
        day = ScheduleParamsBag._cron_field(params, Keys.day)
        month = ScheduleParamsBag._cron_field(params, Keys.month)
        weekday = ScheduleParamsBag._cron_field(params, Keys.weekday)

        try:
            validate_cron(minute, hour, day, month, weekday)
        except ValueError as cron_err:
            return ToolResult.err(
                str(cron_err),
                code="invalid-cron",
                hint=(
                    "each field is a numeric crontab expression: '*' (every), a "
                    "number, a 'N-M' range, a '*/S' step, or a comma list — e.g. "
                    "minute='*/5' fires every 5 minutes; hour='9', minute='0' fires "
                    "at 09:00; weekday='1-5' restricts to weekdays."
                ),
            )

        return message, start_at, minute, hour, day, month, weekday

    @staticmethod
    def _target(params: dict[str, object], *, verb: str) -> tuple[str, str] | ToolResult:
        """The id-or-message target of a mutating action (cancel/enable/disable):
        ``item_id`` (exact) or ``message`` (fuzzy) — at least one must be
        non-blank; the one not given comes back as ``""``. ``verb`` names the
        operation in the error so cancel/enable/disable each read naturally."""
        raw_id = ParamBag.str_default(params, Keys.item_id, default="")
        if isinstance(raw_id, ToolResult):
            return raw_id
        raw_message = ParamBag.str_default(params, Keys.message, default="")
        if isinstance(raw_message, ToolResult):
            return raw_message
        item_id = raw_id.strip()
        message = raw_message.strip()
        if not item_id and not message:
            return ToolResult.err(
                f"item_id or message is required to {verb}.",
                code=f"{verb}-target-required",
                hint="pass item_id (exact) or message (fuzzy match).",
            )
        return item_id, message

    @classmethod
    def from_params(cls, params: dict[str, object]) -> ScheduleParamsBag | ToolResult:
        match params.get(Keys.action):
            case "create":
                return ScheduleCreateParams.from_params(params)
            case "list":
                return ScheduleListParams.from_params(params)
            case "search":
                return ScheduleSearchParams.from_params(params)
            case "cancel":
                return ScheduleCancelParams.from_params(params)
            case "update":
                return ScheduleUpdateParams.from_params(params)
            case "enable":
                return ScheduleEnableParams.from_params(params)
            case "disable":
                return ScheduleDisableParams.from_params(params)
            case unknown:
                return ToolResult.err(
                    f"Unknown scheduler action: {unknown!r}.",
                    code="unknown-action",
                    hint="choose one of the valid actions below.",
                    valid=_ACTIONS,
                )


@dataclass(frozen=True, slots=True)
class ScheduleCreateParams(ScheduleParamsBag):
    """The validated create inputs: ``message`` non-blank and at most 1000
    chars; ``start_at`` already parsed to a datetime (local now when omitted);
    the five crontab fields normalised string expressions that passed
    ``validate_cron`` together."""

    message: str
    start_at: datetime
    minute: str
    hour: str
    day: str
    month: str
    weekday: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        fields = ScheduleParamsBag._create_fields(params)
        if isinstance(fields, ToolResult):
            return fields
        message, start_at, minute, hour, day, month, weekday = fields
        return cls(
            message=message, start_at=start_at, minute=minute,
            hour=hour, day=day, month=month, weekday=weekday,
        )


@dataclass(frozen=True, slots=True)
class ScheduleUpdateParams(ScheduleCreateParams):
    """Every create field (update recreates the item from them) plus the
    ``item_id`` of the row being replaced. The create fields are validated
    FIRST, so an illegal replacement is reported before any target problem —
    the same precedence the handler had."""

    item_id: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        fields = ScheduleParamsBag._create_fields(params)
        if isinstance(fields, ToolResult):
            return fields
        message, start_at, minute, hour, day, month, weekday = fields
        item_id = cls.require_str(params, Keys.item_id)
        if isinstance(item_id, ToolResult):
            return item_id
        return cls(
            item_id=item_id,
            message=message,
            start_at=start_at,
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            weekday=weekday,
        )


@dataclass(frozen=True, slots=True)
class ScheduleListParams(ScheduleParamsBag):
    """List reads no parameters — the leaf exists so ``run()`` can narrow."""

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        return cls()


@dataclass(frozen=True, slots=True)
class ScheduleSearchParams(ScheduleParamsBag):
    """``query`` — the semantic search text (pre-gated present by
    ACTION_REQUIRED; the bag rejects the whitespace-only residue); ``limit`` —
    max matching items, default 5, clamped into [1, 50]."""

    query: str
    limit: int

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        query = cls.require_str(params, Keys.query)
        if isinstance(query, ToolResult):
            return query
        limit = cls.clamp_int(params, Keys.limit, default=5, lo=1, hi=50)
        if isinstance(limit, ToolResult):
            return limit
        return cls(query=query, limit=limit)


@dataclass(frozen=True, slots=True)
class ScheduleCancelParams(ScheduleParamsBag):
    """Target of the hard delete: ``item_id`` (exact) or ``message`` (fuzzy) —
    the bag guarantees at least one is non-blank."""

    item_id: str
    message: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        target = ScheduleParamsBag._target(params, verb="cancel")
        if isinstance(target, ToolResult):
            return target
        item_id, message = target
        return cls(item_id=item_id, message=message)


@dataclass(frozen=True, slots=True)
class ScheduleEnableParams(ScheduleParamsBag):
    """Target of the enable toggle — same id-or-message contract as cancel."""

    item_id: str
    message: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        target = ScheduleParamsBag._target(params, verb="enable")
        if isinstance(target, ToolResult):
            return target
        item_id, message = target
        return cls(item_id=item_id, message=message)


@dataclass(frozen=True, slots=True)
class ScheduleDisableParams(ScheduleParamsBag):
    """Target of the disable toggle — same id-or-message contract as cancel."""

    item_id: str
    message: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        target = ScheduleParamsBag._target(params, verb="disable")
        if isinstance(target, ToolResult):
            return target
        item_id, message = target
        return cls(item_id=item_id, message=message)
