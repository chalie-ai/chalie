"""
TimerAbility — Ephemeral countdown widget rendered as a rich-media card.

Unlike ScheduleAbility (which persists scheduled_items rows in SQLite and
fires events from a background worker), the timer is purely client-side:
the ability returns a payload describing ``{title, duration_seconds}``
and the frontend module computes the remaining time on each render from
the wall-clock anchor injected by ``RichMediaParser`` from the tool_calls
row's ``created_at``. No DB, no worker, no scheduler.

The wall-clock anchor (``started_at``) is deliberately NOT in the JSON
the LLM sees: it is added server-side at parse time so subsequent ACT
iterations cannot misinterpret a literal timestamp as something they
need to reason about, and the model never has to fill it.

Rich-media rendering:
  A successful timer returns ``ToolResult.ok(payload, rich=payload)`` — the
  ``{title, duration_seconds}`` dict is both the structured body the model
  reads AND the card payload. The dispatcher (``ToolDispatcher._render``) owns
  ordinal assignment + the span-tag instruction and injects the card ONLY when
  the invoking channel broadcasts to the user. ``started_at`` is grafted onto
  the card payload later, at parse time, by ``enrich_rich_payload``.
"""

from datetime import datetime, timezone
from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult
from services.time_utils import parse_utc

_PARSE_UTC_SENTINEL = datetime.min.replace(tzinfo=timezone.utc)

_MAX_DURATION_SECONDS = 24 * 60 * 60  # 24 hours
_MIN_DURATION_SECONDS = 1


class TimerAbility(Ability):
    SYSTEM = True

    # title + duration_seconds are both required; presence is enforced by the
    # dispatcher pre-gate (ACTION_REQUIRED) BEFORE run(). Key "" — action-less.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "": ("title", "duration_seconds"),
    }

    def get_name(self) -> str:
        return "timer"

    def get_summary(self) -> str:
        return "Start a live countdown timer with a title — renders an in-chat card with pause, stop, and an alarm when it ends."

    def get_examples(self) -> list[str]:
        return [
            "start a 25 minute focus timer",
            "set a 10 minute timer for the pasta",
            "ring me in 5 minutes",
            "kick off a 90 second breath hold",
            "start a 1 hour deep work block",
            "remind me in 15 minutes the laundry is ready",
        ]

    def get_search_tooltip(self) -> str:
        return "countdown timer"

    _PARAMETERS: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short label for the timer (max 80 chars). E.g. 'Focus block', 'Pasta', 'Breath hold'.",
            },
            "duration_seconds": {
                "type": "integer",
                "description": "Total countdown length in seconds. Must be between 1 and 86400 (24 hours).",
                "minimum": _MIN_DURATION_SECONDS,
                "maximum": _MAX_DURATION_SECONDS,
            },
        },
        "required": ["title", "duration_seconds"],
    }

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    def run(self, params: dict) -> ToolResult:
        title = (params.get("title") or "").strip()
        duration_seconds = params.get("duration_seconds")

        # bool is an int subclass — exclude it so a literal True/False is rejected
        # rather than coerced into a 1-second / 0-second timer.
        if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, int) \
                or duration_seconds < _MIN_DURATION_SECONDS or duration_seconds > _MAX_DURATION_SECONDS:
            return ToolResult.err(
                f"duration_seconds must be an integer between {_MIN_DURATION_SECONDS} and {_MAX_DURATION_SECONDS}",
                code="invalid-duration",
                hint=f"pass an integer number of seconds from {_MIN_DURATION_SECONDS} to {_MAX_DURATION_SECONDS}.",
            )

        if len(title) > 80:
            title = title[:80]

        payload = {
            "title": title,
            "duration_seconds": duration_seconds,
        }
        # body == rich: the model reads the same dict the FE card renders. The
        # dispatcher injects the ordinal + span instruction only on a
        # user-broadcast turn; started_at is grafted later by enrich_rich_payload.
        return ToolResult.ok(payload, rich=payload)

    @classmethod
    def enrich_rich_payload(cls, payload: dict, row: dict) -> dict:
        """Inject the wall-clock anchor from the tool_calls row's ``created_at``.

        ``started_at`` is intentionally absent from the LLM-visible JSON; the FE
        needs it to compute the countdown so the parser grafts it on at render
        time. ``parse_utc`` returns a ``datetime.min`` sentinel on garbage rather
        than raising — that sentinel must be rejected so the FE falls through to
        its "Invalid timer payload" guard rather than rendering a year-0001
        countdown that instantly fires the alarm.
        """
        created_at = row.get("created_at")
        if not created_at:
            return payload
        parsed = parse_utc(created_at)
        if parsed == _PARSE_UTC_SENTINEL:
            return payload
        return {**payload, "started_at": parsed.isoformat()}
