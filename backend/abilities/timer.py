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
  When an ordinal is supplied (``params['_rich_media_ordinal']``) the
  return value is a string ``<JSON>\\n\\n<rich-media instruction trailer>``.
  Without an ordinal — subagent calls, action-button calls, direct test
  calls — the raw dict is returned and no trailer is appended.
"""

import json
import logging
from datetime import datetime, timezone
from typing import ClassVar

from abilities._base import Ability
from services.time_utils import parse_utc

logger = logging.getLogger(__name__)

_PARSE_UTC_SENTINEL = datetime.min.replace(tzinfo=timezone.utc)

_RICH_MEDIA_INSTRUCTION = (
    "This tool supports rich-media rendering. You MUST present this result by "
    "wrapping your synthesis in <span id='{tag}'>your synthesis here</span>. "
    "The span will render as a live countdown card; without it, the user "
    "sees only plain text. Write a brief acknowledgement. Example: "
    "\"<span id='{tag}'>Started a 25-minute focus timer.</span>\""
)

_MAX_DURATION_SECONDS = 24 * 60 * 60  # 24 hours
_MIN_DURATION_SECONDS = 1


class TimerAbility(Ability):
    NAME = "timer"
    SUMMARY = "Start a live countdown timer with a title — renders an in-chat card with pause, stop, and an alarm when it ends."
    EXAMPLES = [
        "start a 25 minute focus timer",
        "set a 10 minute timer for the pasta",
        "ring me in 5 minutes",
        "kick off a 90 second breath hold",
        "start a 1 hour deep work block",
        "remind me in 15 minutes the laundry is ready",
    ]
    INPUT_SCHEMA: ClassVar[dict] = {
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
    TIMEOUT = 5

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict | str:
        ordinal = params.get("_rich_media_ordinal")
        title = (params.get("title") or "").strip()
        duration_seconds = params.get("duration_seconds")

        if not title:
            return {"error": "title is required"}
        if not isinstance(duration_seconds, int) or duration_seconds < _MIN_DURATION_SECONDS or duration_seconds > _MAX_DURATION_SECONDS:
            return {"error": f"duration_seconds must be an integer between {_MIN_DURATION_SECONDS} and {_MAX_DURATION_SECONDS}"}

        if len(title) > 80:
            title = title[:80]

        payload = {
            "title": title,
            "duration_seconds": duration_seconds,
        }

        if ordinal is None:
            return payload

        tag = f"timer_{ordinal}"
        instruction = _RICH_MEDIA_INSTRUCTION.format(tag=tag)
        return f"{json.dumps(payload)}\n\n{instruction}"

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
