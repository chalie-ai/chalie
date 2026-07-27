"""
TimerAbility — Ephemeral countdown widget rendered as a rich-media card.

Unlike ScheduleAbility (which persists scheduled_items rows in SQLite and
fires events from a background worker), the timer is purely client-side:
the ability returns a payload describing ``{title, duration_seconds}``
and the frontend card computes the remaining time on each render from the
wall-clock anchor on its rich segment — ``segment["created_at"]``, which
``RichMediaParser`` normalises from the tool_calls row. No DB, no worker,
no scheduler.

The anchor is deliberately NOT in the JSON the LLM sees. It cannot be:
``DispatchService._render_rich`` REPLACES the model-visible tool body with
the card payload, so a payload field is a field the model reads — and a
literal timestamp is something a subsequent ACT iteration would try to
reason about. Carrying it as segment metadata keeps the model's view to
``{title, duration_seconds}`` while the card still gets its anchor.

Rich-media rendering:
  A successful timer returns ``ToolResult.ok(payload, rich=payload)`` — the
  ``{title, duration_seconds}`` dict is both the structured body the model
  reads AND the card payload. The dispatcher (``ToolDispatcher._render``) owns
  ordinal assignment + the span-tag instruction and injects the card ONLY when
  the invoking channel broadcasts to the user.
"""

from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from contracts.params.timer_params_bag import (
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
    TimerParamsBag,
)


class TimerAbility(Ability[TimerParamsBag]):
    SYSTEM = True

    # title + duration_seconds are both required; presence is enforced by the
    # dispatcher pre-gate (ACTION_REQUIRED) BEFORE run(). Key "" — action-less.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "": (Keys.title_, Keys.duration_seconds),
    }

    # The typed input contract: the dispatch seam builds the bag via
    # TimerParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = TimerParamsBag
    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("countdown", "set timer", "alarm")
    NAME: ClassVar[str] = "timer"

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

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.title_: {
                "type": "string",
                "description": "Short label for the timer (max 80 chars). E.g. 'Focus block', 'Pasta', 'Breath hold'.",
            },
            Keys.duration_seconds: {
                "type": "integer",
                "description": "Total countdown length in seconds. Must be between 1 and 86400 (24 hours).",
                "minimum": MIN_DURATION_SECONDS,
                "maximum": MAX_DURATION_SECONDS,
            },
        },
        "required": [Keys.title_, Keys.duration_seconds],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: TimerParamsBag) -> ToolResult:
        title = params.title
        duration_seconds = params.duration_seconds

        if len(title) > 80:
            title = title[:80]

        payload = {
            "title": title,
            "duration_seconds": duration_seconds,
        }
        # body == rich: the model reads the same dict the FE card renders. The
        # dispatcher injects the ordinal + span instruction only on a
        # user-broadcast turn. The countdown's wall-clock anchor is NOT in here —
        # it rides the rich segment as generic metadata (see the module docstring).
        return ToolResult.ok(payload, rich=payload)
