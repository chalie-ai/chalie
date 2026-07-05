"""``BudgetCappedAbility`` — per-turn call-budget enforced from the DB."""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult


class BudgetCappedAbility(Ability, ABC):
    """Mixin giving an ability a per-turn call budget derived from the DB trail."""

    #: Maximum number of successful calls allowed per processor turn.
    BUDGET_CAP: ClassVar[int] = 0

    def budget_exceeded(self) -> ToolResult | None:
        """Derive this turn's call count from persisted ``tool_calls`` rows."""
        proc = self.mp
        if proc is None:
            return None
        channel = getattr(getattr(proc, "config", None), "channel", None)
        turn_id = getattr(proc, "turn_id", None)
        if channel is None or turn_id is None:
            return None

        from models.tool_call import ToolCall  # noqa: PLC0415
        count = sum(
            1
            for r in ToolCall.by_turn(channel, turn_id)
            # An empty result is this call's own in-flight row, opened by the
            # dispatcher before run() — exclude it so the cap counts only the
            # completed calls that came before, not the one asking.
            if r.tool_name == self.get_name() and r.result
        )
        if count >= self.BUDGET_CAP:
            return ToolResult.ok(
                {
                    "saved": 0,
                    "skipped": 1,
                    "note": (
                        f"per-turn cap of {self.BUDGET_CAP} reached; "
                        "nothing was stored"
                    ),
                },
                capped=True,
            )
        return None
