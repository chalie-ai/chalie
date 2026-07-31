"""TimerParamsBag — the typed input contract of the ``timer`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag

# Bounds are owned here (the bag is the validator) and imported by the ability
# for its model-facing schema, so the two can never drift.
MIN_DURATION_SECONDS = 1
MAX_DURATION_SECONDS = 24 * 60 * 60  # 24 hours


@dataclass(frozen=True, slots=True)
class TimerParamsBag(ParamBag):
    """``title`` — short label for the timer (max 80 chars), required but
    empty/whitespace-only is accepted and stripped; ``duration_seconds`` — total
    countdown length in seconds, required and must be an integer in [1, 86400]."""

    title: str
    duration_seconds: int

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        title = params.get(Keys.title_)
        if not isinstance(title, str):
            return ToolResult.err(
                f"'{Keys.title_}' must be text.",
                code="invalid-param",
                hint=f"pass '{Keys.title_}' as a string.",
            )
        title = title.strip()

        duration_seconds = params.get(Keys.duration_seconds)
        # bool is an int subclass — exclude it so a literal True/False is rejected
        # rather than coerced into a 1-second / 0-second timer.
        if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, int) \
                or duration_seconds < MIN_DURATION_SECONDS or duration_seconds > MAX_DURATION_SECONDS:
            return ToolResult.err(
                f"duration_seconds must be an integer between {MIN_DURATION_SECONDS} and {MAX_DURATION_SECONDS}",
                code="invalid-duration",
                hint=f"pass an integer number of seconds from {MIN_DURATION_SECONDS} to {MAX_DURATION_SECONDS}.",
            )

        return cls(title=title, duration_seconds=duration_seconds)
