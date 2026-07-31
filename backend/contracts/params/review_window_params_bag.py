"""ReviewWindowParamsBag — the shared typed input contract of the review tools.

``review_tool_calls`` and ``review_transcript`` are near-twins built on
``ReviewWindowAbility``: both anchor on a single required ISO ``date_time``.
``review_tool_calls`` uses this bag as-is (its ±5-minute window is frozen);
``review_transcript`` extends it with a clamped buffer and a channel toggle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class ReviewWindowParamsBag(ParamBag):
    """``date_time`` — ISO timestamp anchoring the review window, required."""

    date_time: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        date_time = cls.require_str(params, Keys.date_time)
        if isinstance(date_time, ToolResult):
            return date_time
        return cls(date_time=date_time)
