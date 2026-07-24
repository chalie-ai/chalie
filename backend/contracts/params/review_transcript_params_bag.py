"""ReviewTranscriptParamsBag — the typed input contract of the
``review_transcript`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.review_window_params_bag import ReviewWindowParamsBag

# The buffer contract; the ability's parameter schema quotes these same values.
DEFAULT_BUFFER_MINUTES = 5
MIN_BUFFER_MINUTES = 1
MAX_BUFFER_MINUTES = 30


@dataclass(frozen=True, slots=True)
class ReviewTranscriptParamsBag(ReviewWindowParamsBag):
    """``buffer_minutes`` — half-window in minutes, default 5, clamped to 1–30.
    ``include_subagent_transcripts`` — when true, subagent-channel rows are
    included alongside the user channel."""

    buffer_minutes: int
    include_subagent_transcripts: bool

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        date_time = cls.require_str(params, Keys.date_time)
        if isinstance(date_time, ToolResult):
            return date_time
        buffer_minutes = cls.clamp_int(
            params,
            Keys.buffer_minutes,
            default=DEFAULT_BUFFER_MINUTES,
            lo=MIN_BUFFER_MINUTES,
            hi=MAX_BUFFER_MINUTES,
        )
        if isinstance(buffer_minutes, ToolResult):
            return buffer_minutes
        return cls(
            date_time=date_time,
            buffer_minutes=buffer_minutes,
            include_subagent_transcripts=cls.flag(params, Keys.include_subagent_transcripts),
        )
