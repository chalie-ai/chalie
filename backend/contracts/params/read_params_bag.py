"""ReadParamsBag — the typed input contract of the ``read`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class ReadParamsBag(ParamBag):
    """``source`` — the file path to read, required. ``start_line`` /
    ``end_line`` — optional line window (counted from 1); each may be given
    alone (the other defaults to the file edge), and when both are given
    ``end_line`` must be ≥ ``start_line``."""

    source: str
    start_line: int | None
    end_line: int | None

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        source = cls.require_str(params, Keys.source)
        if isinstance(source, ToolResult):
            return source
        start_line = cls.opt_int(params, Keys.start_line, lo=1)
        if isinstance(start_line, ToolResult):
            return start_line
        end_line = cls.opt_int(params, Keys.end_line, lo=1)
        if isinstance(end_line, ToolResult):
            return end_line
        if start_line is not None and end_line is not None and end_line < start_line:
            return ToolResult.err(
                "'end_line' must be greater than or equal to 'start_line'.",
                code="invalid-param",
                hint="swap the two values so end_line ≥ start_line.",
            )
        return cls(source=source, start_line=start_line, end_line=end_line)
