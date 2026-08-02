"""ReadParamsBag — the typed input contract of the ``read`` ability.

Fields: ``source`` (URL or filesystem path, required), ``start_line`` and
``end_line`` (optional 1-indexed line window into the file — either may be
given alone; the missing bound defaults to the file edge). With any window the
reader returns only those lines; with none the file is
loaded whole under the hard ``_MAX_RETURN_CHARS`` (20 000) cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class ReadParamsBag(ParamBag):
    """``source`` — the read target (URL or filesystem path), required.
    ``start_line`` / ``end_line`` — optional 1-indexed window into the file;
    each may be given alone (the other defaults to the file edge), and when
    both are given ``end_line`` must be ≥ ``start_line``."""

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
