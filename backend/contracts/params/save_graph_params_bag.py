"""SaveGraphParamsBag — the typed input contract of the memory-v3 ``save_graph``
ability (subject + contents)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class SaveGraphParamsBag(ParamBag):
    """``subject`` — the stable fact key; ``contents`` — the stripped, non-blank
    current value."""

    subject: str
    contents: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        subject = cls.require_str(params, "subject")
        if isinstance(subject, ToolResult):
            return subject
        contents = cls.require_str(params, "contents")
        if isinstance(contents, ToolResult):
            return contents
        return cls(subject=subject, contents=contents)
