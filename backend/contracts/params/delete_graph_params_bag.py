"""DeleteGraphParamsBag — the typed input contract of the ``delete_graph``
ability (subject)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class DeleteGraphParamsBag(ParamBag):
    """``subject`` — the stable fact key whose live row should be hard-deleted."""

    subject: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        subject = cls.require_str(params, "subject")
        if isinstance(subject, ToolResult):
            return subject
        return cls(subject=subject)
