"""RecallParamsBag — the typed input contract of the memory-v3 ``recall``
ability (query)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class RecallParamsBag(ParamBag):
    """``query`` — the topic to search existing Graph facts + Map episodes for."""

    query: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        query = cls.require_str(params, "query")
        if isinstance(query, ToolResult):
            return query
        return cls(query=query)
