"""SearchParamsBag — the typed input contract of the ``search`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class SearchParamsBag(ParamBag):
    """``query`` — plain natural language search query, required."""

    query: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        query = cls.require_str(params, Keys.query)
        if isinstance(query, ToolResult):
            return query
        return cls(query=query)
