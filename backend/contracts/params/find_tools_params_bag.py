"""FindToolsParamsBag — the typed input contract of the ``find_tools`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class FindToolsParamsBag(ParamBag):
    """``query`` — a non-empty array of tool names; each entry is resolved
    independently by exact match against the registry's alias map."""

    query: list[str]

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        query = cls.require_str_list(params, Keys.query)
        if isinstance(query, ToolResult):
            return query
        return cls(query=query)
