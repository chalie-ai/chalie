"""FindToolsParamsBag — the typed input contract of the ``find_tools`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class FindToolsParamsBag(ParamBag):
    """``query`` — a non-empty array of tool names or described actions to
    discover; each entry runs the discovery cascade independently."""

    query: list[str]

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(query=cls.require_str_list(params, Keys.query))
