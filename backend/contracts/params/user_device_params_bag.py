"""UserDeviceParamsBag — the typed input contract of the ``user_device``
ability.

The tool takes no parameters: it answers from the persisted client
heartbeat alone. The bag is intentionally empty — its existence is the
contract: the dispatch seam never hands the ability a raw
``dict[str, object]``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class UserDeviceParamsBag(ParamBag):
    """Empty bag — ``user_device`` accepts no parameters."""

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        return cls()
