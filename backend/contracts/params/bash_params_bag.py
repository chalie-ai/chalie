"""BashParamsBag — the typed input contract of the ``bash`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class BashParamsBag(ParamBag):
    """``command`` — the shell command to execute via ``bash -c``, required
    non-blank. ``timeout_s`` — execution timeout in seconds: default 30,
    clamped to ``[1, 300]``; these bounds are the single source of truth."""

    command: str
    timeout_s: int

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        command = cls.require_str(params, Keys.command)
        if isinstance(command, ToolResult):
            return command
        timeout_s = cls.clamp_int(params, Keys.timeout_s, default=30, lo=1, hi=300)
        if isinstance(timeout_s, ToolResult):
            return timeout_s
        return cls(command=command, timeout_s=timeout_s)
