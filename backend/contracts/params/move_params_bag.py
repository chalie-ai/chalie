"""MoveParamsBag — the typed input contract of the ``move`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class MoveParamsBag(ParamBag):
    """``current_path`` — absolute source path of the file or folder to move,
    required. ``new_path`` — absolute destination path (new name or new
    location), required."""

    current_path: str
    new_path: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        current_path = cls.require_str(params, Keys.current_path)
        if isinstance(current_path, ToolResult):
            return current_path
        new_path = cls.require_str(params, Keys.new_path)
        if isinstance(new_path, ToolResult):
            return new_path
        return cls(current_path=current_path, new_path=new_path)
