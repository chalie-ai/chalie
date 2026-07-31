"""MakeDirParamsBag — the typed input contract of the ``make_dir`` ability.

A single-shot bag: the ability does exactly one thing (create a directory), so
there is no router, no ``action`` field, no match statement. The bag declares
its single required field — ``path`` — as a frozen dataclass attribute, with
the contract enforced at construction time by ``from_params``. A missing or
non-string ``path`` returns a ``ToolResult`` error; a valid one produces an
immutable bag instance that the handler can read without re-validating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class MakeDirParamsBag(ParamBag):
    """``path`` — directory to create, required."""

    path: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        path = cls.require_str(params, Keys.path)
        if isinstance(path, ToolResult):
            return path
        return cls(path=path)
