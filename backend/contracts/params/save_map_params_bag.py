"""SaveMapParamsBag — the typed input contract of the ``save_map`` ability
(contents + optional derived_from map ids)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class SaveMapParamsBag(ParamBag):
    """``contents`` — the stripped distillation; ``derived_from`` — map ids this
    distillation replaces/extends (retires them from the searchable pool)."""

    contents: str
    derived_from: tuple[int, ...]

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        contents = cls.require_str(params, "contents")
        if isinstance(contents, ToolResult):
            return contents
        raw = params.get("derived_from")
        if raw is None:
            derived: tuple[int, ...] = ()
        elif isinstance(raw, list):
            ids: list[int] = []
            for item in raw:
                if not isinstance(item, int) or isinstance(item, bool):
                    return ToolResult.err(
                        "'derived_from' must be a list of integers (map ids).",
                        code="invalid-param",
                        hint="pass derived_from as a list of existing map ids, or omit it.",
                    )
                ids.append(item)
            derived = tuple(ids)
        else:
            return ToolResult.err(
                "'derived_from' must be a list of integers (map ids).",
                code="invalid-param",
                hint="pass derived_from as a list of existing map ids, or omit it.",
            )
        return cls(contents=contents, derived_from=derived)
