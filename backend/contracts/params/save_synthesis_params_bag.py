"""SaveSynthesisParamsBag — the typed input contract of the ``save_synthesis``
ability.

The dispatcher pre-gate is truthiness-based so a whitespace-only summary must
be rejected here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag

_REQUIRED = ("summary",)


@dataclass(frozen=True, slots=True)
class SaveSynthesisParamsBag(ParamBag):
    """``summary`` — stripped user-synthesis text."""

    summary: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        summary = (str(params.get(Keys.summary) or "")).strip()
        if not summary:
            return ToolResult.err(
                "Missing required parameter(s): summary.",
                code="missing-params",
                valid=_REQUIRED,
            )
        return cls(summary=summary)
