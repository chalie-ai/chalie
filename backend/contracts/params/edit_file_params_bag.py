"""EditFileParamsBag — the typed input contract of the ``edit_file`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class EditFileParamsBag(ParamBag):
    """``search`` — the exact literal text to find, required. ``replace_`` —
    the replacement text, required (an empty string deletes the matched text).
    ``path`` — required absolute path to a single file to edit.

    ``search`` and ``path`` reject empty values (``""``); ``replace_`` accepts
    them because an empty replacement is the delete operation. Both are kept
    VERBATIM — never stripped, whitespace-only values stay valid — because
    leading/trailing whitespace is meaningful replacement text (e.g. tabs →
    spaces)."""

    search: str
    replace_: str
    path: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        path = cls._verbatim(params, Keys.path)
        if isinstance(path, ToolResult):
            return path
        search = cls._verbatim(params, Keys.search)
        if isinstance(search, ToolResult):
            return search
        replace_ = cls._verbatim(params, Keys.replace_, allow_empty=True)
        if isinstance(replace_, ToolResult):
            return replace_
        return cls(search=search, replace_=replace_, path=path)

    @staticmethod
    def _verbatim(
        params: dict[str, object], key: str, *, allow_empty: bool = False
    ) -> str | ToolResult:
        """Required string returned exactly as passed — ``require_str`` would
        strip and reject whitespace-only values, corrupting whitespace-
        significant search/replacement text. ``allow_empty`` keeps the param
        required but accepts ``""``; it mirrors the ability's ``ALLOW_EMPTY``,
        which is what the dispatcher pre-gate reads."""
        value = params.get(key)
        if not isinstance(value, str) or (not value and not allow_empty):
            received = "|".join(params.keys()) or "none"
            return ToolResult.err(
                f"Required parameter '{key}' is missing.",
                code="missing-params",
                hint=f"pass '{key}' (received: {received})",
                valid=(key,),
            )
        return value
