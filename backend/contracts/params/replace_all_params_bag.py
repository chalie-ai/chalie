"""ReplaceAllParamsBag — the typed input contract of the ``replace_all`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from exceptions import ToolParamError


@dataclass(frozen=True, slots=True)
class ReplaceAllParamsBag(ParamBag):
    """``search`` — the exact literal text to find, required. ``replace_`` —
    the replacement text, required. Both are kept VERBATIM — never stripped,
    whitespace-only values stay valid — because leading/trailing whitespace is
    meaningful replacement text (e.g. tabs → spaces); only an absent, non-str,
    or exactly-empty value is rejected, mirroring the dispatcher pre-gate's
    truthiness check. ``glob`` — optional filename pattern (e.g. ``*.ts``)
    limiting which files are touched; ``None`` when omitted. ``path`` —
    optional absolute path to a single file or directory to scan; ``None``
    when omitted (defaults to the code_agent workspace)."""

    search: str
    replace_: str
    glob: str | None
    path: str | None

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(
            search=cls._verbatim(params, Keys.search),
            replace_=cls._verbatim(params, Keys.replace_),
            glob=cls.opt_str(params, Keys.glob),
            path=cls.opt_str(params, Keys.path),
        )

    @staticmethod
    def _verbatim(params: dict[str, object], key: str) -> str:
        """Required string returned exactly as passed — ``require_str`` would
        strip and reject whitespace-only values, corrupting whitespace-
        significant search/replacement text."""
        value = params.get(key)
        if not isinstance(value, str) or not value:
            received = "|".join(params.keys()) or "none"
            raise ToolParamError(
                f"Required parameter '{key}' is missing.",
                code="missing-params",
                hint=f"pass '{key}' (received: {received})",
                valid=(key,),
            )
        return value
