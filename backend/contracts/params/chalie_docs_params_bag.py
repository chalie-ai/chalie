"""ChalieDocsParamsBag — the typed input contract of the ``chalie_docs`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class ChalieDocsParamsBag(ParamBag):
    """``query`` — the documentation section to look up, required; one of
    ``basics`` / ``tools`` / ``releases`` / ``code-base``."""

    query: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(
            query=cls.require_str(params, Keys.query),
        )
