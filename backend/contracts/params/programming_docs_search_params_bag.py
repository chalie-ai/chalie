"""ProgrammingDocsSearchParamsBag — the typed input contract of the
``programming_docs_search`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class ProgrammingDocsSearchParamsBag(ParamBag):
    """``language`` — programming language or framework id to look up
    documentation for, required; ``query`` — what to look up within that
    language, required."""

    language: str
    query: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        language = cls.require_str(params, Keys.language)
        if isinstance(language, ToolResult):
            return language
        query = cls.require_str(params, Keys.query)
        if isinstance(query, ToolResult):
            return query
        return cls(language=language, query=query)
