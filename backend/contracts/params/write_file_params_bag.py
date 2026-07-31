"""WriteFileParamsBag — the typed input contract of the ``write_file`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class WriteFileParamsBag(ParamBag):
    """``path`` — absolute path to write to, required. ``contents`` — the file
    body, required by PRESENCE, not truthiness: an empty string is valid user
    data (writes a 0-byte file / truncates), so only a genuinely absent key or
    a non-string value is rejected."""

    path: str
    contents: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        path = cls.require_str(params, Keys.path)
        if isinstance(path, ToolResult):
            return path

        if Keys.contents not in params:
            return ToolResult.err(
                "contents is required.",
                code="missing-params",
                hint="pass a 'contents' string (an empty string writes a 0-byte file).",
            )
        contents = params[Keys.contents]
        if not isinstance(contents, str):
            return ToolResult.err(
                f"contents must be a string, got {type(contents).__name__}.",
                code="invalid-param",
                hint="pass the file body as a string; serialise structured data yourself first.",
            )

        return cls(path=path, contents=contents)
