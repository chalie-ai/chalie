"""FileWriteParamsBag — the typed input contract of the ``file_write`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from exceptions import ToolParamError


@dataclass(frozen=True, slots=True)
class FileWriteParamsBag(ParamBag):
    """``path`` — absolute path to write to, required. ``contents`` — the file
    body, required by PRESENCE, not truthiness: an empty string is valid user
    data (writes a 0-byte file / truncates), so only a genuinely absent key or
    a non-string value is rejected."""

    path: str
    contents: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        path = cls.require_str(params, Keys.path)

        if Keys.contents not in params:
            raise ToolParamError(
                "contents is required.",
                code="missing-params",
                hint="pass a 'contents' string (an empty string writes a 0-byte file).",
            )
        contents = params[Keys.contents]
        if not isinstance(contents, str):
            raise ToolParamError(
                f"contents must be a string, got {type(contents).__name__}.",
                code="invalid-param",
                hint="pass the file body as a string; serialise structured data yourself first.",
            )

        return cls(path=path, contents=contents)
