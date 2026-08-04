"""ImagePreviewParamsBag — the typed input contract of the ``image_preview``
ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class ImagePreviewParamsBag(ParamBag):
    """``file_path`` — a local path or an http(s) URL to an image, required.

    ``subtitle`` — an optional caption. The bag keeps it verbatim: clamping it to
    eight words is a display decision the ability owns, and a bag stays pure.
    """

    file_path: str
    subtitle: str | None

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        file_path = cls.require_str(params, Keys.file_path)
        if isinstance(file_path, ToolResult):
            return file_path
        subtitle = cls.opt_str(params, Keys.subtitle)
        if isinstance(subtitle, ToolResult):
            return subtitle
        return cls(file_path=file_path, subtitle=subtitle)
