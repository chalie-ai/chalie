"""ImageSearchParamsBag — the typed input contract of the ``image_search``
ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class ImageSearchParamsBag(ParamBag):
    """``query`` — natural-language text to search for images, required."""

    query: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        return cls(query=cls.require_str(params, Keys.query))
