"""VisionParamsBag — the typed input contract of the ``vision`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from exceptions import ToolParamError


@dataclass(frozen=True, slots=True)
class VisionParamsBag(ParamBag):
    """``image`` — the stripped 8-character document id; ``query`` — the
    stripped natural-language question about the image."""

    image: str
    query: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self:
        # The dispatcher pre-gate is truthiness-based, so a non-empty but
        # whitespace-only image/query slips past it and must be rejected here.
        # One combined error names every missing field at once — the model
        # self-corrects in one step instead of discovering them serially.
        image_raw = params.get(Keys.image, "")
        query_raw = params.get(Keys.query, "")
        image = image_raw.strip() if isinstance(image_raw, str) else ""
        query = query_raw.strip() if isinstance(query_raw, str) else ""
        if not image or not query:
            missing = ", ".join(
                name for name, val in (("image", image), ("query", query)) if not val
            )
            raise ToolParamError(
                f"Missing required parameter(s): {missing}.",
                code="missing-params",
                valid=("image", "query"),
            )
        return cls(image=image, query=query)
