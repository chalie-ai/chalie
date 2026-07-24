"""VisionParamsBag — the typed input contract of the ``vision`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class VisionParamsBag(ParamBag):
    """``image`` — the stripped 8-character document id; ``instructions`` — the
    stripped natural-language question about the image."""

    image: str
    instructions: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        # The dispatcher pre-gate is truthiness-based, so a non-empty but
        # whitespace-only image/instructions slips past it and must be rejected
        # here. One combined error names every missing field at once — the model
        # self-corrects in one step instead of discovering them serially.
        image_raw = params.get(Keys.image, "")
        instructions_raw = params.get(Keys.instructions, "")
        image = image_raw.strip() if isinstance(image_raw, str) else ""
        instructions = instructions_raw.strip() if isinstance(instructions_raw, str) else ""
        if not image or not instructions:
            missing = ", ".join(
                name
                for name, val in (("image", image), ("instructions", instructions))
                if not val
            )
            return ToolResult.err(
                f"Missing required parameter(s): {missing}.",
                code="missing-params",
                valid=("image", "instructions"),
            )
        return cls(image=image, instructions=instructions)
