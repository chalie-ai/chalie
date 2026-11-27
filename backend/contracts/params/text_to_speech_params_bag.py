"""TextToSpeechParamsBag — the typed input contract of the ``text_to_speech`` ability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


@dataclass(frozen=True, slots=True)
class TextToSpeechParamsBag(ParamBag):
    """``message`` — the text to speak, required. Whether anything is left to
    speak once markup is stripped is the voice service's call, not the bag's."""

    message: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        message = params.get(Keys.message)
        if not isinstance(message, str):
            return ToolResult.err(
                f"'{Keys.message}' must be text.",
                code="invalid-param",
                hint=f"pass '{Keys.message}' as the text to speak.",
            )
        return cls(message=message)
