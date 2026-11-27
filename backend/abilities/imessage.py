"""ImessageAbility — text the user on iMessage through the Messages app.

Delegates to ``ImessageCapability`` via :class:`CapabilityAbility`. The one
action, ``send``, goes to the one handle the user set up: a text, a voice note
made from an audio file, or both. There is no recipient parameter, so this
tool cannot message anyone else.

iMessage works only on a Mac: elsewhere the tool is left out of what Chalie
is told about, and a call that arrives anyway is refused with the reason.
"""

from __future__ import annotations

from typing import ClassVar

from abilities._capability import CapabilityAbility
from abilities._result import ToolResult
from capabilities.imessage_capability.availability import APPLE_ONLY, AVAILABLE
from configs.enums.ability_category import AbilityCategory
from configs.enums.param_key import Keys
from contracts.params.capability_params_bag import CapabilityParamsBag
from contracts.params.param_bag import ParamBag


class ImessageAbility(CapabilityAbility):
    NAME: ClassVar[str] = "imessage"
    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.PRODUCTIVITY
    DISCOVERABLE: ClassVar[bool] = AVAILABLE
    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("imessage message", "apple messages")

    CAPABILITY_KEY: ClassVar[str] = "imessage"
    DEFAULT_ACTION: ClassVar[str] = "send"
    ACTION_HANDLERS: ClassVar[dict[str, str]] = {"send": "send"}
    NOT_CONNECTED_HINT: ClassVar[str] = (
        "Ask the user to set up iMessage under Capabilities in the Brain dashboard."
    )

    # "" covers the omitted action, which DEFAULT_ACTION turns into send.
    # Neither field is required on its own; run() asks for one of the two.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (), "send": ()}

    def classify_action(self, params: dict[str, object]) -> str:
        """Every call that passes the pre-gate sends, so the permission is
        always ``<name>.send``, with or without an ``action`` param."""
        return self.DEFAULT_ACTION

    def get_summary(self) -> str:
        return (
            "Text the user on iMessage, through the Messages app on this Mac. It only "
            "ever reaches the user, never anyone else. Success means the message was "
            "handed to Messages, not that it arrived. Plain text. You can also send "
            "a voice note: make one with text_to_speech and pass its path as audio. "
            "Use text where it reads better; a voice note keeps a back-and-forth flowing."
        )

    def get_examples(self) -> list[str]:
        return [
            "text me on iMessage when the backup finishes",
            "send me the daily report on iMessage",
            "iMessage me if the website goes down",
            "message my iPhone in the morning with the weather",
            "send this summary to my iMessage",
            "let me know by iMessage when it's done",
            "remind me by iMessage to call the dentist",
        ]

    def get_search_tooltip(self) -> str:
        return "Text the user on iMessage"

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.action: {
                "type": "string",
                "enum": list(ACTION_HANDLERS),
                "description": "send — text the user on iMessage. May be omitted.",
            },
            Keys.message: {
                "type": "string",
                "description": "The text to send the user, as plain text.",
            },
            Keys.audio: {
                "type": "string",
                "description": (
                    "The full path of an audio file to send as a voice note, such "
                    "as the path text_to_speech returned. Sent after the text, "
                    "when both are given."
                ),
            },
        },
        "required": [],
    }

    def run(self, params: CapabilityParamsBag) -> ToolResult:
        if not AVAILABLE:
            return ToolResult.err(APPLE_ONLY, code="apple-only", action=self.DEFAULT_ACTION)
        message = ParamBag.opt_str(params.extra, Keys.message)
        if isinstance(message, ToolResult):
            return message
        audio = ParamBag.opt_str(params.extra, Keys.audio)
        if isinstance(audio, ToolResult):
            return audio
        if not ((message and message.strip()) or (audio and audio.strip())):
            return ToolResult.err(
                f"Pass '{Keys.message}', '{Keys.audio}', or both.",
                code="missing-params",
                hint=f"'{Keys.message}' is the text to send; '{Keys.audio}' is the path of an audio file.",
            )
        result = super().run(params)
        if result.status == "error" or not isinstance(result.body, dict):
            return result
        return ToolResult.ok(f"Handed to Messages for {result.body['to']}.", action=self.DEFAULT_ACTION)
