"""TelegramAbility — message the user on Telegram through their own bot.

Delegates to ``TelegramCapability`` via :class:`CapabilityAbility`. The one
action, ``send``, goes to the one chat the capability recorded. There is no
recipient parameter, so this tool cannot message anyone else.

With ``wait_for_reply`` the call goes on after sending until the user replies
or the turn is stopped, and the reply is the result. The capability gets no
MessageProcessor, so this ability hands it the turn's stop check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from abilities._capability import CapabilityAbility
from abilities._result import ToolResult
from configs.enums.ability_category import AbilityCategory
from configs.enums.param_key import Keys
from contracts.params.capability_params_bag import CapabilityParamsBag
from contracts.params.param_bag import ParamBag

if TYPE_CHECKING:
    from capabilities.telegram_capability.capability import TelegramCapability


class TelegramAbility(CapabilityAbility):
    NAME: ClassVar[str] = "telegram"
    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.PRODUCTIVITY
    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = ("telegram bot", "telegram message")

    CAPABILITY_KEY: ClassVar[str] = "telegram"
    DEFAULT_ACTION: ClassVar[str] = "send"
    ACTION_HANDLERS: ClassVar[dict[str, str]] = {"send": "send"}
    NOT_CONNECTED_HINT: ClassVar[str] = (
        "Ask the user to set up Telegram under Capabilities in the Brain dashboard."
    )

    # "" covers the omitted action, which DEFAULT_ACTION turns into send.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "": (Keys.message,),
        "send": (Keys.message,),
    }

    def classify_action(self, params: dict[str, object]) -> str:
        """Every call that passes the pre-gate sends, so the permission is
        always ``<name>.send``, with or without an ``action`` param."""
        return self.DEFAULT_ACTION

    def get_summary(self) -> str:
        return (
            "Send the user a message on Telegram, through the bot they connected. "
            "It only ever reaches the user, never anyone else. Use it when the user "
            "asks to be told something on Telegram, now or from a scheduled task. "
            "Plain text; long messages are split into several. When you need the "
            "user's answer to carry on, set wait_for_reply: the call then waits "
            "until they reply, with no time limit, and returns their reply."
        )

    def get_examples(self) -> list[str]:
        return [
            "message me on Telegram when the backup finishes",
            "send me the daily report on Telegram",
            "text me on Telegram if the website goes down",
            "ping me on Telegram in the morning with the weather",
            "send this summary to my Telegram",
            "let me know on Telegram when it's done",
            "remind me on Telegram to call the dentist",
        ]

    def get_search_tooltip(self) -> str:
        return "Message the user on Telegram"

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.action: {
                "type": "string",
                "enum": list(ACTION_HANDLERS),
                "description": "send — send the user a Telegram message. May be omitted.",
            },
            Keys.message: {
                "type": "string",
                "description": "The text to send the user, as plain text.",
            },
            Keys.wait_for_reply: {
                "type": "boolean",
                "default": False,
                "description": (
                    "Set true when you need the user's answer to carry on. After "
                    "sending, the call waits until the user replies on Telegram, "
                    "with no time limit, and returns their reply."
                ),
            },
        },
        "required": [Keys.message],
    }

    def run(self, params: CapabilityParamsBag) -> ToolResult:
        wait = ParamBag.bool_default(params.extra, Keys.wait_for_reply, default=False)
        if isinstance(wait, ToolResult):
            return wait
        cap = self._connected_capability() if wait else None
        # No wait, not connected, or an unknown action: the base sends, or
        # returns its own error.
        if cap is None or self._resolve_action(params) not in self.ACTION_HANDLERS:
            return super().run(params)
        if self.mp is None:
            raise RuntimeError("the telegram ability needs a bound MessageProcessor to wait for a reply")

        result = cap.send(str(params.extra.get(Keys.message) or ""), self.mp.turn_execution_service.should_stop)
        if "error" in result:
            return ToolResult.err(str(result["error"]), code="capability-error", action=self.DEFAULT_ACTION)
        to = result["to"]
        if "reply" in result:
            return ToolResult.ok(
                f"The message was sent to {to} on Telegram, and they replied:\n\n{result['reply']}",
                action=self.DEFAULT_ACTION,
            )
        return ToolResult.ok(
            f"The message was sent to {to}. The wait was stopped before they replied.",
            action=self.DEFAULT_ACTION,
        )

    def _connected_capability(self) -> TelegramCapability | None:
        from capabilities import load_capabilities  # noqa: PLC0415

        cap = load_capabilities().get(self.CAPABILITY_KEY)
        if cap is None or not cap.is_connected():
            return None
        return cast("TelegramCapability", cap)
