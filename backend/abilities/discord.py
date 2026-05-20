"""DiscordAbility — read, send, and navigate Discord channels via the connected bot.

Delegates to DiscordCapability's tool handlers.  Actions map 1-to-1 to the
capability's tool names.
"""

import json
import logging

from abilities._base import Ability
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)


class DiscordAbility(Ability):
    NAME = "discord"
    SUMMARY = (
        "Interact with Discord servers via the connected bot. "
        "Available when the user asks about Discord messages, channels, servers, "
        "or wants to read or send messages on Discord."
    )
    EXAMPLES = [
        "what Discord servers am I in",
        "list channels in my gaming server",
        "read the last 10 messages in #general",
        "send a message to #announcements saying the meeting starts now",
        "what was the last message in the support channel",
        "check Discord for any new messages",
        "post to #updates on my server",
        "show me the messages from the team Discord",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "send_message",
                    "read_messages",
                    "list_channels",
                    "list_servers",
                ],
                "description": (
                    "send_message — send a text message to a Discord channel "
                    "(requires channel_id and content). "
                    "read_messages — fetch recent messages from a channel "
                    "(requires channel_id; optional limit, default 20). "
                    "list_channels — list text channels in a server "
                    "(optional guild_id; omit to list all servers' channels). "
                    "list_servers — list all Discord servers the bot is in."
                ),
            },
            "channel_id": {
                "type": "string",
                "description": (
                    "send_message / read_messages: numeric Discord channel ID. "
                    "Use list_channels first if you only know the channel name."
                ),
            },
            "guild_id": {
                "type": "string",
                "description": (
                    "list_channels: numeric Discord guild (server) ID. "
                    "Omit to list channels across all servers."
                ),
            },
            "content": {
                "type": "string",
                "description": "send_message: the text to post to the channel.",
            },
            "limit": {
                "type": "integer",
                "description": "read_messages: number of messages to fetch (default 20, max 100).",
            },
        },
        "required": ["action"],
    }
    TIMEOUT = 30

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict | str:
        action = params.get("action", "list_servers").lower()

        from capabilities import load_capabilities
        cap = load_capabilities().get("discord")

        if cap is None or not cap.is_connected():
            result = {
                "status": "error",
                "error": "Discord capability not connected. Configure it in the Brain dashboard.",
            }
            return {"text": _skill_tag("discord", json.dumps(result), action=action)}

        tool_map = {t["name"]: t["handler"] for t in cap.get_tools()}
        handler = tool_map.get(action)
        if handler is None:
            result = {"status": "error", "error": f"Unknown discord action: {action}"}
            return {"text": _skill_tag("discord", json.dumps(result), action=action)}

        try:
            action_params = {
                k: v for k, v in params.items()
                if not k.startswith("_") and k != "action"
            }
            raw = handler(topic="", params=action_params, telemetry=telemetry)
            result = raw if isinstance(raw, dict) else {"status": "ok", "data": raw}
        except Exception as exc:
            logger.exception("[DISCORD] action=%s failed: %s", action, exc)
            result = {"status": "error", "error": str(exc)}

        return {"text": _skill_tag("discord", json.dumps(result), action=action)}
