"""DiscordCapability — Discord bot integration.

Runs a discord.py Client in a daemon thread (see DiscordBotRunner).  Inbound
messages are routed through ExternalChatMessageProcessor so every Discord turn
runs an isolated ACT loop under the 'discord' policy context.  The LLM response
is sent back to the originating Discord channel automatically.

Credential storage: discord:bot_token, discord:system_prompt,
discord:allowed_guilds, discord:allowed_channels, discord:owner_user_id
(all encrypted via VaultService).
"""

from __future__ import annotations

import logging
import pathlib
import uuid

import yaml

from capabilities.base import AbstractCapability
from capabilities.discord_capability.discord_bot_runner import DiscordBotRunner

logger = logging.getLogger(__name__)

_MANIFEST_PATH = pathlib.Path(__file__).parent / "manifest.yaml"

_K_BOT_TOKEN = "discord:bot_token"
_K_SYSTEM_PROMPT = "discord:system_prompt"
_K_ALLOWED_GUILDS = "discord:allowed_guilds"
_K_ALLOWED_CHANNELS = "discord:allowed_channels"
_K_OWNER_USER_ID = "discord:owner_user_id"


class DiscordCapability(AbstractCapability):
    """Discord bot capability.

    Connects via a long-lived bot token, listens for inbound messages, routes
    them through ExternalChatMessageProcessor, and sends LLM responses back.
    """

    def __init__(self) -> None:
        super().__init__()
        self._manifest_cache: dict | None = None
        self._bot_runner: DiscordBotRunner | None = None

    # ── Identity ─────────────────────────────────────────────────────────────

    def get_id(self) -> str:
        return "discord"

    def get_manifest(self) -> dict:
        if self._manifest_cache is None:
            with open(_MANIFEST_PATH, encoding="utf-8") as fh:
                self._manifest_cache = yaml.safe_load(fh)
        return self._manifest_cache

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def configure(self, credentials: dict) -> None:
        """Validate and persist credentials, then start the bot.

        Args:
            credentials: Must contain ``bot_token`` (str).  Optional fields:
                ``system_prompt`` (str), ``allowed_guilds`` (str, comma-separated),
                ``allowed_channels`` (str, comma-separated).

        Raises:
            ValueError: When bot_token is absent or empty.
        """
        token = (credentials.get("bot_token") or "").strip()
        if not token:
            raise ValueError("Discord Bot Token is required")

        self.store_credential(_K_BOT_TOKEN, token)
        self.store_credential(_K_SYSTEM_PROMPT, credentials.get("system_prompt") or "")
        self.store_credential(_K_ALLOWED_GUILDS, credentials.get("allowed_guilds") or "")
        self.store_credential(_K_ALLOWED_CHANNELS, credentials.get("allowed_channels") or "")
        self.store_credential(_K_OWNER_USER_ID, credentials.get("owner_user_id") or "")

        self.connect()

    def connect(self) -> bool:
        """Load stored credentials and start the Discord bot daemon thread.

        Returns:
            bool: True when the bot thread starts successfully, False otherwise.
        """
        token = self.load_credential(_K_BOT_TOKEN)
        if not token:
            self._connected = False
            return False

        allowed_guilds = self._parse_id_set(self.load_credential(_K_ALLOWED_GUILDS))
        allowed_channels = self._parse_id_set(self.load_credential(_K_ALLOWED_CHANNELS))

        self._start_bot_runner(token, allowed_guilds, allowed_channels)
        if self._bot_runner.wait_until_ready(timeout=30):
            self._connected = True
        else:
            logger.warning("[discord] bot did not become ready within 30s")
            self._stop_bot_runner()
            self._connected = False
        return self._connected

    def disconnect(self) -> None:
        """Stop the bot thread, clear connection state, and delete credentials."""
        self._connected = False
        self._stop_bot_runner()
        self.delete_credentials()
        logger.info("[discord] Disconnected and credentials removed.")

    # ── Cognitive pipeline ────────────────────────────────────────────────────

    def ingest(self) -> list:
        """Discord is event-driven; polling is not used."""
        return []

    def understand(self, items: list) -> list:
        return items

    def _do_monitor(self) -> None:
        """Reconnect the bot if the daemon thread has died."""
        if self._bot_runner and not self._bot_runner.is_alive:
            logger.warning("[discord] bot thread is dead — reconnecting")
            self.connect()

    def act(self, action: str, params: dict) -> dict:
        """Dispatch a write action to the matching tool handler.

        Args:
            action: Tool name, e.g. ``"send_message"``.
            params: Action-specific parameters.

        Returns:
            dict: Handler result, or ``{"error": ...}`` for unknown actions.
        """
        tool_map = {t["name"]: t["handler"] for t in self.get_tools()}
        handler = tool_map.get(action)
        if handler is None:
            return {"error": f"Unknown action: {action}"}
        return handler(topic="", params=params)

    # ── Tools ─────────────────────────────────────────────────────────────────

    def _require_connection(self) -> dict | None:
        """Return an error dict when not connected, else None."""
        if not self.is_connected():
            return {
                "error": (
                    "Discord capability not connected. "
                    "Configure it in the Brain dashboard."
                )
            }
        return None

    def _th_send_message(self, topic, params, config=None, telemetry=None) -> dict:
        """Send a message to a Discord channel by channel_id."""
        err = self._require_connection()
        if err:
            return err
        channel_id_raw = params.get("channel_id")
        content = (params.get("content") or "").strip()
        if not channel_id_raw:
            return {"error": "channel_id is required"}
        if not content:
            return {"error": "content is required"}
        try:
            channel_id = int(channel_id_raw)
        except (TypeError, ValueError):
            return {"error": "channel_id must be a numeric Discord channel ID"}
        self._bot_runner.send_to_channel(channel_id, content)
        return {"status": "sent", "channel_id": str(channel_id), "content": content}

    def _th_read_messages(self, topic, params, config=None, telemetry=None) -> dict:
        """Fetch recent messages from a Discord channel."""
        err = self._require_connection()
        if err:
            return err
        channel_id_raw = params.get("channel_id")
        if not channel_id_raw:
            return {"error": "channel_id is required"}
        try:
            channel_id = int(channel_id_raw)
        except (TypeError, ValueError):
            return {"error": "channel_id must be a numeric Discord channel ID"}
        try:
            limit = min(int(params.get("limit", 20)), 100)
        except (TypeError, ValueError):
            return {"error": "limit must be a number"}
        messages = self._bot_runner.fetch_messages(channel_id, limit)
        return {"status": "ok", "messages": messages}

    def _th_list_channels(self, topic, params, config=None, telemetry=None) -> dict:
        """List text channels in a guild, or all guilds if guild_id is omitted."""
        err = self._require_connection()
        if err:
            return err
        guild_id_raw = params.get("guild_id")
        guild_id: int | None = None
        if guild_id_raw:
            try:
                guild_id = int(guild_id_raw)
            except (TypeError, ValueError):
                return {"error": "guild_id must be a numeric Discord guild ID"}
        channels = self._bot_runner.list_guild_channels(guild_id)
        return {"status": "ok", "channels": channels}

    def _th_list_servers(self, topic, params, config=None, telemetry=None) -> dict:
        """List all Discord servers (guilds) the bot is a member of."""
        err = self._require_connection()
        if err:
            return err
        servers = self._bot_runner.list_guilds()
        return {"status": "ok", "servers": servers}

    def get_tools(self) -> list:
        """Return tool definitions for all four Discord actions."""
        return [
            {
                "name": "send_message",
                "description": "Send a text message to a Discord channel by channel_id.",
                "parameters": {},
                "handler": self._th_send_message,
                "timeout": 15,
            },
            {
                "name": "read_messages",
                "description": "Fetch recent messages from a Discord channel.",
                "parameters": {},
                "handler": self._th_read_messages,
                "timeout": 15,
            },
            {
                "name": "list_channels",
                "description": "List text channels in a Discord server.",
                "parameters": {},
                "handler": self._th_list_channels,
                "timeout": 10,
            },
            {
                "name": "list_servers",
                "description": "List all Discord servers (guilds) the bot is a member of.",
                "parameters": {},
                "handler": self._th_list_servers,
                "timeout": 10,
            },
        ]

    # ── Inbound message handling ───────────────────────────────────────────────

    def _on_discord_message(
        self,
        guild_name: str,
        channel_name: str,
        channel_id: int,
        author_name: str,
        author_id: int,
        content: str,
    ) -> None:
        """Process an inbound Discord message through the ECMP pipeline.

        Called from a ThreadPoolExecutor thread by DiscordBotRunner.  Instantiates
        ExternalChatMessageProcessor, runs the ACT loop, and sends the LLM response
        back to the originating Discord channel.

        Args:
            guild_name: Discord server name.
            channel_name: Discord channel name.
            channel_id: Numeric Discord channel ID.
            author_name: Display name of the message author.
            author_id: Discord user ID (snowflake) of the message author.
            content: Raw message text.
        """
        if not content.strip():
            return

        owner_raw = self.load_credential(_K_OWNER_USER_ID)
        owner_user_id = self._parse_snowflake(owner_raw)
        custom_prompt = self.load_credential(_K_SYSTEM_PROMPT) or ""

        from services.external_chat_message_processor import ExternalChatMessageProcessor

        request_id = str(uuid.uuid4())
        metadata = {
            "uuid": request_id,
            "exchange_id": request_id,
            "source": "discord",
            "attachments": [],
            "channel": "discord",
            "hidden_input": False,
        }
        proc = ExternalChatMessageProcessor(
            raw_input=content,
            metadata=metadata,
            author_name=author_name,
            author_id=author_id,
            owner_user_id=owner_user_id,
            guild_name=guild_name,
            channel_name=channel_name,
            custom_system_prompt=custom_prompt,
        )
        try:
            response = proc.send(request_id=request_id)
        except Exception as exc:
            logger.exception("[discord] ECMP turn failed for channel %s: %s", channel_id, exc)
            return

        if response and response.strip() and self._bot_runner:
            self._bot_runner.send_to_channel(channel_id, response)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _start_bot_runner(
        self,
        token: str,
        allowed_guilds: set[int],
        allowed_channels: set[int],
    ) -> None:
        """Stop any existing runner and start a fresh one."""
        self._stop_bot_runner()
        self._bot_runner = DiscordBotRunner(
            token=token,
            on_message=self._on_discord_message,
            allowed_guilds=allowed_guilds,
            allowed_channels=allowed_channels,
        )
        self._bot_runner.start()

    def _stop_bot_runner(self) -> None:
        """Gracefully stop the bot runner if it exists."""
        if self._bot_runner:
            try:
                self._bot_runner.stop()
            except Exception as exc:
                logger.warning("[discord] bot runner stop error: %s", exc)
            self._bot_runner = None

    @staticmethod
    def _parse_id_set(raw: str | None) -> set[int]:
        """Parse a comma-separated string of Discord snowflake IDs into a set of ints.

        Returns an empty set for None, empty strings, or entries that cannot be
        parsed as integers.
        """
        if not raw:
            return set()
        ids: set[int] = set()
        for part in raw.split(","):
            part = part.strip()
            if part.isascii() and part.isdigit():
                ids.add(int(part))
        return ids

    @staticmethod
    def _parse_snowflake(raw: str | None) -> int | None:
        """Parse a single Discord snowflake ID string into an int.

        Returns None for None, empty strings, or values that cannot be parsed
        as a plain ASCII integer.
        """
        if not raw:
            return None
        raw = raw.strip()
        if raw.isascii() and raw.isdigit():
            return int(raw)
        return None
