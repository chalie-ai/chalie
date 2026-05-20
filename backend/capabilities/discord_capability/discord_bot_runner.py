"""Discord bot runner — persistent asyncio event loop in a daemon thread.

Owns the discord.py Client lifecycle. Exposes synchronous helpers for the
capability layer: start(), stop(), is_alive, send_to_channel(), fetch_messages(),
list_guild_channels(), list_guilds().

The on_message callback is injected at construction time and called from the
asyncio thread via run_in_executor so the UMP turn runs in a worker thread,
keeping the Discord event loop unblocked.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)

_RECONNECT_BASE = 5
_RECONNECT_MAX = 60

# Maximum messages returned by fetch_messages when limit is not overridden.
_DEFAULT_FETCH_LIMIT = 20
_DISCORD_MAX_LEN = 2000


class DiscordBotRunner:
    """Manages a persistent discord.py Client in its own asyncio event loop.

    Args:
        token: Discord bot token.
        on_message: Callable invoked synchronously for each inbound message.
            Signature: (guild_name, channel_name, channel_id, author_name, content)
            Called inside a ThreadPoolExecutor so blocking operations are safe.
        allowed_guilds: Optional set of guild IDs to accept messages from.
            Empty set means all guilds are allowed.
        allowed_channels: Optional set of channel IDs to accept messages from.
            Empty set means all channels are allowed.
    """

    def __init__(
        self,
        token: str,
        on_message: Callable[[str, str, int, str, str], None],
        allowed_guilds: set[int],
        allowed_channels: set[int],
    ) -> None:
        self._token = token
        self._on_message = on_message
        self._allowed_guilds = allowed_guilds
        self._allowed_channels = allowed_channels

        self._loop: asyncio.AbstractEventLoop | None = None
        self._client = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._inbound_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="discord-inbound",
        )

    @property
    def is_alive(self) -> bool:
        """True when the daemon thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the bot thread (idempotent — does nothing if already alive)."""
        if self.is_alive:
            return
        self._stop_event.clear()
        self._ready_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="discord-bot",
        )
        self._thread.start()

    def wait_until_ready(self, timeout: float = 30) -> bool:
        """Block until the bot fires on_ready or *timeout* seconds elapse."""
        return self._ready_event.wait(timeout=timeout)

    def stop(self) -> None:
        """Signal the bot to stop and wait up to 5 s for the thread to exit."""
        self._stop_event.set()
        if self._loop and self._client:
            asyncio.run_coroutine_threadsafe(self._client.close(), self._loop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._inbound_executor.shutdown(wait=False)
        self._thread = None
        self._loop = None
        self._client = None

    # ── Synchronous API ──────────────────────────────────────────────────────

    def send_to_channel(self, channel_id: int, text: str) -> None:
        """Schedule a message send on the bot's asyncio loop (fire-and-forget).

        Safe to call from any thread.
        """
        if not self._loop or not self._client:
            logger.warning("[discord-bot] send_to_channel called before bot is ready")
            return
        asyncio.run_coroutine_threadsafe(
            self._async_send(channel_id, text),
            self._loop,
        )

    def fetch_messages(self, channel_id: int, limit: int = _DEFAULT_FETCH_LIMIT) -> list[dict]:
        """Fetch recent messages from a channel synchronously.

        Returns an empty list when the bot is not ready or the channel is inaccessible.
        """
        if not self._loop or not self._client:
            return []
        future = asyncio.run_coroutine_threadsafe(
            self._async_fetch_messages(channel_id, limit),
            self._loop,
        )
        try:
            return future.result(timeout=15)
        except Exception as exc:
            logger.warning("[discord-bot] fetch_messages failed: %s", exc)
            return []

    def list_guild_channels(self, guild_id: int | None) -> list[dict]:
        """List text channels in a guild (or all guilds if guild_id is None).

        Returns an empty list when the bot is not ready.
        """
        if not self._loop or not self._client:
            return []
        future = asyncio.run_coroutine_threadsafe(
            self._async_list_channels(guild_id),
            self._loop,
        )
        try:
            return future.result(timeout=10)
        except Exception as exc:
            logger.warning("[discord-bot] list_guild_channels failed: %s", exc)
            return []

    def list_guilds(self) -> list[dict]:
        """List all guilds the bot is a member of.

        Returns an empty list when the bot is not ready.
        """
        if not self._loop or not self._client:
            return []
        future = asyncio.run_coroutine_threadsafe(
            self._async_list_guilds(),
            self._loop,
        )
        try:
            return future.result(timeout=10)
        except Exception as exc:
            logger.warning("[discord-bot] list_guilds failed: %s", exc)
            return []

    # ── Private asyncio helpers ───────────────────────────────────────────────

    async def _async_send(self, channel_id: int, text: str) -> None:
        """Send text to a Discord channel; splits messages longer than 2000 chars."""
        try:
            channel = self._client.get_channel(channel_id)
            if channel is None:
                channel = await self._client.fetch_channel(channel_id)
            for chunk_start in range(0, len(text), _DISCORD_MAX_LEN):
                await channel.send(text[chunk_start:chunk_start + _DISCORD_MAX_LEN])
        except Exception as exc:
            logger.warning("[discord-bot] send to channel %s failed: %s", channel_id, exc)

    async def _async_fetch_messages(self, channel_id: int, limit: int) -> list[dict]:
        """Fetch up to *limit* recent messages from a channel."""
        try:
            channel = self._client.get_channel(channel_id)
            if channel is None:
                channel = await self._client.fetch_channel(channel_id)
            messages = []
            async for msg in channel.history(limit=limit):
                messages.append({
                    "id": str(msg.id),
                    "author": msg.author.display_name,
                    "content": msg.content,
                    "timestamp": msg.created_at.isoformat(),
                })
            return messages
        except Exception as exc:
            logger.warning("[discord-bot] fetch messages from %s failed: %s", channel_id, exc)
            return []

    async def _async_list_channels(self, guild_id: int | None) -> list[dict]:
        """List text channels for a specific guild or all guilds."""
        import discord as _discord
        channels = []
        guilds = (
            [g for g in self._client.guilds if g.id == guild_id]
            if guild_id is not None
            else list(self._client.guilds)
        )
        for guild in guilds:
            for ch in guild.channels:
                if isinstance(ch, _discord.TextChannel):
                    channels.append({
                        "id": str(ch.id),
                        "name": ch.name,
                        "guild_id": str(guild.id),
                        "guild_name": guild.name,
                    })
        return channels

    async def _async_list_guilds(self) -> list[dict]:
        """Return summary dicts for every guild the bot is in."""
        return [
            {"id": str(g.id), "name": g.name, "member_count": g.member_count}
            for g in self._client.guilds
        ]

    # ── Thread entry point ────────────────────────────────────────────────────

    def _run(self) -> None:
        """Run the asyncio event loop with reconnect logic until stopped."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        backoff = _RECONNECT_BASE
        while not self._stop_event.is_set():
            try:
                self._loop.run_until_complete(self._connect_and_run())
                backoff = _RECONNECT_BASE
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                logger.warning(
                    "[discord-bot] connection error: %s — retrying in %ds", exc, backoff
                )
                self._stop_event.wait(timeout=backoff)
                backoff = min(backoff * 2, _RECONNECT_MAX)
        self._loop.close()

    async def _connect_and_run(self) -> None:
        """Build the discord.py Client, register on_message, and run it."""
        import discord as _discord

        intents = _discord.Intents.default()
        intents.message_content = True
        self._client = _discord.Client(intents=intents)

        @self._client.event
        async def on_ready():
            logger.info("[discord-bot] logged in as %s", self._client.user)
            self._ready_event.set()

        @self._client.event
        async def on_message(message):
            if message.author == self._client.user:
                return
            await self._dispatch_inbound(message)

        await self._client.start(self._token)

    async def _dispatch_inbound(self, message) -> None:
        """Validate filters and dispatch an inbound message to the callback."""
        import discord as _discord

        guild_name = message.guild.name if message.guild else "DM"
        channel_name = (
            message.channel.name
            if isinstance(message.channel, _discord.TextChannel)
            else "DM"
        )
        channel_id = message.channel.id
        guild_id = message.guild.id if message.guild else None

        if not self._is_message_allowed(guild_id, channel_id):
            return

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._inbound_executor,
            self._on_message,
            guild_name,
            channel_name,
            channel_id,
            message.author.display_name,
            message.content,
        )

    def _is_message_allowed(self, guild_id: int | None, channel_id: int) -> bool:
        """Return True when the message passes guild and channel filter checks.

        DMs (guild_id=None) are blocked when an explicit guild allowlist is set.
        """
        if guild_id is None:
            return not self._allowed_guilds
        if self._allowed_guilds and guild_id not in self._allowed_guilds:
            return False
        if self._allowed_channels and channel_id not in self._allowed_channels:
            return False
        return True
