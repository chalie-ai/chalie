"""Telegram capability — the user's own bot, used to message that one user.

The user creates a bot with @BotFather and pastes its token into setup, then
says hi to the bot. The first private chat the bot receives is recorded as the
one chat Chalie may message, with the name Telegram gives for it, so the user
can see whose chat it is. It is never replaced by another chat's message: only
a disconnect or a new token resets it. Outside a send that waits for the
user's reply (:mod:`reply_wait`), nothing else is read from the bot. Every
request goes through :mod:`bot_api`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from capabilities.base import AbstractCapability
from capabilities.telegram_capability.bot_api import call
from capabilities.telegram_capability.reply_wait import await_reply, drain, next_offset
from exceptions import TelegramApiError
from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)

_MANIFEST_PATH = FileMapperService.get_capabilities_path("telegram_capability", "manifest.yaml")

_MAX_MESSAGE_LENGTH = 4096
# The shape @BotFather issues, "<bot id>:<secret>". Holding a token to it keeps
# the token one plain URL path segment, so it can always be cut out of a text.
_TOKEN_SHAPE = re.compile(r"\d+:[A-Za-z0-9_-]+")
_UNNAMED = "a Telegram user with no name"

_K_TOKEN = "telegram:bot_token"
_K_USERNAME = "telegram:bot_username"
_K_CHAT_ID = "telegram:chat_id"
_K_CHAT_NAME = "telegram:chat_name"


def _split(text: str) -> list[str]:
    """Cut *text* into consecutive pieces Telegram accepts, losing nothing.

    Length is counted in UTF-16 units, the stricter of the two ways the limit
    can be read, so a piece full of emoji is never over it either way."""
    pieces: list[str] = []
    start = units = 0
    for i, char in enumerate(text):
        width = 2 if ord(char) > 0xFFFF else 1
        if units + width > _MAX_MESSAGE_LENGTH:
            pieces.append(text[start:i])
            start, units = i, 0
        units += width
    pieces.append(text[start:])
    return pieces


def _private_chat(update: object) -> dict[str, object] | None:
    message = update.get("message") if isinstance(update, dict) else None
    chat = message.get("chat") if isinstance(message, dict) else None
    if isinstance(chat, dict) and chat.get("type") == "private" and isinstance(chat.get("id"), int):
        return chat
    return None


def _display_name(chat: dict[str, object]) -> str:
    """The person's first and last name, then their @username when they
    have one."""
    name = " ".join(part for part in (chat.get("first_name"), chat.get("last_name")) if isinstance(part, str) and part)
    username = chat.get("username")
    if isinstance(username, str) and username:
        return f"{name} (@{username})" if name else f"@{username}"
    return name or _UNNAMED


class TelegramCapability(AbstractCapability):

    def __init__(self) -> None:
        super().__init__(_MANIFEST_PATH)

    def get_id(self) -> str:
        return "telegram"

    # ── Lifecycle ────────────────────────────────────────────────────

    def configure(self, credentials: dict[str, object]) -> None:
        """Check the token with Telegram, store it, then look for the user's
        hello. The ``ValueError`` text reaches the log, so it never holds the
        token."""
        token = str(credentials.get("bot_token") or "").strip()
        if not _TOKEN_SHAPE.fullmatch(token):
            raise ValueError("This is not a Telegram bot token; paste the whole token @BotFather sent.")
        try:
            me = call(token, "getMe")
        except TelegramApiError as exc:
            raise ValueError(f"Could not check the bot token. {exc}") from None
        username = me.get("username") if isinstance(me, dict) else None
        if not isinstance(username, str) or not username:
            raise ValueError("Telegram named no username for this bot.")

        if self.load_credential(_K_TOKEN) != token:
            self.store_credential(_K_CHAT_ID, "")
            self.store_credential(_K_CHAT_NAME, "")
        self.store_credential(_K_TOKEN, token)
        self.store_credential(_K_USERNAME, username)
        self._connected = True
        logger.info("[telegram] Connected bot @%s.", username)

        try:
            self._record_chat(token)
        except TelegramApiError:
            logger.warning("[telegram] Could not look for the user's hello yet; the first send looks again.")

    def connect(self) -> bool:
        """Connected exactly when a bot token is stored. No call to Telegram, so
        a restart never depends on reaching it. A locked vault raises
        ``VaultLockedError``, as for every capability."""
        self._connected = bool(self.load_credential(_K_TOKEN))
        return self._connected

    def disconnect(self) -> None:
        self._connected = False
        self.delete_credentials()
        logger.info("[telegram] Disconnected and credentials removed.")

    def status_note(self) -> str | None:
        """Whose chat messages go to, or how to give Chalie one, so a stranger
        who messaged the bot first is plain to see."""
        if not self.is_connected():
            return None
        if self.load_credential(_K_CHAT_ID):
            return f"Messages go to {self._chat_name()}. Not you? Disconnect Telegram and set it up again."
        return f"Open {self._bot_link()} and press Start so Chalie knows where to send messages."

    # ── Cognitive pipeline ───────────────────────────────────────────

    def ingest(self) -> list[object]:
        """Chalie reads no Telegram inbox."""
        return []

    def understand(self, items: list[object]) -> list[object]:
        return items

    def _do_monitor(self) -> None:
        """Nothing to poll: the bot only sends, and its one chat is recorded at
        setup or at the first send."""
        return

    # ── The one chat ─────────────────────────────────────────────────

    def _record_chat(self, token: str) -> str | None:
        """Return the recorded chat id, first recording it if there is none.

        The chat is the first private chat among the bot's pending updates,
        stored with the person's name. Every update read is confirmed, so
        Telegram drops it. Raises :class:`TelegramApiError` when Telegram
        cannot be read."""
        recorded = self.load_credential(_K_CHAT_ID)
        if recorded:
            return recorded
        updates = call(token, "getUpdates", {"timeout": 0})
        if not isinstance(updates, list):
            return None
        chat = next((c for c in map(_private_chat, updates) if c is not None), None)
        chat_id: str | None = None
        if chat is not None:
            chat_id = str(chat["id"])
            self.store_credential(_K_CHAT_NAME, _display_name(chat))
            self.store_credential(_K_CHAT_ID, chat_id)
            logger.info("[telegram] Recorded the chat to message.")
        offset = next_offset(updates, None)
        if offset is not None:
            call(token, "getUpdates", {"offset": offset, "timeout": 0})
        return chat_id

    def _chat_name(self) -> str:
        return self.load_credential(_K_CHAT_NAME) or _UNNAMED

    def _bot_link(self) -> str:
        return f"https://t.me/{self.load_credential(_K_USERNAME)}"

    def send(self, message: str, should_stop: Callable[[], bool] | None = None) -> dict[str, object]:
        """Send *message* to the recorded chat, split into several messages
        when too long. A success names who it went to in ``to``.

        With *should_stop*, the updates already waiting are dropped first, and
        after the last piece the call waits, with no time limit, for the user's
        reply: the result carries ``reply``, or ``stopped`` when *should_stop*
        said the turn was stopped first. A failure is an ``error`` text that
        says how much was sent."""
        token = self.load_credential(_K_TOKEN)
        if not token:
            return {"error": "Telegram is not connected. Set it up under Capabilities in the Brain dashboard."}
        pieces = _split(message)
        sent = 0
        offset: int | None = None
        try:
            chat_id = self._record_chat(token)
            if chat_id is None:
                return {"error": (
                    "Nothing was sent: no one has said hi to the bot yet, so there is no chat to "
                    f"message. Ask the user to open {self._bot_link()} and press Start, then try again."
                )}
            to = self._chat_name()
            if should_stop is not None:
                offset = drain(token)
            for piece in pieces:
                call(token, "sendMessage", {"chat_id": int(chat_id), "text": piece})
                sent += 1
        except TelegramApiError as exc:
            outcome = f"Only {sent} of {len(pieces)} messages were sent." if sent else "Nothing was sent."
            return {"error": f"{exc} {outcome}"}
        if should_stop is None:
            return {"sent": True, "messages": sent, "to": to}

        logger.info("[telegram] Sent; waiting for the user's reply.")
        try:
            reply = await_reply(token, int(chat_id), offset, should_stop)
        except TelegramApiError as exc:
            return {"error": f"{exc} The message was sent to {to}, but no reply can be read, so the wait ended."}
        if reply is None:
            logger.info("[telegram] Stopped waiting for the user's reply.")
            return {"sent": True, "messages": sent, "to": to, "stopped": True}
        logger.info("[telegram] The user replied.")
        return {"sent": True, "messages": sent, "to": to, "reply": reply}

    # ── Tool handlers ────────────────────────────────────────────────

    def _th_send(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        return self.send(str(params.get("message") or ""))

    def get_tools(self) -> list[dict[str, object]]:
        return [
            {"name": "send", "handler": self._th_send, "timeout": 60,
             "description": "Send the user a plain-text Telegram message, split into several when too long.",
             "parameters": {"message": {"type": "string", "description": "The text to send."}}},
        ]
