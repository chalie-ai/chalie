"""Telegram capability — the user's own bot, used to message that one user.

The user creates a bot with @BotFather and pastes its token into setup. Chalie
then shows a 6-digit code for the user to send the bot, and reads the bot in
the background for up to five minutes (:func:`reply_wait.await_code`). The
private chat that sends the code becomes the one chat Chalie may message, with
the name Telegram gives for it, and a confirmation is sent there. Anything else
— the wrong text, no code in time, Telegram refusing — ends the attempt with
nothing stored. Past setup, only a send that waits for the user's reply
(:mod:`reply_wait`) reads from the bot. Every request goes through
:mod:`bot_api`.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from capabilities.base import AbstractCapability
from capabilities.telegram_capability.bot_api import call
from capabilities.telegram_capability.reply_wait import await_code, await_reply, drain
from exceptions import TelegramApiError, VaultLockedError
from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)

_MANIFEST_PATH = FileMapperService.get_capabilities_path("telegram_capability", "manifest.yaml")

_MAX_MESSAGE_LENGTH = 4096
# The shape @BotFather issues, "<bot id>:<secret>". Holding a token to it keeps
# the token one plain URL path segment, so it can always be cut out of a text.
_TOKEN_SHAPE = re.compile(r"\d+:[A-Za-z0-9_-]+")
_UNNAMED = "a Telegram user with no name"

# How long a setup attempt waits for the user to send the bot its code.
_VERIFY_SECONDS = 300
_FAILED = "Could not connect Telegram, please try again."
_CONFIRMATION = "Chalie is connected. Its messages will come to this chat."

_K_TOKEN = "telegram:bot_token"
_K_CHAT_ID = "telegram:chat_id"
_K_CHAT_NAME = "telegram:chat_name"


@dataclass(frozen=True)
class _Attempt:
    """A setup attempt waiting for its code. The token stays here, unstored,
    until the code arrives."""

    token: str
    username: str
    code: str


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


def _how_much_was_sent(sent: int, texts: int, voice: bool) -> str:
    """What reached the user when a send failed after *sent* messages, of
    *texts* text messages and a voice note when *voice*."""
    if not sent:
        return "Nothing was sent."
    if voice and sent == texts:
        return "The text was sent, but not the voice note."
    return f"Only {sent} of {texts + voice} messages were sent."


def _display_name(chat: dict[str, object]) -> str:
    """The person's first and last name, then their @username when they
    have one."""
    name = " ".join(part for part in (chat.get("first_name"), chat.get("last_name")) if isinstance(part, str) and part)
    username = chat.get("username")
    if isinstance(username, str) and username:
        return f"{name} (@{username})" if name else f"@{username}"
    return name or _UNNAMED


def _check_token(token: str) -> tuple[str, int | None]:
    """The bot's username, and the offset past every update already waiting,
    so nothing sent to the bot before this attempt counts. Raises
    ``ValueError`` saying why the token cannot be used, never holding it."""
    if not _TOKEN_SHAPE.fullmatch(token):
        raise ValueError("This is not a Telegram bot token; paste the whole token @BotFather sent.")
    try:
        me = call(token, "getMe")
    except TelegramApiError as exc:
        raise ValueError(f"Could not check the bot token. {exc}") from None
    username = me.get("username") if isinstance(me, dict) else None
    if not isinstance(username, str) or not username:
        raise ValueError("Telegram named no username for this bot.")
    try:
        return username, drain(token)
    except TelegramApiError as exc:
        raise ValueError(str(exc)) from None


class TelegramCapability(AbstractCapability):

    def __init__(self) -> None:
        super().__init__(_MANIFEST_PATH)
        # Guards the attempt against a disconnect landing while it connects.
        self._lock = threading.Lock()
        self._attempt: _Attempt | None = None
        self._failure: str | None = None

    def get_id(self) -> str:
        return "telegram"

    # ── Lifecycle ────────────────────────────────────────────────────

    def configure(self, credentials: dict[str, object]) -> None:
        """Start a setup attempt: check the token with Telegram, then wait in
        the background for the code :meth:`setup_code` shows. Every attempt
        starts afresh, so the old connection and any attempt still waiting end
        first. A failure raises ``ValueError``, whose text reaches the log, so
        it never holds the token."""
        self.disconnect()
        token = str(credentials.get("bot_token") or "").strip()
        try:
            username, offset = _check_token(token)
        except ValueError as exc:
            self._failure = f"{_FAILED} {exc}"
            raise
        attempt = _Attempt(token, username, f"{secrets.randbelow(1_000_000):06d}")
        with self._lock:
            self._attempt = attempt
        threading.Thread(
            target=self._verify, args=(attempt, offset), name="telegram-setup", daemon=True,
        ).start()
        logger.info("[telegram] Waiting for the setup code to reach bot @%s.", username)

    def setup_pending(self) -> bool:
        return self._attempt is not None

    def setup_code(self) -> str | None:
        attempt = self._attempt
        return attempt.code if attempt is not None else None

    def connect(self) -> bool:
        """Connected exactly when a bot token and its chat are stored. No call
        to Telegram, so a restart never depends on reaching it. A locked vault
        raises ``VaultLockedError``, as for every capability."""
        self._connected = bool(self.load_credential(_K_TOKEN) and self.load_credential(_K_CHAT_ID))
        return self._connected

    def disconnect(self) -> None:
        """End the connection, or the attempt still waiting for its code, and
        remove everything stored."""
        with self._lock:
            self._attempt = None
            self._failure = None
            self._connected = False
            self.delete_credentials()
        logger.info("[telegram] Disconnected and credentials removed.")

    def status_note(self) -> str | None:
        """Where to send the code while an attempt waits, why the last
        attempt failed, or whose chat messages go to."""
        attempt = self._attempt
        if attempt is not None:
            return (
                f"Open https://t.me/{attempt.username}, press Start, then send the bot "
                f"this code within {_VERIFY_SECONDS // 60} minutes:"
            )
        if self._failure:
            return self._failure
        if self.is_connected():
            return f"Messages go to {self._chat_name()}."
        return None

    # ── Setup attempt ────────────────────────────────────────────────

    def _verify(self, attempt: _Attempt, offset: int | None) -> None:
        """Wait for *attempt*'s code, then connect or fail. Runs on its own
        thread; an unexpected error still ends the attempt, then propagates."""
        try:
            chat = await_code(attempt.token, attempt.code, offset, _VERIFY_SECONDS, lambda: self._attempt is not attempt)
            if chat is not None:
                self._connect_chat(attempt, chat)
        except (TelegramApiError, TimeoutError, ValueError, VaultLockedError) as exc:
            self._fail(attempt, str(exc))
        except Exception:
            self._fail(attempt, "Something went wrong on Chalie's side.")
            raise

    def _connect_chat(self, attempt: _Attempt, chat: dict[str, object]) -> None:
        """Store the connection, then send the confirmation to *chat*, unless
        a newer attempt or a disconnect ended *attempt* first. A confirmation
        Telegram refuses fails the attempt, which removes what was stored."""
        chat_id = cast(int, chat["id"])
        with self._lock:
            if self._attempt is not attempt:
                return
            self.store_credential(_K_TOKEN, attempt.token)
            self.store_credential(_K_CHAT_ID, str(chat_id))
            self.store_credential(_K_CHAT_NAME, _display_name(chat))
            call(attempt.token, "sendMessage", {"chat_id": chat_id, "text": _CONFIRMATION})
            self._attempt = None
            self._connected = True
        logger.info("[telegram] Connected bot @%s.", attempt.username)

    def _fail(self, attempt: _Attempt, reason: str) -> None:
        """End *attempt* with *reason* on the card and nothing stored, unless
        it was already over."""
        with self._lock:
            if self._attempt is not attempt:
                return
            self._attempt = None
            self._failure = f"{_FAILED} {reason}"
            self.delete_credentials()
        logger.warning("[telegram] Could not connect: %s", reason)

    # ── Cognitive pipeline ───────────────────────────────────────────

    def ingest(self) -> list[object]:
        """Chalie reads no Telegram inbox."""
        return []

    def understand(self, items: list[object]) -> list[object]:
        return items

    def _do_monitor(self) -> None:
        """Nothing to poll: the bot only sends, and setup reads its own code."""
        return

    # ── Sending ──────────────────────────────────────────────────────

    def _chat_name(self) -> str:
        return self.load_credential(_K_CHAT_NAME) or _UNNAMED

    def send(
        self, message: str, should_stop: Callable[[], bool] | None = None, audio: str | None = None,
    ) -> dict[str, object]:
        """Send *message* to the user's chat, split into several messages
        when too long, then the audio file at *audio*, when given, as a voice
        note. A success names who it went to in ``to``.

        With *should_stop*, the updates already waiting are dropped first, and
        after the last message the call waits, with no time limit, for the
        user's reply: the result carries ``reply``, or ``stopped`` when
        *should_stop* said the turn was stopped first. A failure is an
        ``error`` text that says how much was sent."""
        token = self.load_credential(_K_TOKEN)
        chat_id = self.load_credential(_K_CHAT_ID)
        if not (token and chat_id):
            return {"error": "Telegram is not connected. Set it up under Capabilities in the Brain dashboard."}
        if audio and not os.path.isfile(audio):
            return {"error": f"There is no audio file at {audio}. Nothing was sent."}
        to = self._chat_name()
        pieces = _split(message) if message.strip() else []
        sent = 0
        offset: int | None = None
        try:
            if should_stop is not None:
                offset = drain(token)
            for piece in pieces:
                call(token, "sendMessage", {"chat_id": int(chat_id), "text": piece})
                sent += 1
            if audio:
                call(token, "sendVoice", {"chat_id": int(chat_id)}, upload=("voice", audio))
                sent += 1
        except TelegramApiError as exc:
            return {"error": f"{exc} {_how_much_was_sent(sent, len(pieces), bool(audio))}"}
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
        return self.send(str(params.get("message") or ""), audio=str(params.get("audio") or "") or None)

    def get_tools(self) -> list[dict[str, object]]:
        return [
            {"name": "send", "handler": self._th_send, "timeout": 60,
             "description": (
                 "Send the user a plain-text Telegram message, split into several when too long, "
                 "and an audio file as a voice note."
             ),
             "parameters": {
                 "message": {"type": "string", "description": "The text to send."},
                 "audio": {"type": "string", "description": "The full path of an audio file to send as a voice note."},
             }},
        ]
