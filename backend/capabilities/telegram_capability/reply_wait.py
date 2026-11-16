"""Waiting for the user's reply on Telegram, after a send that asks for one.

The bot's updates are read with ``getUpdates`` and confirmed by offset, so
Telegram drops each one once read. Only a message in the recorded chat counts
as the reply.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from capabilities.telegram_capability.bot_api import TIMEOUT, Outcome, post
from exceptions import TelegramApiError

logger = logging.getLogger(__name__)

# While waiting for a reply, Telegram holds each read open this long when
# nothing has arrived, so a stopped turn is seen within one read.
_POLL_SECONDS = 10
# Pauses before reading again after a read failed: another program reading
# this bot's updates, or Telegram being unreachable or failing on its side.
_CONFLICT_PAUSE = 1
_OUTAGE_PAUSE = 5

_WEBHOOK_SET = (
    "This bot has a webhook set, so Telegram sends its messages there and Chalie "
    "cannot read the user's reply."
)

# What a message without text or caption is called, checked in this order:
# Telegram also marks a GIF as a document, and a video message is not a video.
_MESSAGE_KINDS = (
    ("photo", "a photo"),
    ("voice", "a voice message"),
    ("audio", "an audio file"),
    ("video_note", "a video message"),
    ("video", "a video"),
    ("animation", "a GIF"),
    ("document", "a file"),
    ("sticker", "a sticker"),
    ("location", "a location"),
    ("contact", "a contact"),
    ("poll", "a poll"),
)


def next_offset(updates: list[object], offset: int | None) -> int | None:
    """The offset that confirms every update in *updates*, so Telegram drops
    them; *offset* itself when none carries an id."""
    ids = [u["update_id"] for u in updates if isinstance(u, dict) and isinstance(u.get("update_id"), int)]
    return max(ids) + 1 if ids else offset


def _reply_text(update: object, chat_id: int) -> str | None:
    """What the user said in *update* when it is a message in the recorded
    chat, else ``None``. A message with neither text nor caption is named by
    its kind, so it still counts as the reply."""
    message = update.get("message") if isinstance(update, dict) else None
    if not isinstance(message, dict):
        return None
    chat = message.get("chat")
    if not isinstance(chat, dict) or chat.get("id") != chat_id:
        return None
    for field in ("text", "caption"):
        said = message.get(field)
        if isinstance(said, str) and said:
            return said
    kind = next((name for key, name in _MESSAGE_KINDS if key in message), "a message")
    return f"[{kind}, with no text]"


def _updates_payload(offset: int | None, hold: int) -> dict[str, str | int]:
    payload: dict[str, str | int] = {"timeout": hold}
    if offset is not None:
        payload["offset"] = offset
    return payload


def _webhook_set(outcome: Outcome) -> bool:
    """Whether a read failed because the bot has a webhook. Telegram answers
    that and a second reader alike with 409, telling them apart only in the
    description."""
    return outcome.status == 409 and "webhook" in outcome.error.lower()


def _pause_after(outcome: Outcome) -> int:
    """Seconds to wait before reading again after a failed read. Raises
    :class:`TelegramApiError` for a refusal that waiting cannot fix."""
    if _webhook_set(outcome):
        raise TelegramApiError(_WEBHOOK_SET)
    if outcome.status == 409:
        return _CONFLICT_PAUSE
    if outcome.status in (0, 429) or outcome.status >= 500:
        return _OUTAGE_PAUSE
    raise TelegramApiError(outcome.error)


def drain(token: str) -> int | None:
    """Confirm every update already waiting, so nothing the user sent before
    this point is taken as their reply. Returns the offset to read from next.
    Raises :class:`TelegramApiError` when the updates cannot be read."""
    offset: int | None = None
    while True:
        outcome = post(token, "getUpdates", _updates_payload(offset, 0))
        if _webhook_set(outcome):
            raise TelegramApiError(_WEBHOOK_SET)
        if outcome.error:
            raise TelegramApiError(outcome.error)
        updates = outcome.result if isinstance(outcome.result, list) else []
        following = next_offset(updates, offset)
        if following == offset:
            return offset
        offset = following


def await_reply(token: str, chat_id: int, offset: int | None, should_stop: Callable[[], bool]) -> str | None:
    """Read the bot's updates from *offset* until a message arrives in the
    recorded chat, and return what it said. Every other update is confirmed
    and skipped. ``None`` once *should_stop* says the turn was stopped, which
    is checked before every read.

    There is no time limit, so a second reader of the bot and an outage are
    waited out, with one warning each time reading starts failing. Raises
    :class:`TelegramApiError` when Telegram refuses in a way waiting cannot
    fix."""
    failing = False
    while not should_stop():
        outcome = post(
            token, "getUpdates", _updates_payload(offset, _POLL_SECONDS), timeout=TIMEOUT + _POLL_SECONDS,
        )
        if outcome.error:
            pause = _pause_after(outcome)
            if not failing:
                failing = True
                logger.warning("[telegram] %s Still waiting for the user's reply; trying again.", outcome.error)
            time.sleep(pause)
            continue
        if failing:
            failing = False
            logger.info("[telegram] Reading the bot's updates again.")
        updates = outcome.result if isinstance(outcome.result, list) else []
        for update in updates:
            reply = _reply_text(update, chat_id)
            if reply is not None:
                return reply
        offset = next_offset(updates, offset)
    return None
