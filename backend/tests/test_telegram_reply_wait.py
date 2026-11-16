"""Feature tests: a Telegram message that asks for a reply waits for it.

A chat turn calls the telegram tool with ``wait_for_reply`` through the real
dispatcher, on a thread of its own as a real turn runs, with its turn row open
so Stop reaches it. The real capability sends, reads the bot's updates and
returns what the user said.

Telegram is the stand-in Bot API from ``test_telegram_ability``: a real HTTP
server on loopback that holds a waiting read open until a message arrives, and
can be told what happens once the bot starts waiting. The capability's API base
URL is the one thing redirected. A call that is still waiting when a test gives
up is ended by a reply, so a broken wait fails its test instead of hanging the
run.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Iterator

import pytest
from flask.testing import FlaskClient

from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from services.database import Database
from services.memory_store import MemoryStore
from services.turn_execution_service import TurnExecutionService
from tests.test_telegram_ability import (
    _APP_PACKAGES,
    _STRANGER_CHAT,
    _TOKEN,
    _USER_CHAT,
    _USER_FIRST,
    _USER_HANDLE,
    _USER_LAST,
    _bot_api,
    _connect_with_hello,
    _head,
    _open_vault_and_own_registry,
    _sent_to,
    _Telegram,
)

pytestmark = pytest.mark.unit

# How long a test lets a call wait before ending it with a reply and failing.
_WAIT_LIMIT_SECONDS = 10.0
_GIVE_UP_REPLY = "the test gave up waiting"
_PHOTO = {"file_id": "AgACAgQAAxkBAAIB", "file_unique_id": "AQADa", "width": 1280, "height": 960}


@pytest.fixture()
def telegram(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Telegram]:
    with _bot_api(monkeypatch) as state:
        yield state


@pytest.fixture()
def client(authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> Iterator[FlaskClient]:
    with _open_vault_and_own_registry():
        yield authed_client[0]


class _Turn:
    """A chat turn making one telegram call on its own thread, its turn row
    open, as a real turn runs."""

    def __init__(self, params: dict[str, object]) -> None:
        self.channel = ""
        self.turn_id = 0
        self._params = params
        self._result: str | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            mp = MessageProcessor(UserConfig(), raw_input="ask them on Telegram")
            assert mp.turn_execution_service.open() is not None, "the turn row must be open for Stop to reach it"
            self.channel, self.turn_id = mp.channel, mp.turn_id
            self._result = mp.dispatch_service.dispatch("telegram", self._params)
        except BaseException as exc:  # handed to the test thread by outcome()
            self._error = exc
        finally:
            Database.close()

    def outcome(self, telegram: _Telegram) -> str:
        """The result the model reads, once the call returns."""
        self._thread.join(_WAIT_LIMIT_SECONDS)
        if self._thread.is_alive():
            telegram.end_every_wait(_TOKEN, _USER_CHAT, _GIVE_UP_REPLY)
            self._thread.join(_WAIT_LIMIT_SECONDS)
            pytest.fail(f"the call was still waiting after {_WAIT_LIMIT_SECONDS:.0f}s")
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def _ask(text: str) -> dict[str, object]:
    return {"message": text, "wait_for_reply": True}


def _names_the_user(result: str) -> bool:
    return f"{_USER_FIRST} {_USER_LAST}" in result and f"@{_USER_HANDLE}" in result


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_a_message_that_does_not_ask_for_a_reply_never_reads_what_comes_back(
    client: FlaskClient, telegram: _Telegram,
) -> None:
    _connect_with_hello(client, telegram)

    plain_sends: list[dict[str, object]] = [
        {"message": "Your backup finished"},
        {"message": "Your backup was checked", "wait_for_reply": False},
    ]
    for params in plain_sends:
        telegram.arrive_while_waiting = [(_USER_CHAT, {"text": "Thanks!"})]
        result = _Turn(params).outcome(telegram)
        assert "status=success" in _head(result), result
        assert "Thanks!" not in result

    methods = [method for _, method, _ in telegram.requests]
    assert methods.count("sendMessage") == 2
    assert "getUpdates" not in methods[methods.index("sendMessage"):], "nothing may read updates after a plain send"


def test_the_reply_is_the_users_next_message_never_an_earlier_one_or_a_strangers(
    client: FlaskClient, telegram: _Telegram,
) -> None:
    _connect_with_hello(client, telegram)
    telegram.message(_TOKEN, _USER_CHAT, text="Remind me about the dentist")
    telegram.arrive_while_waiting = [
        (_STRANGER_CHAT, {"text": "Who is this?"}),
        (_USER_CHAT, {"text": "Yes, book the 9:15 train"}),
    ]

    result = _Turn(_ask("Shall I book the 9:15 train?")).outcome(telegram)

    assert "status=success" in _head(result), result
    assert "Yes, book the 9:15 train" in result
    assert "dentist" not in result, "a message sent before the question is not its reply"
    assert "Who is this?" not in result, "a stranger's message is not the user's reply"
    assert _names_the_user(result), result
    assert _sent_to(telegram) == [_USER_CHAT]


def test_a_photo_reply_without_a_caption_is_named_and_a_caption_is_the_reply(
    client: FlaskClient, telegram: _Telegram,
) -> None:
    _connect_with_hello(client, telegram)

    telegram.arrive_while_waiting = [(_USER_CHAT, {"photo": [_PHOTO]})]
    bare = _Turn(_ask("Send me a picture of the meter")).outcome(telegram)
    telegram.arrive_while_waiting = [(_USER_CHAT, {"photo": [_PHOTO], "caption": "Meter reads 4512"})]
    captioned = _Turn(_ask("Can you send it with the reading?")).outcome(telegram)

    assert "status=success" in _head(bare), bare
    assert "[a photo, with no text]" in bare
    assert "status=success" in _head(captioned), captioned
    assert "Meter reads 4512" in captioned
    assert "[a photo" not in captioned


def test_stopping_the_turn_ends_the_wait_without_a_reply(
    client: FlaskClient, telegram: _Telegram,
) -> None:
    _connect_with_hello(client, telegram)

    turn = _Turn(_ask("Are you home yet?"))
    assert telegram.waiting.wait(_WAIT_LIMIT_SECONDS), "the call never started waiting for the reply"
    assert TurnExecutionService.cancel_turn(turn.channel, turn.turn_id) is not None
    reads_at_stop = len(telegram.calls("getUpdates"))
    result = turn.outcome(telegram)

    assert "status=success" in _head(result), result
    assert "stopped" in result.lower(), result
    assert _names_the_user(result), result
    # Stop is checked before every read: only a read already on its way may follow it.
    assert len(telegram.calls("getUpdates")) - reads_at_stop <= 1
    assert _sent_to(telegram) == [_USER_CHAT]


def test_another_program_reading_the_bot_is_waited_out_and_the_reply_still_arrives(
    client: FlaskClient, telegram: _Telegram,
) -> None:
    _connect_with_hello(client, telegram)
    telegram.conflicts_while_waiting = 1
    telegram.arrive_while_waiting = [(_USER_CHAT, {"text": "Tuesday works"})]

    result = _Turn(_ask("Which day suits you?")).outcome(telegram)

    assert "status=success" in _head(result), result
    assert "Tuesday works" in result
    held_reads = [p for p in telegram.calls("getUpdates") if p.get("timeout")]
    assert len(held_reads) >= 2, "the refused read must have been tried again"


def test_a_webhook_ends_the_wait_naming_who_was_messaged_and_stops_the_next_ask_before_it_sends(
    client: FlaskClient, telegram: _Telegram,
) -> None:
    _connect_with_hello(client, telegram)
    telegram.webhook_while_waiting = True

    mid_wait = _Turn(_ask("Which day suits you?")).outcome(telegram)

    assert "status=error" in _head(mid_wait), mid_wait
    assert "webhook" in mid_wait.lower()
    assert _names_the_user(mid_wait), "the error must say the message did reach them"
    assert len(telegram.calls("sendMessage")) == 1

    before_sending = _Turn(_ask("Is any day this week fine?")).outcome(telegram)

    assert "status=error" in _head(before_sending), before_sending
    assert "webhook" in before_sending.lower()
    assert len(telegram.calls("sendMessage")) == 1, "a question whose reply cannot be read must not be sent"

    # A message that waits for nothing still goes out.
    assert "status=success" in _head(_Turn({"message": "I'll pick Friday"}).outcome(telegram))
    assert len(telegram.calls("sendMessage")) == 2


def test_neither_the_reply_nor_the_token_reaches_the_logs(
    client: FlaskClient, telegram: _Telegram, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    for package in _APP_PACKAGES:
        caplog.set_level(logging.DEBUG, logger=package)
    secret = _TOKEN.split(":", 1)[1]
    reply = "The spare key is under the blue flowerpot"
    _connect_with_hello(client, telegram)
    telegram.conflicts_while_waiting = 1
    telegram.arrive_while_waiting = [(_USER_CHAT, {"text": reply})]

    result = _Turn(_ask("Where is the spare key?")).outcome(telegram)

    assert reply in result, result
    assert caplog.records, "the send and the wait must have been logged"
    for record in caplog.records:
        message = record.getMessage()
        assert secret not in message, f"{record.name} logged the token"
        assert reply not in message, f"{record.name} logged the user's reply"
    assert secret not in caplog.text
    assert reply not in caplog.text
