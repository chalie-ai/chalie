"""Feature tests: the Telegram tool messages the user, and only the user.

The real setup endpoint checks a bot token with Telegram and starts the setup
attempt; the user sends the bot the code the card shows, and the private chat
that sends it becomes the one chat Chalie may message. The real dispatcher and
policy gate then decide whether a turn may send, and the real capability sends
to that chat.

Telegram is a real HTTP server on loopback that answers the Bot API methods the
capability uses the way Telegram does (an unknown token is refused, confirmed
updates are dropped, a text over 4096 characters is refused) and records every
request it receives. The capability's API base URL is the one thing redirected,
so every request in these tests reaches that server and none reaches Telegram.
The vault, database, setup and status endpoints, dispatcher and policy rows are
the real ones.
"""

from __future__ import annotations

import json
import logging
import re
import socket
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import pytest
from flask.testing import FlaskClient

import capabilities
import capabilities.telegram_capability.bot_api as bot_api
import capabilities.telegram_capability.capability as capability_module
import services.vault_service as vault_module
from abilities.telegram import TelegramAbility
from configs.channels.external_agent import EAMPConfig
from configs.channels.skill_suggestion import SkillSuggestionConfig
from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from services.memory_store import MemoryStore
from services.processor_config import ProcessorConfig
from services.vault_service import _vault_state, get_vault_service

pytestmark = pytest.mark.unit

_TOKEN = "7100000001:AAH-first_bot_secret"
_NEW_TOKEN = "7100000002:AAH-second_bot_secret"
_REVOKED_TOKEN = "7100000003:AAH-revoked_bot_secret"
_USERNAME = "my_chalie_bot"
_NEW_USERNAME = "my_other_chalie_bot"
_USER_CHAT = 4242
_STRANGER_CHAT = 9999
_GROUP_CHAT = -100123
_TELEGRAM_LIMIT = 4096
# What Telegram says about the person on each side of a private chat.
_USER_FIRST, _USER_LAST, _USER_HANDLE = "Ada", "Lovelace", "ada_l"
_PEOPLE: dict[int, dict[str, object]] = {
    _USER_CHAT: {"first_name": _USER_FIRST, "last_name": _USER_LAST, "username": _USER_HANDLE},
    _STRANGER_CHAT: {"first_name": "Mallory"},
}
_VAULT_PASSWORD = "test-password"
# How long the stand-in holds open a read that waits for updates.
_HELD_READ_SECONDS = 0.2
# How long a test lets a setup attempt run before failing.
_SETUP_LIMIT_SECONDS = 5.0
_FAILED = "Could not connect Telegram, please try again."
_CODE = re.compile(r"\b\d{6}\b")

# Every first-party package: its loggers are the app's loggers.
_APP_PACKAGES = sorted(
    p.name for p in Path(__file__).resolve().parent.parent.iterdir() if (p / "__init__.py").is_file()
)


# ── The stand-in Bot API ───────────────────────────────────────────────────────


class _Telegram:
    """What the stand-in Bot API knows, and every request it was sent.

    A read of the bot's updates that asks Telegram to hold it open (a
    ``timeout`` above 0) is how the bot waits for a reply: it returns as soon
    as an update arrives, or empty after a fraction of a second, so a wait
    loops quickly. What happens once the bot starts waiting is set up in
    advance: messages that arrive, reads refused because another program is
    reading the same bot, or a webhook being set."""

    def __init__(self) -> None:
        self.usernames: dict[str, str] = {}
        self.pending: dict[str, list[dict[str, object]]] = {}
        self.requests: list[tuple[str, str, dict[str, object]]] = []
        self.refuse_sends = False
        self.webhook = False
        self.waiting = threading.Event()
        self.arrive_while_waiting: list[tuple[int, dict[str, object]]] = []
        self.conflicts_while_waiting = 0
        self.webhook_while_waiting = False
        self._next_update_id = 500
        self._cond = threading.Condition()

    def add_bot(self, token: str, username: str) -> None:
        self.usernames[token] = username
        self.pending[token] = []

    def message(self, token: str, chat_id: int, chat_type: str = "private", text: str | None = "hi",
                **content: object) -> None:
        """Someone messages the bot; Telegram holds the update until the bot
        confirms it. *content* is the rest of the message, such as a photo."""
        with self._cond:
            self._add(token, chat_id, chat_type, {**content, **({"text": text} if text is not None else {})})

    def end_every_wait(self, token: str, chat_id: int, text: str) -> None:
        """Clear every trouble and have *chat_id* reply, so a call still
        waiting returns."""
        with self._cond:
            self.webhook = self.webhook_while_waiting = False
            self.conflicts_while_waiting = 0
            self._add(token, chat_id, "private", {"text": text})

    def forget_requests(self) -> None:
        with self._cond:
            self.requests = []

    def calls(self, method: str, token: str | None = None) -> list[dict[str, object]]:
        with self._cond:
            return [p for t, m, p in self.requests if m == method and (token is None or t == token)]

    def answer(self, token: str, method: str, payload: dict[str, object]) -> tuple[int, dict[str, object]]:
        with self._cond:
            self.requests.append((token, method, payload))
            if token not in self.usernames:
                return 401, {"ok": False, "error_code": 401, "description": "Unauthorized"}
            if method == "getMe":
                bot_id = int(token.split(":", 1)[0])
                return 200, {"ok": True, "result": {
                    "id": bot_id, "is_bot": True, "first_name": "Chalie", "username": self.usernames[token],
                }}
            if method == "getUpdates":
                return self._updates(token, payload)
            if method == "sendMessage":
                text = str(payload.get("text"))
                if self.refuse_sends:
                    # Echoes the token so the test proves it is cut out of what Chalie keeps.
                    return 400, {"ok": False, "error_code": 400,
                                 "description": f"Bad Request: chat not found for bot{token}"}
                if len(text.encode("utf-16-le")) // 2 > _TELEGRAM_LIMIT:
                    return 400, {"ok": False, "error_code": 400, "description": "Bad Request: message is too long"}
                return 200, {"ok": True, "result": {
                    "message_id": len(self.requests), "chat": {"id": payload.get("chat_id"), "type": "private"},
                    "text": text,
                }}
            return 404, {"ok": False, "error_code": 404, "description": "Not Found"}

    def _updates(self, token: str, payload: dict[str, object]) -> tuple[int, dict[str, object]]:
        """``getUpdates``, called with the lock held."""
        hold = payload.get("timeout")
        held = isinstance(hold, int) and hold > 0
        if held:
            self.waiting.set()
            self.webhook = self.webhook or self.webhook_while_waiting
            for chat_id, content in self.arrive_while_waiting:
                self._add(token, chat_id, "private", content)
            self.arrive_while_waiting = []
        if self.webhook:
            return 409, {"ok": False, "error_code": 409, "description": (
                "Conflict: can't use getUpdates method while webhook is active; "
                "use deleteWebhook to delete the webhook first"
            )}
        if held and self.conflicts_while_waiting:
            self.conflicts_while_waiting -= 1
            return 409, {"ok": False, "error_code": 409, "description": (
                "Conflict: terminated by other getUpdates request; make sure that only one bot instance is running"
            )}
        offset = payload.get("offset")
        if isinstance(offset, int):
            self.pending[token] = [u for u in self.pending[token] if cast(int, u["update_id"]) >= offset]
        if held:
            self._cond.wait_for(lambda: bool(self.pending[token]), timeout=_HELD_READ_SECONDS)
        return 200, {"ok": True, "result": list(self.pending[token])}

    def _add(self, token: str, chat_id: int, chat_type: str, content: dict[str, object]) -> None:
        """One update, called with the lock held."""
        chat: dict[str, object] = {"id": chat_id, "type": chat_type}
        if chat_type == "private":
            chat.update(_PEOPLE.get(chat_id, {}))
        update_id = self._next_update_id
        self._next_update_id += 1
        message = {"message_id": update_id, "chat": chat, **content}
        self.pending[token].append({"update_id": update_id, "message": message})
        self._cond.notify_all()


class _BotApiServer(ThreadingHTTPServer):
    telegram: _Telegram


class _BotApiHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's contract
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        payload = cast("dict[str, object]", json.loads(body or b"{}"))
        bot, _, method = self.path.lstrip("/").partition("/")
        status, reply = cast(_BotApiServer, self.server).telegram.answer(bot.removeprefix("bot"), method, payload)
        data = json.dumps(reply).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        """Silence the per-request stderr line the default handler prints."""


# ── Fixtures ───────────────────────────────────────────────────────────────────


@contextmanager
def _bot_api(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Telegram]:
    """The stand-in Bot API on loopback, with the capability pointed at it."""
    state = _Telegram()
    server = _BotApiServer(("127.0.0.1", 0), _BotApiHandler)
    server.telegram = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(bot_api, "_API_BASE", f"http://127.0.0.1:{server.server_address[1]}")
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def _open_vault_and_own_registry() -> Iterator[None]:
    """An unlocked vault and a capability registry of this test's own."""
    _vault_state.dek = None
    vault_module._vault_service_instance = None
    vault = get_vault_service()
    vault.initialize(_VAULT_PASSWORD)
    vault.unlock(_VAULT_PASSWORD)
    saved_registry = capabilities._capabilities_cache
    capabilities._capabilities_cache = None
    try:
        yield
    finally:
        capabilities._capabilities_cache = saved_registry
        _vault_state.dek = None
        vault_module._vault_service_instance = None


@pytest.fixture()
def telegram(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Telegram]:
    with _bot_api(monkeypatch) as state:
        yield state


@pytest.fixture()
def client(authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> Iterator[FlaskClient]:
    """The real app with an unlocked vault and a capability registry of its own."""
    with _open_vault_and_own_registry():
        yield authed_client[0]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _save_token(client: FlaskClient, token: str) -> int:
    return client.post("/api/capabilities/setup/telegram", json={"bot_token": token}).status_code


def _status(client: FlaskClient) -> dict[str, object]:
    resp = client.get("/api/capabilities/status/telegram")
    assert resp.status_code == 200
    return cast("dict[str, dict[str, object]]", resp.get_json())["result"]


def _connected(client: FlaskClient) -> bool:
    return _status(client)["connected"] is True


def _code(client: FlaskClient) -> str:
    """The code the waiting setup asks the user to send the bot."""
    status = _status(client)
    assert status["pending"] is True, status
    code = str(status["setup_code"])
    assert _CODE.fullmatch(code), status
    return code


def _settled(client: FlaskClient) -> dict[str, object]:
    """The status once the setup attempt has ended."""
    deadline = time.monotonic() + _SETUP_LIMIT_SECONDS
    while (status := _status(client))["pending"]:
        assert time.monotonic() < deadline, f"the setup attempt was still waiting after {_SETUP_LIMIT_SECONDS:.0f}s"
        time.sleep(0.05)
    assert status["setup_code"] is None, "an attempt that ended shows no code"
    return status


def _setup_thread_ended() -> None:
    """Wait for the setup attempt's background reads to finish."""
    for thread in threading.enumerate():
        if thread.name == "telegram-setup":
            thread.join(_SETUP_LIMIT_SECONDS)
            assert not thread.is_alive(), "the setup attempt kept reading after it ended"


def _stored_rows(db: sqlite3.Connection) -> int:
    return int(db.execute("SELECT COUNT(*) FROM tool_configs WHERE tool_name = 'telegram'").fetchone()[0])


def _send(text: str, config: ProcessorConfig | None = None) -> str:
    """One telegram call through the real dispatcher and policy gate, as a turn
    on *config*'s channel (a chat turn by default) makes it. Returns the result
    the model reads."""
    mp = MessageProcessor(config or UserConfig(), raw_input="tell me on Telegram")
    return mp.dispatch_service.dispatch("telegram", {"message": text})


def _head(result: str) -> str:
    return result.split("\n", 1)[0]


def _connect(
    client: FlaskClient, telegram: _Telegram, token: str = _TOKEN, username: str = _USERNAME, chat: int = _USER_CHAT,
) -> None:
    """The user sets up their bot: saves its token, presses Start in it and
    sends it the code. Then the requests setup made are forgotten, so a test
    counts only its own."""
    if token not in telegram.usernames:
        telegram.add_bot(token, username)
    assert _save_token(client, token) == 204
    code = _code(client)
    telegram.message(token, chat, text="/start")
    telegram.message(token, chat, text=code)
    assert _settled(client)["connected"] is True
    assert [(p["chat_id"], p["text"]) for p in telegram.calls("sendMessage")] == [
        (chat, capability_module._CONFIRMATION),
    ]
    telegram.waiting.clear()
    telegram.forget_requests()


def _sent_to(telegram: _Telegram) -> list[object]:
    return [p["chat_id"] for p in telegram.calls("sendMessage")]


def _card(client: FlaskClient) -> dict[str, object]:
    """The Telegram card's row in the capability list."""
    resp = client.get("/api/capabilities/all?limit=100")
    assert resp.status_code == 200
    rows = {item["id"]: item for item in cast("dict[str, list[dict[str, object]]]", resp.get_json())["result"]}
    return rows["telegram"]


def _note(client: FlaskClient) -> str:
    """The note the Telegram card shows under its name."""
    return str(_card(client)["status_note"])


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_a_revoked_token_or_a_pasted_bot_link_is_refused_and_nothing_is_kept(
    client: FlaskClient, db: sqlite3.Connection, telegram: _Telegram,
) -> None:
    assert _save_token(client, _REVOKED_TOKEN) == 422
    assert telegram.calls("getMe", _REVOKED_TOKEN), "the token must have been checked with Telegram"
    assert _stored_rows(db) == 0
    status = _status(client)
    assert status["connected"] is False and status["pending"] is False
    assert str(status["status_note"]).startswith(_FAILED), status
    assert _REVOKED_TOKEN.split(":", 1)[1] not in str(status["status_note"])

    requests_before = len(telegram.requests)
    assert _save_token(client, f"https://t.me/{_USERNAME}") == 422
    assert len(telegram.requests) == requests_before, "a text that is not a token must never be sent anywhere"
    assert _stored_rows(db) == 0
    assert _connected(client) is False
    assert _note(client).startswith(_FAILED)


def test_sending_the_code_shown_on_the_card_connects_that_chat_and_confirms_it_there(
    client: FlaskClient, db: sqlite3.Connection, telegram: _Telegram,
) -> None:
    telegram.add_bot(_TOKEN, _USERNAME)
    # Sent before setup began: never read as the code.
    telegram.message(_TOKEN, _STRANGER_CHAT, text="hi")

    assert _save_token(client, _TOKEN) == 204
    code = _code(client)
    card = _card(client)
    note = str(card["status_note"])
    assert f"https://t.me/{_USERNAME}" in note and code not in note, note
    assert card["setup_code"] == code, "the card shows the code on its own"
    assert _stored_rows(db) == 0, "nothing is kept until the code arrives"
    assert _connected(client) is False
    assert "code=not-connected" in _head(_send("Too early"))

    telegram.message(_TOKEN, _GROUP_CHAT, chat_type="group", text=code)
    telegram.message(_TOKEN, _USER_CHAT, text="/start")
    telegram.message(_TOKEN, _USER_CHAT, text=f" {code[:3]} {code[3:]} ")
    status = _settled(client)

    assert status["connected"] is True, status
    assert [(p["chat_id"], p["text"]) for p in telegram.calls("sendMessage")] == [
        (_USER_CHAT, capability_module._CONFIRMATION),
    ]
    note = _note(client)
    assert f"{_USER_FIRST} {_USER_LAST}" in note and f"@{_USER_HANDLE}" in note, note
    assert _stored_rows(db) == 3

    telegram.forget_requests()
    assert "status=success" in _head(_send("Setup worked"))
    assert _sent_to(telegram) == [_USER_CHAT]


def _assert_failed_with_nothing_kept(client: FlaskClient, db: sqlite3.Connection, telegram: _Telegram) -> str:
    """The attempt ended without connecting: the card says so, nothing is
    stored, no confirmation went out and a send says Telegram is not set up.
    Returns the card's note."""
    status = _settled(client)
    assert status["connected"] is False, status
    note = str(status["status_note"])
    assert note.startswith(_FAILED), note
    assert _note(client) == note
    assert _stored_rows(db) == 0
    assert telegram.calls("sendMessage") == []
    assert "code=not-connected" in _head(_send("Is it set up?"))
    return note


def test_a_wrong_code_ends_the_setup_with_nothing_kept(
    client: FlaskClient, db: sqlite3.Connection, telegram: _Telegram,
) -> None:
    telegram.add_bot(_TOKEN, _USERNAME)
    assert _save_token(client, _TOKEN) == 204
    wrong = f"{(int(_code(client)) + 1) % 1_000_000:06d}"

    telegram.message(_TOKEN, _USER_CHAT, text=wrong)

    _assert_failed_with_nothing_kept(client, db, telegram)


def test_no_code_in_time_ends_the_setup_with_nothing_kept(
    client: FlaskClient, db: sqlite3.Connection, telegram: _Telegram, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capability_module, "_VERIFY_SECONDS", 2)
    telegram.add_bot(_TOKEN, _USERNAME)
    assert _save_token(client, _TOKEN) == 204
    assert _status(client)["pending"] is True

    note = _assert_failed_with_nothing_kept(client, db, telegram)

    assert "No code arrived in time" in note, note


def test_a_confirmation_telegram_refuses_ends_the_setup_with_nothing_kept_and_no_token_shown(
    client: FlaskClient, db: sqlite3.Connection, telegram: _Telegram,
) -> None:
    telegram.add_bot(_TOKEN, _USERNAME)
    assert _save_token(client, _TOKEN) == 204
    telegram.refuse_sends = True

    telegram.message(_TOKEN, _USER_CHAT, text=_code(client))
    status = _settled(client)

    assert status["connected"] is False, status
    note = str(status["status_note"])
    assert note.startswith(_FAILED), note
    assert _TOKEN.split(":", 1)[1] not in note
    assert telegram.calls("sendMessage"), "the confirmation must have reached Telegram"
    assert _stored_rows(db) == 0


def test_cancelling_while_the_setup_waits_stops_it_and_a_late_code_connects_nothing(
    client: FlaskClient, db: sqlite3.Connection, telegram: _Telegram,
) -> None:
    telegram.add_bot(_TOKEN, _USERNAME)
    assert _save_token(client, _TOKEN) == 204
    code = _code(client)

    assert client.post("/api/capabilities/disconnect/telegram").status_code == 204
    telegram.message(_TOKEN, _USER_CHAT, text=code)
    _setup_thread_ended()

    status = _status(client)
    assert status["connected"] is False and status["pending"] is False, status
    assert telegram.calls("sendMessage") == []
    assert _stored_rows(db) == 0


def test_a_chat_turn_sends_the_exact_text_to_the_user_without_asking(
    client: FlaskClient, db: sqlite3.Connection, telegram: _Telegram,
) -> None:
    _connect(client, telegram)
    text = "Your backup finished at 03:00 ✅ — 2 files skipped:\n- a.txt\n- b.txt"

    result = _send(text)

    # The probe turn has no surface to ask on: an ``ask`` row would deny here.
    assert "status=success" in _head(result), result
    assert f"{_USER_FIRST} {_USER_LAST}" in result and f"@{_USER_HANDLE}" in result, result
    sends = telegram.calls("sendMessage")
    assert len(sends) == 1
    assert sends[0]["chat_id"] == _USER_CHAT
    assert sends[0]["text"] == text
    assert db.execute("SELECT COUNT(*) FROM policy_blocked_log").fetchone()[0] == 0


def test_a_5000_character_message_arrives_as_two_messages_that_join_back_exactly(
    client: FlaskClient, telegram: _Telegram,
) -> None:
    _connect(client, telegram)
    text = "".join(f"{i:04d}," for i in range(1000))
    assert len(text) == 5000

    result = _send(text)

    assert "status=success" in _head(result), result
    sends = telegram.calls("sendMessage")
    assert len(sends) == 2
    assert "".join(str(p["text"]) for p in sends) == text
    assert _sent_to(telegram) == [_USER_CHAT, _USER_CHAT]


def test_a_stranger_messaging_the_bot_later_does_not_take_over_the_chat(
    client: FlaskClient, telegram: _Telegram,
) -> None:
    _connect(client, telegram)
    telegram.message(_TOKEN, _STRANGER_CHAT, text="hi, who is this?")

    assert "status=success" in _head(_send("Your report is ready"))

    assert _sent_to(telegram) == [_USER_CHAT]


def test_an_outside_agent_or_a_background_process_cannot_send(
    client: FlaskClient, db: sqlite3.Connection, telegram: _Telegram,
) -> None:
    _connect(client, telegram)
    assert _connected(client) is True

    for config in (EAMPConfig("reviewer", "sample-project", False), SkillSuggestionConfig()):
        result = _send("Message from a process the user is not watching", config)
        assert "status=success" not in result, result

    assert telegram.calls("sendMessage") == []
    blocked = db.execute("SELECT action_id, context, reason FROM policy_blocked_log ORDER BY id").fetchall()
    assert [tuple(r) for r in blocked] == [
        ("telegram.send", "external_agent", "deny"),
        ("telegram.send", "subconscious", "deny"),
    ]


def test_setting_up_again_forgets_the_old_chat_at_once_and_connects_whoever_sends_the_new_code(
    client: FlaskClient, db: sqlite3.Connection, telegram: _Telegram,
) -> None:
    _connect(client, telegram)

    telegram.add_bot(_NEW_TOKEN, _NEW_USERNAME)
    assert _save_token(client, _NEW_TOKEN) == 204
    code = _code(client)
    assert _stored_rows(db) == 0, "the old connection is removed as the new attempt starts"
    assert "code=not-connected" in _head(_send("Between bots"))

    telegram.message(_NEW_TOKEN, _STRANGER_CHAT, text=code)
    assert _settled(client)["connected"] is True
    telegram.forget_requests()

    assert "status=success" in _head(_send("Through the new bot"))
    assert telegram.calls("sendMessage", _TOKEN) == []
    assert [p["chat_id"] for p in telegram.calls("sendMessage", _NEW_TOKEN)] == [_STRANGER_CHAT]
    assert "Mallory" in _note(client)


def test_a_refused_or_unreachable_send_never_shows_the_token(
    client: FlaskClient, telegram: _Telegram, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    caplog.set_level(logging.INFO)
    for package in _APP_PACKAGES:
        caplog.set_level(logging.DEBUG, logger=package)
    secret = _TOKEN.split(":", 1)[1]
    _connect(client, telegram)

    telegram.refuse_sends = True
    refused = _send("Telegram will refuse this one")
    assert telegram.calls("sendMessage"), "the send must have reached Telegram"

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    monkeypatch.setattr(bot_api, "_API_BASE", f"http://127.0.0.1:{closed_port}")
    unreachable = _send("Telegram cannot be reached for this one")

    for result in (refused, unreachable):
        assert "status=error" in _head(result), result
        assert secret not in result
    assert caplog.records, "the failures must have been logged"
    for record in caplog.records:
        assert secret not in record.getMessage(), f"{record.name} logged the token"
    assert secret not in caplog.text


def test_sending_without_telegram_set_up_or_after_disconnecting_says_so_with_the_setup_hint(
    client: FlaskClient, db: sqlite3.Connection, telegram: _Telegram,
) -> None:
    never_set_up = _send("Is anyone there?")

    _connect(client, telegram)
    assert client.post("/api/capabilities/disconnect/telegram").status_code == 204
    after_disconnect = _send("Is anyone still there?")

    for result in (never_set_up, after_disconnect):
        assert "code=not-connected" in _head(result), result
        assert f"hint: {TelegramAbility.NOT_CONNECTED_HINT}" in result
    assert telegram.calls("sendMessage") == []
    assert _stored_rows(db) == 0
