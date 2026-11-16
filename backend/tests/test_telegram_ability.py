"""Feature tests: the Telegram tool messages the user, and only the user.

The real setup endpoint checks a bot token with Telegram and records the chat
of the first private hello; the real dispatcher and policy gate then decide
whether a turn may send, and the real capability sends to that chat.

Telegram is a real HTTP server on loopback that answers the Bot API methods the
capability uses the way Telegram does (an unknown token is refused, confirmed
updates are dropped, a text over 4096 characters is refused) and records every
request it receives. The capability's API base URL is the one thing redirected,
so every request in these tests reaches that server and none reaches Telegram.
The vault, database, setup endpoint, dispatcher and policy rows are the real
ones.
"""

from __future__ import annotations

import json
import logging
import socket
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import pytest
from flask.testing import FlaskClient

import capabilities
import capabilities.telegram_capability.bot_api as bot_api
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


def _connected(client: FlaskClient) -> bool:
    resp = client.get("/api/capabilities/status/telegram")
    assert resp.status_code == 200
    body = cast("dict[str, dict[str, object]]", resp.get_json())
    return body["result"]["connected"] is True


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


def _connect_with_hello(client: FlaskClient, telegram: _Telegram) -> None:
    """The user's bot is set up after they said hi to it."""
    telegram.add_bot(_TOKEN, _USERNAME)
    telegram.message(_TOKEN, _USER_CHAT)
    assert _save_token(client, _TOKEN) == 204


def _sent_to(telegram: _Telegram) -> list[object]:
    return [p["chat_id"] for p in telegram.calls("sendMessage")]


def _note(client: FlaskClient) -> str:
    """The note the Telegram card shows under its name."""
    resp = client.get("/api/capabilities/all?limit=100")
    assert resp.status_code == 200
    rows = {item["id"]: item for item in cast("dict[str, list[dict[str, object]]]", resp.get_json())["result"]}
    return str(rows["telegram"]["status_note"])


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_a_revoked_token_or_a_pasted_bot_link_is_refused_and_nothing_is_kept(
    client: FlaskClient, db: sqlite3.Connection, telegram: _Telegram,
) -> None:
    assert _save_token(client, _REVOKED_TOKEN) == 422
    assert telegram.calls("getMe", _REVOKED_TOKEN), "the token must have been checked with Telegram"
    assert _stored_rows(db) == 0
    assert _connected(client) is False

    requests_before = len(telegram.requests)
    assert _save_token(client, f"https://t.me/{_USERNAME}") == 422
    assert len(telegram.requests) == requests_before, "a text that is not a token must never be sent anywhere"
    assert _stored_rows(db) == 0
    assert _connected(client) is False


def test_saving_the_token_after_the_user_said_hi_records_their_private_chat_and_confirms_every_update(
    client: FlaskClient, telegram: _Telegram,
) -> None:
    telegram.add_bot(_TOKEN, _USERNAME)
    telegram.message(_TOKEN, _GROUP_CHAT, chat_type="group", text="hello everyone")
    telegram.message(_TOKEN, _USER_CHAT, text="hi")
    telegram.message(_TOKEN, _USER_CHAT, text="are you there?")
    last_update_id = cast(int, telegram.pending[_TOKEN][-1]["update_id"])

    assert _save_token(client, _TOKEN) == 204
    assert _connected(client) is True
    offsets = [p["offset"] for p in telegram.calls("getUpdates") if "offset" in p]
    assert offsets == [last_update_id + 1]
    assert telegram.pending[_TOKEN] == [], "every update read at setup must be confirmed"
    note = _note(client)
    assert f"{_USER_FIRST} {_USER_LAST}" in note and f"@{_USER_HANDLE}" in note, note

    # Nothing is pending any more, so the chat reached now is the one setup recorded.
    assert "status=success" in _head(_send("Setup worked"))
    assert _sent_to(telegram) == [_USER_CHAT]


def test_a_chat_turn_sends_the_exact_text_to_the_user_without_asking(
    client: FlaskClient, db: sqlite3.Connection, telegram: _Telegram,
) -> None:
    _connect_with_hello(client, telegram)
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
    _connect_with_hello(client, telegram)
    text = "".join(f"{i:04d}," for i in range(1000))
    assert len(text) == 5000

    result = _send(text)

    assert "status=success" in _head(result), result
    sends = telegram.calls("sendMessage")
    assert len(sends) == 2
    assert "".join(str(p["text"]) for p in sends) == text
    assert _sent_to(telegram) == [_USER_CHAT, _USER_CHAT]


def test_before_a_hi_the_card_and_a_send_name_the_bot_link_and_a_later_hi_shows_on_the_card_before_any_send(
    client: FlaskClient, telegram: _Telegram,
) -> None:
    telegram.add_bot(_TOKEN, _USERNAME)
    assert _save_token(client, _TOKEN) == 204
    assert f"https://t.me/{_USERNAME}" in _note(client)

    result = _send("Are you there?")

    assert "status=error" in _head(result), result
    assert f"https://t.me/{_USERNAME}" in result
    assert telegram.calls("sendMessage") == []

    # The user presses Start after setup: the card names them with no send.
    telegram.message(_TOKEN, _USER_CHAT)
    assert f"{_USER_FIRST} {_USER_LAST}" in _note(client)
    assert telegram.calls("sendMessage") == []
    assert telegram.pending[_TOKEN] == [], "the hello read for the card must be confirmed"

    assert "status=success" in _head(_send("Now you are"))
    assert _sent_to(telegram) == [_USER_CHAT]
    assert f"{_USER_FIRST} {_USER_LAST}" in _note(client)


def test_a_stranger_messaging_the_bot_later_does_not_take_over_the_chat(
    client: FlaskClient, telegram: _Telegram,
) -> None:
    _connect_with_hello(client, telegram)
    telegram.message(_TOKEN, _STRANGER_CHAT, text="hi, who is this?")

    assert "status=success" in _head(_send("Your report is ready"))

    assert _sent_to(telegram) == [_USER_CHAT]


def test_an_outside_agent_or_a_background_process_cannot_send(
    client: FlaskClient, db: sqlite3.Connection, telegram: _Telegram,
) -> None:
    _connect_with_hello(client, telegram)
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


def test_a_new_token_forgets_the_old_chat_but_saving_the_same_token_keeps_it(
    client: FlaskClient, telegram: _Telegram,
) -> None:
    _connect_with_hello(client, telegram)

    # Saving the same token again, with a stranger's message waiting, keeps the user's chat.
    telegram.message(_TOKEN, _STRANGER_CHAT)
    assert _save_token(client, _TOKEN) == 204
    assert "status=success" in _head(_send("Still you"))
    assert _sent_to(telegram) == [_USER_CHAT]

    # A different bot has never been said hi to: nothing goes to the old bot's chat.
    telegram.add_bot(_NEW_TOKEN, _NEW_USERNAME)
    assert _save_token(client, _NEW_TOKEN) == 204
    result = _send("Through the new bot")

    assert "status=error" in _head(result), result
    assert f"https://t.me/{_NEW_USERNAME}" in result
    assert telegram.calls("sendMessage", _NEW_TOKEN) == []


def test_a_refused_or_unreachable_send_never_shows_the_token(
    client: FlaskClient, telegram: _Telegram, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    caplog.set_level(logging.INFO)
    for package in _APP_PACKAGES:
        caplog.set_level(logging.DEBUG, logger=package)
    secret = _TOKEN.split(":", 1)[1]
    _connect_with_hello(client, telegram)

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

    _connect_with_hello(client, telegram)
    assert client.post("/api/capabilities/disconnect/telegram").status_code == 204
    after_disconnect = _send("Is anyone still there?")

    for result in (never_set_up, after_disconnect):
        assert "code=not-connected" in _head(result), result
        assert f"hint: {TelegramAbility.NOT_CONNECTED_HINT}" in result
    assert telegram.calls("sendMessage") == []
    assert _stored_rows(db) == 0
