"""Feature tests: a UniFi tool call signs in with the saved account, asking the user for a 2FA code.

The console is a real server on loopback that answers the UniFi OS sign-in the
way a console does for a UI.com account with 2FA on: the saved username and
password get a 499 and a half-made sign-in, and only the account's current
code, sent with that half-made sign-in's cookie, gets a session. A tool call
uses the stored session; with none, or once the console refuses it, it signs
in, and the user is asked for the code through the one service every ask goes
through. The call waits for the code with no time limit, until it is typed,
dismissed or the run stops. Real turns, real Flask app and cookie session, real
socket registry, real vault and database; the model provider is the one
stand-in, at its network boundary.
"""

from __future__ import annotations

import json
import logging
import socket
import sqlite3
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from typing import NamedTuple, cast

import pytest
from flask import Flask
from flask.testing import FlaskClient

import capabilities
from abilities.ubiquiti import UbiquitiAbility
from configs.channels.user import UserConfig
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from models.policy import Policy
from models.turn_execution import TurnExecution
from services.ask_user import AskUser, CodeReply
from services.vault_service import get_vault_service
from tests._browser_live import signed_in
from tests._delegates import (
    USER,
    WAIT_S,
    ChannelProvider,
    RecordingClient,
    _forget_vault,
    begin,
    calls,
    calls_of,
    execution,
    listening,
    patched,
    tool,
    wait,
)
from tests.test_capability_https_setup import (
    DEVICE,
    DEVICES_PATH,
    LOGIN_PATH,
    MFA_COOKIE,
    Console,
    UnifiHandler,
    certificate,  # noqa: F401 — the self-signed certificate fixture, used by name below
    serve,
    set_up_unifi,
)

pytestmark = pytest.mark.unit

CODE = "048217"  # the code the authenticator app shows; distinctive, so a leak is found by value
_MFA_VALUE = MFA_COOKIE.split("=", 1)[1]

# What the model reads when a sign-in does not finish.
CANCELLED = "UniFi sign-in cancelled before a 2FA code was entered."
TOO_MANY = "UI.com blocked sign-ins for a few minutes after too many attempts."
REJECTED = "UniFi rejected the saved username or password."
CODE_REFUSED = "Failed - 2FA Error"
UNREACHABLE = "Failed - network unreachable"
CERT_UNTRUSTED = "Failed - certificate not trusted"

_SIGN_IN = ("POST", LOGIN_PATH)
_DEVICES = ("GET", DEVICES_PATH)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def app(db: sqlite3.Connection) -> Flask:
    from api import create_app  # noqa: PLC0415 — built per test, on the test's database

    built = create_app()
    built.config["TESTING"] = True
    return built


@pytest.fixture(autouse=True)
def no_ask_left_open() -> Iterator[None]:
    """Asks live in the process, not the database: each test starts with none
    open and leaves none open, so one test's ask never answers another's. A
    call still waiting on a code when a test fails ends here, on the cancel."""
    assert AskUser.open_asks() == [], "an earlier test left an ask open"
    yield
    left = AskUser.open_asks()
    for ask in left:
        AskUser.cancel(ask.ask_id)
    assert left == [], "the test left an ask open"


@pytest.fixture
def console(request: pytest.FixtureRequest) -> Iterator[Console]:
    """A UniFi console on loopback, signed in to with a UI.com account that has 2FA on:
    over plain HTTP, or over HTTPS with a self-signed certificate for a test that
    passes ``"https"``."""
    https = getattr(request, "param", None) == "https"
    with serve(UnifiHandler, request.getfixturevalue("certificate") if https else None) as served:
        served.code = CODE
        yield served


@pytest.fixture
def client(
    app: Flask, console: Console, chat_provider: sqlite3.Connection
) -> Iterator[FlaskClient]:
    """The user's signed-in Brain, with UniFi set up against *console* through the
    setup form; the vault and the capability registry are this test's own."""
    _forget_vault()
    vault = get_vault_service()
    vault.initialize("test-vault-password")
    vault.unlock("test-vault-password")
    # The tool runs without asking leave, so the only ask a turn opens is the code's.
    Policy.upsert(PolicyChannel.CHAT.value, UbiquitiAbility.NAME, "allow")
    saved_registry = capabilities._capabilities_cache
    capabilities._capabilities_cache = None
    brain, _token = signed_in(app)
    try:
        assert set_up_unifi(brain, console.url).status_code == 204
        yield brain
    finally:
        capabilities._capabilities_cache = saved_registry
        _forget_vault()


# ── Helpers ──────────────────────────────────────────────────────────────────


class Frames:
    """Every frame broadcast to a screen, and an event for each ask opening and closing."""

    def __init__(self) -> None:
        self.asked = threading.Event()
        self.closed = threading.Event()

    def on_frame(self, frame: dict[str, object]) -> None:
        if frame["type"] == "ask_opened":
            self.asked.set()
        elif frame["type"] == "ask_closed":
            self.closed.set()


def of_type(ui: RecordingClient, kind: str) -> list[dict[str, object]]:
    return [frame for frame in ui.frames if frame["type"] == kind]


def first_ask(ui: RecordingClient, frames: Frames) -> str:
    """The id of the code ask a sign-in opened, once it has."""
    wait(frames.asked, "the sign-in to ask for the 2FA code")
    return cast("str", of_type(ui, "ask_opened")[0]["ask_id"])


@dataclass
class Turn:
    """A running turn whose model lists the UniFi devices, the frames broadcast meanwhile, and what the model was sent."""

    run: MessageProcessor
    ui: RecordingClient
    frames: Frames
    provider: ChannelProvider

    def ask_id(self) -> str:
        return first_ask(self.ui, self.frames)


@contextmanager
def listing_devices() -> Iterator[Turn]:
    """Start a turn whose model lists the UniFi devices; on leaving, wait for it and the work after it to end."""
    frames = Frames()
    provider = ChannelProvider(
        chat=[calls(tool(UbiquitiAbility.NAME, action=UbiquitiAbility.DEFAULT_ACTION))]
    )
    with listening(frames.on_frame) as ui, patched(provider):
        yield Turn(begin(UserConfig(), "what is on my network"), ui, frames, provider)


def list_devices(client: FlaskClient, typed: str | None = None) -> Turn:
    """Run one turn that lists the UniFi devices, typing *typed* when asked for a code."""
    with listing_devices() as turn:
        if typed is not None:
            assert answer(client, turn.ask_id(), typed) == 204
    return turn


def answer(client: FlaskClient, ask_id: str, code: object) -> int:
    """The user types *code* into ask *ask_id*; the reply's status."""
    return client.post(f"/api/asks/reply/{ask_id}", json={"value": {"code": code}}).status_code


def model_read(db: sqlite3.Connection) -> tuple[str, list[str]]:
    """The latest UniFi tool call's state, and the lines of its result the model reads."""
    row = calls_of(db, UbiquitiAbility.NAME)[-1]
    return cast("str", row["state"]), cast("str", row["result"]).splitlines()


def reads_devices(db: sqlite3.Connection) -> bool:
    """Whether the latest UniFi tool call ended with the console's device in what the model reads."""
    state, lines = model_read(db)
    return state == "done" and any(
        f'"{DEVICE["mac"]}"' in line and f'"{DEVICE["name"]}"' in line for line in lines
    )


def device_lister() -> Callable[..., dict[str, object]]:
    """The UniFi capability's list-devices handler, as a tool call dispatches to it."""
    handlers = {
        t["name"]: t["handler"]
        for t in capabilities.load_capabilities()["ubiquiti"].get_tools()
    }
    return cast("Callable[..., dict[str, object]]", handlers["list_devices"])


def in_background(call: Callable[[], dict[str, object]]) -> Future[dict[str, object]]:
    """Run *call* on its own thread. Not joined on the way out: a call left
    waiting on a code ends when the open ask is cancelled after the test."""
    done: Future[dict[str, object]] = Future()

    def run() -> None:
        try:
            done.set_result(call())
        except BaseException as exc:  # noqa: BLE001 — handed to the test through the future
            done.set_exception(exc)

    threading.Thread(target=run, name="unifi-call", daemon=True).start()
    return done


def macs(result: dict[str, object]) -> list[object]:
    return [
        cast("dict[str, object]", d)["mac"]
        for d in cast("list[object]", result["devices"])
    ]


# ── Signing in with a 2FA code ───────────────────────────────────────────────


def test_the_user_types_the_2fa_code_once_and_later_calls_and_a_restart_reuse_the_session(
    client: FlaskClient,
    console: Console,
    chat_provider: sqlite3.Connection,
) -> None:
    first = list_devices(client, typed=CODE)

    [ask] = of_type(first.ui, "ask_opened")
    assert (ask["kind"], ask["data"]) == (
        "code",
        {
            "service": "unifi",
            "origin": {"type": "user", "turn_id": first.run.turn_id, "forked": False},
        },
    )
    assert of_type(first.ui, "ask_closed") == [
        {"type": "ask_closed", "ask_id": ask["ask_id"]}
    ]
    # The console finishes the sign-in only for the right code sent with the half-made sign-in's cookie.
    assert console.requests() == [(*_SIGN_IN, 499), (*_SIGN_IN, 200), (*_DEVICES, 200)]
    assert reads_devices(chat_provider)

    second = list_devices(client)
    assert of_type(second.ui, "ask_opened") == []
    assert console.requests()[3:] == [(*_DEVICES, 200)]
    assert reads_devices(chat_provider)

    # As after a restart, once the vault is unlocked.
    capabilities._capabilities_cache = None
    assert capabilities.load_capabilities()["ubiquiti"].connect()
    after_restart = list_devices(client)
    assert of_type(after_restart.ui, "ask_opened") == []
    assert console.requests()[4:] == [(*_DEVICES, 200)]
    assert reads_devices(chat_provider)


def test_a_session_the_console_refuses_is_replaced_and_the_call_retried_once(
    client: FlaskClient,
    console: Console,
    chat_provider: sqlite3.Connection,
) -> None:
    console.code = ""  # an account without 2FA, so each sign-in is the password alone
    list_devices(client)
    assert console.requests() == [(*_SIGN_IN, 200), (*_DEVICES, 200)]

    console.refuse_next = 1
    list_devices(client)
    assert console.requests()[2:] == [
        (*_DEVICES, 401),
        (*_SIGN_IN, 200),
        (*_DEVICES, 200),
    ]
    assert reads_devices(chat_provider)

    # A console that refuses the new session too is not signed in to again and again.
    console.refuse_next = 2
    list_devices(client)
    assert console.requests()[5:] == [
        (*_DEVICES, 401),
        (*_SIGN_IN, 200),
        (*_DEVICES, 401),
    ]
    assert not reads_devices(chat_provider)


class Refusal(NamedTuple):
    """How the console turns a sign-in down, and what the model and the console end up with.

    ``sign_in_status`` and ``code_status`` are the console's forced answers to
    the username and password and to the code; ``typed`` is what the user types
    when asked (None when no ask is expected); ``said`` is what the model reads;
    ``sign_ins`` is the console's answer to each sign-in it got.
    """

    sign_in_status: int | None
    code_status: int | None
    typed: str | None
    said: str
    sign_ins: list[int]


@pytest.mark.parametrize(
    "refusal",
    [
        pytest.param(
            Refusal(None, None, "123456", CODE_REFUSED, [499, 499]), id="wrong code"
        ),
        pytest.param(Refusal(429, None, None, TOO_MANY, [429]), id="too many sign-ins"),
        pytest.param(
            Refusal(None, 429, CODE, TOO_MANY, [499, 429]),
            id="too many sign-ins at the code",
        ),
        pytest.param(
            Refusal(401, None, None, REJECTED, [401]), id="password refused with 401"
        ),
        pytest.param(
            Refusal(403, None, None, REJECTED, [403]), id="password refused with 403"
        ),
        pytest.param(
            Refusal(None, None, None, UNREACHABLE, []), id="console unreachable"
        ),
    ],
)
def test_a_sign_in_that_does_not_finish_tells_the_model_why_and_reaches_no_data(
    client: FlaskClient,
    console: Console,
    chat_provider: sqlite3.Connection,
    refusal: Refusal,
) -> None:
    console.sign_in_status, console.code_status = (
        refusal.sign_in_status,
        refusal.code_status,
    )
    if refusal.said == UNREACHABLE:
        # A port nothing listens on once the probe closes.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            closed_port = cast("tuple[str, int]", probe.getsockname())[1]
        assert (
            set_up_unifi(client, f"http://127.0.0.1:{closed_port}").status_code == 204
        )

    turn = list_devices(client, typed=refusal.typed)

    # A refused code is not asked for again.
    assert len(of_type(turn.ui, "ask_opened")) == (0 if refusal.typed is None else 1)
    state, lines = model_read(chat_provider)
    assert state == "error" and refusal.said in lines, lines
    assert console.requests() == [(*_SIGN_IN, status) for status in refusal.sign_ins]


# ── A console with a self-signed certificate ─────────────────────────────────

# A UniFi console comes with a certificate no machine trusts; Verify SSL on the
# Brain's setup form says whether a call checks it.


@pytest.mark.parametrize("console", ["https"], indirect=True)
def test_with_verify_ssl_off_a_console_with_a_self_signed_certificate_is_signed_in_to_and_used(
    client: FlaskClient,
    console: Console,
    chat_provider: sqlite3.Connection,
) -> None:
    assert console.url.startswith("https://")
    assert set_up_unifi(client, console.url, verify_ssl=False).status_code == 204

    list_devices(client, typed=CODE)

    assert console.requests() == [(*_SIGN_IN, 499), (*_SIGN_IN, 200), (*_DEVICES, 200)]
    assert reads_devices(chat_provider)


@pytest.mark.parametrize("console", ["https"], indirect=True)
def test_with_verify_ssl_on_the_same_console_fails_as_certificate_not_trusted(
    client: FlaskClient,
    console: Console,
    chat_provider: sqlite3.Connection,
) -> None:
    assert console.url.startswith("https://")
    assert set_up_unifi(client, console.url, verify_ssl=True).status_code == 204

    turn = list_devices(client)

    assert of_type(turn.ui, "ask_opened") == []
    state, lines = model_read(chat_provider)
    assert state == "error" and CERT_UNTRUSTED in lines, lines
    assert console.requests() == []  # turned away at the certificate, before any sign-in


# ── Ending the wait without a code ───────────────────────────────────────────


def test_a_dismissed_code_ask_cancels_the_sign_in(
    client: FlaskClient,
    console: Console,
    chat_provider: sqlite3.Connection,
) -> None:
    with listing_devices() as turn:
        ask_id = turn.ask_id()
        assert client.post(f"/api/asks/dismiss/{ask_id}").status_code == 204

    assert of_type(turn.ui, "ask_closed") == [{"type": "ask_closed", "ask_id": ask_id}]
    state, lines = model_read(chat_provider)
    assert state == "error" and CANCELLED in lines, lines
    assert console.requests() == [(*_SIGN_IN, 499)]  # no code sent, no data asked for


def test_a_run_stopped_while_it_waits_for_the_code_closes_the_ask_and_reaches_no_data(
    client: FlaskClient,
    console: Console,
) -> None:
    with listing_devices() as turn:
        ask_id = turn.ask_id()
        stop = client.delete(f"/api/threads/{turn.run.turn_id}")
        wait(turn.frames.closed, "the stopped run to close its code ask")

    assert (
        stop.status_code == 200
        and cast("dict[str, dict[str, object]]", stop.get_json())["result"]["cancelled"]
        is True
    )
    assert of_type(turn.ui, "ask_closed") == [{"type": "ask_closed", "ask_id": ask_id}]
    assert AskUser.open_asks() == []
    assert console.requests() == [(*_SIGN_IN, 499)]
    assert execution(USER, turn.run.turn_id).state == TurnExecution.CANCELLED


# ── What counts as a code ────────────────────────────────────────────────────


def test_a_reply_that_is_not_six_digits_is_refused_and_the_ask_still_waits_for_the_code(
    client: FlaskClient,
    console: Console,
    chat_provider: sqlite3.Connection,
) -> None:
    with listing_devices() as turn:
        ask_id = turn.ask_id()
        outcomes = [
            (
                value,
                answer(client, ask_id, value),
                [ask.ask_id for ask in AskUser.open_asks()],
            )
            for value in ("12345", "1234567", "12a456", None, "١٢٣٤٥٦", 123456)
        ]
        assert answer(client, ask_id, CODE) == 204

    # Arabic-Indic digits are digits, but not the ASCII ones a console takes. A
    # null or a number is no code at all: a code keeps its leading zero only as text.
    assert outcomes == [
        ("12345", 400, [ask_id]),
        ("1234567", 400, [ask_id]),
        ("12a456", 400, [ask_id]),
        (None, 422, [ask_id]),
        ("١٢٣٤٥٦", 400, [ask_id]),
        (123456, 422, [ask_id]),
    ]
    assert console.requests() == [(*_SIGN_IN, 499), (*_SIGN_IN, 200), (*_DEVICES, 200)]
    assert reads_devices(chat_provider)


# ── Two calls at once ────────────────────────────────────────────────────────


def test_a_call_with_a_working_session_is_not_held_up_by_another_waiting_for_a_code(
    client: FlaskClient,
    console: Console,
) -> None:
    console.code = ""
    # Signed in with the password alone, and the session stored.
    assert macs(device_lister()({}, None)) == [DEVICE["mac"]]
    console.code = CODE
    # The next call's session is refused, so it signs in again and waits for the code.
    console.refuse_next = 1

    frames = Frames()
    with listening(frames.on_frame) as ui:
        waiting = in_background(lambda: device_lister()({}, None))
        ask_id = first_ask(ui, frames)
        other = in_background(lambda: device_lister()({}, None))
        assert macs(other.result(timeout=WAIT_S)) == [DEVICE["mac"]]
        assert not waiting.done() and [ask.ask_id for ask in AskUser.open_asks()] == [
            ask_id
        ]

        assert AskUser.reply(ask_id, CodeReply(code=CODE))
        assert macs(waiting.result(timeout=WAIT_S)) == [DEVICE["mac"]]

    assert console.requests()[2:] == [
        (*_DEVICES, 401),
        (*_SIGN_IN, 499),
        (*_DEVICES, 200),
        (*_SIGN_IN, 200),
        (*_DEVICES, 200),
    ]


def test_two_calls_with_no_session_share_one_sign_in_and_one_code(
    client: FlaskClient, console: Console
) -> None:
    second_waits = threading.Event()

    def second_stop_check() -> bool:
        """Only checked while the second call waits for the first one's sign-in."""
        second_waits.set()
        return False

    frames = Frames()
    with listening(frames.on_frame) as ui:
        first = in_background(lambda: device_lister()({}, None))
        ask_id = first_ask(ui, frames)
        second = in_background(
            lambda: device_lister()({}, None, should_stop=second_stop_check)
        )
        wait(second_waits, "the second call to wait for the first one's sign-in")

        assert AskUser.reply(ask_id, CodeReply(code=CODE))
        results = [first.result(timeout=WAIT_S), second.result(timeout=WAIT_S)]

    assert [macs(result) for result in results] == [[DEVICE["mac"]], [DEVICE["mac"]]]
    assert len(of_type(ui, "ask_opened")) == 1
    assert console.requests() == [
        (*_SIGN_IN, 499),
        (*_SIGN_IN, 200),
        (*_DEVICES, 200),
        (*_DEVICES, 200),
    ]


# ── Disconnecting and setting up again ───────────────────────────────────────

# Disconnect forgets the setup and keeps the signed-in session; a setup of the
# same console and username uses it again, and any other setup starts without it.

_SESSION_KEY = "ubiquiti:session"


def disconnect(client: FlaskClient) -> int:
    """The user presses Disconnect on the UniFi card; the reply's status."""
    return client.post("/api/capabilities/disconnect/ubiquiti").status_code


def saved_keys(db: sqlite3.Connection) -> list[str]:
    """Every UniFi row in the credential store, by key."""
    return sorted(
        row["config_key"]
        for row in db.execute(
            "SELECT config_key FROM tool_configs WHERE tool_name = 'ubiquiti'"
        ).fetchall()
    )


def test_after_a_disconnect_the_same_console_and_username_are_set_up_again_with_no_new_sign_in(
    client: FlaskClient,
    console: Console,
    chat_provider: sqlite3.Connection,
) -> None:
    list_devices(client, typed=CODE)

    assert disconnect(client) == 204
    assert saved_keys(chat_provider) == [_SESSION_KEY]
    assert set_up_unifi(client, console.url).status_code == 204
    again = list_devices(client)
    assert of_type(again.ui, "ask_opened") == []
    assert console.requests()[3:] == [(*_DEVICES, 200)]
    assert reads_devices(chat_provider)

    # As after a restart between the disconnect and the new setup.
    assert disconnect(client) == 204
    capabilities._capabilities_cache = None
    assert not capabilities.load_capabilities()["ubiquiti"].connect()
    assert set_up_unifi(client, console.url).status_code == 204
    after_restart = list_devices(client)
    assert of_type(after_restart.ui, "ask_opened") == []
    assert console.requests()[4:] == [(*_DEVICES, 200)]
    assert reads_devices(chat_provider)


@pytest.mark.parametrize(
    "disconnect_first", [True, False], ids=["after a disconnect", "by an edit"]
)
def test_setting_up_another_console_starts_with_no_session(
    client: FlaskClient,
    console: Console,
    chat_provider: sqlite3.Connection,
    disconnect_first: bool,
) -> None:
    list_devices(client, typed=CODE)
    if disconnect_first:
        assert disconnect(client) == 204

    with serve(UnifiHandler, None) as other:
        other.code = CODE
        assert set_up_unifi(client, other.url).status_code == 204
        assert _SESSION_KEY not in saved_keys(chat_provider)
        list_devices(client, typed=CODE)
        assert other.requests() == [(*_SIGN_IN, 499), (*_SIGN_IN, 200), (*_DEVICES, 200)]
    assert reads_devices(chat_provider)
    assert console.requests() == [(*_SIGN_IN, 499), (*_SIGN_IN, 200), (*_DEVICES, 200)]


@pytest.mark.parametrize(
    "disconnect_first", [True, False], ids=["after a disconnect", "by an edit"]
)
def test_setting_up_another_username_on_the_same_console_starts_with_no_session(
    client: FlaskClient,
    console: Console,
    chat_provider: sqlite3.Connection,
    disconnect_first: bool,
) -> None:
    list_devices(client, typed=CODE)
    if disconnect_first:
        assert disconnect(client) == 204

    assert set_up_unifi(client, console.url, username="someone-else").status_code == 204
    assert _SESSION_KEY not in saved_keys(chat_provider)
    turn = list_devices(client)

    # Signed in afresh as the new username, which this console does not know.
    assert of_type(turn.ui, "ask_opened") == []
    state, lines = model_read(chat_provider)
    assert state == "error" and REJECTED in lines, lines
    assert console.requests()[3:] == [(*_SIGN_IN, 403)]


def test_a_session_signed_in_while_the_setup_moves_to_another_console_is_never_sent_there(
    client: FlaskClient,
    console: Console,
    chat_provider: sqlite3.Connection,
) -> None:
    with serve(UnifiHandler, None) as other:
        other.code = CODE
        with listing_devices() as turn:
            ask_id = turn.ask_id()
            assert set_up_unifi(client, other.url).status_code == 204
            assert answer(client, ask_id, CODE) == 204
        # The waiting call finishes on the console it signed in to.
        assert reads_devices(chat_provider)
        assert console.requests() == [(*_SIGN_IN, 499), (*_SIGN_IN, 200), (*_DEVICES, 200)]

        list_devices(client, typed=CODE)
        assert other.requests() == [(*_SIGN_IN, 499), (*_SIGN_IN, 200), (*_DEVICES, 200)]
    assert reads_devices(chat_provider)


# ── Secrets ──────────────────────────────────────────────────────────────────


def test_the_code_and_the_half_made_sign_in_are_never_written_down(
    client: FlaskClient,
    console: Console,
    chat_provider: sqlite3.Connection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    with listing_devices() as turn:
        ask_id = turn.ask_id()
        listing = client.get("/api/asks/open").get_data(as_text=True)
        assert answer(client, ask_id, CODE) == 204
    assert reads_devices(chat_provider)

    cap = capabilities.load_capabilities()["ubiquiti"]
    stored = [
        cap.load_credential(row["config_key"]) or ""
        for row in chat_provider.execute(
            "SELECT config_key FROM tool_configs WHERE tool_name = 'ubiquiti'",
        ).fetchall()
    ]
    # The session the console handed out is what was kept.
    assert any("TOKEN=session-1" in value for value in stored)
    written = [
        *(logging.Formatter().format(record) for record in caplog.records),
        json.dumps(turn.ui.frames),
        listing,
        *(
            cast("str", row["result"])
            for row in calls_of(chat_provider, UbiquitiAbility.NAME)
        ),
        *(request.text for request in turn.provider.requests),
        *stored,
    ]
    assert [line for line in written if CODE in line or _MFA_VALUE in line] == []
