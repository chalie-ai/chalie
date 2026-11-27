"""Feature tests: every ask that waits on the user goes through one service.

A run opens an ask — may it take a gated action, or will the user take over a
browser tab for a step only they can do — and hands over what to do with the
answer. Every screen the user has open hears the ask open and close; a screen
that connects later rebuilds the open asks from the listing; the user answers
or dismisses an ask from their own signed-in session, and only the first answer
reaches the run. A run stopped while it waits closes its ask and does not run
the tool. An answer is never echoed into a log line, a frame, the listing or an
error. Real Flask app, real cookie session, real socket registry broadcast and
real database; the model provider is the one stand-in, at its network boundary.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import pytest
from flask import Flask
from flask.testing import FlaskClient

from configs.channels.user import UserConfig
from configs.enums.policy_channel import PolicyChannel
from exceptions import AskReplyInvalid
from models.policy import Policy
from models.turn_execution import TurnExecution
from services.ask_user import AskReply, AskUser, HandoffAsk, Origin, PermissionAsk
from services.policy_manager import _BLOCK
from tests._browser_live import signed_in
from tests._delegates import (
    USER,
    WAIT_S,
    ChannelProvider,
    RecordingClient,
    begin,
    calls,
    calls_of,
    execution,
    listening,
    patched,
    tool,
    wait,
)
from tests.test_policy_gate_origin import gate_probe

pytestmark = pytest.mark.unit

_ORIGIN = Origin(type="user", turn_id=7, forked=False)
_ORIGIN_SENT: dict[str, object] = {"type": "user", "turn_id": 7, "forked": False}
_EMAIL = PermissionAsk(action_id="send_email", summary="Email Sam the report", origin=_ORIGIN)
_SIGN_IN = HandoffAsk(tab_id="tab-1", reason="Please sign in", origin=_ORIGIN)
_SECRET = "hunter2-417093"  # stands in for an answer that must never be written down


@pytest.fixture
def app(db: sqlite3.Connection) -> Flask:
    from api import create_app  # noqa: PLC0415 — built per test, on the test's database

    built = create_app()
    built.config["TESTING"] = True
    return built


@pytest.fixture(autouse=True)
def no_ask_left_open() -> Iterator[None]:
    """Asks live in the process, not the database: each test starts with none
    open and leaves none open, so one test's ask never answers another's."""
    assert AskUser.open_asks() == [], "an earlier test left an ask open"
    yield
    left = AskUser.open_asks()
    for ask in left:
        AskUser.cancel(ask.ask_id)
    assert left == [], "the test left an ask open"


def _listing(client: FlaskClient) -> list[dict[str, object]]:
    """The open asks, as a screen that has just connected reads them."""
    response = client.get("/api/asks/open")
    assert response.status_code == 200, response.data
    return cast("list[dict[str, object]]", cast("dict[str, object]", response.get_json())["result"])


def _reply(client: FlaskClient, ask_id: str, value: bool | None) -> int:
    """The user answers ask *ask_id* with *value*; the reply's status."""
    return client.post(f"/api/asks/reply/{ask_id}", json={"value": value}).status_code


def _of_type(ui: RecordingClient, kind: str) -> list[dict[str, object]]:
    return [frame for frame in ui.frames if frame.get("type") == kind]


def test_every_screen_hears_the_asks_open_and_close_and_each_answer_reaches_its_run(
    app: Flask,
) -> None:
    client, _token = signed_in(app)
    email: list[AskReply] = []
    sign_in: list[AskReply] = []
    delete: list[AskReply] = []
    with listening() as ui:
        ids = [
            AskUser.permission(_EMAIL, email.append),
            AskUser.handoff(_SIGN_IN, sign_in.append),
            AskUser.permission(
                PermissionAsk(action_id="delete_file", summary="Delete the draft", origin=None),
                delete.append,
            ),
        ]
        opened = _of_type(ui, "ask_opened")
        assert [(f["ask_id"], f["kind"], f["data"]) for f in opened] == [
            (ids[0], "permission", {"action_id": "send_email", "summary": "Email Sam the report", "origin": _ORIGIN_SENT}),
            (ids[1], "handoff", {"tab_id": "tab-1", "reason": "Please sign in", "origin": _ORIGIN_SENT}),
            (ids[2], "permission", {"action_id": "delete_file", "summary": "Delete the draft", "origin": None}),
        ]
        # A screen that connects now rebuilds the same asks, oldest first.
        assert _listing(client) == [{k: v for k, v in f.items() if k != "type"} for f in opened]

        assert [_reply(client, ids[0], True), _reply(client, ids[1], None), _reply(client, ids[2], False)] == [204] * 3

    assert _of_type(ui, "ask_closed") == [{"type": "ask_closed", "ask_id": i} for i in ids]
    assert (email, sign_in, delete) == (
        [AskReply(answered=True, value=True)],
        [AskReply(answered=True, value=None)],
        [AskReply(answered=True, value=False)],
    )
    assert _listing(client) == []


def test_when_two_screens_answer_one_ask_at_once_only_the_first_answer_reaches_the_run(
    app: Flask,
) -> None:
    laptop, _laptop_token = signed_in(app)
    phone, _phone_token = signed_in(app)
    run: list[AskReply] = []
    ask_id = AskUser.permission(_EMAIL, run.append)

    together = threading.Barrier(2)

    def answer(screen: FlaskClient, allowed: bool) -> int:
        together.wait(timeout=WAIT_S)
        return _reply(screen, ask_id, allowed)

    with ThreadPoolExecutor(max_workers=2) as pool:
        answers = [pool.submit(answer, laptop, True), pool.submit(answer, phone, False)]
        statuses = [a.result(timeout=WAIT_S) for a in answers]

    assert sorted(statuses) == [204, 404], statuses
    first = statuses == [204, 404]  # the laptop allowed it; the phone denied it
    assert run == [AskReply(answered=True, value=first)]
    # Once answered, it cannot be dismissed either.
    assert phone.post(f"/api/asks/dismiss/{ask_id}").status_code == 404
    assert run == [AskReply(answered=True, value=first)]
    assert _listing(laptop) == []


def test_a_dismissed_hand_off_tells_its_run_no_one_answered_and_takes_no_answer_after(
    app: Flask,
) -> None:
    client, _token = signed_in(app)
    run: list[AskReply] = []
    with listening() as ui:
        ask_id = AskUser.handoff(_SIGN_IN, run.append)
        dismissed = client.post(f"/api/asks/dismiss/{ask_id}")
        late = [_reply(client, ask_id, None), client.post(f"/api/asks/dismiss/{ask_id}").status_code]

    assert dismissed.status_code == 204, dismissed.data
    # Not "done": the run must not carry on as if the user had finished.
    assert run == [AskReply(answered=False, value=None)]
    assert late == [404, 404]
    assert _of_type(ui, "ask_closed") == [{"type": "ask_closed", "ask_id": ask_id}]
    assert _listing(client) == []


def test_an_answer_that_does_not_fit_the_ask_is_refused_and_the_ask_still_waits(
    app: Flask,
) -> None:
    client, _token = signed_in(app)
    email: list[AskReply] = []
    sign_in: list[AskReply] = []
    with listening() as ui:
        email_id = AskUser.permission(_EMAIL, email.append)
        sign_in_id = AskUser.handoff(_SIGN_IN, sign_in.append)
        refused = [
            client.post(f"/api/asks/reply/{email_id}", json={"value": None}),  # needs a yes or a no
            client.post(f"/api/asks/reply/{sign_in_id}", json={"value": True}),  # only ever "done"
        ]
        assert [(r.status_code, cast("dict[str, object]", r.get_json())["success"]) for r in refused] == [
            (400, False),
            (400, False),
        ]
        assert [ask["ask_id"] for ask in _listing(client)] == [email_id, sign_in_id]
        assert (email, sign_in, _of_type(ui, "ask_closed")) == ([], [], [])

        assert [_reply(client, email_id, True), _reply(client, sign_in_id, None)] == [204, 204]

    assert (email, sign_in) == ([AskReply(answered=True, value=True)], [AskReply(answered=True, value=None)])


def test_a_run_that_fails_on_its_answer_fails_the_reply_loudly_and_the_ask_still_closes(
    app: Flask,
) -> None:
    client, _token = signed_in(app)

    class RunBroke(RuntimeError):
        """The run waiting on the ask could not take its answer."""

    def broken(reply: AskReply) -> None:
        raise RunBroke("the run could not take its answer")

    with listening() as ui:
        ask_id = AskUser.permission(_EMAIL, broken)
        with pytest.raises(RunBroke):
            client.post(f"/api/asks/reply/{ask_id}", json={"value": True})

    assert _of_type(ui, "ask_closed") == [{"type": "ask_closed", "ask_id": ask_id}]
    assert _listing(client) == []
    assert _reply(client, ask_id, True) == 404


def test_an_answer_is_never_echoed_into_a_log_line_a_frame_the_listing_or_an_error(
    app: Flask, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    client, _token = signed_in(app)
    email: list[AskReply] = []
    sign_in: list[AskReply] = []
    with listening() as ui:
        email_id = AskUser.permission(_EMAIL, email.append)
        sign_in_id = AskUser.handoff(_SIGN_IN, sign_in.append)
        refused = client.post(f"/api/asks/reply/{email_id}", json={"value": _SECRET})
        with pytest.raises(AskReplyInvalid) as wrong:
            AskUser.reply(sign_in_id, _SECRET)
        listing = _listing(client)
        assert [_reply(client, email_id, False), _reply(client, sign_in_id, None)] == [204, 204]

    assert refused.status_code == 422, refused.data  # an answer is a yes, a no or "done"
    assert [ask["ask_id"] for ask in listing] == [email_id, sign_in_id]  # both still waited
    assert (email, sign_in) == ([AskReply(answered=True, value=False)], [AskReply(answered=True, value=None)])

    lines = [logging.Formatter().format(record) for record in caplog.records]
    written = [refused.get_data(as_text=True), str(wrong.value), json.dumps(ui.frames), json.dumps(listing), *lines]
    assert [w for w in written if _SECRET in w] == []
    # The service's own lines name the ask, never the answer it was given.
    said = [line for record, line in zip(caplog.records, lines, strict=True) if record.name == "services.ask_user"]
    assert said, "the service logged nothing about the asks it opened and closed"
    assert [line for line in said if "False" in line or "None" in line] == []


def test_a_run_stopped_while_it_waits_on_the_user_closes_the_ask_and_does_not_run_the_tool(
    app: Flask, chat_provider: sqlite3.Connection
) -> None:
    Policy.upsert(PolicyChannel.CHAT.value, gate_probe.NAME, "ask")
    client, _token = signed_in(app)
    asked = threading.Event()
    closed = threading.Event()

    def on_frame(frame: dict[str, object]) -> None:
        if frame["type"] == "ask_opened":
            asked.set()
        elif frame["type"] == "ask_closed":
            closed.set()

    provider = ChannelProvider(chat=[calls(tool(gate_probe.NAME, q="x", act_summary="Probe the gate"))])
    with listening(on_frame) as ui, patched(provider):
        run = begin(UserConfig(), "probe the gate")
        wait(asked, "the run to ask the user")
        stop = client.delete(f"/api/threads/{run.turn_id}")
        wait(closed, "the stopped run to close its ask")
        run.result()

    assert stop.status_code == 200 and cast("dict[str, dict[str, object]]", stop.get_json())["result"]["cancelled"] is True
    [opened] = _of_type(ui, "ask_opened")
    assert _of_type(ui, "ask_closed") == [{"type": "ask_closed", "ask_id": opened["ask_id"]}]
    assert _listing(client) == []
    # The tool was denied, not run, and a stop is no verdict of the user's: nothing is audited.
    assert [(r["state"], r["result"]) for r in calls_of(chat_provider, gate_probe.NAME)] == [
        ("error", _BLOCK.format(permission=gate_probe.NAME)),
    ]
    assert chat_provider.execute("SELECT count(*) FROM policy_blocked_log").fetchone()[0] == 0
    assert execution(USER, run.turn_id).state == TurnExecution.CANCELLED
