"""Feature test: stopping a running subagent, and what its caller is told.

A person can stop a delegate's child turn from its transcript panel. The child
must actually stop — its late answer is never stored and it ends cancelled on
the wire. A caller that waits for its subagent's answer — here a subtask the
person started from the chat, searching the web on its own turn — must learn
the subagent was stopped by the user, under its own outcome and without the
retry advice an empty answer carries, then finish its own turn. Stopping a
typed turn (chat, scheduled, background) keeps broadcasting exactly as before,
and a stopped row whose type no longer resolves is reported loudly without a
frame sent under a guessed address.

Drives the real production entry point (construct inertly, ``begin()``,
``result()``) and the real Flask stop route against the real, fully-migrated
SQLite DB, observing frames through the REAL socket registry. The only
substitution is the LLM network boundary
(``services.provider_service.build_client``); the search is held inside its own
model call on a ``threading.Event`` so the stop lands while it is running.
"""

import re
import sqlite3

import pytest
from flask.testing import FlaskClient

from configs.channels import config_for
from configs.channels.user import UserConfig
from configs.enums.channels import Channel
from controllers.message_processor import MessageProcessor
from models.turn_execution import TurnExecution
from services.turn_execution_service import TurnExecutionService
from tests._delegates import (
    ChannelProvider,
    Entry,
    Held,
    assistant_texts,
    begin,
    block,
    calls,
    calls_of,
    child_turn,
    drain,
    execution,
    helper_channel,
    listening,
    messages,
    patched,
    text,
    tool,
    wait,
    watching_warnings,
)

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

_QUESTION = "what is the capital of France?"
_TASK = "find out the capital of France"
_INSTRUCTIONS = "find the capital of France"
_CHILD_ANSWER = "Paris is the capital of France."
_SEARCH = helper_channel("web_search")
_SUBTASK = Channel.DELEGATE_SUBTASK.value
_ERROR_CODE = re.compile(r"\A\[web_search\(status=error, code=([\w-]+)")
_HINT_LINE = re.compile(r"^hint: \S", re.MULTILINE)


def _error_code(result: str) -> str | None:
    """The outcome code in the header of a failed call's result envelope."""
    match = _ERROR_CODE.match(result)
    return match.group(1) if match else None


def _provider(search: list[Entry], subtask_answer: str) -> ChannelProvider:
    """The person's chat starts a subtask; the subtask searches the web, waits
    for the search, and answers *subtask_answer*; the chat answers the
    subtask's reply."""
    return ChannelProvider(
        chat=[
            calls(tool("subtask", action="start", instructions=_TASK)),
            text("I have started a subtask."),
            text("The subtask is done."),
        ],
        subtask=[calls(tool("web_search", instructions=_INSTRUCTIONS)), text(subtask_answer)],
        web_search=search,
    )


def _assert_subtask_completed_with_its_answer(db: sqlite3.Connection, answer: str) -> None:
    """The subtask that waited for the search finished its own turn, its
    answer the last thing it said."""
    subtask = child_turn(db, _SUBTASK, calls_of(db, "subtask")[0]["id"])
    assert execution(_SUBTASK, subtask).state == TurnExecution.COMPLETED
    assert assistant_texts(db, _SUBTASK, subtask) == ["", answer]


def test_stopped_subagent_is_reported_as_stopped_and_its_caller_still_answers(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """Stopping the running ``web_search`` child by its channel ends it: the
    answer its in-flight model call comes back with is never stored, its last
    lifecycle frame says cancelled, its execution row stays cancelled, and its
    transcript reads back as stopped with only its task in it. The waiting
    caller's call fails as ``delegate-stopped`` with no retry hint — the user
    chose to stop that work — and the caller's own turn still completes with its
    reply."""
    client, db, _store = authed_client
    held = Held(text(_CHILD_ANSWER))
    provider = _provider([held], "Stopped as you asked.")

    with listening() as listener, patched(provider):
        chat = begin(UserConfig(), _QUESTION)
        wait(held.entered, "the search to reach its model call")
        searching = db.execute(
            "SELECT turn_id FROM turn_executions WHERE channel = ? AND ended_at IS NULL", (_SEARCH,),
        ).fetchone()["turn_id"]
        stop = client.delete(f"/api/threads/{searching}?channel={_SEARCH}")
        held.gate.set()
        chat.result()
        drain()

    assert stop.status_code == 200 and (stop.get_json() or {})["result"]["cancelled"] is True
    assert held.opened
    assert execution(_SEARCH, searching).state == TurnExecution.CANCELLED
    assert db.execute(
        "SELECT COUNT(*) FROM transcript WHERE channel = ? AND role = 'assistant'", (_SEARCH,),
    ).fetchone()[0] == 0
    lifecycle = [f for f in listener.frames if f.get("channel") == _SEARCH and "tool_name" not in f]
    assert lifecycle and lifecycle[-1]["state"] == TurnExecution.CANCELLED
    thread = block(client, f"/api/threads/{searching}?channel={_SEARCH}")
    assert (thread["cancelled"], thread["crashed"], thread["working"]) == (True, False, False)
    assert [m["content"] for m in messages(thread)] == [_INSTRUCTIONS]
    call = calls_of(db, "web_search")[0]
    assert call["state"] == "error"
    assert _error_code(call["result"]) == "delegate-stopped"
    assert not _HINT_LINE.search(call["result"])
    _assert_subtask_completed_with_its_answer(db, "Stopped as you asked.")
    assert provider.unused() == {}


def test_subagent_that_finishes_without_an_answer_still_reports_no_answer_with_its_hint(
    db: sqlite3.Connection,
) -> None:
    """A child nobody stopped that completes with an empty answer is not
    mistaken for a stopped one: the waiting caller's call fails as
    ``delegate-no-answer`` and keeps the tool's retry hint."""
    provider = _provider([calls(tool("recall", query="capital of France")), text("")], "I could not find it.")

    with patched(provider):
        begin(UserConfig(), _QUESTION).result()
        drain()

    call = calls_of(db, "web_search")[0]
    assert execution(_SEARCH, child_turn(db, _SEARCH, call["id"])).state == TurnExecution.COMPLETED
    assert call["state"] == "error"
    assert _error_code(call["result"]) == "delegate-no-answer"
    assert _HINT_LINE.search(call["result"])
    _assert_subtask_completed_with_its_answer(db, "I could not find it.")
    assert provider.unused() == {}


@pytest.mark.parametrize(
    ("config_type", "broadcasts"),
    [("user", True), ("scheduled", True), ("discovery", False)],
)
def test_stopping_a_typed_turn_broadcasts_as_that_turn_always_did(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object], config_type: str, broadcasts: bool,
) -> None:
    """A chat or scheduled turn's stop sends one cancelled frame addressed by its
    type, so the surface showing that thread sees it end; a background
    (discovery) turn, which never broadcasts its state, stays silent."""
    client, _db, _store = authed_client
    config = config_for(config_type)
    assert MessageProcessor(config, 5).turn_execution_service.open() is not None

    with listening() as listener:
        response = client.delete(f"/api/threads/5?type={config_type}")

    assert (response.get_json() or {})["result"]["cancelled"] is True
    stopped = TurnExecution.latest(config.channel, 5)
    assert stopped is not None and stopped.state == TurnExecution.CANCELLED
    if broadcasts:
        assert [(f["type"], f["turn_id"], f["state"]) for f in listener.frames] == [
            (config_type, 5, TurnExecution.CANCELLED),
        ]
    else:
        assert listener.frames == []


def test_stopping_a_turn_whose_type_no_longer_resolves_warns_and_sends_no_frame(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """A running row left with a type no config answers to any more (written by
    an older build) is still stopped — but which surface it belongs to cannot be
    worked out, so no frame goes out under a guessed address and the gap is
    reported with a warning instead of failing the stop."""
    client, db, _store = authed_client
    assert MessageProcessor(UserConfig(), 5).turn_execution_service.open() is not None
    db.execute("UPDATE turn_executions SET type = 'retired' WHERE channel = ? AND turn_id = 5", (UserConfig().channel,))
    db.commit()

    with watching_warnings(source_name=TurnExecutionService.__module__) as watch, listening() as listener:
        response = client.delete("/api/threads/5")

    assert response.status_code == 200
    assert (response.get_json() or {})["result"]["cancelled"] is True
    stopped = TurnExecution.latest(UserConfig().channel, 5)
    assert stopped is not None and stopped.state == TurnExecution.CANCELLED
    assert listener.frames == []
    assert len(watch.records) == 1
