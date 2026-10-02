"""Feature test: stopping a running subagent, and what its caller is told.

A person can stop a delegate's child turn from its transcript panel. The child
must actually stop — its late answer is never stored and it ends cancelled on
the wire — and the calling turn must learn the subagent was stopped by the user,
under its own outcome and without the retry advice an empty answer carries, then
finish its own turn. Stopping a typed turn (chat, scheduled, background) keeps
broadcasting exactly as before, and a stopped row whose type no longer resolves
is reported loudly without a frame sent under a guessed address.

Drives the real production entry point (construct inertly, ``begin()``,
``result()``) and the real Flask stop route against the real, fully-migrated
SQLite DB, observing frames through the REAL socket registry. The only
substitution is the LLM network boundary
(``services.provider_service.build_client``); the child is held inside its own
model call on a ``threading.Event`` so the stop lands while it is running.
"""

import re
import sqlite3
import threading
from typing import cast
from unittest.mock import patch

import pytest
from flask.testing import FlaskClient

from configs.channels import config_for
from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.provider_response import ProviderResponse
from models.turn_execution import TurnExecution
from services.turn_execution_service import TurnExecutionService
from tests.test_delegate_transcript_watch import (
    _BUILD_CLIENT,
    _CHILD_ANSWER,
    _CHILD_CHANNEL,
    _INSTRUCTIONS,
    _assert_parent_turn_completed_with_its_answer,
    _block,
    _drain_background_turns,
    _HeldChildProvider,
    _listening,
    _ScriptedProvider,
    _tool,
    _watching_warnings,
    _web_search_call,
)

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

_QUESTION = "what is the capital of France?"
_ERROR_CODE = re.compile(r"\A\[web_search\(status=error, code=([\w-]+)")
_HINT_LINE = re.compile(r"^hint: \S", re.MULTILINE)


def _error_code(result: str) -> str | None:
    """The outcome code in the header of a failed call's result envelope."""
    match = _ERROR_CODE.match(result)
    return match.group(1) if match else None


def test_stopped_subagent_is_reported_as_stopped_and_its_caller_still_answers(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """Stopping the running ``web_search`` child by its channel ends it: the
    answer its in-flight model call comes back with is never stored, its last
    lifecycle frame says cancelled, its execution row stays cancelled, and its
    transcript reads back as stopped with only its task in it. The
    caller's call fails as ``delegate-stopped`` with no retry hint — the user
    chose to stop that work — and the caller's own turn still completes with its
    reply."""
    client, db, _store = authed_client
    gate = threading.Event()
    provider = _HeldChildProvider(
        gate,
        lambda: None,
        ProviderResponse(text="", model="m", tool_calls=[_tool("web_search", instructions=_INSTRUCTIONS)]),
        ProviderResponse(text=_CHILD_ANSWER, model="m", tool_calls=None),
        ProviderResponse(text="Stopped as you asked.", model="m", tool_calls=None),
    )
    parent = MessageProcessor(UserConfig(), raw_input=_QUESTION)

    with _listening() as listener, patch(_BUILD_CLIENT, return_value=provider):
        parent.begin()
        try:
            assert provider.entered.wait(timeout=10), "the subagent never reached its model call"
            child_turn = db.execute(
                "SELECT turn_id FROM turn_executions WHERE channel = ? AND ended_at IS NULL", (_CHILD_CHANNEL,),
            ).fetchone()["turn_id"]
            stop = client.delete(f"/api/threads/{child_turn}?channel={_CHILD_CHANNEL}")
        finally:
            gate.set()
            parent.result()
            _drain_background_turns()

    assert stop.status_code == 200 and (stop.get_json() or {})["result"]["cancelled"] is True
    assert provider.gate_opened
    child = TurnExecution.latest(_CHILD_CHANNEL, child_turn)
    assert child is not None and child.state == TurnExecution.CANCELLED
    assert db.execute(
        "SELECT COUNT(*) FROM transcript WHERE channel = ? AND role = 'assistant'", (_CHILD_CHANNEL,),
    ).fetchone()[0] == 0
    lifecycle = [f for f in listener.frames if f.get("channel") == _CHILD_CHANNEL and "tool_name" not in f]
    assert lifecycle and lifecycle[-1]["state"] == TurnExecution.CANCELLED
    block = _block(client, f"/api/threads/{child_turn}?channel={_CHILD_CHANNEL}")
    assert (block["cancelled"], block["crashed"], block["working"]) == (True, False, False)
    assert [m["content"] for m in cast("list[dict[str, object]]", block["messages"])] == [_INSTRUCTIONS]
    call = _web_search_call(db)
    assert call["state"] == "error"
    assert _error_code(call["result"]) == "delegate-stopped"
    assert not _HINT_LINE.search(call["result"])
    _assert_parent_turn_completed_with_its_answer(db, parent, "Stopped as you asked.")


def test_subagent_that_finishes_without_an_answer_still_reports_no_answer_with_its_hint(
    db: sqlite3.Connection,
) -> None:
    """A child nobody stopped that completes with an empty answer is not
    mistaken for a stopped one: the caller's call fails as
    ``delegate-no-answer`` and keeps the tool's retry hint."""
    provider = _ScriptedProvider(
        ProviderResponse(text="", model="m", tool_calls=[_tool("web_search", instructions=_INSTRUCTIONS)]),
        ProviderResponse(text="", model="m", tool_calls=[_tool("recall", query="capital of France")]),
        ProviderResponse(text="", model="m", tool_calls=None),
        ProviderResponse(text="I could not find it.", model="m", tool_calls=None),
    )
    parent = MessageProcessor(UserConfig(), raw_input=_QUESTION)

    with patch(_BUILD_CLIENT, return_value=provider):
        parent.begin()
        parent.result()
        _drain_background_turns()

    child_turn = db.execute(
        "SELECT turn_id FROM transcript WHERE channel = ? AND role = 'web_search'", (_CHILD_CHANNEL,),
    ).fetchone()["turn_id"]
    child = TurnExecution.latest(_CHILD_CHANNEL, child_turn)
    assert child is not None and child.state == TurnExecution.COMPLETED
    call = _web_search_call(db)
    assert call["state"] == "error"
    assert _error_code(call["result"]) == "delegate-no-answer"
    assert _HINT_LINE.search(call["result"])
    _assert_parent_turn_completed_with_its_answer(db, parent, "I could not find it.")


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

    with _listening() as listener:
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

    with _watching_warnings(source_name=TurnExecutionService.__module__) as watch, _listening() as listener:
        response = client.delete("/api/threads/5")

    assert response.status_code == 200
    assert (response.get_json() or {})["result"]["cancelled"] is True
    stopped = TurnExecution.latest(UserConfig().channel, 5)
    assert stopped is not None and stopped.state == TurnExecution.CANCELLED
    assert listener.frames == []
    assert len(watch.records) == 1
