"""Feature tests: a subagent's permission prompt can be answered after the chat
turn that started it has ended.

The person's chat starts a subtask and ends its own turn at once; the subtask
then calls a tool whose permission is set to ask. The prompt is listed by
``GET /api/policies/pending`` under the chat turn that started the subtask, and
``POST /api/policies/respond`` answers it by its id alone. Approved, the tool
runs; denied, it does not run and the subtask reads that it was not allowed.
Either way the subtask goes on to answer, and its answer reaches the chat as a
reply.

Drives real turns (construct inertly, ``begin()``, ``result()``) against the
real, fully migrated SQLite DB, observed through the real socket registry and
answered through the real Flask app with a real signed-in session. The only
substitution is the LLM network boundary
(``services.provider_service.build_client``), answered from one script per
channel, every script used up by the end of a test. The subtask step that
calls the tool is held until the chat turn that asked for the work has ended,
so the prompt is always raised after it, as it is for a person.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, cast

import pytest
from flask import Flask
from flask.testing import FlaskClient

from abilities.subtask import SubtaskAbility
from configs.channels.user import UserConfig
from configs.enums.channels import Channel
from models.tool_call import ToolCall
from models.turn_execution import TurnExecution
from services.policy_manager import _BLOCK
from tests._browser_live import signed_in, use_own_data_dir
from tests._delegates import (
    ChannelProvider,
    Held,
    agent_number,
    begin,
    calls,
    calls_of,
    child,
    execution,
    listening,
    patched,
    replies,
    text,
    tool,
    wait,
)
from tests.test_policy_gate_origin import _gate_probe_asks, gate_probe

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider", "earlier_agents")]

_SUBTASK = Channel.DELEGATE_SUBTASK.value
_USER = Channel.USER.value
_TASK = "probe the gate"
_SUMMARY = "Probing the gate"
_ANSWER = "The gate was probed."
_RAN = "gate probe done"

Ask = dict[str, object]


class _Answered(NamedTuple):
    """One run of the scenario: the chat turn that started the subtask, the
    prompts listed while the subtask waited, those still listed once every
    turn ended, and the provider that served every turn."""

    chat_turn: int
    listed: list[Ask]
    left: list[Ask]
    provider: ChannelProvider


@pytest.fixture
def app(db: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Flask:
    """The real app on the test's database, keeping its data under *tmp_path*."""
    use_own_data_dir(tmp_path, monkeypatch)
    from api import create_app  # noqa: PLC0415 — built per test, on the test's database

    built = create_app()
    built.config["TESTING"] = True
    return built


def _pending(client: FlaskClient) -> list[Ask]:
    response = client.get("/api/policies/pending")
    assert response.status_code == 200, response.data
    return cast("list[Ask]", cast("dict[str, object]", response.get_json())["result"])


def _noting_prompts(asked: threading.Event) -> Callable[[dict[str, object]], None]:
    """A frame listener that sets *asked* once a permission prompt is broadcast."""
    def note(frame: dict[str, object]) -> None:
        if frame.get("type") == "permission_request":
            asked.set()
    return note


def _answer_when_asked(client: FlaskClient, asked: threading.Event, approved: bool) -> list[Ask]:
    """Wait for a prompt to be raised, then read the pending prompts as the
    signed-in person and answer each with *approved*; returns those read."""
    wait(asked, "the subtask's permission prompt")
    listed = _pending(client)
    for ask in listed:
        response = client.post(
            "/api/policies/respond", json={"request_id": ask["request_id"], "approved": approved},
        )
        assert response.status_code == 204, response.data
    return listed


def _answer_the_subtasks_prompt(app: Flask, approved: bool) -> _Answered:
    """Run the person's chat turn that starts a subtask to its end, then let the
    subtask call the probe, whose permission asks. Once its prompt is raised,
    answer it with *approved*, then wait for every turn to end, the subtask's
    reply among them."""
    _gate_probe_asks()
    client, _token = signed_in(app)
    asked = threading.Event()
    held = Held(calls(tool(gate_probe.NAME, q="x", act_summary=_SUMMARY)))
    provider = ChannelProvider(
        chat=[
            calls(tool(SubtaskAbility.NAME, action="start", instructions=_TASK)),
            text("I have started a subtask."),
            text("The subtask answered."),
        ],
        subtask=[held, text(_ANSWER)],
    )

    with listening(_noting_prompts(asked)), patched(provider):
        chat = begin(UserConfig(), "probe the gate in the background")
        chat.result()
        assert execution(_USER, chat.turn_id).state == TurnExecution.COMPLETED
        wait(held.entered, "the subtask's first step")
        held.gate.set()
        listed = _answer_when_asked(client, asked, approved)
    assert held.opened and provider.unused() == {}
    return _Answered(chat.turn_id, listed, _pending(client), provider)


def _assert_listed_under_the_chat_turn(run: _Answered) -> None:
    """The prompt was listed once, for the probe, with the subtask's summary,
    under the chat turn that started the subtask; nothing is left pending."""
    [ask] = run.listed
    assert (ask["action_id"], ask["summary"], ask["origin"]) == (
        gate_probe.NAME, _SUMMARY, {"type": "user", "turn_id": run.chat_turn, "forked": False},
    )
    assert run.left == []


def _assert_the_subtask_answered_the_chat(db: sqlite3.Connection, run: _Answered) -> None:
    """The subtask finished its own turn, and its answer arrived on the chat as
    a reply opening a turn of its own, the chat's turn having ended."""
    [call] = calls_of(db, SubtaskAbility.NAME)
    channel, n = child(db, _SUBTASK, call["id"])
    assert execution(channel, n).state == TurnExecution.COMPLETED
    [reply] = replies(db, call["id"])
    assert reply["content"] == _ANSWER
    assert reply["turn_id"] != run.chat_turn
    assert f"Reply from {SubtaskAbility.NAME} #{agent_number(channel)}" in run.provider.texts("chat")[-1]


def test_an_approved_prompt_runs_the_tool_and_the_subtask_still_answers_the_chat(
    app: Flask, db: sqlite3.Connection,
) -> None:
    run = _answer_the_subtasks_prompt(app, approved=True)

    _assert_listed_under_the_chat_turn(run)
    [probe] = calls_of(db, gate_probe.NAME)
    assert probe["state"] == ToolCall.DONE
    assert _RAN in probe["result"]
    # What the subtask read on its next step: the tool's own result.
    assert _RAN in run.provider.read("subtask")[-1]
    _assert_the_subtask_answered_the_chat(db, run)


def test_a_denied_prompt_does_not_run_the_tool_the_subtask_is_told_and_still_answers_the_chat(
    app: Flask, db: sqlite3.Connection,
) -> None:
    run = _answer_the_subtasks_prompt(app, approved=False)

    _assert_listed_under_the_chat_turn(run)
    refusal = _BLOCK.format(permission=gate_probe.NAME)
    [probe] = calls_of(db, gate_probe.NAME)
    assert (probe["state"], probe["result"]) == (ToolCall.ERROR, refusal)
    # What the subtask read on its next step: the refusal, never the tool's result.
    read = run.provider.read("subtask")[-1]
    assert refusal in read
    assert _RAN not in read
    _assert_the_subtask_answered_the_chat(db, run)
