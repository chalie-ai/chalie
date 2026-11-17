"""Feature test: a subtask runs beside the person's chat and always answers
it; a scheduled task waits for its subtask instead.

A ``subtask`` call from the chat opens a background turn on the
``delegate:subtask`` channel and returns at once, so the chat keeps working.
Whatever way the subtask ends, its outcome comes back on the chat as an input
row linked to the call that asked for it ("Reply from subtask #N"):

* it joins the caller's turn while that turn still works, and opens a new user
  turn once the caller has finished;
* a scheduled task's subtask is waited for: its answer is the call's result
  and nothing is sent back afterwards;
* a subtask cannot start one of its own, and only the conversation that owns a
  subtask can message it;
* a subtask stopped by its caller's stop sends nothing back — no one is
  waiting — while one stopped on its own, one that crashed, and one a restart
  cut off each send a notice saying so;
* the memory step runs on the subtask channel too, in a turn of its own, and
  that turn has no caller: it sends nothing back, finished or cut off.

Drives the real production entry point (construct inertly, ``begin()``,
``result()``) against the real, fully-migrated SQLite DB: the real
``SubtaskAbility``, ``DispatchService``, ``TurnExecutionService``, boot-time
sweep and reply, and Flask thread reads all run. The only substitution is the
LLM network boundary (``services.provider_service.build_client``), served by
the shared :class:`tests._delegates.ChannelProvider`; a turn whose order
matters is pinned by holding its provider call on a ``threading.Event``.
"""

import logging
import sqlite3
from typing import cast

import pytest
from flask.testing import FlaskClient

from configs.channels import config_for
from configs.channels.scheduled import ScheduledConfig
from configs.channels.subtask import UserSubtaskConfig
from configs.channels.user import UserConfig
from configs.enums.channels import Channel
from configs.enums.config_type import ConfigTypeEnum
from controllers.message_processor import _MAX_PROVIDER_ATTEMPTS, MessageProcessor
from models.provider_response import ProviderResponse
from models.tool_call import ToolCall
from models.transcript import Transcript
from models.turn_execution import TurnExecution
from run import _reply_to_swept_delegates
from services.memory_step_service import memory_step_config
from services.turn_execution_service import TurnExecutionService
from tests._delegates import (
    WAIT_S,
    ChannelProvider,
    Entry,
    Held,
    as_sent,
    begin,
    block,
    calls,
    calls_of,
    child_turn,
    chip,
    dies,
    drain,
    execution,
    listening,
    messages,
    patched,
    process_deaths_expected,
    raises,
    replies,
    text,
    tool,
    wait,
)

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

_SUBTASK = Channel.DELEGATE_SUBTASK.value
_USER = Channel.USER.value
_SCHEDULE = Channel.SCHEDULE.value
_TASK = "list the primes below 6"
_ANSWER = "2, 3, 5"
_EXHAUSTED = "The AI provider failed to respond after several attempts. Please try again in a moment."
_CUT_OFF = "This subtask was cut off by a restart before it finished."

_START = calls(tool("subtask", action="start", instructions=_TASK))


def _all_replies(db: sqlite3.Connection) -> list[str]:
    """The content of every reply row any subtask sent to the user's
    conversation, oldest first."""
    return [
        cast("str", r["content"]) for r in db.execute(
            "SELECT content FROM transcript WHERE channel = ? AND role = 'user' AND tool_call_id IS NOT NULL "
            "ORDER BY id",
            (_USER,),
        ).fetchall()
    ]


def _memory_turns(db: sqlite3.Connection) -> list[int]:
    """Every memory-step turn on the subtask channel — the turns whose
    opening row carries the ``memory`` role."""
    return [
        cast("int", r["turn_id"]) for r in db.execute(
            "SELECT DISTINCT turn_id FROM transcript WHERE channel = ? AND role = 'memory' ORDER BY turn_id",
            (_SUBTASK,),
        ).fetchall()
    ]


def _errors(caplog: pytest.LogCaptureFixture, logger: str) -> list[str]:
    """Every ERROR record *logger* emitted during the test."""
    return [r.getMessage() for r in caplog.records if r.name == logger and r.levelno >= logging.ERROR]


def _first_call(db: sqlite3.Connection) -> int:
    """The id of the first ``subtask`` call."""
    return cast("int", calls_of(db, "subtask")[0]["id"])


def test_reply_after_the_caller_finished_opens_a_new_user_turn(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object], db: sqlite3.Connection,
) -> None:
    """The caller answers and finishes while its subtask still works; the
    subtask's answer then arrives as the opening message of a new user turn,
    labelled with the subtask's number, and that turn answers it. The caller's
    pill points at the subtask, which no longer works; the subtask's thread
    opens with the task, labelled as Chalie's."""
    client, _db, _store = authed_client
    held = Held(text(_ANSWER))
    summary = "Listing the primes below 6"
    start = calls(tool("subtask", action="start", instructions=_TASK, act_summary=summary))
    provider = ChannelProvider(chat=[start, text("Started it."), text("Got the reply.")], subtask=[held])

    with patched(provider):
        parent = begin(UserConfig(), "find the primes below 6 in the background")
        parent.result()
        wait(held.entered, "the subtask's provider call")
        held.gate.set()

    assert held.opened
    assert provider.unused() == {}
    call_id = _first_call(db)
    [reply] = replies(db, call_id)
    n = child_turn(db, _SUBTASK, call_id)
    assert (reply["content"], reply["joined"]) == (_ANSWER, 0)
    assert reply["turn_id"] != parent.turn_id
    assert execution(_SUBTASK, n).state == TurnExecution.COMPLETED
    assert execution(_USER, reply["turn_id"]).state == TurnExecution.COMPLETED
    assert f"Reply from subtask #{n}" in provider.texts("chat")[-1]
    # The chat's call returned at once with the subtask's number and the label
    # its answer arrives under, written as that label writes them — never the
    # answer itself.
    started = cast("str", calls_of(db, "subtask")[0]["result"])
    assert started == (
        "[subtask(status=success)]\n"
        f'Subagent subtask #{n} has been spawned. It will reply to you once it\'s done as '
        f'"Reply from subtask #{n}". Acknowledge to the user that the agent has been spawned '
        "and you'll be invoked automatically once it replies.\n"
        "[end:subtask]"
    )

    opener = messages(block(client, f"/api/threads/{reply['turn_id']}"))[0]
    assert (opener["role"], opener["content"]) == ("user", _ANSWER)
    # The reply carries plain data for the interface to word — the subagent's
    # tool and the summary of the call that started it — and never the label
    # the model reads.
    assert opener["reply_from"] == {"tool_name": "subtask", "summary": summary}
    assert opener.get("speaker") is None
    parent_block = block(client, f"/api/threads/{parent.turn_id}")
    assert chip(parent_block, "subtask")["delegate"] == {
        "channel": _SUBTASK, "turn_id": n, "working": False, "crashed": False, "cancelled": False,
    }
    task = messages(block(client, f"/api/threads/{n}?channel={_SUBTASK}"))[0]
    assert (task["content"], task["speaker"]) == (_TASK, "Task from Chalie")


def test_reply_joins_the_caller_turn_while_it_still_works(db: sqlite3.Connection) -> None:
    """A subtask that finishes while its caller is mid-call lands its answer on
    the caller's own turn as a joined message; the caller does not end on the
    call it was making but steps again with the reply in view."""
    caller_held = Held(text("Done for now."))
    subtask_held = Held(text(_ANSWER))
    provider = ChannelProvider(chat=[_START, caller_held, text("Got the reply.")], subtask=[subtask_held])

    with patched(provider):
        parent = begin(UserConfig(), "find the primes below 6 in the background")
        wait(caller_held.entered, "the caller's second provider call")
        wait(subtask_held.entered, "the subtask's provider call")
        subtask_held.gate.set()
        assert subtask_held.thread is not None
        subtask_held.thread.join(timeout=WAIT_S)
        assert not subtask_held.thread.is_alive(), "the subtask's turn did not end"
        caller_held.gate.set()
        parent.result()

    assert caller_held.opened and subtask_held.opened
    assert provider.unused() == {}
    call_id = _first_call(db)
    [reply] = replies(db, call_id)
    n = child_turn(db, _SUBTASK, call_id)
    assert (reply["content"], reply["turn_id"], reply["joined"]) == (_ANSWER, parent.turn_id, 1)
    assert f"Reply from subtask #{n}" in provider.texts("chat")[-1]
    assert execution(_USER, parent.turn_id).state == TurnExecution.COMPLETED
    count = db.execute(
        "SELECT COUNT(*) FROM turn_executions WHERE channel = ? AND turn_id = ?", (_USER, parent.turn_id),
    ).fetchone()[0]
    assert count == 1
    final = db.execute(
        "SELECT content FROM transcript WHERE channel = ? AND turn_id = ? AND role = 'assistant' "
        "ORDER BY id DESC LIMIT 1",
        (_USER, parent.turn_id),
    ).fetchone()
    assert final["content"] == "Got the reply."


def test_a_subtask_cannot_start_a_subtask(db: sqlite3.Connection) -> None:
    """A subtask's own ``subtask`` call is refused: no second subtask opens,
    the refusal is what the subtask sees next, and it still answers its
    caller."""
    nested = Held(calls(tool("subtask", action="start", instructions="count to three")))
    provider = ChannelProvider(
        chat=[_START, text("Started it."), text("Got the reply.")],
        subtask=[nested, text("I did it myself.")],
    )

    with patched(provider):
        parent = begin(UserConfig(), "find the primes below 6 in the background")
        parent.result()
        wait(nested.entered, "the subtask's provider call")
        nested.gate.set()

    assert nested.opened
    assert provider.unused() == {}
    started = calls_of(db, "subtask")
    assert len(started) == 2
    refused = started[1]
    assert refused["state"] == ToolCall.ERROR
    assert "code=subtask-nested" in refused["result"]
    linked = db.execute(
        "SELECT COUNT(*) FROM transcript WHERE channel = ? AND tool_call_id IS NOT NULL", (_SUBTASK,),
    ).fetchone()[0]
    assert linked == 1
    assert "subtask-nested" in provider.texts("subtask")[-1]
    [reply] = replies(db, cast("int", started[0]["id"]))
    assert reply["content"] == "I did it myself."


def test_message_reaches_a_working_subtask_and_an_unknown_one_is_refused(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object], db: sqlite3.Connection,
) -> None:
    """The caller messages its working subtask and a number no subtask of its
    has: the first joins the subtask's turn, labelled as Chalie's, and the
    subtask answers with it in view — the call that sent the message is told
    the answer will arrive later under the subtask's label and to carry on,
    and the reply then goes back to that call; the second is refused as not
    found."""
    client, _db, _store = authed_client
    subtask_held = Held(text(_ANSWER))

    def message_both() -> ProviderResponse:
        wait(subtask_held.entered, "the subtask's provider call")
        started = TurnExecution.filter("channel", _SUBTASK).first()
        assert started is not None
        return calls(
            tool("subtask", action="message", subtask=started.turn_id, message="also add 7"),
            tool("subtask", action="message", subtask=9999, message="hello"),
        )

    provider = ChannelProvider(
        chat=[_START, message_both, text("Told it."), text("Got the reply.")],
        subtask=[subtask_held, text("2, 3, 5, 7")],
    )

    with patched(provider):
        parent = begin(UserConfig(), "find the primes below 6 in the background")
        parent.result()
        subtask_held.gate.set()

    assert subtask_held.opened
    assert provider.unused() == {}
    start, sent, refused = calls_of(db, "subtask")
    n = child_turn(db, _SUBTASK, cast("int", start["id"]))
    assert sent["state"] == ToolCall.DONE
    assert (
        f'Message sent to subtask #{n}. Its answer will arrive as a new message labelled "Reply from subtask #{n}". '
        "Carry on — do not wait for it."
    ) in sent["result"]
    assert refused["state"] == ToolCall.ERROR
    assert "code=subtask-not-found" in refused["result"]
    assert "No subtask #9999 belongs to this conversation." in refused["result"]

    message = db.execute(
        "SELECT turn_id, joined FROM transcript WHERE channel = ? AND tool_call_id = ?",
        (_SUBTASK, str(sent["id"])),
    ).fetchone()
    assert (message["turn_id"], message["joined"]) == (n, 1)
    assert "also add 7" in provider.texts("subtask")[-1]
    [reply] = replies(db, cast("int", sent["id"]))
    assert reply["content"] == "2, 3, 5, 7"
    assert replies(db, cast("int", start["id"])) == []

    thread = messages(block(client, f"/api/threads/{n}?channel={_SUBTASK}"))
    assert [(m["content"], m["speaker"]) for m in thread if m["role"] == "user"] == [
        (_TASK, "Task from Chalie"), ("also add 7", "Message from Chalie"),
    ]


def test_stopping_the_caller_stops_its_subtask_without_a_reply(db: sqlite3.Connection) -> None:
    """Stopping a caller mid-turn stops the subtask it started as well, marked
    as stopped by its caller, and the subtask sends nothing back: no one is
    waiting for its answer."""
    caller_held = Held(text("Done for now."))
    subtask_held = Held(text(_ANSWER))
    provider = ChannelProvider(chat=[_START, caller_held], subtask=[subtask_held])

    with patched(provider):
        parent = begin(UserConfig(), "find the primes below 6 in the background")
        wait(caller_held.entered, "the caller's second provider call")
        wait(subtask_held.entered, "the subtask's provider call")
        TurnExecutionService.cancel_turn(_USER, parent.turn_id)
        caller_held.gate.set()
        subtask_held.gate.set()
        parent.result()

    assert caller_held.opened and subtask_held.opened
    call_id = _first_call(db)
    n = child_turn(db, _SUBTASK, call_id)
    caller = execution(_USER, parent.turn_id)
    assert (caller.state, caller.stop_reason) == (TurnExecution.CANCELLED, None)
    subtask = execution(_SUBTASK, n)
    assert (subtask.state, subtask.stop_reason) == (TurnExecution.CANCELLED, "parent stopped")
    assert replies(db, call_id) == []


def test_a_stopped_subtask_tells_its_caller(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object], db: sqlite3.Connection,
) -> None:
    """A subtask stopped on its own — its caller still fine — sends a notice
    that it was stopped before it finished, and the caller's history shows its
    call's subtask as stopped, not crashed."""
    client, _db, _store = authed_client
    held = Held(text(_ANSWER))
    provider = ChannelProvider(chat=[_START, text("Started it."), text("Noted.")], subtask=[held])

    with patched(provider):
        parent = begin(UserConfig(), "find the primes below 6 in the background")
        parent.result()
        wait(held.entered, "the subtask's provider call")
        call_id = _first_call(db)
        n = child_turn(db, _SUBTASK, call_id)
        TurnExecutionService.cancel_turn(_SUBTASK, n)
        held.gate.set()

    assert held.opened
    assert provider.unused() == {}
    subtask = execution(_SUBTASK, n)
    assert (subtask.state, subtask.stop_reason) == (TurnExecution.CANCELLED, None)
    [reply] = replies(db, call_id)
    assert reply["content"] == "This subtask was stopped before it finished."
    assert reply["turn_id"] != parent.turn_id
    assert chip(block(client, f"/api/threads/{parent.turn_id}"), "subtask")["delegate"] == {
        "channel": _SUBTASK, "turn_id": n, "working": False, "crashed": False, "cancelled": True,
    }


def test_a_crashed_subtask_tells_its_caller_why(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object], db: sqlite3.Connection,
) -> None:
    """A subtask whose provider fails on every attempt crashes, and its caller
    is told so with the reason the turn ended on; the caller's history shows
    its call's subtask as crashed, not stopped."""
    client, _db, _store = authed_client
    boom = RuntimeError("boom")
    first = Held(boom)
    script: list[Entry] = [first, *(raises(boom) for _ in range(_MAX_PROVIDER_ATTEMPTS - 1))]
    provider = ChannelProvider(chat=[_START, text("Started it."), text("Noted.")], subtask=script)

    with patched(provider):
        parent = begin(UserConfig(), "find the primes below 6 in the background")
        parent.result()
        wait(first.entered, "the subtask's provider call")
        first.gate.set()

    assert first.opened
    assert provider.unused() == {}
    assert len(provider.texts("subtask")) == _MAX_PROVIDER_ATTEMPTS
    call_id = _first_call(db)
    n = child_turn(db, _SUBTASK, call_id)
    subtask = execution(_SUBTASK, n)
    assert (subtask.state, subtask.stop_reason) == (TurnExecution.CRASHED, _EXHAUSTED)
    [reply] = replies(db, call_id)
    assert reply["content"] == f"This subtask crashed: {_EXHAUSTED}"
    assert chip(block(client, f"/api/threads/{parent.turn_id}"), "subtask")["delegate"] == {
        "channel": _SUBTASK, "turn_id": n, "working": False, "crashed": True, "cancelled": False,
    }


def test_the_memory_step_after_a_subtask_answers_no_one(
    real_memory_step: None, db: sqlite3.Connection, caplog: pytest.LogCaptureFixture,
) -> None:
    """With the real memory step armed, a finished subtask's settle runs a
    memory step on the subtask channel, in a turn of its own. Only the
    subtask's own turn answers the caller: exactly one reply reaches it, and
    nothing fails trying to report the memory step's turn to a caller it does
    not have."""
    held = Held(text(_ANSWER))
    provider = ChannelProvider(chat=[_START, text("Started it."), text("Got the reply.")], subtask=[held])

    with patched(provider):
        parent = begin(UserConfig(), "find the primes below 6 in the background")
        parent.result()
        wait(held.entered, "the subtask's provider call")
        held.gate.set()

    assert held.opened
    assert provider.unused() == {}
    n = child_turn(db, _SUBTASK, _first_call(db))
    [memory] = _memory_turns(db)
    assert memory != n
    assert execution(_SUBTASK, memory).state == TurnExecution.COMPLETED
    assert _all_replies(db) == [_ANSWER]
    assert _errors(caplog, "controllers.message_processor") == []


def test_boot_tells_a_cut_off_subtask_s_caller_once_and_a_cut_off_memory_step_no_one(
    db: sqlite3.Connection, caplog: pytest.LogCaptureFixture,
) -> None:
    """The process dies with two turns open on the subtask channel: a subtask
    mid-call, and the memory step of an earlier subtask that had already
    answered. Boot closes both as cut off by the restart and tells the cut-off
    subtask's caller exactly once, in a new user turn that answers it; the
    memory step's turn has no caller, so it gets no notice and nothing fails
    trying to send one."""
    first = Held(text(_ANSWER))
    subtask: list[Entry] = [first, dies]
    background: list[Entry] = [dies]
    provider = ChannelProvider(
        chat=[
            _START, text("Started it."), text("Got the reply."),
            _START, text("Started another."), text("Noted."),
        ],
        subtask=subtask,
        background=background,
    )

    with patched(provider), process_deaths_expected():
        begin(UserConfig(), "find the primes below 6 in the background").result()
        wait(first.entered, "the first subtask's provider call")
        first.gate.set()
        drain()
        answered_id = _first_call(db)
        assert [r["content"] for r in replies(db, answered_id)] == [_ANSWER]
        ids = Transcript.ids_for_turn(_SUBTASK, child_turn(db, _SUBTASK, answered_id))
        MessageProcessor.process(memory_step_config(UserSubtaskConfig(), ids)).result()
        caller = begin(UserConfig(), "now find the primes below 10 in the background")
        caller.result()
        drain()
        cut_id = cast("int", calls_of(db, "subtask")[1]["id"])
        cut = child_turn(db, _SUBTASK, cut_id)
        [memory] = _memory_turns(db)
        assert sorted((r.channel, r.turn_id) for r in TurnExecution.open_rows()) == sorted([
            (_SUBTASK, memory), (_SUBTASK, cut),
        ])

        swept = MessageProcessor(config_for(ConfigTypeEnum.USER)).turn_execution_service.sweep_orphaned()
        _reply_to_swept_delegates(swept)

    assert first.opened
    assert provider.unused() == {}
    assert sorted((r.channel, r.turn_id, r.state, r.stop_reason) for r in swept) == sorted([
        (_SUBTASK, memory, TurnExecution.CRASHED, "process death"),
        (_SUBTASK, cut, TurnExecution.CRASHED, "process death"),
    ])
    [notice] = replies(db, cut_id)
    assert _all_replies(db) == [_ANSWER, _CUT_OFF]
    assert notice["turn_id"] != caller.turn_id
    assert notice["joined"] == 0
    assert execution(_USER, cast("int", notice["turn_id"])).state == TurnExecution.COMPLETED
    assert _errors(caplog, "run") == []
    assert list(TurnExecution.open_rows()) == []


def test_a_scheduled_task_waits_for_its_subtask_and_ends_once(db: sqlite3.Connection) -> None:
    """A scheduled task can start a subtask, and it waits for it. The
    ``subtask`` tool its turn is sent can only start one and says the call
    waits, while the chat is still sent the follow-up action and told the
    answer arrives later. The start ends with the subtask's answer and nothing
    in front of it, the schedule's model reads that answer on its next step,
    nothing is sent back to the schedule's thread or to the user's
    conversation, and the schedule's turn ends once — one execution, one
    finished frame, after the subtask's.

    Nothing can be sent to that subtask: the schedule's own ``message`` call,
    naming the subtask and the message, is refused whole for parameters it was
    never shown, and a bare ``message`` is refused too; neither reaches the
    subtask or claims a reply will come. The user's conversation does not own
    it, so its message, sent while the schedule waits, is refused as not found
    and never reaches it either."""
    held = Held(text(_ANSWER))

    def started() -> int:
        found = TurnExecution.filter("channel", _SUBTASK).first()
        assert found is not None
        return found.turn_id

    def user_message() -> ProviderResponse:
        return calls(tool("subtask", action="message", subtask=started(), message="also add 7"))

    def schedule_message() -> ProviderResponse:
        return calls(tool("subtask", action="message", subtask=started(), message="also add 11"))

    provider = ChannelProvider(
        chat=[
            calls(tool("find_tools", query=["subtask"])),
            _START,
            user_message,
            text("Told it."),
            schedule_message,
            calls(tool("subtask", action="message")),
            text("Status: the primes are listed."),
        ],
        subtask=[held],
    )

    with listening() as listener, patched(provider):
        schedule = MessageProcessor.process(ScheduledConfig(), "write a status update")
        wait(held.entered, "the subtask's provider call")
        begin(UserConfig(), "tell the schedule's subtask to also add 7").result()
        held.gate.set()
        schedule.result()

    assert held.opened
    assert provider.unused() == {}

    # What each turn was sent, read off its own requests.
    before_discovery, *sent = provider.definitions("chat", "subtask", "write a status update")
    assert before_discovery is None
    assert len(sent) == 4
    for description, actions, params in map(as_sent, sent):
        assert actions == ["start"]
        assert params == {"action", "instructions", "act_summary"}
        assert "waits" in description
        assert "Reply from" not in description
    for description, actions, params in map(
        as_sent, provider.definitions("chat", "subtask", "tell the schedule's subtask"),
    ):
        assert actions == ["start", "message"]
        assert {"subtask", "message"} <= params
        assert "Reply from subtask #N" in description

    start, user_msg, named, bare = calls_of(db, "subtask")
    call_id = cast("int", start["id"])
    n = child_turn(db, _SUBTASK, call_id)
    assert start["state"] == ToolCall.DONE, start["result"]
    assert start["result"].split("\n")[1:-1] == [_ANSWER]
    assert _ANSWER in provider.texts("chat")[-3]
    assert replies(db, call_id, _SCHEDULE) == []
    assert replies(db, call_id) == []

    executions = db.execute(
        "SELECT state, ended_at FROM turn_executions WHERE channel = ? AND turn_id = ?",
        (_SCHEDULE, schedule.turn_id),
    ).fetchall()
    assert [r["state"] for r in executions] == [TurnExecution.COMPLETED]
    ended = [
        f["state"] for f in listener.frames
        if "cancel_requested" in f and f.get("channel") == _SCHEDULE and f.get("turn_id") == schedule.turn_id
        and f["state"] != TurnExecution.WORKING
    ]
    assert ended == [TurnExecution.COMPLETED]
    subtask_ended = execution(_SUBTASK, n).ended_at
    assert subtask_ended is not None and subtask_ended <= executions[0]["ended_at"]

    assert user_msg["state"] == ToolCall.ERROR, user_msg["result"]
    assert "code=subtask-not-found" in user_msg["result"]
    assert named["state"] == ToolCall.ERROR, named["result"]
    assert named["result"].split("\n")[:3] == [
        "[subtask(status=error, code=unknown-params)]",
        "Unknown parameter(s): subtask, message. Nothing was done.",
        "valid: action | instructions",
    ]
    assert bare["state"] == ToolCall.ERROR, bare["result"]
    for refused in (user_msg, named, bare):
        assert "will arrive" not in refused["result"]
        assert "Reply from" not in refused["result"]
        reached = db.execute(
            "SELECT COUNT(*) FROM transcript WHERE channel = ? AND tool_call_id = ?",
            (_SUBTASK, str(refused["id"])),
        ).fetchone()[0]
        assert reached == 0
    assert len(provider.texts("subtask")) == 1
    assert replies(db, cast("int", named["id"]), _SCHEDULE) == []
