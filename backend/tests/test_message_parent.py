"""Feature tests: a subagent the person's chat spawned sends the chat a
message and waits for its reply.

* Every kind of subagent the chat spawns — a subtask, a ``web_search``, a
  ``web_browse``, a ``code_agent`` — reads who spawned it and that it may ask,
  and ``message_parent`` is offered to it. Once the chat's turn has ended, its
  message opens a turn of its own on the chat, labelled for the chat's model as
  a message from that subagent and followed by a nudge to reply, with
  ``message_subagent`` offered on that turn. The search waits. The chat's reply
  joins the search's turn; the search reads it once, as its call's result with
  a nudge not to acknowledge it, carries on, and its one final answer comes
  back to the chat as before. Once replied to, the nudge is gone. What the
  person sees — the stored row, the thread read over HTTP and every websocket
  frame — carries only the message's text and which subagent sent it.
* A message that arrives while the chat's turn is still working joins that
  turn: the chat reads it on its next step with the nudge, ``message_subagent``
  is offered from then on, and the reply ends the wait the same way.
* A message the chat sends a working subagent of its own accord is followed,
  for that subagent's model, by the nudge not to acknowledge it — on a
  ``web_search`` and on a subtask alike.
* A message the chat sent before the subagent asked is already waiting, so it
  is taken at once as the reply; the subagent carries on and the chat's next
  read of the unanswered message carries no nudge.
* Stopping the waiting subagent — by the chat's ``stop_subagent`` or by the
  person's stop — ends the wait without a reply and ends the subagent
  cancelled; the nudge is gone from the chat's next read. The chat's own stop
  sends nothing back; the person's stop tells the chat it was stopped.
* The person's stop of the chat turn the message opened stops the subagent
  waiting on it: the wait ends without a reply, the subagent ends cancelled
  with its caller, and nothing is sent back. Once the chat has replied, that
  stop leaves the subagent working.
* A ``web_search`` or a subtask a scheduled task or an outside agent spawned
  is waited for, so it holds no ``message_parent``: it reads that it should finish, is not offered
  the tool, and a call to it is refused as an unknown tool and posts nothing.
* A restart while the subagent posts its message leaves it, and the chat turn
  its message opened, for the next boot: the sweep ends both, the chat is told
  the subagent was cut off, and no nudge is left behind.

Drives real turns against the real, fully migrated SQLite DB, observed through
the real socket registry and read back through the real Flask app. The only
substitution is the LLM network boundary
(``services.provider_service.build_client``), answered from one script per
channel, every script used up by the end of a test. A step is held on an event
while another turn acts, never timed with a sleep.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import cast

import pytest
from flask.testing import FlaskClient

from abilities.code_agent import CodeAgentAbility
from abilities.find_tools import FindToolsAbility
from abilities.message_parent import PARENT_NUDGE, SUBAGENT_NUDGE, MessageParentAbility
from abilities.message_subagent import MessageSubagentAbility
from abilities.stop_subagent import StopSubagentAbility
from abilities.subtask import SubtaskAbility
from abilities.web_browse import WebBrowseAbility
from abilities.web_search import WebSearchAbility
from configs.channels import config_for
from configs.channels.scheduled import ScheduledConfig
from configs.channels.user import UserConfig
from configs.enums.channels import Channel
from configs.enums.config_type import ConfigTypeEnum
from controllers.message_processor import MessageProcessor
from models.agent import Agent
from models.provider_response import ProviderResponse
from models.tool_call import ToolCall
from models.turn_execution import TurnExecution
from run import _backfill_redesign_columns, _reply_to_swept_delegates
from services.turn_execution_service import STOP_REASON_PARENT_STOPPED, STOP_REASON_PROCESS_DEATH
from tests._delegates import (
    WAIT_S,
    ChannelProvider,
    Entry,
    Held,
    ProcessDeath,
    agent_number,
    begin,
    block,
    calls,
    calls_of,
    child_turn,
    delegate_channel,
    drain,
    execution,
    listening,
    messages,
    patched,
    process_deaths_expected,
    replies,
    text,
    tool,
    turn_threads,
    wait,
)
from tests.test_agent_tools import _answer, _executions, _input_rows
from tests.test_delegate_waits import _AGENT, _blocked, _talk

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider", "earlier_agents")]

_SCHEDULE_ID = 7
_USER = Channel.USER.value
_SCHEDULE = Channel.SCHEDULE.value
_SUBTASK = Channel.DELEGATE_SUBTASK.value
_WEB = WebSearchAbility.NAME
_WEB_CHANNEL = delegate_channel(_WEB)

_TASK = "find this week's news about the river bridge"
_SUMMARY = "Searching the bridge news"
_QUESTION = "Which bridge do you mean, the old one or the new one?"
_REPLY = "the old stone bridge"
_ANSWER = "The old stone bridge reopened in March."
_DRAFT = "The bridge reopened."
_MESSAGE = "only this year"

_ASKS = (
    f"You can ask for more context if needed by using `{MessageParentAbility.NAME}`. "
    "This will send a message to the agent that spawned you and wait for its reply. Else, finish"
)
_FINISH = " the task naturally and your final output will be sent to the agent that spawned you automatically."
_WAIT_STOPPED = "The message was sent. The wait was stopped before a reply arrived."


def _spawned(turn_id: int, *, asks: bool) -> str:
    """The task as a subagent of any kind reads it, spawned by turn *turn_id*
    — told it may ask, or only to finish."""
    return f"You have been spawned by agent {turn_id} to perform the task; {_TASK}. " + (
        _ASKS if asks else "Finish"
    ) + _FINISH


def _start(tool_name: str) -> ProviderResponse:
    """The call that spawns a *tool_name* subagent on :data:`_TASK`."""
    if tool_name == SubtaskAbility.NAME:
        return calls(tool(tool_name, action="start", instructions=_TASK))
    return calls(tool(tool_name, instructions=_TASK))


def _route(tool_name: str) -> str:
    """The script that answers a chat-spawned *tool_name* subagent."""
    return "subtask" if tool_name == SubtaskAbility.NAME else tool_name


def _ask() -> ProviderResponse:
    return calls(tool(MessageParentAbility.NAME, message=_QUESTION))


def _spawn() -> ProviderResponse:
    return calls(tool(_WEB, instructions=_TASK, act_summary=_SUMMARY))


def _reply_to_it(seen: dict[str, int]) -> ProviderResponse:
    """The chat's reply to the one subagent running, whose number it notes."""
    [agent] = Agent.running()
    seen["n"] = agent.agent_id
    return calls(tool(MessageSubagentAbility.NAME, agent_id=agent.agent_id, message=_REPLY))


def _carrying(db: sqlite3.Connection, call_id: int) -> int:
    """How many rows, on any channel, carry the call *call_id*."""
    return cast("int", db.execute(
        "SELECT COUNT(*) FROM transcript WHERE tool_call_id = ?", (str(call_id),),
    ).fetchone()[0])


def _the_one(reads: list[str], said: str) -> int:
    """The index of the one read that contains *said*."""
    found = [i for i, read in enumerate(reads) if said in read]
    assert len(found) == 1, f"{len(found)} reads contain {said!r}"
    return found[0]


def test_a_subagent_asks_the_chat_waits_for_its_reply_and_finishes_with_it(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """The chat spawns a ``web_search`` and ends its turn. The search asks a
    question, which opens a turn on the chat labelled as its message and
    nudging the chat to reply; the chat replies with ``message_subagent``,
    offered on that turn. The search reads the reply once, with the nudge not
    to acknowledge it, and answers; the answer comes back to the chat."""
    client, db, _store = authed_client
    asking = Held(_ask())
    seen: dict[str, int] = {}
    provider = ChannelProvider(
        chat=[
            _spawn(),
            text("Started."),
            lambda: _reply_to_it(seen),
            # The message's turn after the reply, and the answer's own turn,
            # in whichever order they ask.
            text("Noted."),
            text("Noted."),
        ],
        web_search=[asking, text(_ANSWER)],
    )

    with listening() as listener, patched(provider):
        chat = begin(UserConfig(), "find the bridge news")
        chat.result()
        asking.gate.set()

    assert asking.opened and provider.unused() == {}
    assert [f for f in listener.frames if f.get("type") == "permission_request"] == []
    assert _blocked(db) == []
    n = seen["n"]
    [spawn] = calls_of(db, _WEB)
    spawn_id = cast("int", spawn["id"])
    turn = child_turn(db, _WEB_CHANNEL, spawn_id)
    assert agent_number(_WEB, turn) == n
    [asked] = calls_of(db, MessageParentAbility.NAME)
    [sent] = calls_of(db, MessageSubagentAbility.NAME)
    assert _answer(sent) == ("success", None, f"Message sent to subagent {_WEB} #{n}.")

    # The search read who spawned it and that it may ask, with the tool to.
    read = provider.read(_WEB)
    assert len(read) == 2
    assert _spawned(chat.turn_id, asks=True) in read[0]
    assert MessageParentAbility.NAME in provider.offered(_WEB)[0]
    # It read the reply once, as its call's result, with the nudge.
    assert cast("str", asked["result"]).split("\n")[1:-1] == [_REPLY, SUBAGENT_NUDGE]
    assert read[1].count(_REPLY) == 1 and read[1].count(SUBAGENT_NUDGE) == 1
    assert "Message from Chalie" not in read[1]
    assert _input_rows(db, _WEB_CHANNEL, turn) == [(_TASK, False, str(spawn_id)), (_REPLY, True, str(spawn_id))]
    assert _executions(_WEB_CHANNEL, turn) == 1
    assert execution(_WEB_CHANNEL, turn).state == TurnExecution.COMPLETED
    assert cast("Agent", Agent.by_id(n)).status == Agent.DONE

    # The message opened a turn of its own on the chat, carrying its own call.
    [message] = replies(db, cast("int", asked["id"]))
    assert (message["content"], message["joined"]) == (_QUESTION, 0)
    assert message["turn_id"] != chat.turn_id
    # The chat's model read it labelled, with the nudge, and was offered the
    # tool to reply on that turn — never on the turn that spawned it.
    chat_read = provider.read("chat")
    offered = provider.offered("chat")
    labelled = f"Message from {_WEB} #{n}: {_QUESTION}"
    opening = _the_one(chat_read, f"{labelled}\n{PARENT_NUDGE}")
    assert MessageSubagentAbility.NAME in offered[opening]
    assert MessageSubagentAbility.NAME not in offered[0] | offered[1]
    # Replied to, the label stays and the nudge is gone.
    answered = _the_one(chat_read, f"Message sent to subagent {_WEB} #{n}.")
    assert labelled in chat_read[answered]
    assert sum(PARENT_NUDGE in r for r in chat_read) == 1
    assert all(SUBAGENT_NUDGE not in r for r in chat_read)

    # What the person sees carries only its text and which subagent sent it.
    opener = messages(block(client, f"/api/threads/{message['turn_id']}"))[0]
    assert opener["content"] == _QUESTION
    assert opener["reply_from"] == {"tool_name": _WEB, "summary": _SUMMARY, "kind": "message"}
    assert opener.get("speaker") is None
    frames = json.dumps(listener.frames)
    assert PARENT_NUDGE not in frames and f"Message from {_WEB}" not in frames

    # Its one answer came back to the call that spawned it.
    assert [r["content"] for r in replies(db, spawn_id)] == [_ANSWER]
    assert any(f"Reply from {_WEB} #{n}: {_ANSWER}" in r for r in chat_read)


@pytest.mark.parametrize(
    "tool_name",
    (SubtaskAbility.NAME, WebSearchAbility.NAME, WebBrowseAbility.NAME, CodeAgentAbility.NAME),
)
def test_every_kind_of_subagent_the_chat_spawns_reads_that_it_may_message_the_chat(
    db: sqlite3.Connection, tool_name: str,
) -> None:
    """Whichever kind the chat spawns, the subagent reads who spawned it, its
    task, that it may message the chat with ``message_parent`` — which it is
    offered — and that its final output goes back on its own; the stored row
    keeps the bare task, and its answer comes back to the chat. (A ``pim``
    needs a connected mailbox, which a test connects through a vault.)"""
    route = _route(tool_name)
    scripts: dict[str, list[Entry]] = {"chat": [_start(tool_name), text("Started."), text("Noted.")], route: [text(_ANSWER)]}
    provider = ChannelProvider(**scripts)

    with listening(), patched(provider):
        chat = begin(UserConfig(), "look into the bridge news")
        chat.result()

    assert provider.unused() == {}
    [read] = provider.read(route)
    assert _spawned(chat.turn_id, asks=True) in read
    [offered] = provider.offered(route)
    assert MessageParentAbility.NAME in offered
    [spawn] = calls_of(db, tool_name)
    spawn_id = cast("int", spawn["id"])
    channel = _SUBTASK if tool_name == SubtaskAbility.NAME else delegate_channel(tool_name)
    turn = child_turn(db, channel, spawn_id)
    assert _input_rows(db, channel, turn) == [(_TASK, False, str(spawn_id))]
    assert [r["content"] for r in replies(db, spawn_id)] == [_ANSWER]


def test_a_message_that_joins_the_chats_working_turn_is_answered_in_that_turn(db: sqlite3.Connection) -> None:
    """The search asks while the chat's turn is still working, so the message
    joins that turn. The chat reads it on its next step with the nudge and is
    offered ``message_subagent`` from then on; it replies, the nudge is gone,
    and the search's answer joins the same turn."""
    seen: dict[str, int] = {}
    threads: dict[str, threading.Thread] = {}
    asked_for = threading.Event()
    posted = threading.Event()

    def spawn() -> ProviderResponse:
        threads["chat"] = threading.current_thread()
        return _spawn()

    def wait_for_the_message() -> ProviderResponse:
        asked_for.set()
        wait(posted, "the search's message to join the chat's turn")
        return text("Searching now.")

    def ask_once_the_chat_waits() -> ProviderResponse:
        threads["search"] = threading.current_thread()
        wait(asked_for, "the chat's second step")
        return _ask()

    def once_it_answered() -> ProviderResponse:
        threads["search"].join(timeout=WAIT_S)
        seen["search_alive"] = int(threads["search"].is_alive())
        return text("Told it.")

    def on_frame(frame: dict[str, object]) -> None:
        # The join's refresh signal, sent from the search's own thread.
        chat_thread = threads.get("chat")
        if (
            chat_thread is not None and threading.current_thread() is not chat_thread
            and frame.get("status") == "updated" and "channel" not in frame
        ):
            posted.set()

    provider = ChannelProvider(
        chat=[spawn, wait_for_the_message, lambda: _reply_to_it(seen), once_it_answered, text("Done.")],
        web_search=[ask_once_the_chat_waits, text(_ANSWER)],
    )

    with listening(on_frame) as listener, patched(provider):
        chat = begin(UserConfig(), "find the bridge news")
        chat.result()

    assert provider.unused() == {} and seen["search_alive"] == 0
    assert [f for f in listener.frames if f.get("type") == "permission_request"] == []
    n = seen["n"]
    [spawn_call] = calls_of(db, _WEB)
    [asked] = calls_of(db, MessageParentAbility.NAME)
    assert cast("str", asked["result"]).split("\n")[1:-1] == [_REPLY, SUBAGENT_NUDGE]
    # The message and the answer both joined the chat's one turn.
    [message] = replies(db, cast("int", asked["id"]))
    assert (message["content"], message["turn_id"], message["joined"]) == (_QUESTION, chat.turn_id, 1)
    [answer] = replies(db, cast("int", spawn_call["id"]))
    assert (answer["content"], answer["turn_id"], answer["joined"]) == (_ANSWER, chat.turn_id, 1)
    assert _executions(_USER, chat.turn_id) == 1

    chat_read = provider.read("chat")
    offered = provider.offered("chat")
    assert len(chat_read) == 5
    labelled = f"Message from {_WEB} #{n}: {_QUESTION}"
    assert labelled not in chat_read[1]
    assert f"{labelled}\n{PARENT_NUDGE}" in chat_read[2]
    assert [MessageSubagentAbility.NAME in tools for tools in offered] == [False, False, True, True, True]
    assert labelled in chat_read[3] and PARENT_NUDGE not in chat_read[3]
    assert f"Reply from {_WEB} #{n}: {_ANSWER}" in chat_read[4]
    assert sum(PARENT_NUDGE in r for r in chat_read) == 1
    # The nudge not to acknowledge is the subagent's alone.
    assert all(SUBAGENT_NUDGE not in r for r in chat_read)
    assert cast("Agent", Agent.by_id(n)).status == Agent.DONE


@pytest.mark.parametrize("tool_name", (WebSearchAbility.NAME, SubtaskAbility.NAME))
def test_a_message_the_chat_sends_a_working_subagent_tells_it_not_to_acknowledge_it(
    db: sqlite3.Connection, tool_name: str,
) -> None:
    """The chat messages a subagent working on its first step. The subagent
    reads the message on its next step followed by the nudge not to
    acknowledge it; the stored row holds only the message."""
    held = Held(text(_DRAFT))
    route = "subtask" if tool_name == SubtaskAbility.NAME else tool_name
    channel = _SUBTASK if tool_name == SubtaskAbility.NAME else delegate_channel(tool_name)
    start = (
        tool(tool_name, action="start", instructions=_TASK) if tool_name == SubtaskAbility.NAME
        else tool(tool_name, instructions=_TASK)
    )

    def find_while_it_works() -> ProviderResponse:
        wait(held.entered, f"the {tool_name}'s first step")
        return calls(tool(FindToolsAbility.NAME, query=[MessageSubagentAbility.NAME]))

    def message_it() -> ProviderResponse:
        [agent] = Agent.running()
        return calls(tool(MessageSubagentAbility.NAME, agent_id=agent.agent_id, message=_MESSAGE))

    scripts: dict[str, list[Entry]] = {
        "chat": [calls(start), find_while_it_works, message_it, text("Asked."), text("Noted.")],
        route: [held, text(_ANSWER)],
    }
    provider = ChannelProvider(**scripts)

    with listening() as listener, patched(provider):
        begin(UserConfig(), "look into the bridge news").result()
        held.gate.set()

    assert held.opened and provider.unused() == {}
    [spawn] = calls_of(db, tool_name)
    turn = child_turn(db, channel, cast("int", spawn["id"]))
    read = provider.read(route)
    assert len(read) == 2
    assert SUBAGENT_NUDGE not in read[0]
    assert f"Message from Chalie: {_MESSAGE}\n{SUBAGENT_NUDGE}" in read[1]
    assert read[1].count(SUBAGENT_NUDGE) == 1
    assert _input_rows(db, channel, turn)[1:] == [(_MESSAGE, True, str(spawn["id"]))]
    assert SUBAGENT_NUDGE not in json.dumps(listener.frames)


def test_a_message_the_chat_sent_before_the_subagent_asked_is_taken_as_its_reply(db: sqlite3.Connection) -> None:
    """The chat messages the search while it works; the search then asks. The
    chat's earlier message is already waiting on the search's turn, so it is
    taken at once as the reply and the search carries on. The chat's message
    is not answered, yet the chat's next read of it carries no nudge: the
    search is still working, but no longer waits."""
    working = Held(_ask())
    carrying_on = Held(text(_ANSWER))
    seen: dict[str, int] = {}
    threads: dict[str, threading.Thread] = {}
    read_the_message = threading.Event()

    def find_while_it_works() -> ProviderResponse:
        wait(working.entered, "the search's first step")
        return calls(tool(FindToolsAbility.NAME, query=[MessageSubagentAbility.NAME]))

    def message_it() -> ProviderResponse:
        [agent] = Agent.running()
        seen["n"] = agent.agent_id
        return calls(tool(MessageSubagentAbility.NAME, agent_id=agent.agent_id, message=_MESSAGE))

    def on_the_message() -> ProviderResponse:
        threads["message"] = threading.current_thread()
        read_the_message.set()
        return text("Okay.")

    provider = ChannelProvider(
        chat=[
            _spawn(), find_while_it_works, message_it, text("Asked."),
            on_the_message, text("Nothing else."), text("Noted."),
        ],
        web_search=[working, carrying_on],
    )

    with listening(), patched(provider):
        begin(UserConfig(), "find the bridge news").result()
        working.gate.set()
        wait(carrying_on.entered, "the search's step after it asked")
        wait(read_the_message, "the chat's turn the search's message opened")
        threads["message"].join(timeout=WAIT_S)
        assert not threads["message"].is_alive()
        begin(UserConfig(), "anything else?").result()
        carrying_on.gate.set()

    assert working.opened and carrying_on.opened and provider.unused() == {}
    n = seen["n"]
    [asked] = calls_of(db, MessageParentAbility.NAME)
    assert cast("str", asked["result"]).split("\n")[1:-1] == [_MESSAGE, SUBAGENT_NUDGE]
    read = provider.read(_WEB)
    assert read[1].count(_MESSAGE) == 1 and "Message from Chalie" not in read[1]
    chat_read = provider.read("chat")
    labelled = f"Message from {_WEB} #{n}: {_QUESTION}"
    assert labelled in chat_read[5] and PARENT_NUDGE not in chat_read[5]
    [spawn] = calls_of(db, _WEB)
    assert [r["content"] for r in replies(db, cast("int", spawn["id"]))] == [_ANSWER]
    assert cast("Agent", Agent.by_id(n)).status == Agent.DONE


def test_stopping_a_subagent_that_waits_for_a_reply_ends_its_wait_and_sends_nothing_back(
    db: sqlite3.Connection, real_memory_step: None,
) -> None:
    """The search asks; the chat's turn its message opened ends without a
    reply, so the search still waits and the chat's next turn reads the
    message, nudge and all, among its previous messages. The memory step that
    turn's settle ran while the search waited read the message but never the
    nudge, since what it reads goes into stored prose. The chat's next turn
    stops the search: the wait ends with no reply, the search ends cancelled,
    nothing is sent back, and the chat's next read carries no nudge."""
    asking = Held(_ask())
    unanswered = Held(text("Let me think about that."))
    unanswered.gate.set()
    seen: dict[str, int] = {}

    def stop_it() -> ProviderResponse:
        [agent] = Agent.running()
        seen["n"] = agent.agent_id
        return calls(tool(StopSubagentAbility.NAME, agent_id=agent.agent_id))

    provider = ChannelProvider(
        chat=[
            _spawn(),
            text("Started."),
            unanswered,
            calls(tool(FindToolsAbility.NAME, query=[StopSubagentAbility.NAME])),
            stop_it,
            text("Stopped it."),
        ],
        web_search=[asking],
    )

    with listening() as listener, patched(provider):
        chat = begin(UserConfig(), "find the bridge news")
        chat.result()
        asking.gate.set()
        wait(unanswered.entered, "the chat's turn the message opened")
        cast("threading.Thread", unanswered.thread).join(timeout=WAIT_S)
        for step in [t for t in turn_threads() if t.name.startswith("memory-step-")]:
            step.join(timeout=WAIT_S)
        begin(UserConfig(), "stop the search").result()

    assert asking.opened and provider.unused() == {}
    assert [f for f in listener.frames if f.get("type") == "permission_request"] == []
    n = seen["n"]
    [spawn] = calls_of(db, _WEB)
    turn = child_turn(db, _WEB_CHANNEL, cast("int", spawn["id"]))
    [stopped] = calls_of(db, StopSubagentAbility.NAME)
    assert _answer(stopped) == ("success", None, f"Subagent {_WEB} #{n} stopped.")
    [asked] = calls_of(db, MessageParentAbility.NAME)
    assert _answer(asked) == ("success", None, _WAIT_STOPPED)

    chat_read = provider.read("chat")
    # The chat's next turn read the unanswered message, nudge and all.
    assert f"Message from {_WEB} #{n}: {_QUESTION} {PARENT_NUDGE}" in chat_read[3]
    # Stopped, the message stays and the nudge is gone.
    assert f"Message from {_WEB} #{n}: {_QUESTION}" in chat_read[5]
    assert PARENT_NUDGE not in chat_read[5]
    remembered = provider.read("background")
    assert any(_QUESTION in r for r in remembered)
    assert all(PARENT_NUDGE not in r for r in remembered)

    assert len(provider.read(_WEB)) == 1
    assert cast("Agent", Agent.by_id(n)).status == Agent.CANCELLED
    ended = execution(_WEB_CHANNEL, turn)
    assert (ended.state, ended.stop_reason) == (TurnExecution.CANCELLED, STOP_REASON_PARENT_STOPPED)
    assert replies(db, cast("int", spawn["id"])) == []


def test_the_persons_stop_ends_the_wait_and_tells_the_chat(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """While the chat's turn the message opened is working, the person stops
    the waiting search. The wait ends with no reply, the search ends
    cancelled, and the chat is told it was stopped, with no nudge."""
    client, db, _store = authed_client
    asking = Held(_ask())
    reading = Held(text("Let me check."))
    provider = ChannelProvider(
        chat=[_spawn(), text("Started."), reading, text("Okay.")],
        web_search=[asking],
    )

    with listening(), patched(provider):
        chat = begin(UserConfig(), "find the bridge news")
        chat.result()
        asking.gate.set()
        wait(reading.entered, "the chat's turn the message opened")
        [spawn] = calls_of(db, _WEB)
        turn = child_turn(db, _WEB_CHANNEL, cast("int", spawn["id"]))
        stop = client.delete(f"/api/threads/{turn}?channel={_WEB_CHANNEL}")
        assert stop.status_code == 200 and (stop.get_json() or {})["result"]["cancelled"] is True
        cast("threading.Thread", asking.thread).join(timeout=WAIT_S)
        reading.gate.set()

    assert asking.opened and reading.opened and provider.unused() == {}
    n = agent_number(_WEB, turn)
    [asked] = calls_of(db, MessageParentAbility.NAME)
    assert _answer(asked) == ("success", None, _WAIT_STOPPED)
    assert len(provider.read(_WEB)) == 1
    assert cast("Agent", Agent.by_id(n)).status == Agent.CANCELLED
    assert execution(_WEB_CHANNEL, turn).state == TurnExecution.CANCELLED

    notice = f"This {_WEB} was stopped before it finished."
    [told] = replies(db, cast("int", spawn["id"]))
    assert told["content"] == notice
    chat_read = provider.read("chat")
    assert f"Message from {_WEB} #{n}: {_QUESTION}\n{PARENT_NUDGE}" in chat_read[2]
    assert notice in chat_read[3] and PARENT_NUDGE not in chat_read[3]


def test_the_persons_stop_of_the_chat_turn_a_message_opened_stops_the_subagent_waiting_on_it(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """The chat's turn has ended when the search asks, so its message opens a
    chat turn of its own. While that turn works, the person stops it. The stop
    reaches the search waiting on it: the wait ends with no reply, the search
    ends cancelled with its caller, and nothing is sent back."""
    client, db, _store = authed_client
    asking = Held(_ask())
    reading = Held(text("Let me check."))
    provider = ChannelProvider(
        chat=[_spawn(), text("Started."), reading],
        web_search=[asking],
    )

    with listening(), patched(provider):
        chat = begin(UserConfig(), "find the bridge news")
        chat.result()
        asking.gate.set()
        wait(reading.entered, "the chat's turn the message opened")
        [asked] = calls_of(db, MessageParentAbility.NAME)
        [message] = replies(db, cast("int", asked["id"]))
        opened = cast("int", message["turn_id"])
        stop = client.delete(f"/api/threads/{opened}")
        assert stop.status_code == 200 and (stop.get_json() or {})["result"]["cancelled"] is True
        cast("threading.Thread", asking.thread).join(timeout=WAIT_S)
        reading.gate.set()
        cast("threading.Thread", reading.thread).join(timeout=WAIT_S)

    assert asking.opened and reading.opened and provider.unused() == {}
    assert opened != chat.turn_id
    assert execution(_USER, opened).state == TurnExecution.CANCELLED
    [spawn] = calls_of(db, _WEB)
    turn = child_turn(db, _WEB_CHANNEL, cast("int", spawn["id"]))
    n = agent_number(_WEB, turn)
    [asked] = calls_of(db, MessageParentAbility.NAME)
    assert _answer(asked) == ("success", None, _WAIT_STOPPED)
    assert len(provider.read(_WEB)) == 1
    assert cast("Agent", Agent.by_id(n)).status == Agent.CANCELLED
    ended = execution(_WEB_CHANNEL, turn)
    assert (ended.state, ended.stop_reason) == (TurnExecution.CANCELLED, STOP_REASON_PARENT_STOPPED)
    assert replies(db, cast("int", spawn["id"])) == []
    assert f"Message from {_WEB} #{n}: {_QUESTION}\n{PARENT_NUDGE}" in provider.read("chat")[2]


def test_the_persons_stop_of_the_chat_turn_after_it_replied_leaves_the_subagent_working(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """The search's message opens a chat turn, and the chat replies in it.
    While that turn and the search are both still working, the person stops
    the chat turn. The search no longer waits on it, so it works on, finishes
    with the reply it read, and its answer comes back."""
    client, db, _store = authed_client
    asking = Held(_ask())
    noting = Held(text("Noted."))
    finishing = Held(text(_ANSWER))
    seen: dict[str, int] = {}
    provider = ChannelProvider(
        chat=[_spawn(), text("Started."), lambda: _reply_to_it(seen), noting, text("Noted.")],
        web_search=[asking, finishing],
    )

    with listening(), patched(provider):
        chat = begin(UserConfig(), "find the bridge news")
        chat.result()
        asking.gate.set()
        wait(noting.entered, "the chat's turn after its reply")
        wait(finishing.entered, "the search's step after the reply")
        [asked] = calls_of(db, MessageParentAbility.NAME)
        [message] = replies(db, cast("int", asked["id"]))
        opened = cast("int", message["turn_id"])
        stop = client.delete(f"/api/threads/{opened}")
        assert stop.status_code == 200 and (stop.get_json() or {})["result"]["cancelled"] is True
        n = seen["n"]
        assert cast("Agent", Agent.by_id(n)).status == Agent.RUNNING
        noting.gate.set()
        finishing.gate.set()
        cast("threading.Thread", finishing.thread).join(timeout=WAIT_S)

    assert asking.opened and noting.opened and finishing.opened and provider.unused() == {}
    assert execution(_USER, opened).state == TurnExecution.CANCELLED
    [spawn] = calls_of(db, _WEB)
    turn = child_turn(db, _WEB_CHANNEL, cast("int", spawn["id"]))
    assert agent_number(_WEB, turn) == n
    assert cast("str", asked["result"]).split("\n")[1:-1] == [_REPLY, SUBAGENT_NUDGE]
    assert cast("Agent", Agent.by_id(n)).status == Agent.DONE
    assert execution(_WEB_CHANNEL, turn).state == TurnExecution.COMPLETED
    assert [r["content"] for r in replies(db, cast("int", spawn["id"]))] == [_ANSWER]


@pytest.mark.parametrize(
    ("caller", "tool_name"),
    [
        ("schedule", WebSearchAbility.NAME),
        ("outside agent", WebSearchAbility.NAME),
        ("schedule", SubtaskAbility.NAME),
        ("outside agent", SubtaskAbility.NAME),
    ],
)
def test_a_subagent_its_caller_waits_for_holds_no_message_parent(
    db: sqlite3.Connection, caller: str, tool_name: str,
) -> None:
    """A ``web_search`` or a subtask a scheduled task or an outside agent
    spawned is waited for. It reads only that it should finish, is not offered
    ``message_parent``, and a call to it is refused as an unknown tool, posting
    nothing anywhere; its answer reaches its caller as the call's result. (An
    outside agent's subtask is answered from the chat's script: its prompt
    carries neither a subagent's own system prompt nor the subtask's label.)"""
    subagent: list[Entry] = [_ask(), text(_ANSWER)]
    shared = caller == "outside agent" and tool_name == SubtaskAbility.NAME
    route = "chat" if shared else _route(tool_name)
    scripts: dict[str, list[Entry]] = (
        {"chat": [_start(tool_name), *subagent, text("Noted.")]} if shared
        else {"chat": [_start(tool_name), text("Noted.")], route: subagent}
    )
    provider = ChannelProvider(**scripts)

    with listening() as listener, patched(provider):
        if caller == "schedule":
            begin(ScheduledConfig(), "check the bridge news", turn_id=_SCHEDULE_ID).result()
        else:
            assert _talk("check the bridge news") == ["Noted."]

    assert provider.unused() == {}
    assert [f for f in listener.frames if f.get("type") == "permission_request"] == []
    [spawn] = calls_of(db, tool_name)
    channel = _SUBTASK if tool_name == SubtaskAbility.NAME else delegate_channel(tool_name)
    turn = child_turn(db, channel, cast("int", spawn["id"]))
    agent = cast("Agent", Agent.by_turn(tool_name, turn))
    spawner = _SCHEDULE if caller == "schedule" else Channel.for_external_agent(_AGENT)
    assert agent.spawned_by_channel == spawner
    if caller == "schedule":
        assert agent.spawned_in_turn == _SCHEDULE_ID

    first = 1 if shared else 0
    assert _spawned(agent.spawned_in_turn, asks=False) in provider.read(route)[first]
    assert MessageParentAbility.NAME not in provider.offered(route)[first]
    [asked] = calls_of(db, MessageParentAbility.NAME)
    assert _answer(asked) == ("error", "unknown-tool", f"Unknown tool: {MessageParentAbility.NAME}")
    assert _carrying(db, cast("int", asked["id"])) == 0
    assert db.execute("SELECT COUNT(*) FROM turn_executions WHERE channel = ?", (_USER,)).fetchone()[0] == 0
    assert _ANSWER in cast("str", spawn["result"])
    assert agent.status == Agent.DONE


def test_a_restart_while_a_subagent_posts_its_message_leaves_no_nudge_and_tells_the_chat(
    db: sqlite3.Connection,
) -> None:
    """The process dies the moment the search's message has opened a turn on
    the chat, before that turn runs. The next boot's backfill, sweep and
    notices end the search failed and the chat's turn crashed, close the
    ``message_parent`` call, and tell the chat the search was cut off; the
    chat's read of the message carries no nudge."""
    asking = Held(_ask())
    died = threading.Event()

    def die_as_the_message_opens_a_turn(frame: dict[str, object]) -> None:
        if (
            not died.is_set() and threading.current_thread() is asking.thread
            and frame.get("state") == TurnExecution.WORKING
        ):
            died.set()
            raise ProcessDeath

    provider = ChannelProvider(
        chat=[_spawn(), text("Started."), text("It was cut off.")],
        web_search=[asking],
    )

    with process_deaths_expected(), listening(die_as_the_message_opens_a_turn), patched(provider):
        chat = begin(UserConfig(), "find the bridge news")
        chat.result()
        asking.gate.set()
        wait(asking.entered, "the search's first step")
        cast("threading.Thread", asking.thread).join(timeout=WAIT_S)

    assert died.is_set()
    [spawn] = calls_of(db, _WEB)
    turn = child_turn(db, _WEB_CHANNEL, cast("int", spawn["id"]))
    n = agent_number(_WEB, turn)
    [asked] = calls_of(db, MessageParentAbility.NAME)
    assert asked["state"] == ToolCall.STARTED
    [message] = replies(db, cast("int", asked["id"]))
    assert message["content"] == _QUESTION
    assert execution(_USER, message["turn_id"]).state == TurnExecution.WORKING

    # What the next boot runs: the backfill, the sweep, then the notices.
    with patched(provider):
        _backfill_redesign_columns()
        swept = MessageProcessor(config_for(ConfigTypeEnum.USER)).turn_execution_service.sweep_orphaned()
        _reply_to_swept_delegates(swept)
        drain()
    assert provider.unused() == {}

    [asked] = calls_of(db, MessageParentAbility.NAME)
    assert asked["state"] == ToolCall.DONE
    assert cast("Agent", Agent.by_id(n)).status == Agent.FAILED
    cut_off = execution(_WEB_CHANNEL, turn)
    assert (cut_off.state, cut_off.stop_reason) == (TurnExecution.CRASHED, STOP_REASON_PROCESS_DEATH)
    assert execution(_USER, message["turn_id"]).state == TurnExecution.CRASHED
    notice = f"This {_WEB} was cut off by a restart before it finished."
    [told] = replies(db, cast("int", spawn["id"]))
    assert told["content"] == notice
    chat_read = provider.read("chat")
    assert len(chat_read) == 3
    assert notice in chat_read[2]
    assert all(PARENT_NUDGE not in r for r in chat_read)
