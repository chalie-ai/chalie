"""Feature tests: the person's chat lists, checks, messages and stops the
subagents it spawned — and only those.

The four tools — ``list_agents``, ``agent_status``, ``message_subagent`` and
``stop_subagent`` — are found with ``find_tools``, never pinned, and never ask:
a ``deny`` stored for any of them is not read.

* While a ``web_search`` the chat spawned is working, the chat lists it (its
  number, type, summary and status), reads the last thing it wrote, and sends
  it a message. The message joins the subagent's own turn as a message from
  Chalie, carrying the call that spawned it, so the subagent reads it before it
  answers and its one reply goes back to that call. Once it is done, it is no
  longer listed and a message gets its final response, while a stop still ends
  it cancelled and tells the panel.
* A message to a working subtask joins its turn the same way: a message from
  Chalie, carrying the call that started it, read before it answers, with its
  one reply going back to that call.
* Stopping a subtask stops it and the search it was waiting on, and nothing
  comes back to the chat — it asked for the stop. A second stop is told it
  stopped again, with nothing written; a message reads it as stopped.
* A subagent a scheduled task spawned is invisible to the chat: it is not
  listed, and the other three answer as for a number no subagent has. It runs
  on untouched.
* A scheduled task waits for its subagents, so it manages none — not even the
  one it spawned.
* Nor does any other caller reach the chat's subagents, even by a real number:
  a scheduled task, an outside agent and a subtask the chat spawned each list
  none, and are told no subagent has the number of a search the chat spawned.
  The search works on, with nothing joined to it, and answers the chat.
* A subagent a restart cut off, which the next boot's sweep ends failed, reads
  as stopped to a message and its status reads failed; a stop then ends it
  cancelled and tells the panel.
* Two searches the chat spawns each run on a channel of their own, numbered by
  the search, each on turn 1 there. A status read and a message each reach the
  search whose number they carry, and each reply comes back labelled with its
  own search's number. Stopping one of the two while both work ends only that
  one; the other answers on.

Drives real turns against the real, fully migrated SQLite DB, observed through
the real socket registry and read back through the real Flask app. The only
substitution is the LLM network boundary
(``services.provider_service.build_client``), answered from one script per
channel, every script used up by the end of a test. A subagent's step is held
on an event while the chat acts on it, never timed with a sleep.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from typing import cast

import pytest
from flask.testing import FlaskClient

from abilities.agent_status import AgentStatusAbility
from abilities.find_tools import FindToolsAbility
from abilities.list_agents import ListAgentsAbility
from abilities.message_subagent import MessageSubagentAbility
from abilities.recall import Recall
from abilities.stop_subagent import StopSubagentAbility
from abilities.subtask import SubtaskAbility
from abilities.web_search import WebSearchAbility
from configs.channels import config_for
from configs.channels.scheduled import ScheduledConfig
from configs.channels.user import UserConfig
from configs.enums.channels import Channel
from configs.enums.config_type import ConfigTypeEnum
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from models.agent import Agent
from models.policy import Policy
from models.provider_response import ProviderResponse
from models.tool_call import ToolCall
from models.turn_execution import TurnExecution
from run import _reply_to_swept_delegates
from services.turn_execution_service import STOP_REASON_PARENT_STOPPED, STOP_REASON_PROCESS_DEATH
from tests._delegates import (
    ChannelProvider,
    Entry,
    Held,
    agent_number,
    begin,
    block,
    calls,
    calls_of,
    child,
    delegate_channel,
    dies,
    drain,
    execution,
    listening,
    messages,
    patched,
    process_deaths_expected,
    replies,
    run_turn,
    text,
    tool,
    wait,
)
from tests.test_delegate_late_reply import _held_threads_end
from tests.test_delegate_waits import _blocked, _talk

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider", "earlier_agents")]

_SCHEDULE_ID = 7
_USER = Channel.USER.value
_SCHEDULE = Channel.SCHEDULE.value
_SUBTASK = Channel.DELEGATE_SUBTASK.value
_WEB = WebSearchAbility.NAME
_WEB_CHANNEL = delegate_channel(_WEB)
_TOOLS = (ListAgentsAbility.NAME, AgentStatusAbility.NAME, MessageSubagentAbility.NAME, StopSubagentAbility.NAME)

_TASK = "find this week's news about the river bridge"
_SUMMARY = "Searching the bridge news"
_FIRST_WORDS = "Checking memory first."
_DRAFT = "The bridge reopened."
_MESSAGE = "only this year"
_ANSWER = "This year: the bridge reopened in March."
_FERRY_TASK = "find this week's news about the harbour ferry"
_FERRY_SUMMARY = "Searching the ferry news"
_FERRY_ANSWER = "This week: the ferry runs every hour."
_NOTHING_REACHED = "No subagent has that number."
_NONE_RUNNING = "No agents are currently running"
_SPAWN_AGAIN = "If you need to perform follow-ups you can spawn a new agent"
_STOPPED = f"This agent was stopped before it finished. {_SPAWN_AGAIN}"

_HEAD = re.compile(r"^\[\w+\(status=(?P<status>\w+)(?:, code=(?P<code>[\w-]+))?")


def _answer(call: sqlite3.Row) -> tuple[str, str | None, str]:
    """What a call returned to the model: its status, its error code (``None``
    on success) and the body line under the envelope's head."""
    lines = cast("str", call["result"]).split("\n")
    head = _HEAD.match(lines[0])
    assert head is not None, call["result"]
    return head["status"], head["code"], lines[1]


def _not_found(n: int) -> tuple[str, str, str]:
    return "error", "agent-not-found", f"No subagent with agent_id {n}."


def _finished(answer: str) -> tuple[str, str, str]:
    return "error", "agent-finished", f"This agent has finished working. The final response was; {answer}. {_SPAWN_AGAIN}"


def _deny_agent_tools() -> None:
    """A ``deny`` on every channel for each of the four tools; none of them is
    gated, so a gate that read them would refuse at once."""
    for channel in PolicyChannel:
        for permission in _TOOLS:
            assert Policy.upsert(channel.value, permission, "deny") == 1


def _find_agent_tools() -> ProviderResponse:
    return calls(tool(FindToolsAbility.NAME, query=list(_TOOLS)))


def _all_four(n: int) -> ProviderResponse:
    return calls(
        tool(ListAgentsAbility.NAME),
        tool(AgentStatusAbility.NAME, agent_id=n),
        tool(MessageSubagentAbility.NAME, agent_id=n, message=_MESSAGE),
        tool(StopSubagentAbility.NAME, agent_id=n),
    )


def _input_rows(db: sqlite3.Connection, channel: str, turn_id: int) -> list[tuple[str, bool, str | None]]:
    """Every input row of one turn, oldest first — every row the model did
    not write: its text, whether it joined the turn while it worked, and the
    call it carries."""
    return [
        (r["content"], bool(r["joined"]), r["tool_call_id"]) for r in db.execute(
            "SELECT content, joined, tool_call_id FROM transcript "
            "WHERE channel = ? AND turn_id = ? AND role != 'assistant' ORDER BY id",
            (channel, turn_id),
        )
    ]


def _statuses(frames: list[dict[str, object]], n: int) -> list[str]:
    """The status each ``agent_changed`` frame for agent ``n`` carried, in order."""
    return [
        cast("dict[str, str]", f["agent"])["status"] for f in frames
        if f.get("type") == "agent_changed" and cast("dict[str, int]", f["agent"])["agent_id"] == n
    ]


def _executions(channel: str, turn_id: int) -> int:
    return len(TurnExecution.filter("channel", channel).filter("turn_id", turn_id).get())


def test_the_chat_lists_checks_and_messages_its_working_subagent_then_reads_how_it_ended(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """The chat spawns a ``web_search`` that writes a line, calls a tool and is
    held on its next step. The chat finds the four tools, lists the search,
    reads its status and messages it, then ends its turn. Released, the search
    reads the message before it answers, and its one reply opens a turn on the
    chat, which lists its subagents again and finds none, reads the finished
    search's status, messages it and stops it: the message gets its final
    response, and the stop ends it cancelled and tells the panel."""
    client, db, _store = authed_client
    _deny_agent_tools()
    held = Held(text(_DRAFT))
    seen: dict[str, int] = {}

    def find_while_it_works() -> ProviderResponse:
        wait(held.entered, "the search's second step")
        return _find_agent_tools()

    def manage() -> ProviderResponse:
        [agent] = Agent.running()
        seen["n"] = agent.agent_id
        return calls(
            tool(ListAgentsAbility.NAME),
            tool(AgentStatusAbility.NAME, agent_id=agent.agent_id),
            tool(MessageSubagentAbility.NAME, agent_id=agent.agent_id, message=_MESSAGE),
        )

    def after_the_reply() -> ProviderResponse:
        return calls(
            tool(ListAgentsAbility.NAME),
            tool(AgentStatusAbility.NAME, agent_id=seen["n"]),
            tool(MessageSubagentAbility.NAME, agent_id=seen["n"], message="and last year?"),
            tool(StopSubagentAbility.NAME, agent_id=seen["n"]),
        )

    provider = ChannelProvider(
        chat=[
            calls(tool(_WEB, instructions=_TASK, act_summary=_SUMMARY)),
            find_while_it_works,
            manage,
            text("I asked it to keep to this year."),
            _find_agent_tools(),
            after_the_reply,
            text("It is done."),
        ],
        web_search=[
            ProviderResponse(text=_FIRST_WORDS, model="m", tool_calls=[tool(Recall.NAME, query="river bridge")]),
            held,
            text(_ANSWER),
        ],
    )

    with listening() as listener, patched(provider):
        chat = begin(UserConfig(), "keep an eye on the bridge news")
        chat.result()
        held.gate.set()

    assert held.opened and provider.unused() == {}
    assert [f for f in listener.frames if f.get("type") == "permission_request"] == []
    assert _blocked(db) == []
    # Found, never pinned: the chat's first step was not sent them.
    assert not set(_TOOLS) & provider.offered("chat")[0]

    [spawn] = calls_of(db, _WEB)
    spawn_id = str(spawn["id"])
    search, n_turn = child(db, _WEB_CHANNEL, cast("int", spawn["id"]))
    n = agent_number(search)
    assert (n, n_turn) == (seen["n"], 1)

    listed, listed_done = calls_of(db, ListAgentsAbility.NAME)
    status, code, body = _answer(listed)
    assert (status, code) == ("success", None)
    assert json.loads(body) == [{"agent_id": n, "type": _WEB, "summary": _SUMMARY, "status": Agent.RUNNING}]
    # Done, it drops out of the same chat's list.
    assert _answer(listed_done)[:2] == ("success", None)
    assert _answer(listed_done)[2] == _NONE_RUNNING

    working, done = calls_of(db, AgentStatusAbility.NAME)
    assert _answer(working)[:2] == ("success", None)
    assert json.loads(_answer(working)[2]) == {
        "agent_id": n, "type": _WEB, "status": Agent.RUNNING, "last_output": _FIRST_WORDS,
    }
    assert json.loads(_answer(done)[2]) == {"agent_id": n, "type": _WEB, "status": Agent.DONE, "last_output": _ANSWER}

    sent, too_late = calls_of(db, MessageSubagentAbility.NAME)
    assert _answer(sent) == ("success", None, f"Message sent to subagent {_WEB} #{n}.")
    assert _answer(too_late) == _finished(_ANSWER)
    [stopped_late] = calls_of(db, StopSubagentAbility.NAME)
    assert _answer(stopped_late) == ("success", None, f"Subagent {_WEB} #{n} stopped.")

    # The message joined the search's own turn, carrying the call that spawned
    # it; the late ones reached nothing.
    assert _input_rows(db, search, n_turn) == [(_TASK, False, spawn_id), (_MESSAGE, True, spawn_id)]
    assert _executions(search, n_turn) == 1
    assert execution(search, n_turn).state == TurnExecution.COMPLETED
    # The stop ended the done search cancelled, and the panel was told.
    assert cast("Agent", Agent.by_id(n)).status == Agent.CANCELLED
    assert _statuses(listener.frames, n) == [Agent.RUNNING, Agent.DONE, Agent.CANCELLED]

    # The search read it as Chalie's message on the step after the held one.
    read = provider.read(_WEB)
    assert len(read) == 3
    assert _MESSAGE not in read[1]
    assert f"Message from Chalie: {_MESSAGE}" in read[2]
    thread = messages(block(client, f"/api/threads/{n_turn}?channel={search}"))
    assert [(m["content"], m["speaker"]) for m in thread if m["role"] == "user"] == [
        (_TASK, None), (_MESSAGE, "Message from Chalie"),
    ]

    # One reply, to the call that spawned it; none to the message's call.
    [reply] = replies(db, cast("int", spawn["id"]))
    assert reply["content"] == _ANSWER
    assert replies(db, cast("int", sent["id"])) == []
    assert any(f"Reply from {_WEB} #{n}" in r for r in provider.read("chat"))


def test_a_message_to_a_working_subtask_joins_its_turn_as_chalies_and_its_one_reply_goes_to_the_start(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """The chat starts a subtask that is held on its first step, messages it
    and ends its turn. The message joins the subtask's own turn as a message
    from Chalie, carrying the call that started it; released, the subtask
    reads it before it answers, and its one reply goes back to that call."""
    client, db, _store = authed_client
    _deny_agent_tools()
    held = Held(text(_DRAFT))

    def find_while_it_works() -> ProviderResponse:
        wait(held.entered, "the subtask's first step")
        return _find_agent_tools()

    def message_it() -> ProviderResponse:
        [agent] = Agent.running()
        return calls(tool(MessageSubagentAbility.NAME, agent_id=agent.agent_id, message=_MESSAGE))

    provider = ChannelProvider(
        chat=[
            calls(tool(SubtaskAbility.NAME, action="start", instructions=_TASK, act_summary=_SUMMARY)),
            find_while_it_works,
            message_it,
            text("I asked it to keep to this year."),
            text("It answered."),
        ],
        subtask=[held, text(_ANSWER)],
    )

    with listening() as listener, patched(provider):
        begin(UserConfig(), "look into the bridge news in the background").result()
        held.gate.set()

    assert held.opened and provider.unused() == {}
    assert [f for f in listener.frames if f.get("type") == "permission_request"] == []
    assert _blocked(db) == []

    [start] = calls_of(db, SubtaskAbility.NAME)
    start_id = str(start["id"])
    subtask, turn = child(db, _SUBTASK, cast("int", start["id"]))
    n = agent_number(subtask)
    [sent] = calls_of(db, MessageSubagentAbility.NAME)
    assert _answer(sent) == ("success", None, f"Message sent to subagent {SubtaskAbility.NAME} #{n}.")

    assert _input_rows(db, subtask, turn) == [(_TASK, False, start_id), (_MESSAGE, True, start_id)]
    assert _executions(subtask, turn) == 1
    assert execution(subtask, turn).state == TurnExecution.COMPLETED
    assert cast("Agent", Agent.by_id(n)).status == Agent.DONE
    read = provider.read("subtask")
    assert len(read) == 2
    assert _MESSAGE not in read[0]
    assert f"Message from Chalie: {_MESSAGE}" in read[1]
    thread = messages(block(client, f"/api/threads/{turn}?channel={subtask}"))
    assert [(m["content"], m["speaker"]) for m in thread if m["role"] == "user"] == [
        (_TASK, "Task from Chalie"), (_MESSAGE, "Message from Chalie"),
    ]

    [reply] = replies(db, cast("int", start["id"]))
    assert reply["content"] == _ANSWER
    assert replies(db, cast("int", sent["id"])) == []


def test_stopping_a_subtask_stops_it_and_its_own_search_and_sends_nothing_back(db: sqlite3.Connection) -> None:
    """The chat spawns a subtask, which spawns a ``web_search`` and waits on
    it. While the search is held, the chat stops the subtask: it is told the
    subtask stopped, the subtask and its search both end stopped by their
    caller, and no reply comes back for either. A second stop is told it
    stopped, with nothing written and no frame sent; a message finds it stopped
    and a status read cancelled."""
    _deny_agent_tools()
    held = Held(text(_DRAFT))
    seen: dict[str, int] = {}

    def stop_it() -> ProviderResponse:
        wait(held.entered, "the subtask's search")
        [subtask] = [a for a in Agent.running() if a.type == SubtaskAbility.NAME]
        seen["n"] = subtask.agent_id
        return calls(tool(StopSubagentAbility.NAME, agent_id=subtask.agent_id))

    def again() -> ProviderResponse:
        return calls(
            tool(StopSubagentAbility.NAME, agent_id=seen["n"]),
            tool(MessageSubagentAbility.NAME, agent_id=seen["n"], message=_MESSAGE),
            tool(AgentStatusAbility.NAME, agent_id=seen["n"]),
        )

    provider = ChannelProvider(
        chat=[
            calls(tool(SubtaskAbility.NAME, action="start", instructions=_TASK, act_summary=_SUMMARY)),
            _find_agent_tools(),
            stop_it,
            again,
            text("Stopped it."),
        ],
        subtask=[calls(tool(FindToolsAbility.NAME, query=[_WEB])), calls(tool(_WEB, instructions=_TASK))],
        web_search=[held],
    )

    with listening() as listener, patched(provider):
        chat = begin(UserConfig(), "look into the bridge news in the background")
        chat.result()
        held.gate.set()

    assert held.opened and provider.unused() == {}
    assert [f for f in listener.frames if f.get("type") == "permission_request"] == []
    assert _blocked(db) == []
    n = seen["n"]

    [start] = calls_of(db, SubtaskAbility.NAME)
    subtask, subtask_turn = child(db, _SUBTASK, cast("int", start["id"]))
    assert agent_number(subtask) == n
    [search] = calls_of(db, _WEB)
    searcher, search_turn = child(db, _WEB_CHANNEL, cast("int", search["id"]))

    stopped, stopped_again = calls_of(db, StopSubagentAbility.NAME)
    assert _answer(stopped) == ("success", None, f"Subagent {SubtaskAbility.NAME} #{n} stopped.")
    assert _answer(stopped_again) == ("success", None, f"Subagent {SubtaskAbility.NAME} #{n} stopped.")
    [message] = calls_of(db, MessageSubagentAbility.NAME)
    assert _answer(message) == ("error", "agent-stopped", _STOPPED)
    [status] = calls_of(db, AgentStatusAbility.NAME)
    assert json.loads(_answer(status)[2]) == {
        "agent_id": n, "type": SubtaskAbility.NAME, "status": Agent.CANCELLED, "last_output": None,
    }
    assert _statuses(listener.frames, n) == [Agent.RUNNING, Agent.CANCELLED]

    for channel, turn_id in ((subtask, subtask_turn), (searcher, search_turn)):
        ended = execution(channel, turn_id)
        assert (ended.state, ended.stop_reason) == (TurnExecution.CANCELLED, STOP_REASON_PARENT_STOPPED)
        assert cast("Agent", Agent.by_id(agent_number(channel))).status == Agent.CANCELLED
    # The message never reached the stopped subtask.
    assert _input_rows(db, subtask, subtask_turn) == [(_TASK, False, str(start["id"]))]
    # Nothing came back: not to the chat, not to the subtask.
    assert replies(db, cast("int", start["id"])) == []
    assert replies(db, cast("int", search["id"]), subtask) == []
    assert execution(_USER, chat.turn_id).state == TurnExecution.COMPLETED


def test_the_chat_cannot_see_or_reach_a_subagent_a_scheduled_task_spawned(db: sqlite3.Connection) -> None:
    """A scheduled task spawns a ``web_search`` and waits on it. Meanwhile the
    chat lists its subagents and finds none, and checks, messages and stops the
    schedule's search by its number: each is told no subagent has it. The
    search runs on untouched — no message reaches it, it is not stopped — and
    the schedule reads its answer."""
    _deny_agent_tools()
    held = Held(text(_ANSWER))
    seen: dict[str, int] = {}

    def reach_it() -> ProviderResponse:
        [agent] = Agent.running()
        seen["n"] = agent.agent_id
        return _all_four(agent.agent_id)

    # The schedule and the chat share the chat-shaped route, in this order.
    provider = ChannelProvider(
        chat=[
            calls(tool(FindToolsAbility.NAME, query=[_WEB])),
            calls(tool(_WEB, instructions=_TASK)),
            _find_agent_tools(),
            reach_it,
            text("You have no subagents running."),
            text("Status: the bridge reopened."),
        ],
        web_search=[held],
    )

    with listening() as listener, patched(provider):
        schedule = begin(ScheduledConfig(), "check the bridge news", turn_id=_SCHEDULE_ID)
        wait(held.entered, "the scheduled task's search")
        chat = begin(UserConfig(), "what are my subagents doing?")
        chat.result()
        held.gate.set()
        schedule.result()

    assert held.opened and provider.unused() == {}
    assert [f for f in listener.frames if f.get("type") == "permission_request"] == []
    assert _blocked(db) == []
    n = seen["n"]

    [listed] = calls_of(db, ListAgentsAbility.NAME)
    assert _answer(listed)[:2] == ("success", None)
    assert _answer(listed)[2] == _NONE_RUNNING
    for tool_name in (AgentStatusAbility.NAME, MessageSubagentAbility.NAME, StopSubagentAbility.NAME):
        [call] = calls_of(db, tool_name)
        assert _answer(call) == _not_found(n)

    [search] = calls_of(db, _WEB)
    assert search["state"] == ToolCall.DONE, search["result"]
    searcher, search_turn = child(db, _WEB_CHANNEL, cast("int", search["id"]))
    agent = cast("Agent", Agent.by_id(n))
    assert agent_number(searcher) == n
    assert (agent.spawned_by_channel, agent.turn_id, agent.status) == (_SCHEDULE, search_turn, Agent.DONE)
    assert _input_rows(db, searcher, search_turn) == [(_TASK, False, str(search["id"]))]
    assert execution(searcher, search_turn).state == TurnExecution.COMPLETED
    assert _ANSWER in provider.texts("chat")[-1]
    assert execution(_USER, chat.turn_id).state == TurnExecution.COMPLETED
    assert replies(db, cast("int", search["id"])) == []


def test_a_scheduled_task_manages_no_subagent_not_even_its_own(db: sqlite3.Connection) -> None:
    """A scheduled task spawns a ``web_search``, waits for its answer, then
    lists its subagents and checks, messages and stops that search by its
    number: it lists none, and each of the other three is told no subagent has
    that number."""
    _deny_agent_tools()

    def reach_it() -> ProviderResponse:
        newest = Agent.filter("type", _WEB).order_by("agent_id DESC").first()
        assert newest is not None and newest.spawned_by_channel == _SCHEDULE
        return _all_four(newest.agent_id)

    provider = ChannelProvider(
        chat=[
            calls(tool(FindToolsAbility.NAME, query=[_WEB, *_TOOLS])),
            calls(tool(_WEB, instructions=_TASK)),
            reach_it,
            text("Status: the bridge reopened."),
        ],
        web_search=[text(_ANSWER)],
    )

    _mp, frames = run_turn(provider, ScheduledConfig(), "check the bridge news", turn_id=_SCHEDULE_ID)

    assert provider.unused() == {}
    assert [f for f in frames if f.get("type") == "permission_request"] == []
    assert _blocked(db) == []
    [search] = calls_of(db, _WEB)
    assert search["state"] == ToolCall.DONE, search["result"]
    n = agent_number(child(db, _WEB_CHANNEL, cast("int", search["id"])).channel)

    [listed] = calls_of(db, ListAgentsAbility.NAME)
    assert _answer(listed)[2] == _NONE_RUNNING
    for tool_name in (AgentStatusAbility.NAME, MessageSubagentAbility.NAME, StopSubagentAbility.NAME):
        [call] = calls_of(db, tool_name)
        assert _answer(call) == _not_found(n)


@pytest.mark.parametrize("caller", ("schedule", "outside agent", "subtask"))
def test_no_caller_but_the_chat_reaches_the_chats_working_subagent_even_by_its_real_number(
    db: sqlite3.Connection, caller: str,
) -> None:
    """While a ``web_search`` the chat spawned is working, a scheduled task, an
    outside agent, or a subtask the chat spawned alongside it is handed the
    search's real number. It finds the four tools and lists none, and
    checking, messaging and stopping the search by that number each answer as
    for a number no subagent has. The search is still working once that
    caller is done, with nothing joined to its turn; released, it answers the
    chat."""
    _deny_agent_tools()
    held = Held(text(_ANSWER))
    reached = threading.Event()
    seen: dict[str, int] = {}

    def find_while_it_works() -> ProviderResponse:
        wait(held.entered, "the chat's search")
        return _find_agent_tools()

    def reach_it() -> ProviderResponse:
        [agent] = [a for a in Agent.running() if a.type == _WEB]
        seen["n"] = agent.agent_id
        return _all_four(agent.agent_id)

    def done() -> ProviderResponse:
        reached.set()
        return text(_NOTHING_REACHED)

    reach: list[Entry] = [find_while_it_works, reach_it, done]
    if caller == "subtask":
        # The chat reads the subtask's reply, then the search's.
        spawn = calls(
            tool(_WEB, instructions=_TASK),
            tool(SubtaskAbility.NAME, action="start", instructions="check on the bridge search"),
        )
        provider = ChannelProvider(
            chat=[spawn, text("Started both."), text("Noted."), text("Noted.")], subtask=reach, web_search=[held],
        )
    else:
        # The caller shares the chat-shaped route, between the chat's own turn
        # and the search's reply.
        provider = ChannelProvider(
            chat=[calls(tool(_WEB, instructions=_TASK)), text("Started."), *reach, text("Noted.")],
            web_search=[held],
        )

    with listening() as listener, patched(provider):
        begin(UserConfig(), "keep an eye on the bridge news").result()
        if caller == "schedule":
            begin(ScheduledConfig(), "check on the bridge search", turn_id=_SCHEDULE_ID).result()
        elif caller == "outside agent":
            assert _talk("check on the bridge search") == [_NOTHING_REACHED]
        wait(reached, f"the {caller}'s last step")
        assert cast("Agent", Agent.by_id(seen["n"])).status == Agent.RUNNING
        held.gate.set()

    assert held.opened and provider.unused() == {}
    assert [f for f in listener.frames if f.get("type") == "permission_request"] == []
    assert _blocked(db) == []
    n = seen["n"]

    [listed] = calls_of(db, ListAgentsAbility.NAME)
    assert _answer(listed)[:2] == ("success", None)
    assert _answer(listed)[2] == _NONE_RUNNING
    for tool_name in (AgentStatusAbility.NAME, MessageSubagentAbility.NAME, StopSubagentAbility.NAME):
        [call] = calls_of(db, tool_name)
        assert _answer(call) == _not_found(n)

    [search] = calls_of(db, _WEB)
    searcher, search_turn = child(db, _WEB_CHANNEL, cast("int", search["id"]))
    agent = cast("Agent", Agent.by_id(n))
    assert agent_number(searcher) == n
    assert (agent.spawned_by_channel, agent.turn_id, agent.status) == (_USER, search_turn, Agent.DONE)
    assert _input_rows(db, searcher, search_turn) == [(_TASK, False, str(search["id"]))]
    assert _executions(searcher, search_turn) == 1
    assert execution(searcher, search_turn).state == TurnExecution.COMPLETED
    [reply] = replies(db, cast("int", search["id"]))
    assert reply["content"] == _ANSWER


def test_a_subagent_a_restart_cut_off_reads_as_failed_and_a_stop_ends_it_cancelled(db: sqlite3.Connection) -> None:
    """The process dies while a ``web_search`` the chat spawned is mid-call, so
    its agent is left running. The next boot's sweep ends it failed and tells
    the chat it was cut off. The chat then messages, checks and stops it by its
    number: the message finds it stopped and nothing joins its turn, its status
    reads failed, and the stop ends it cancelled and tells the panel."""
    _deny_agent_tools()
    seen: dict[str, int] = {}

    def reach_it() -> ProviderResponse:
        return calls(
            tool(MessageSubagentAbility.NAME, agent_id=seen["n"], message=_MESSAGE),
            tool(AgentStatusAbility.NAME, agent_id=seen["n"]),
            tool(StopSubagentAbility.NAME, agent_id=seen["n"]),
        )

    provider = ChannelProvider(
        chat=[
            calls(tool(_WEB, instructions=_TASK)),
            text("Started."),
            text("It was cut off."),
            _find_agent_tools(),
            reach_it,
            text("It had stopped."),
        ],
        web_search=[dies],
    )

    with process_deaths_expected():
        run_turn(provider, UserConfig(), "look up the bridge news")
    [search] = calls_of(db, _WEB)
    searcher, turn = child(db, _WEB_CHANNEL, cast("int", search["id"]))
    n = seen["n"] = agent_number(searcher)
    assert cast("Agent", Agent.by_id(n)).status == Agent.RUNNING

    # What the next boot runs: the sweep, then the notices to the callers.
    with patched(provider):
        swept = MessageProcessor(config_for(ConfigTypeEnum.USER)).turn_execution_service.sweep_orphaned()
        _reply_to_swept_delegates(swept)
        drain()
    assert cast("Agent", Agent.by_id(n)).status == Agent.FAILED

    _mp, frames = run_turn(provider, UserConfig(), "what happened to the bridge search?")

    assert provider.unused() == {}
    assert [f for f in frames if f.get("type") == "permission_request"] == []
    assert _blocked(db) == []
    [message] = calls_of(db, MessageSubagentAbility.NAME)
    assert _answer(message) == ("error", "agent-stopped", _STOPPED)
    [status] = calls_of(db, AgentStatusAbility.NAME)
    assert json.loads(_answer(status)[2]) == {"agent_id": n, "type": _WEB, "status": Agent.FAILED, "last_output": None}
    [stopped] = calls_of(db, StopSubagentAbility.NAME)
    assert _answer(stopped) == ("success", None, f"Subagent {_WEB} #{n} stopped.")
    assert _statuses(frames, n) == [Agent.CANCELLED]

    assert _input_rows(db, searcher, turn) == [(_TASK, False, str(search["id"]))]
    assert _executions(searcher, turn) == 1
    assert execution(searcher, turn).stop_reason == STOP_REASON_PROCESS_DEATH
    assert cast("Agent", Agent.by_id(n)).status == Agent.CANCELLED


def _two_searches(first: Held, second: Held, then: Entry) -> list[Entry]:
    """The chat's first steps: it spawns a ``web_search``, then — once that one
    is held — a second, then — once the second is held — finds the four
    tools and takes the step *then*."""
    def spawn_the_second() -> ProviderResponse:
        wait(first.entered, "the first search's held step")
        return calls(tool(_WEB, instructions=_FERRY_TASK, act_summary=_FERRY_SUMMARY))

    def find_while_both_work() -> ProviderResponse:
        wait(second.entered, "the second search's held step")
        return _find_agent_tools()

    return [calls(tool(_WEB, instructions=_TASK, act_summary=_SUMMARY)), spawn_the_second, find_while_both_work, then]


def _own_channels(db: sqlite3.Connection) -> list[tuple[int, str, int]]:
    """For each ``web_search`` call, oldest first: the number of the search it
    spawned, the channel that search ran on, and its turn there."""
    out = []
    for call in calls_of(db, _WEB):
        channel, turn_id = child(db, _WEB_CHANNEL, cast("int", call["id"]))
        out.append((agent_number(channel), channel, turn_id))
    return out


def test_two_searches_the_chat_spawns_run_on_channels_of_their_own_and_each_reply_carries_its_own_number(
    db: sqlite3.Connection,
) -> None:
    """The chat spawns two ``web_search`` subagents, the second once the first
    is working. Each runs on a channel of its own, numbered by it, on turn 1
    there, and its spawn result names its own number. While both work the chat
    reads each one's status — each reads its own — and messages the first,
    which reads that message before it answers; the second never does. Both
    answers join the chat's working turn, each labelled with its own search's
    number, and are answered once."""
    _deny_agent_tools()
    first_held = Held(text(_DRAFT))
    second_held = Held(text(_FERRY_ANSWER))
    waiting = Held(text("Both are searching."))
    seen: dict[str, int] = {}

    def manage() -> ProviderResponse:
        first, second = Agent.running()
        seen.update(first=first.agent_id, second=second.agent_id)
        return calls(
            tool(AgentStatusAbility.NAME, agent_id=first.agent_id),
            tool(AgentStatusAbility.NAME, agent_id=second.agent_id),
            tool(MessageSubagentAbility.NAME, agent_id=first.agent_id, message=_MESSAGE),
        )

    provider = ChannelProvider(
        chat=[*_two_searches(first_held, second_held, manage), waiting, text("Both searches answered.")],
        web_search=[
            ProviderResponse(text=_FIRST_WORDS, model="m", tool_calls=[tool(Recall.NAME, query="river bridge")]),
            first_held,
            second_held,
            text(_ANSWER),
        ],
    )

    with listening() as listener, patched(provider):
        chat = begin(UserConfig(), "look up the bridge news and the ferry news")
        wait(waiting.entered, "the chat's step after it messaged the first search")
        first_held.gate.set()
        second_held.gate.set()
        _held_threads_end(first_held, second_held)
        waiting.gate.set()
        chat.result()

    assert first_held.opened and second_held.opened and waiting.opened and provider.unused() == {}
    assert [f for f in listener.frames if f.get("type") == "permission_request"] == []
    assert _blocked(db) == []
    (a, a_channel, a_turn), (b, b_channel, b_turn) = _own_channels(db)
    assert (a, b) == (seen["first"], seen["second"])
    assert (a_channel, a_turn, b_channel, b_turn) == (Channel.for_agent(_WEB, a), 1, Channel.for_agent(_WEB, b), 1)
    assert a_channel != b_channel

    spawned_a, spawned_b = calls_of(db, _WEB)
    for call, n in ((spawned_a, a), (spawned_b, b)):
        assert _answer(call)[2].startswith(
            f"Subagent {_WEB} #{n} has been spawned. It will reply to you once it's done as \"Reply from {_WEB} #{n}\"."
        ), call["result"]

    status_a, status_b = calls_of(db, AgentStatusAbility.NAME)
    assert json.loads(_answer(status_a)[2]) == {
        "agent_id": a, "type": _WEB, "status": Agent.RUNNING, "last_output": _FIRST_WORDS,
    }
    assert json.loads(_answer(status_b)[2]) == {"agent_id": b, "type": _WEB, "status": Agent.RUNNING, "last_output": None}

    [sent] = calls_of(db, MessageSubagentAbility.NAME)
    assert _answer(sent) == ("success", None, f"Message sent to subagent {_WEB} #{a}.")
    a_id, b_id = str(spawned_a["id"]), str(spawned_b["id"])
    assert _input_rows(db, a_channel, a_turn) == [(_TASK, False, a_id), (_MESSAGE, True, a_id)]
    assert _input_rows(db, b_channel, b_turn) == [(_FERRY_TASK, False, b_id)]
    searches = provider.read(_WEB)
    assert any(f"Message from Chalie: {_MESSAGE}" in r for r in searches if _TASK in r)
    assert not any(_MESSAGE in r for r in searches if _FERRY_TASK in r)

    for channel, turn_id, n in ((a_channel, a_turn, a), (b_channel, b_turn, b)):
        assert _executions(channel, turn_id) == 1
        assert execution(channel, turn_id).state == TurnExecution.COMPLETED
        assert _statuses(listener.frames, n) == [Agent.RUNNING, Agent.DONE]

    [reply_a] = replies(db, cast("int", spawned_a["id"]))
    [reply_b] = replies(db, cast("int", spawned_b["id"]))
    assert [(r["content"], r["turn_id"], r["joined"]) for r in (reply_a, reply_b)] == [
        (_ANSWER, chat.turn_id, 1), (_FERRY_ANSWER, chat.turn_id, 1),
    ]
    answering = provider.read("chat")[-1]
    for n, answer in ((a, _ANSWER), (b, _FERRY_ANSWER)):
        assert re.search(rf"Reply from {_WEB} #{n}[\s:]*{re.escape(answer)}", answering), answering[-1500:]


def test_stopping_one_of_two_working_searches_ends_only_that_one(db: sqlite3.Connection) -> None:
    """While two ``web_search`` subagents the chat spawned are working, each on
    turn 1 of its own channel, the chat stops the first and reads the second's
    status. Only the first ends — cancelled, stopped by its caller, with no
    reply — while the second reads as working, answers on, and its reply comes
    back labelled with its own number."""
    _deny_agent_tools()
    first_held = Held(text(_DRAFT))
    second_held = Held(text(_FERRY_ANSWER))
    seen: dict[str, int] = {}

    def stop_the_first() -> ProviderResponse:
        first, second = Agent.running()
        seen.update(first=first.agent_id, second=second.agent_id)
        return calls(
            tool(StopSubagentAbility.NAME, agent_id=first.agent_id),
            tool(AgentStatusAbility.NAME, agent_id=second.agent_id),
        )

    provider = ChannelProvider(
        chat=[*_two_searches(first_held, second_held, stop_the_first), text("Stopped the bridge search."), text("Done.")],
        web_search=[first_held, second_held],
    )

    with listening() as listener, patched(provider):
        begin(UserConfig(), "look up the bridge news and the ferry news").result()
        second_held.gate.set()
        _held_threads_end(second_held)
        first_held.gate.set()

    assert first_held.opened and second_held.opened and provider.unused() == {}
    assert [f for f in listener.frames if f.get("type") == "permission_request"] == []
    assert _blocked(db) == []
    (a, a_channel, a_turn), (b, b_channel, b_turn) = _own_channels(db)
    assert (a, b) == (seen["first"], seen["second"])
    assert (a_turn, b_turn) == (1, 1) and a_channel != b_channel

    [stopped] = calls_of(db, StopSubagentAbility.NAME)
    assert _answer(stopped) == ("success", None, f"Subagent {_WEB} #{a} stopped.")
    [status] = calls_of(db, AgentStatusAbility.NAME)
    assert json.loads(_answer(status)[2]) == {"agent_id": b, "type": _WEB, "status": Agent.RUNNING, "last_output": None}

    ended = execution(a_channel, a_turn)
    assert (ended.state, ended.stop_reason) == (TurnExecution.CANCELLED, STOP_REASON_PARENT_STOPPED)
    assert execution(b_channel, b_turn).state == TurnExecution.COMPLETED
    assert _statuses(listener.frames, a) == [Agent.RUNNING, Agent.CANCELLED]
    assert _statuses(listener.frames, b) == [Agent.RUNNING, Agent.DONE]

    spawned_a, spawned_b = calls_of(db, _WEB)
    assert replies(db, cast("int", spawned_a["id"])) == []
    [reply] = replies(db, cast("int", spawned_b["id"]))
    assert reply["content"] == _FERRY_ANSWER
    assert any(re.search(rf"Reply from {_WEB} #{b}[\s:]*{re.escape(_FERRY_ANSWER)}", r) for r in provider.read("chat"))
