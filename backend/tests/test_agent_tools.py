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
  longer listed, and a message or a stop gets its final response instead.
* A message to a working subtask joins its turn the same way: a message from
  Chalie, carrying the call that started it, read before it answers, with its
  one reply going back to that call.
* Stopping a subtask stops it and the search it was waiting on, and nothing
  comes back to the chat — it asked for the stop. A stopped subagent then
  reads as stopped.
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
  as stopped to a message or a stop, and its status reads failed.
* Once retention purges a subagent channel, its turn numbers come round again.
  A number the chat holds for a subagent it stopped then reaches only that
  stopped subagent, never the newer one working on the same turn number — and
  the newer one's own number still reaches it.

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
from services.garbage_collector_service import GarbageCollectorService
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
    child_turn,
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
_FOLLOW_UP = "and the year before"
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


def _executions(channel: str, turn_id: int) -> int:
    return len(TurnExecution.filter("channel", channel).filter("turn_id", turn_id).get())


def test_the_chat_lists_checks_and_messages_its_working_subagent_then_reads_how_it_ended(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """The chat spawns a ``web_search`` that writes a line, calls a tool and is
    held on its next step. The chat finds the four tools, lists the search,
    reads its status and messages it, then ends its turn. Released, the search
    reads the message before it answers, and its one reply opens a turn on the
    chat, which lists its subagents again and finds none, then messages and
    stops the finished search and reads its status: the message and the stop
    each get its final response."""
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
    n_turn = child_turn(db, _WEB_CHANNEL, cast("int", spawn["id"]))
    n = agent_number(_WEB, n_turn)
    assert n == seen["n"]

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
    assert _answer(stopped_late) == _finished(_ANSWER)

    # The message joined the search's own turn, carrying the call that spawned
    # it; the late ones reached nothing.
    assert _input_rows(db, _WEB_CHANNEL, n_turn) == [(_TASK, False, spawn_id), (_MESSAGE, True, spawn_id)]
    assert _executions(_WEB_CHANNEL, n_turn) == 1
    assert execution(_WEB_CHANNEL, n_turn).state == TurnExecution.COMPLETED
    assert cast("Agent", Agent.by_id(n)).status == Agent.DONE

    # The search read it as Chalie's message on the step after the held one.
    read = provider.read(_WEB)
    assert len(read) == 3
    assert _MESSAGE not in read[1]
    assert f"Message from Chalie: {_MESSAGE}" in read[2]
    thread = messages(block(client, f"/api/threads/{n_turn}?channel={_WEB_CHANNEL}"))
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
    turn = child_turn(db, _SUBTASK, cast("int", start["id"]))
    n = agent_number(SubtaskAbility.NAME, turn)
    [sent] = calls_of(db, MessageSubagentAbility.NAME)
    assert _answer(sent) == ("success", None, f"Message sent to subagent {SubtaskAbility.NAME} #{n}.")

    assert _input_rows(db, _SUBTASK, turn) == [(_TASK, False, start_id), (_MESSAGE, True, start_id)]
    assert _executions(_SUBTASK, turn) == 1
    assert execution(_SUBTASK, turn).state == TurnExecution.COMPLETED
    assert cast("Agent", Agent.by_id(n)).status == Agent.DONE
    read = provider.read("subtask")
    assert len(read) == 2
    assert _MESSAGE not in read[0]
    assert f"Message from Chalie: {_MESSAGE}" in read[1]
    thread = messages(block(client, f"/api/threads/{turn}?channel={_SUBTASK}"))
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
    caller, and no reply comes back for either. A second stop, a message and a
    status read then each find it stopped."""
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
    subtask_turn = child_turn(db, _SUBTASK, cast("int", start["id"]))
    assert agent_number(SubtaskAbility.NAME, subtask_turn) == n
    [search] = calls_of(db, _WEB)
    search_turn = child_turn(db, _WEB_CHANNEL, cast("int", search["id"]))

    stopped, stopped_again = calls_of(db, StopSubagentAbility.NAME)
    assert _answer(stopped) == ("success", None, f"Subagent {SubtaskAbility.NAME} #{n} stopped.")
    assert _answer(stopped_again) == ("error", "agent-stopped", _STOPPED)
    [message] = calls_of(db, MessageSubagentAbility.NAME)
    assert _answer(message) == ("error", "agent-stopped", _STOPPED)
    [status] = calls_of(db, AgentStatusAbility.NAME)
    assert json.loads(_answer(status)[2]) == {
        "agent_id": n, "type": SubtaskAbility.NAME, "status": Agent.CANCELLED, "last_output": None,
    }

    for channel, turn_id, tool_name in ((_SUBTASK, subtask_turn, SubtaskAbility.NAME), (_WEB_CHANNEL, search_turn, _WEB)):
        ended = execution(channel, turn_id)
        assert (ended.state, ended.stop_reason) == (TurnExecution.CANCELLED, STOP_REASON_PARENT_STOPPED)
        assert cast("Agent", Agent.by_turn(tool_name, turn_id)).status == Agent.CANCELLED
    # The message never reached the stopped subtask.
    assert _input_rows(db, _SUBTASK, subtask_turn) == [(_TASK, False, str(start["id"]))]
    # Nothing came back: not to the chat, not to the subtask.
    assert replies(db, cast("int", start["id"])) == []
    assert replies(db, cast("int", search["id"]), _SUBTASK) == []
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
    search_turn = child_turn(db, _WEB_CHANNEL, cast("int", search["id"]))
    agent = cast("Agent", Agent.by_id(n))
    assert (agent.spawned_by_channel, agent.turn_id, agent.status) == (_SCHEDULE, search_turn, Agent.DONE)
    assert _input_rows(db, _WEB_CHANNEL, search_turn) == [(_TASK, False, str(search["id"]))]
    assert execution(_WEB_CHANNEL, search_turn).state == TurnExecution.COMPLETED
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
    n = agent_number(_WEB, child_turn(db, _WEB_CHANNEL, cast("int", search["id"])))

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
    search_turn = child_turn(db, _WEB_CHANNEL, cast("int", search["id"]))
    agent = cast("Agent", Agent.by_id(n))
    assert (agent.spawned_by_channel, agent.turn_id, agent.status) == (_USER, search_turn, Agent.DONE)
    assert _input_rows(db, _WEB_CHANNEL, search_turn) == [(_TASK, False, str(search["id"]))]
    assert _executions(_WEB_CHANNEL, search_turn) == 1
    assert execution(_WEB_CHANNEL, search_turn).state == TurnExecution.COMPLETED
    [reply] = replies(db, cast("int", search["id"]))
    assert reply["content"] == _ANSWER


def test_a_stopped_subagents_number_never_reaches_the_newer_one_on_its_reused_turn(db: sqlite3.Connection) -> None:
    """The chat stops a ``web_search`` it spawned. Retention then purges the
    search channel, so the next search the chat spawns runs as the same turn
    number. While that one works, the chat messages, stops and checks the
    first by its number: each finds the first stopped, with nothing written.
    It then messages the newer search by its own number, which reads that
    message — and only that one — before it answers, and is not stopped."""
    _deny_agent_tools()
    first_held = Held(text(_DRAFT))
    second_held = Held(text(_DRAFT))
    seen: dict[str, int] = {}

    def stop_the_first() -> ProviderResponse:
        wait(first_held.entered, "the first search")
        [agent] = Agent.running()
        seen["n"] = agent.agent_id
        return calls(tool(StopSubagentAbility.NAME, agent_id=agent.agent_id))

    first = ChannelProvider(
        chat=[
            calls(tool(_WEB, instructions=_TASK)),
            _find_agent_tools(),
            stop_the_first,
            text("Stopped it."),
        ],
        web_search=[first_held],
    )
    with patched(first):
        begin(UserConfig(), "look up the bridge news").result()
        first_held.gate.set()
    assert first_held.opened and first.unused() == {}
    n = seen["n"]
    [first_call] = calls_of(db, _WEB)
    turn = child_turn(db, _WEB_CHANNEL, cast("int", first_call["id"]))
    assert cast("Agent", Agent.by_id(n)).status == Agent.CANCELLED

    db.execute("UPDATE transcript SET created_at = datetime('now', '-200 days') WHERE channel = ?", (_WEB_CHANNEL,))
    db.commit()
    GarbageCollectorService().run()
    assert db.execute("SELECT COUNT(*) FROM transcript WHERE channel = ?", (_WEB_CHANNEL,)).fetchone()[0] == 0

    task = "find last year's news about the river bridge"

    def reach_the_first() -> ProviderResponse:
        wait(second_held.entered, "the second search")
        return _find_agent_tools()

    def message_the_newer() -> ProviderResponse:
        [agent] = Agent.running()
        seen["newer"] = agent.agent_id
        return calls(tool(MessageSubagentAbility.NAME, agent_id=agent.agent_id, message=_FOLLOW_UP))

    second = ChannelProvider(
        chat=[
            calls(tool(_WEB, instructions=task)),
            reach_the_first,
            calls(
                tool(MessageSubagentAbility.NAME, agent_id=n, message=_MESSAGE),
                tool(StopSubagentAbility.NAME, agent_id=n),
                tool(AgentStatusAbility.NAME, agent_id=n),
            ),
            message_the_newer,
            text("That search was already stopped."),
            text("The new search answered."),
        ],
        web_search=[
            ProviderResponse(text=_FIRST_WORDS, model="m", tool_calls=[tool(Recall.NAME, query="river bridge")]),
            second_held,
            text(_ANSWER),
        ],
    )
    with patched(second):
        begin(UserConfig(), "look up last year's bridge news").result()
        second_held.gate.set()
    assert second_held.opened and second.unused() == {}

    second_call = calls_of(db, _WEB)[-1]
    assert child_turn(db, _WEB_CHANNEL, cast("int", second_call["id"])) == turn
    newer = cast("Agent", Agent.by_turn(_WEB, turn))
    assert newer.agent_id == seen["newer"] != n

    message, to_the_newer = calls_of(db, MessageSubagentAbility.NAME)
    assert _answer(message) == ("error", "agent-stopped", _STOPPED)
    assert _answer(to_the_newer) == ("success", None, f"Message sent to subagent {_WEB} #{newer.agent_id}.")
    [*_, stopped_again] = calls_of(db, StopSubagentAbility.NAME)
    assert _answer(stopped_again) == ("error", "agent-stopped", _STOPPED)
    [status] = calls_of(db, AgentStatusAbility.NAME)
    assert json.loads(_answer(status)[2]) == {
        "agent_id": n, "type": _WEB, "status": Agent.CANCELLED, "last_output": None,
    }

    # The newer search, on the same turn number, read only its own message and
    # was not stopped.
    second_call_id = str(second_call["id"])
    assert _input_rows(db, _WEB_CHANNEL, turn) == [(task, False, second_call_id), (_FOLLOW_UP, True, second_call_id)]
    assert execution(_WEB_CHANNEL, turn).state == TurnExecution.COMPLETED
    assert newer.status == Agent.DONE
    read = second.read(_WEB)[-1]
    assert f"Message from Chalie: {_FOLLOW_UP}" in read
    assert _MESSAGE not in read
    [reply] = replies(db, cast("int", second_call["id"]))
    assert reply["content"] == _ANSWER


def test_a_subagent_a_restart_cut_off_reads_as_stopped_and_its_status_as_failed(db: sqlite3.Connection) -> None:
    """The process dies while a ``web_search`` the chat spawned is mid-call, so
    its agent is left running. The next boot's sweep ends it failed and tells
    the chat it was cut off. The chat then messages, stops and checks it by its
    number: the message and the stop each find it stopped, nothing joins its
    turn, and its status reads failed."""
    _deny_agent_tools()
    seen: dict[str, int] = {}

    def reach_it() -> ProviderResponse:
        return calls(
            tool(MessageSubagentAbility.NAME, agent_id=seen["n"], message=_MESSAGE),
            tool(StopSubagentAbility.NAME, agent_id=seen["n"]),
            tool(AgentStatusAbility.NAME, agent_id=seen["n"]),
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
    turn = child_turn(db, _WEB_CHANNEL, cast("int", search["id"]))
    n = seen["n"] = agent_number(_WEB, turn)
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
    [stopped] = calls_of(db, StopSubagentAbility.NAME)
    assert _answer(stopped) == ("error", "agent-stopped", _STOPPED)
    [status] = calls_of(db, AgentStatusAbility.NAME)
    assert json.loads(_answer(status)[2]) == {"agent_id": n, "type": _WEB, "status": Agent.FAILED, "last_output": None}

    assert _input_rows(db, _WEB_CHANNEL, turn) == [(_TASK, False, str(search["id"]))]
    assert _executions(_WEB_CHANNEL, turn) == 1
    assert execution(_WEB_CHANNEL, turn).stop_reason == STOP_REASON_PROCESS_DEATH
    assert cast("Agent", Agent.by_id(n)).status == Agent.FAILED
