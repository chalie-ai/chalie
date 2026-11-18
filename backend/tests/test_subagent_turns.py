"""Feature tests: every message to a subagent, and every reply from one, is a
turn of its own — opened at once on an idle channel, queued behind a working
one, never joined.

* A finished ``web_search`` the chat messages again runs its next turn: it
  goes running, then done, keeping the new turn's answer. That turn reads its
  own earlier turn and the message, nothing from the chat, and its answer
  comes back labelled with the same number.
* Two searches, each messaged again once, each read only their own turns.
* A search the chat stops, and one a restart cut off, each keep their turn's
  last row; a later message to either is answered with how it ended and opens
  no turn, and its status reads the same. The stop, and the boot that ends the
  cut-off one failed, each drop the message waiting for it.
* A search the person stops drops, as its turn ends, the message waiting for
  it, and the chat is told it was cancelled.
* A stop of a search lands wholly before or wholly after a message to it from
  another chat turn: a message let go while the stop acts is told the search
  was cancelled once the stop commits; a stop that looked the search up
  before its waiting message opened the next turn cancels that next turn; a
  message that looked the search up before a stop committed is told it was
  cancelled and opens nothing.
* A reply or a question waiting on the chat is still delivered when the chat
  turn that queued it, or the search that asked, is stopped: a stop drops only
  the messages sent to subagents.
* A reply to an idle chat opens a chat turn at once; to a working chat it
  waits, and opens a turn once that one ends.
* Two messages sent to an idle search in one step: the first opens its next
  turn at once, the second waits and opens the turn after. Nothing is lost.
* A subtask a scheduled task started cannot be messaged by it: the call is
  told no subagent has that number, and no turn opens.
* A file write on a subagent's turn the chat's message opened — at once or
  after waiting — asks the person on the chat; so does one on a chat turn a
  subagent's question opened.
* A chat turn stopped while its message waits for a working search drops the
  message, which keeps its text in the call that sent it; a turn the message
  already opened, minutes after its call ended, stops with it.
* A search that ended more than a week ago is archived to every tool; one
  still working never is.
* A stop of a finished subtask drops the message waiting on its memory step,
  and no turn opens for it.

Drives real turns against the real, fully migrated SQLite DB, observed through
the real socket registry and read back through the real Flask app. The only
substitution is the LLM network boundary
(``services.provider_service.build_client``), answered from one script per
channel, every script used up by the end of a test. A subagent's step is held
on an event while the chat acts on it, never timed with a sleep.
"""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
from collections.abc import Callable
from datetime import timedelta
from typing import cast

import pytest
from flask.testing import FlaskClient

from abilities._agents import ARCHIVED_TEXT, AgentAbility, label
from abilities._delegate import reply_label
from abilities._result import ToolResult
from abilities.agent_status import AgentStatusAbility
from abilities.code_agent import CodeAgentAbility
from abilities.find_tools import FindToolsAbility
from abilities.message_parent import MessageParentAbility
from abilities.message_subagent import _SENT, MessageSubagentAbility
from abilities.recall import Recall
from abilities.stop_subagent import StopSubagentAbility
from abilities.subtask import SubtaskAbility
from abilities.web_search import WebSearchAbility
from abilities.write_file import WriteFileAbility
from configs.channels import config_for
from configs.channels.scheduled import ScheduledConfig
from configs.channels.subtask import UserSubtaskConfig
from configs.channels.user import UserConfig
from configs.enums.channels import Channel
from configs.enums.config_type import ConfigTypeEnum
from controllers.message_processor import MessageProcessor
from models.agent import Agent
from models.policy import Policy
from models.provider_response import ProviderResponse
from models.queued_message import QueuedMessage
from models.tool_call import ToolCall
from models.transcript import _CHILD_ROW_LAG, Transcript
from models.turn_execution import TurnExecution
from run import _reply_to_swept_delegates
from services.agent_service import ARCHIVE_AFTER, CANCELLED_TEXT, CRASHED_TEXT, AgentService
from services.file_mapper_service import FileMapperService
from services.memory_step_service import memory_step_config
from services.policy_manager import _BLOCK
from services.time_utils import parse_utc, utc_now
from services.turn_execution_service import STOP_REASON_PARENT_STOPPED
from tests._delegates import (
    WAIT_S,
    ChannelProvider,
    Held,
    agent_number,
    assistant_texts,
    begin,
    calls,
    calls_of,
    child,
    delegate_channel,
    dies,
    drain,
    execution,
    listening,
    patched,
    process_deaths_expected,
    replies,
    run_turn,
    text,
    tool,
    turn_threads,
    wait,
)
from tests.test_agent_tools import (
    _ANSWER,
    _DRAFT,
    _FERRY_ANSWER,
    _FERRY_SUMMARY,
    _FERRY_TASK,
    _FIRST_WORDS,
    _MESSAGE,
    _SUMMARY,
    _TASK,
    _answer,
    _executions,
    _find_agent_tools,
    _input_rows,
    _not_found,
    _statuses,
)
from tests.test_agents_table import _last_row
from tests.test_delegate_late_reply import _held_threads_end
from tests.test_delegate_waits import _blocked, _reached
from tests.test_message_parent import _QUESTION, _WAITS, _ask, _sent_to, _the_one, _the_search
from tests.test_subagent_permissions import _CHAT, _FILE_WRITE, _origin
from tests.test_subtask import _memory_turns

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider", "earlier_agents")]

_SCHEDULE_ID = 7
_USER = Channel.USER.value
_SCHEDULE = Channel.SCHEDULE.value
_SUBTASK = Channel.DELEGATE_SUBTASK.value
_WEB = WebSearchAbility.NAME
_WEB_CHANNEL = delegate_channel(_WEB)

_AGAIN = "In 2026 the bridge reopened on 3 March."
_THIRD = "Next the bridge closes in May for repairs."
_NEXT = "and what happens next"
_LATER = "any news since then?"
_FERRY_MESSAGE = "only the morning sailings"

_CODE = CodeAgentAbility.NAME
_CODE_CHANNEL = delegate_channel(_CODE)
_CODE_TASK = "make a notes file"
_CODE_FIRST = "I set up the workspace."
_WRITE_IT = "write the notes file now"
_WRITE_AGAIN = "try writing it once more"
_NOT_WRITTEN = "I was not allowed to write it."
_STILL_NOT = "I still was not allowed to write it."


def _spawn() -> ProviderResponse:
    return calls(tool(_WEB, instructions=_TASK, act_summary=_SUMMARY))


def _working_search() -> ProviderResponse:
    """A search's first step: it writes a line and calls a tool, so its turn
    holds rows past its task before its next step."""
    return ProviderResponse(text=_FIRST_WORDS, model="m", tool_calls=[tool(Recall.NAME, query="river bridge")])


def _message(n: int, message: str) -> dict[str, object]:
    return tool(MessageSubagentAbility.NAME, agent_id=n, message=message)


def _queued(channel: str) -> list[tuple[str, int | None]]:
    """Every message waiting on *channel*, oldest first: its text and the call
    it carries."""
    return [(q.text, q.tool_call_id) for q in QueuedMessage.filter("channel", channel).order_by("id ASC").get()]


def _turns(db: sqlite3.Connection, channel: str) -> list[int]:
    """Every turn that wrote a row on *channel*, in order."""
    return [
        cast("int", r["turn_id"]) for r in db.execute(
            "SELECT DISTINCT turn_id FROM transcript WHERE channel = ? ORDER BY turn_id", (channel,),
        )
    ]


def _kept(db: sqlite3.Connection, n: int) -> tuple[str, int, str, str]:
    """The row agent *n*'s latest ending kept: its channel, turn, role and
    content."""
    agent = cast("Agent", Agent.by_id(n))
    assert agent.response_id is not None, f"agent #{n} kept no row"
    row = db.execute(
        "SELECT channel, turn_id, role, content FROM transcript WHERE id = ?", (agent.response_id,),
    ).fetchone()
    assert row is not None, f"agent #{n} kept row {agent.response_id}, which no transcript row has"
    return row["channel"], row["turn_id"], row["role"], row["content"]


def _quiet_but(held: Held) -> None:
    """Wait for *held*'s subagent to reach its held call, then for every
    other turn — and the background work turns spawn — to end."""
    wait(held.entered, "the search's held call")
    while others := [t for t in turn_threads() if t is not held.thread]:
        for t in others:
            t.join(timeout=WAIT_S)
            assert not t.is_alive(), f"{t.name} did not end"


def _sent(n: int) -> tuple[str, None, str]:
    """What ``message_subagent`` returns for a message to subagent *n*, of any kind."""
    agent = cast("Agent", Agent.by_id(n))
    return "success", None, _SENT.format(label=label(agent), reply=reply_label(agent.type, n))


def _write() -> dict[str, object]:
    """A file write into the code agent's workspace — gated by the person's
    file-write permission."""
    return tool(
        WriteFileAbility.NAME, path=str(FileMapperService.get_code_agent_workspace_path("notes.txt")), contents="",
    )


class _Prompts:
    """Each permission prompt broadcast while it listens, and the end of each
    gated write, in the order they happen. :meth:`deny_until` denies each
    prompt as the signed-in person until the writes have all ended, so a write
    refused without asking shows as a write with no prompt, never as a wait
    that times out."""

    _WROTE = ""

    def __init__(self) -> None:
        self._events: queue.Queue[str] = queue.Queue()
        self.shown: list[tuple[object, object]] = []

    def note(self, frame: dict[str, object]) -> None:
        if frame.get("type") == "permission_request":
            self._events.put(cast("str", frame["request_id"]))

    def after_write(self, answer: str) -> Callable[[], ProviderResponse]:
        """The model step after a gated write: it marks the write ended, then
        answers *answer*."""
        def entry() -> ProviderResponse:
            self._events.put(self._WROTE)
            return text(answer)
        return entry

    def deny_until(self, client: FlaskClient, writes: int) -> None:
        """Deny every prompt, each read back as the only one pending, until
        *writes* gated writes have ended; note each one as the pending list
        showed it — its permission and the turn it asks on."""
        while writes:
            event = self._events.get(timeout=WAIT_S)
            if event == self._WROTE:
                writes -= 1
                continue
            response = client.get("/api/policies/pending")
            assert response.status_code == 200, response.data
            [ask] = cast("list[dict[str, object]]", cast("dict[str, object]", response.get_json())["result"])
            assert ask["request_id"] == event
            self.shown.append((ask["action_id"], ask["origin"]))
            answered = client.post("/api/policies/respond", json={"request_id": event, "approved": False})
            assert answered.status_code == 204, answered.data

    def left(self) -> int:
        return self._events.qsize()


def test_a_finished_search_messaged_again_runs_its_next_turn_on_its_own_history(db: sqlite3.Connection) -> None:
    """The chat spawns a ``web_search``, which answers. On the turn that reply
    opens, the chat messages the finished search: the message opens the
    search's next turn at once, carrying the message's call. The search goes
    running, then done, and keeps that turn's answer. That turn read its first
    turn's task and answer and then the message — nothing from the chat —
    and its answer comes back to the chat labelled with the same number."""
    seen: dict[str, object] = {}

    def message_it() -> ProviderResponse:
        agent = _the_search()
        seen["status"] = agent.status
        seen["n"] = agent.agent_id
        return calls(_message(agent.agent_id, _MESSAGE))

    provider = ChannelProvider(
        chat=[_spawn(), text("Started."), _find_agent_tools(), message_it, text("Asked."), text("Noted.")],
        web_search=[text(_ANSWER), text(_AGAIN)],
    )

    with listening() as listener, patched(provider):
        begin(UserConfig(), "find the bridge news").result()

    assert provider.unused() == {}
    assert [f for f in listener.frames if f.get("type") == "permission_request"] == []
    assert _blocked(db) == []
    n = cast("int", seen["n"])
    assert seen["status"] == Agent.DONE

    [spawn] = calls_of(db, _WEB)
    spawn_id = cast("int", spawn["id"])
    searcher, turn = child(db, _WEB_CHANNEL, spawn_id)
    assert agent_number(searcher) == n
    [sent] = calls_of(db, MessageSubagentAbility.NAME)
    sent_id = cast("int", sent["id"])
    assert _answer(sent) == ("success", None, _sent_to(n))

    # The message opened the search's next turn on its own channel.
    again = child(db, _WEB_CHANNEL, sent_id)
    assert again.channel == searcher and again.turn_id > turn
    assert _input_rows(db, searcher, again.turn_id) == [(_MESSAGE, False, str(sent_id))]
    assert _turns(db, searcher) == [turn, again.turn_id]
    for turn_id in (turn, again.turn_id):
        assert _executions(searcher, turn_id) == 1
        assert execution(searcher, turn_id).state == TurnExecution.COMPLETED

    # It went running, then done, and keeps the new turn's answer.
    agent = cast("Agent", Agent.by_id(n))
    assert (agent.status, agent.turn_id) == (Agent.DONE, again.turn_id)
    assert _kept(db, n) == (searcher, again.turn_id, "assistant", _AGAIN)
    assert _statuses(listener.frames, n) == [Agent.RUNNING, Agent.DONE, Agent.RUNNING, Agent.DONE]

    # Its next turn read its first turn, then the message bare — and nothing
    # from the chat.
    read = provider.read(_WEB)
    assert len(read) == 2
    assert f"{_WEB}: {_TASK}\n" in read[1] and f"Assistant: {_ANSWER}\n" in read[1]
    assert f"Research query:\n{_MESSAGE}\n" in read[1]
    assert "find the bridge news" not in read[1] and "Started." not in read[1]

    # Both answers went back to the call that spawned it, each labelled with
    # its number; none to the message's call.
    assert [(r["content"], r["joined"]) for r in replies(db, spawn_id)] == [(_ANSWER, 0), (_AGAIN, 0)]
    assert replies(db, sent_id) == []
    assert any(f"Reply from {_WEB} #{n}: {_AGAIN}" in r for r in provider.read("chat"))


def test_two_searches_each_messaged_again_read_only_their_own_turns(db: sqlite3.Connection) -> None:
    """The chat spawns a bridge search and, once it answers, a ferry search.
    Once that one answers too, the chat messages both in one step: each opens
    its next turn on its own channel, which reads its own first turn and its
    own message — nothing of the other search's — and each answer comes back
    labelled with its own number."""
    seen: dict[str, int] = {}

    def message_both() -> ProviderResponse:
        bridge, ferry = Agent.filter("spawned_by_channel", _USER).order_by("agent_id ASC").get()
        seen["bridge"], seen["ferry"] = bridge.agent_id, ferry.agent_id
        return calls(_message(bridge.agent_id, _MESSAGE), _message(ferry.agent_id, _FERRY_MESSAGE))

    provider = ChannelProvider(
        chat=[
            _spawn(),
            text("Started."),
            calls(tool(_WEB, instructions=_FERRY_TASK, act_summary=_FERRY_SUMMARY)),
            text("Started the ferry search."),
            _find_agent_tools(),
            message_both,
            text("Asked both."),
            text("Noted."),
            text("Noted."),
        ],
        web_search=[text(_ANSWER), text(_FERRY_ANSWER), text(_AGAIN), text(_AGAIN)],
    )

    with listening() as listener, patched(provider):
        begin(UserConfig(), "find the bridge news").result()

    assert provider.unused() == {}
    assert [f for f in listener.frames if f.get("type") == "permission_request"] == []
    spawn_bridge, spawn_ferry = (cast("int", c["id"]) for c in calls_of(db, _WEB))
    sent_bridge, sent_ferry = (cast("int", c["id"]) for c in calls_of(db, MessageSubagentAbility.NAME))
    for spawn_id, sent_id, n in ((spawn_bridge, sent_bridge, seen["bridge"]), (spawn_ferry, sent_ferry, seen["ferry"])):
        searcher, turn = child(db, _WEB_CHANNEL, spawn_id)
        assert agent_number(searcher) == n
        again = child(db, _WEB_CHANNEL, sent_id)
        assert again.channel == searcher and again.turn_id > turn
        assert _turns(db, searcher) == [turn, again.turn_id]
        assert cast("Agent", Agent.by_id(n)).status == Agent.DONE
        assert _statuses(listener.frames, n) == [Agent.RUNNING, Agent.DONE, Agent.RUNNING, Agent.DONE]

    # Each next turn read its own first turn and message, and nothing else.
    read = provider.read(_WEB)
    assert len(read) == 4
    bridge = read[_the_one(read, _MESSAGE)]
    assert _TASK in bridge and _ANSWER in bridge
    assert all(other not in bridge for other in (_FERRY_TASK, _FERRY_ANSWER, _FERRY_MESSAGE))
    ferry = read[_the_one(read, _FERRY_MESSAGE)]
    assert _FERRY_TASK in ferry and _FERRY_ANSWER in ferry
    assert all(other not in ferry for other in (_TASK, _ANSWER, _MESSAGE))

    assert [r["content"] for r in replies(db, spawn_bridge)] == [_ANSWER, _AGAIN]
    assert [r["content"] for r in replies(db, spawn_ferry)] == [_FERRY_ANSWER, _AGAIN]
    chat_read = provider.read("chat")
    for n in (seen["bridge"], seen["ferry"]):
        assert any(f"Reply from {_WEB} #{n}: {_AGAIN}" in r for r in chat_read)


def test_a_search_the_chat_stops_keeps_its_last_row_drops_its_waiting_message_and_opens_no_turn(
    db: sqlite3.Connection,
) -> None:
    """While a ``web_search`` is held mid-turn, the chat messages it — the
    message waits behind its turn — and then stops it: the stop drops that
    message before the search's turn has ended. A second message is answered
    with the cancelled text and opens no turn, and its status reads the same.
    The search keeps its turn's last row, and nothing comes back to the chat."""
    held = Held(text(_DRAFT))
    seen: dict[str, int] = {}
    agent_channel: list[str] = []
    waiting: list[list[tuple[str, int | None]]] = []
    released: list[bool] = []

    def find_while_it_works() -> ProviderResponse:
        wait(held.entered, "the search's second step")
        return _find_agent_tools()

    def message_it() -> ProviderResponse:
        [agent] = Agent.running()
        seen["n"] = agent.agent_id
        agent_channel.append(agent.channel)
        return calls(_message(agent.agent_id, _MESSAGE))

    def stop_it() -> ProviderResponse:
        waiting.append(_queued(agent_channel[0]))
        return calls(tool(StopSubagentAbility.NAME, agent_id=seen["n"]))

    def again() -> ProviderResponse:
        waiting.append(_queued(agent_channel[0]))
        released.append(held.gate.is_set())
        return calls(_message(seen["n"], _LATER), tool(AgentStatusAbility.NAME, agent_id=seen["n"]))

    provider = ChannelProvider(
        chat=[_spawn(), find_while_it_works, message_it, stop_it, again, text("Stopped it.")],
        web_search=[_working_search(), held],
    )

    with listening() as listener, patched(provider):
        begin(UserConfig(), "find the bridge news").result()
        held.gate.set()

    assert held.opened and provider.unused() == {}
    assert [f for f in listener.frames if f.get("type") == "permission_request"] == []
    n = seen["n"]
    [spawn] = calls_of(db, _WEB)
    spawn_id = cast("int", spawn["id"])
    searcher, turn = child(db, _WEB_CHANNEL, spawn_id)
    assert agent_number(searcher) == n
    sent, sent_later = calls_of(db, MessageSubagentAbility.NAME)
    sent_id, later_id = cast("int", sent["id"]), cast("int", sent_later["id"])

    # The first message waited behind the search's turn; the stop dropped it
    # while that turn was still held.
    assert _answer(sent) == ("success", None, _sent_to(n))
    assert waiting == [[(_MESSAGE, sent_id)], []]
    assert released == [False]
    [stopped] = calls_of(db, StopSubagentAbility.NAME)
    assert _answer(stopped) == ("success", None, f"Subagent {_WEB} #{n} stopped.")

    # Stopped, it answers a message and a status read with how it ended.
    assert _answer(sent_later) == ("error", "agent-stopped", CANCELLED_TEXT)
    [status] = calls_of(db, AgentStatusAbility.NAME)
    assert json.loads(_answer(status)[2]) == {
        "agent_id": n, "type": _WEB, "status": Agent.CANCELLED, "last_output": CANCELLED_TEXT,
    }

    # It keeps its turn's last row; neither message reached it or opened a turn,
    # and a cancelled search is never moved onto a next turn.
    assert not Agent.resume(n, turn + 1)
    agent = cast("Agent", Agent.by_id(n))
    assert (agent.status, agent.turn_id) == (Agent.CANCELLED, turn)
    assert agent.response_id == _last_row(db, searcher, turn)
    assert _reached(db, _WEB_CHANNEL, sent_id) == _reached(db, _WEB_CHANNEL, later_id) == 0
    assert _turns(db, searcher) == [turn]
    assert _executions(searcher, turn) == 1
    ended = execution(searcher, turn)
    assert (ended.state, ended.stop_reason) == (TurnExecution.CANCELLED, STOP_REASON_PARENT_STOPPED)
    assert _queued(searcher) == []
    assert _statuses(listener.frames, n) == [Agent.RUNNING, Agent.CANCELLED]
    assert replies(db, spawn_id) == []


def test_a_search_a_restart_cut_off_keeps_its_last_row_and_boot_drops_its_waiting_message(
    db: sqlite3.Connection,
) -> None:
    """The chat messages a ``web_search`` held mid-turn, so the message waits
    behind its turn, and the process dies under the search. The next boot ends
    it failed, keeping its turn's last row, drops the message waiting for it
    — which never reaches it — and tells the chat it crashed. A later message
    is answered with the crashed text and opens no turn, and its status reads
    the same."""
    held = Held(text(_DRAFT), then=dies)
    seen: dict[str, int] = {}

    def find_while_it_works() -> ProviderResponse:
        wait(held.entered, "the search's second step")
        return _find_agent_tools()

    def message_it() -> ProviderResponse:
        [agent] = Agent.running()
        seen["n"] = agent.agent_id
        return calls(_message(agent.agent_id, _MESSAGE))

    def reach_it() -> ProviderResponse:
        return calls(_message(seen["n"], _LATER), tool(AgentStatusAbility.NAME, agent_id=seen["n"]))

    provider = ChannelProvider(
        chat=[
            _spawn(),
            find_while_it_works,
            message_it,
            text("Asked."),
            text("It crashed."),
            _find_agent_tools(),
            reach_it,
            text("It had crashed."),
        ],
        web_search=[_working_search(), held],
    )

    with process_deaths_expected(), patched(provider):
        begin(UserConfig(), "find the bridge news").result()
        held.gate.set()

    n = seen["n"]
    [spawn] = calls_of(db, _WEB)
    spawn_id = cast("int", spawn["id"])
    searcher, turn = child(db, _WEB_CHANNEL, spawn_id)
    assert agent_number(searcher) == n
    sent, = calls_of(db, MessageSubagentAbility.NAME)
    sent_id = cast("int", sent["id"])
    assert _answer(sent) == ("success", None, _sent_to(n))
    # Cut off mid-turn: still running, the message still waiting for it.
    assert cast("Agent", Agent.by_id(n)).status == Agent.RUNNING
    assert _queued(searcher) == [(_MESSAGE, sent_id)]

    # What the next boot runs: the sweep, the waiting messages, then the
    # notices to the callers.
    with patched(provider):
        swept = MessageProcessor(config_for(ConfigTypeEnum.USER)).turn_execution_service.sweep_orphaned()
        for channel in QueuedMessage.channels():
            MessageProcessor.drain(channel)
        _reply_to_swept_delegates(swept)

    # A failed search is never moved onto a next turn.
    assert not Agent.resume(n, turn + 1)
    agent = cast("Agent", Agent.by_id(n))
    assert (agent.status, agent.turn_id) == (Agent.FAILED, turn)
    assert agent.response_id == _last_row(db, searcher, turn)
    assert _queued(searcher) == []
    assert _reached(db, _WEB_CHANNEL, sent_id) == 0
    assert [r["content"] for r in replies(db, spawn_id)] == [CRASHED_TEXT]

    run_turn(provider, UserConfig(), "what happened to the bridge search?")

    assert held.opened and provider.unused() == {}
    _, later = calls_of(db, MessageSubagentAbility.NAME)
    assert _answer(later) == ("error", "agent-stopped", CRASHED_TEXT)
    [status] = calls_of(db, AgentStatusAbility.NAME)
    assert json.loads(_answer(status)[2]) == {
        "agent_id": n, "type": _WEB, "status": Agent.FAILED, "last_output": CRASHED_TEXT,
    }
    assert _reached(db, _WEB_CHANNEL, cast("int", later["id"])) == 0
    assert _turns(db, searcher) == [turn]
    assert QueuedMessage.channels() == set()


def test_a_search_the_person_stops_drops_its_waiting_message_as_its_turn_ends(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """The chat messages a ``web_search`` held mid-turn, so the message waits
    behind its turn; then the person stops the search. Their stop leaves the
    message waiting; the search's turn, ending cancelled, drops it rather
    than open a turn, and tells the chat it was cancelled."""
    client, db, _store = authed_client
    held = Held(text(_DRAFT))

    def find_while_it_works() -> ProviderResponse:
        wait(held.entered, "the search's second step")
        return _find_agent_tools()

    def message_it() -> ProviderResponse:
        [agent] = Agent.running()
        return calls(_message(agent.agent_id, _MESSAGE))

    provider = ChannelProvider(
        chat=[_spawn(), find_while_it_works, message_it, text("Asked."), text("It was cancelled.")],
        web_search=[_working_search(), held],
    )

    with listening(), patched(provider):
        begin(UserConfig(), "find the bridge news").result()
        [spawn] = calls_of(db, _WEB)
        spawn_id = cast("int", spawn["id"])
        searcher, turn = child(db, _WEB_CHANNEL, spawn_id)
        before = _queued(searcher)
        stop = client.delete(f"/api/threads/{turn}?channel={searcher}")
        after = _queued(searcher)
        held.gate.set()

    assert stop.status_code == 200 and (stop.get_json() or {})["result"]["cancelled"] is True
    assert held.opened and provider.unused() == {}
    n = agent_number(searcher)
    [sent] = calls_of(db, MessageSubagentAbility.NAME)
    sent_id = cast("int", sent["id"])
    assert before == after == [(_MESSAGE, sent_id)]
    assert _queued(searcher) == []
    assert _reached(db, _WEB_CHANNEL, sent_id) == 0
    assert _turns(db, searcher) == [turn]

    agent = cast("Agent", Agent.by_id(n))
    assert (agent.status, agent.turn_id) == (Agent.CANCELLED, turn)
    assert agent.response_id == _last_row(db, searcher, turn)
    assert execution(searcher, turn).state == TurnExecution.CANCELLED
    assert [r["content"] for r in replies(db, spawn_id)] == [CANCELLED_TEXT]


_A_MOMENT_S = 1.0


def _a_finished_search(db: sqlite3.Connection) -> tuple[str, int, int]:
    """A ``web_search`` the chat spawned that answered and ended done, its
    reply read on a chat turn of its own: its channel, its turn and its
    number."""
    provider = ChannelProvider(chat=[_spawn(), text("Started."), text("Got the reply.")], web_search=[text(_ANSWER)])
    with patched(provider):
        begin(UserConfig(), "find the bridge news").result()
    assert provider.unused() == {}
    [spawn] = calls_of(db, _WEB)
    searcher, turn = child(db, _WEB_CHANNEL, cast("int", spawn["id"]))
    n = agent_number(searcher)
    assert cast("Agent", Agent.by_id(n)).status == Agent.DONE
    return searcher, turn, n


def test_a_message_racing_the_chats_stop_of_a_finished_search_lands_after_it_and_is_told_it_was_stopped(
    db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One chat turn stops a finished ``web_search`` while another messages
    it, the message let go between the stop's read of the search and its
    cancel. The message cannot land inside the stop: once the stop commits it
    is told the search was cancelled. The search ends cancelled, no turn of
    its opened after its last, and it still keeps that turn's answer."""
    searcher, turn, n = _a_finished_search(db)
    kept = cast("Agent", Agent.by_id(n)).response_id
    messaging = Held(calls(_message(n, _LATER)))
    answered = threading.Event()
    landed: list[bool] = []
    real_cancel = AgentService.cancel

    def cancel_while_messaged(agent_id: int) -> Agent | None:
        messaging.gate.set()
        landed.append(answered.wait(timeout=_A_MOMENT_S))
        return real_cancel(agent_id)

    def after_its_call() -> ProviderResponse:
        answered.set()
        return text("Done.")

    monkeypatch.setattr(AgentService, "cancel", staticmethod(cancel_while_messaged))
    provider = ChannelProvider(chat=[
        _find_agent_tools(),
        messaging,
        _find_agent_tools(),
        calls(tool(StopSubagentAbility.NAME, agent_id=n)),
        after_its_call,
        after_its_call,
    ])

    with listening() as listener, patched(provider):
        messaging_turn = begin(UserConfig(), "tell the search to look again")
        wait(messaging.entered, "the chat's step that messages the search")
        begin(UserConfig(), "stop the bridge search").result()
        messaging_turn.result()

    assert messaging.opened and provider.unused() == {}
    [sent] = calls_of(db, MessageSubagentAbility.NAME)
    sent_id = cast("int", sent["id"])
    [stopped] = calls_of(db, StopSubagentAbility.NAME)
    assert _answer(sent) == ("error", "agent-stopped", CANCELLED_TEXT)
    assert landed == [False]
    assert _answer(stopped) == ("success", None, f"Subagent {_WEB} #{n} stopped.")

    agent = cast("Agent", Agent.by_id(n))
    assert (agent.status, agent.turn_id, agent.response_id) == (Agent.CANCELLED, turn, kept)
    assert _kept(db, n) == (searcher, turn, "assistant", _ANSWER)
    assert _turns(db, searcher) == [turn]
    assert TurnExecution.open_on(searcher) is None
    assert _reached(db, _WEB_CHANNEL, sent_id) == 0
    assert _queued(searcher) == []
    assert provider.read(_WEB) == []
    assert _statuses(listener.frames, n) == [Agent.CANCELLED]


def test_a_stop_cancels_the_next_turn_a_searchs_waiting_message_opened_after_the_stop_looked_it_up(
    db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The chat messages a ``web_search`` held mid-turn, so the message waits
    behind its turn, then stops it. Once the stop has looked the search up,
    and before it acts, the search's turn ends and the waiting message opens
    its next turn. The stop cancels that next turn — the one open when it
    acts — and the search ends cancelled with nothing left waiting for it."""
    held = Held(text(_DRAFT))
    next_turn = Held(text(_AGAIN))
    read: list[int] = []

    def owned_as_its_turn_moves_on(self: StopSubagentAbility, agent_id: int) -> Agent | ToolResult:
        agent = AgentAbility.owned(self, agent_id)
        assert isinstance(agent, Agent), agent
        read.append(agent.turn_id)
        held.gate.set()
        wait(next_turn.entered, "the search's next turn, which its waiting message opened")
        return agent

    def find_while_it_works() -> ProviderResponse:
        wait(held.entered, "the search's second step")
        return _find_agent_tools()

    def message_it() -> ProviderResponse:
        [agent] = Agent.running()
        return calls(_message(agent.agent_id, _MESSAGE))

    def stop_it() -> ProviderResponse:
        [agent] = Agent.running()
        return calls(tool(StopSubagentAbility.NAME, agent_id=agent.agent_id))

    monkeypatch.setattr(StopSubagentAbility, "owned", owned_as_its_turn_moves_on)
    provider = ChannelProvider(
        chat=[_spawn(), find_while_it_works, message_it, stop_it, text("Stopped it."), text("Noted.")],
        web_search=[_working_search(), held, next_turn],
    )

    with listening() as listener, patched(provider):
        begin(UserConfig(), "find the bridge news").result()
        next_turn.gate.set()

    assert held.opened and next_turn.opened and provider.unused() == {}
    [spawn] = calls_of(db, _WEB)
    spawn_id = cast("int", spawn["id"])
    searcher, turn = child(db, _WEB_CHANNEL, spawn_id)
    n = agent_number(searcher)
    [sent] = calls_of(db, MessageSubagentAbility.NAME)
    sent_id = cast("int", sent["id"])
    opened = child(db, _WEB_CHANNEL, sent_id)
    [stopped] = calls_of(db, StopSubagentAbility.NAME)

    # The stop looked the search up on its first turn; by the time it acted,
    # that turn had ended and the message had opened the next.
    assert read == [turn]
    assert opened.channel == searcher and opened.turn_id > turn
    assert execution(searcher, turn).state == TurnExecution.COMPLETED
    assert _answer(stopped) == ("success", None, f"Subagent {_WEB} #{n} stopped.")
    ended = execution(searcher, opened.turn_id)
    assert (ended.state, ended.stop_reason) == (TurnExecution.CANCELLED, STOP_REASON_PARENT_STOPPED)
    agent = cast("Agent", Agent.by_id(n))
    assert (agent.status, agent.turn_id) == (Agent.CANCELLED, opened.turn_id)
    assert _queued(searcher) == []
    assert _turns(db, searcher) == [turn, opened.turn_id]
    assert [r["content"] for r in replies(db, spawn_id)] == [_DRAFT]
    assert replies(db, sent_id) == []
    assert _statuses(listener.frames, n) == [Agent.RUNNING, Agent.DONE, Agent.RUNNING, Agent.CANCELLED]


def test_a_message_that_finds_its_search_stopped_after_looking_it_up_is_told_it_was_stopped(
    db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chat turn messages a finished ``web_search``. After the message has
    looked the search up, and before it is sent, another chat turn stops the
    search, and that stop commits. The message is told the search was
    cancelled: it neither waits for the search nor opens a turn of its."""
    searcher, turn, n = _a_finished_search(db)
    stop_done = threading.Event()
    stopper: list[MessageProcessor] = []

    def owned_then_stopped(self: MessageSubagentAbility, agent_id: int) -> Agent | ToolResult:
        agent = AgentAbility.owned(self, agent_id)
        stopper.append(begin(UserConfig(), "stop the bridge search"))
        wait(stop_done, "the other chat turn's stop of the search")
        return agent

    def after_the_stop() -> ProviderResponse:
        stop_done.set()
        return text("Stopped it.")

    monkeypatch.setattr(MessageSubagentAbility, "owned", owned_then_stopped)
    provider = ChannelProvider(chat=[
        _find_agent_tools(),
        calls(_message(n, _LATER)),
        _find_agent_tools(),
        calls(tool(StopSubagentAbility.NAME, agent_id=n)),
        after_the_stop,
        text("Told it."),
    ])

    with patched(provider):
        begin(UserConfig(), "tell the search to look again").result()
        [stop_turn] = stopper
        stop_turn.result()

    assert provider.unused() == {}
    [sent] = calls_of(db, MessageSubagentAbility.NAME)
    [stopped] = calls_of(db, StopSubagentAbility.NAME)
    assert _answer(stopped) == ("success", None, f"Subagent {_WEB} #{n} stopped.")
    assert _answer(sent) == ("error", "agent-stopped", CANCELLED_TEXT)
    assert QueuedMessage.channels() == set()
    assert _reached(db, _WEB_CHANNEL, cast("int", sent["id"])) == 0
    assert _turns(db, searcher) == [turn]
    agent = cast("Agent", Agent.by_id(n))
    assert (agent.status, agent.turn_id) == (Agent.CANCELLED, turn)


def test_a_reply_waiting_on_a_chat_turn_the_person_stops_still_opens_a_chat_turn_of_its_own(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """A chat turn spawns two ``web_search``es and messages the first, held
    mid-turn, so the message waits behind it; the second answers while the
    chat turn still works, so its reply waits on the chat. The person stops
    the chat turn. The stop drops the message to the search but keeps the
    reply to the chat: once the stopped turn ends, the reply opens a chat
    turn of its own, and the chat reads the answer."""
    client, db, _store = authed_client
    ferry = Held(text(_FERRY_ANSWER))
    answering = Held(text(_ANSWER))
    asking = Held(text("Asked."))

    def find_while_it_works() -> ProviderResponse:
        wait(ferry.entered, "the ferry search's second step")
        return _find_agent_tools()

    def message_it() -> ProviderResponse:
        [agent] = Agent.running()
        return calls(_message(agent.agent_id, _FERRY_MESSAGE))

    def answer_while_it_works() -> ProviderResponse:
        wait(answering.entered, "the bridge search's step")
        answering.gate.set()
        _held_threads_end(answering)
        return _find_agent_tools()

    provider = ChannelProvider(
        chat=[
            calls(tool(_WEB, instructions=_FERRY_TASK, act_summary=_FERRY_SUMMARY)),
            find_while_it_works,
            message_it,
            _spawn(),
            answer_while_it_works,
            asking,
            text("Noted."),
        ],
        web_search=[_working_search(), ferry, answering],
    )

    with patched(provider):
        chat = begin(UserConfig(), "find the ferry and bridge news")
        wait(asking.entered, "the chat's step after the bridge search answered")
        ferry_spawn, bridge_spawn = calls_of(db, _WEB)
        ferry_search, _ferry_turn = child(db, _WEB_CHANNEL, cast("int", ferry_spawn["id"]))
        before = (_queued(ferry_search), _queued(_USER))
        stop = client.delete(f"/api/threads/{chat.turn_id}")
        after = (_queued(ferry_search), _queued(_USER))
        asking.gate.set()
        ferry.gate.set()
        chat.result()

    assert stop.status_code == 200 and (stop.get_json() or {})["result"]["cancelled"] is True
    bridge_id = cast("int", bridge_spawn["id"])
    n = agent_number(child(db, _WEB_CHANNEL, bridge_id).channel)
    ferry_n = agent_number(ferry_search)
    [sent] = calls_of(db, MessageSubagentAbility.NAME)
    sent_id = cast("int", sent["id"])
    assert _answer(sent) == ("success", None, _sent_to(ferry_n))

    # The message waited behind the ferry search and the reply on the chat;
    # the stop dropped the message and kept the reply.
    assert before == ([(_FERRY_MESSAGE, sent_id)], [(_ANSWER, bridge_id)])
    assert after == ([], [(_ANSWER, bridge_id)])
    assert _reached(db, _WEB_CHANNEL, sent_id) == 0
    assert QueuedMessage.channels() == set()
    assert ferry.opened and answering.opened and asking.opened and provider.unused() == {}

    # Once the stopped turn ended, the reply opened a chat turn of its own,
    # and the chat read the answer there.
    assert execution(_USER, chat.turn_id).state == TurnExecution.CANCELLED
    [reply] = replies(db, bridge_id)
    assert reply["content"] == _ANSWER and reply["turn_id"] > chat.turn_id
    assert execution(_USER, reply["turn_id"]).state == TurnExecution.COMPLETED
    assert assistant_texts(db, _USER, reply["turn_id"]) == ["Noted."]
    chat_read = provider.read("chat")
    assert _the_one(chat_read, f"Reply from {_WEB} #{n}: {_ANSWER}") == len(chat_read) - 1

    # The ferry search stopped with the chat turn and sent nothing back.
    assert cast("Agent", Agent.by_id(ferry_n)).status == Agent.CANCELLED
    assert replies(db, cast("int", ferry_spawn["id"])) == []
    assert cast("Agent", Agent.by_id(n)).status == Agent.DONE


def test_a_question_waiting_on_the_chat_still_opens_a_chat_turn_after_the_chat_stops_the_search_that_asked(
    db: sqlite3.Connection,
) -> None:
    """A ``web_search`` asks the chat while the chat's turn works, so its
    question waits on the chat; the chat messages the search, held mid-turn,
    so that message waits behind it, then stops it. The stop drops the
    chat's message to the search but keeps the search's question: once the
    chat's turn ends, the question opens a chat turn of its own, and the chat
    reads it."""
    held = Held(text(_WAITS))
    searcher: list[str] = []
    before: list[tuple[list[tuple[str, int | None]], list[tuple[str, int | None]]]] = []
    after: list[tuple[list[tuple[str, int | None]], list[tuple[str, int | None]]]] = []

    def find_while_it_works() -> ProviderResponse:
        wait(held.entered, "the search's step after it asked")
        return _find_agent_tools()

    def message_it() -> ProviderResponse:
        [agent] = Agent.running()
        searcher.append(agent.channel)
        return calls(_message(agent.agent_id, _MESSAGE))

    def stop_it() -> ProviderResponse:
        before.append((_queued(_USER), _queued(searcher[0])))
        [agent] = Agent.running()
        return calls(tool(StopSubagentAbility.NAME, agent_id=agent.agent_id))

    def after_the_stop() -> ProviderResponse:
        after.append((_queued(_USER), _queued(searcher[0])))
        return text("Stopped it.")

    provider = ChannelProvider(
        chat=[_spawn(), find_while_it_works, message_it, stop_it, after_the_stop, text("Noted.")],
        web_search=[_ask(), held],
    )

    with patched(provider):
        chat = begin(UserConfig(), "find the bridge news")
        chat.result()
        held.gate.set()

    [spawn] = calls_of(db, _WEB)
    spawn_id = cast("int", spawn["id"])
    search, turn = child(db, _WEB_CHANNEL, spawn_id)
    assert [search] == searcher
    n = agent_number(search)
    [asked] = calls_of(db, MessageParentAbility.NAME)
    asked_id = cast("int", asked["id"])
    [sent] = calls_of(db, MessageSubagentAbility.NAME)
    sent_id = cast("int", sent["id"])
    [stopped] = calls_of(db, StopSubagentAbility.NAME)
    assert _answer(stopped) == ("success", None, f"Subagent {_WEB} #{n} stopped.")

    # The question waited on the chat and the message behind the search; the
    # stop dropped the message and kept the question.
    assert before == [([(_QUESTION, asked_id)], [(_MESSAGE, sent_id)])]
    assert after == [([(_QUESTION, asked_id)], [])]
    assert _reached(db, _WEB_CHANNEL, sent_id) == 0
    assert QueuedMessage.channels() == set()
    assert held.opened and provider.unused() == {}

    # Once the chat's turn ended, the question opened a chat turn of its own,
    # and the chat read it there.
    [question] = replies(db, asked_id)
    assert question["content"] == _QUESTION and question["turn_id"] > chat.turn_id
    assert execution(_USER, question["turn_id"]).state == TurnExecution.COMPLETED
    assert assistant_texts(db, _USER, question["turn_id"]) == ["Noted."]
    chat_read = provider.read("chat")
    assert _the_one(chat_read, f"Message from {_WEB} #{n}: {_QUESTION}") == len(chat_read) - 1

    # The search's turn was stopped by its caller: it ends cancelled and
    # sends nothing back.
    ended = execution(search, turn)
    assert (ended.state, ended.stop_reason) == (TurnExecution.CANCELLED, STOP_REASON_PARENT_STOPPED)
    assert cast("Agent", Agent.by_id(n)).status == Agent.CANCELLED
    assert replies(db, spawn_id) == []


def test_a_reply_to_an_idle_chat_opens_a_chat_turn_at_once(db: sqlite3.Connection) -> None:
    """The chat spawns a ``web_search`` and its turn ends while the search is
    held. With nothing working on the chat, the search answers: its reply
    opens a chat turn of its own at once, never waiting in the chat's queue."""
    held = Held(text(_ANSWER))
    provider = ChannelProvider(chat=[_spawn(), text("Started."), text("Noted.")], web_search=[held])

    with listening(), patched(provider):
        chat = begin(UserConfig(), "find the bridge news")
        chat.result()
        _quiet_but(held)
        idle = TurnExecution.open_on(_USER) is None
        held.gate.set()
        _held_threads_end(held)
        waiting = _queued(_USER)

    assert held.opened and provider.unused() == {}
    assert idle
    assert waiting == []
    [spawn] = calls_of(db, _WEB)
    [reply] = replies(db, cast("int", spawn["id"]))
    assert (reply["content"], reply["joined"]) == (_ANSWER, 0)
    assert reply["turn_id"] > chat.turn_id
    assert execution(_USER, reply["turn_id"]).state == TurnExecution.COMPLETED
    assert assistant_texts(db, _USER, reply["turn_id"]) == ["Noted."]
    assert _executions(_USER, chat.turn_id) == 1


def test_a_reply_to_a_working_chat_waits_and_opens_a_chat_turn_once_that_one_ends(db: sqlite3.Connection) -> None:
    """The search answers while the chat's turn that spawned it still works:
    its reply waits in the chat's queue, and opens a chat turn of its own
    once that turn ends — it never joins it."""
    held = Held(text(_ANSWER))
    waiting: list[tuple[str, int | None]] = []

    def answer_while_it_works() -> ProviderResponse:
        wait(held.entered, "the search's model call")
        held.gate.set()
        _held_threads_end(held)
        waiting.extend(_queued(_USER))
        return text("Started.")

    provider = ChannelProvider(chat=[_spawn(), answer_while_it_works, text("Noted.")], web_search=[held])

    with listening(), patched(provider):
        chat = begin(UserConfig(), "find the bridge news")
        chat.result()

    assert held.opened and provider.unused() == {}
    [spawn] = calls_of(db, _WEB)
    spawn_id = cast("int", spawn["id"])
    assert waiting == [(_ANSWER, spawn_id)]
    assert _queued(_USER) == []
    [reply] = replies(db, spawn_id)
    assert (reply["content"], reply["joined"]) == (_ANSWER, 0)
    assert reply["turn_id"] > chat.turn_id
    assert _executions(_USER, chat.turn_id) == 1
    assert execution(_USER, reply["turn_id"]).state == TurnExecution.COMPLETED
    assert assistant_texts(db, _USER, reply["turn_id"]) == ["Noted."]


def test_two_messages_to_an_idle_search_open_its_next_two_turns_in_order(db: sqlite3.Connection) -> None:
    """The chat sends a finished ``web_search`` two messages in one step. The
    first opens its next turn at once; the second, finding that turn working,
    waits, and opens the turn after once that one ends. That last turn reads
    the first message, its answer and then the second message; each turn's
    answer comes back to the chat."""
    held = Held(text(_AGAIN))
    both_sent = threading.Event()
    seen: dict[str, int] = {}
    agent_channel: list[str] = []
    waiting: list[tuple[str, int | None]] = []

    def message_twice() -> ProviderResponse:
        agent = _the_search()
        seen["n"] = agent.agent_id
        agent_channel.append(agent.channel)
        return calls(_message(agent.agent_id, _MESSAGE), _message(agent.agent_id, _NEXT))

    def asked_twice() -> ProviderResponse:
        waiting.extend(_queued(agent_channel[0]))
        both_sent.set()
        held.gate.set()
        return text("Asked twice.")

    provider = ChannelProvider(
        chat=[
            _spawn(),
            text("Started."),
            _find_agent_tools(),
            message_twice,
            asked_twice,
            text("Noted."),
            text("Noted."),
        ],
        web_search=[text(_ANSWER), held, text(_THIRD)],
    )

    with listening() as listener, patched(provider):
        begin(UserConfig(), "find the bridge news").result()
        wait(both_sent, "the chat's step after its two messages")

    assert held.opened and provider.unused() == {}
    n = seen["n"]
    [spawn] = calls_of(db, _WEB)
    spawn_id = cast("int", spawn["id"])
    searcher, turn = child(db, _WEB_CHANNEL, spawn_id)
    assert agent_number(searcher) == n
    first, second = calls_of(db, MessageSubagentAbility.NAME)
    first_id, second_id = cast("int", first["id"]), cast("int", second["id"])
    assert _answer(first) == _answer(second) == ("success", None, _sent_to(n))

    # The second waited behind the turn the first opened, then opened the next.
    assert waiting == [(_NEXT, second_id)]
    assert _queued(searcher) == []
    opened_first = child(db, _WEB_CHANNEL, first_id)
    opened_second = child(db, _WEB_CHANNEL, second_id)
    assert opened_first.channel == opened_second.channel == searcher
    assert turn < opened_first.turn_id < opened_second.turn_id
    assert _turns(db, searcher) == [turn, opened_first.turn_id, opened_second.turn_id]
    assert _input_rows(db, searcher, opened_first.turn_id) == [(_MESSAGE, False, str(first_id))]
    assert _input_rows(db, searcher, opened_second.turn_id) == [(_NEXT, False, str(second_id))]
    agent = cast("Agent", Agent.by_id(n))
    assert (agent.status, agent.turn_id) == (Agent.DONE, opened_second.turn_id)
    assert _kept(db, n) == (searcher, opened_second.turn_id, "assistant", _THIRD)
    assert _statuses(listener.frames, n) == [Agent.RUNNING, Agent.DONE] * 3

    read = provider.read(_WEB)
    assert len(read) == 3
    assert f"Research query:\n{_NEXT}\n" in read[2]
    assert all(said in read[2] for said in (_TASK, _ANSWER, _MESSAGE, _AGAIN))
    assert [r["content"] for r in replies(db, spawn_id)] == [_ANSWER, _AGAIN, _THIRD]


def test_a_subtask_a_scheduled_task_started_is_never_messaged_by_it(db: sqlite3.Connection) -> None:
    """A scheduled task starts a subtask and waits for its answer, then
    messages it by its number: the call is told no subagent has that number,
    nothing waits for the subtask, and no turn of its opens."""
    seen: dict[str, int] = {}

    def reach_it() -> ProviderResponse:
        newest = Agent.filter("type", SubtaskAbility.NAME).order_by("agent_id DESC").first()
        assert newest is not None and newest.spawned_by_channel == _SCHEDULE
        seen["n"] = newest.agent_id
        return calls(_message(newest.agent_id, _MESSAGE))

    provider = ChannelProvider(
        chat=[
            calls(tool(FindToolsAbility.NAME, query=[SubtaskAbility.NAME, MessageSubagentAbility.NAME])),
            calls(tool(SubtaskAbility.NAME, action="start", instructions=_TASK)),
            reach_it,
            text("Status: the bridge reopened."),
        ],
        subtask=[text(_ANSWER)],
    )

    run_turn(provider, ScheduledConfig(), "check the bridge news", turn_id=_SCHEDULE_ID)

    assert provider.unused() == {}
    n = seen["n"]
    [start] = calls_of(db, SubtaskAbility.NAME)
    assert start["state"] == ToolCall.DONE, start["result"]
    subtask, turn = child(db, _SUBTASK, cast("int", start["id"]))
    assert agent_number(subtask) == n
    [sent] = calls_of(db, MessageSubagentAbility.NAME)
    assert _answer(sent) == _not_found(n)
    assert _reached(db, _SUBTASK, cast("int", sent["id"])) == 0
    assert QueuedMessage.channels() == set()
    assert _turns(db, subtask) == [turn]
    agent = cast("Agent", Agent.by_id(n))
    assert (agent.spawned_by_channel, agent.status, agent.turn_id) == (_SCHEDULE, Agent.DONE, turn)


def test_a_code_agents_turns_two_messages_open_each_ask_the_person_before_a_file_write(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """The person has set file writes to ask on the chat. The chat starts a
    ``code_agent``, which answers; on the turn that reply opens, the chat sends
    it two messages in one step. The first opens its next turn at once; the
    second waits behind that turn and opens the turn after. Each turn writes a
    file, and each write asks the person under the chat turn that sent the
    messages — never refused for want of somewhere to ask. Denied, neither
    runs, and each turn's answer reaches the chat."""
    client, db, _store = authed_client
    assert Policy.upsert(_CHAT, _FILE_WRITE, "ask") == 1
    prompts = _Prompts()
    both_sent = threading.Event()
    seen: dict[str, int] = {}
    agent_channel: list[str] = []
    waiting: list[tuple[str, int | None]] = []

    def message_twice() -> ProviderResponse:
        agent = _the_search()
        seen["n"] = agent.agent_id
        agent_channel.append(agent.channel)
        return calls(_message(agent.agent_id, _WRITE_IT), _message(agent.agent_id, _WRITE_AGAIN))

    def asked_twice() -> ProviderResponse:
        waiting.extend(_queued(agent_channel[0]))
        both_sent.set()
        return text("Asked twice.")

    provider = ChannelProvider(
        chat=[
            calls(tool(_CODE, instructions=_CODE_TASK)),
            text("Started."),
            _find_agent_tools(),
            message_twice,
            asked_twice,
            text("Noted."),
            text("Noted."),
        ],
        code_agent=[
            text(_CODE_FIRST),
            calls(_write()),
            prompts.after_write(_NOT_WRITTEN),
            calls(_write()),
            prompts.after_write(_STILL_NOT),
        ],
    )

    with listening(prompts.note), patched(provider):
        begin(UserConfig(), "make me a notes file").result()
        wait(both_sent, "the chat's step after its two messages")
        prompts.deny_until(client, 2)

    assert provider.unused() == {} and prompts.left() == 0
    n = seen["n"]
    [coding] = calls_of(db, _CODE)
    coding_id = cast("int", coding["id"])
    coder, turn = child(db, _CODE_CHANNEL, coding_id)
    assert agent_number(coder) == n
    first, second = calls_of(db, MessageSubagentAbility.NAME)
    first_id, second_id = cast("int", first["id"]), cast("int", second["id"])
    assert _answer(first) == _answer(second) == _sent(n)

    # Both writes asked on the chat, under the turn that sent the messages,
    # and were refused as the person denied them.
    sender = cast("int", replies(db, coding_id)[0]["turn_id"])
    assert {first_id, second_id} <= {c.id for c in ToolCall.by_turn(_USER, sender)}
    assert prompts.shown == [(_FILE_WRITE, _origin("user", sender))] * 2
    assert _blocked(db) == [(_FILE_WRITE, _CHAT, "user_denied")] * 2
    writes = calls_of(db, WriteFileAbility.NAME)
    assert [(w["state"], _BLOCK.format(permission=_FILE_WRITE) in w["result"]) for w in writes] == [
        (ToolCall.ERROR, True),
    ] * 2

    # The first opened the next turn at once; the second waited, then opened
    # the turn after.
    assert waiting == [(_WRITE_AGAIN, second_id)]
    assert _queued(coder) == []
    opened_first = child(db, _CODE_CHANNEL, first_id)
    opened_second = child(db, _CODE_CHANNEL, second_id)
    assert opened_first.channel == opened_second.channel == coder
    assert turn < opened_first.turn_id < opened_second.turn_id
    assert _input_rows(db, coder, opened_first.turn_id) == [(_WRITE_IT, False, str(first_id))]
    assert _input_rows(db, coder, opened_second.turn_id) == [(_WRITE_AGAIN, False, str(second_id))]
    agent = cast("Agent", Agent.by_id(n))
    assert (agent.status, agent.turn_id) == (Agent.DONE, opened_second.turn_id)
    assert [r["content"] for r in replies(db, coding_id)] == [_CODE_FIRST, _NOT_WRITTEN, _STILL_NOT]


def test_a_chat_turn_a_subagents_question_opens_asks_the_person_before_a_file_write(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """The person has set file writes to ask on the chat. A ``web_search``
    the chat spawned asks it a question, which opens a chat turn of its own;
    on that turn the chat writes a file. The write asks the person under that
    turn — never refused for want of somewhere to ask. Denied, it does not
    run, and the chat answers on that turn."""
    client, db, _store = authed_client
    assert Policy.upsert(_CHAT, _FILE_WRITE, "ask") == 1
    prompts = _Prompts()
    asking = Held(_ask())
    provider = ChannelProvider(
        chat=[_spawn(), text("Started."), calls(_write()), prompts.after_write(_NOT_WRITTEN)],
        web_search=[asking, text(_WAITS)],
    )

    with listening(prompts.note), patched(provider):
        chat = begin(UserConfig(), "find the bridge news")
        chat.result()
        asking.gate.set()
        prompts.deny_until(client, 1)

    assert asking.opened and provider.unused() == {} and prompts.left() == 0
    [asked] = calls_of(db, MessageParentAbility.NAME)
    [message] = replies(db, cast("int", asked["id"]))
    asked_on = cast("int", message["turn_id"])
    assert message["content"] == _QUESTION and asked_on != chat.turn_id
    assert prompts.shown == [(_FILE_WRITE, _origin("user", asked_on))]
    assert _blocked(db) == [(_FILE_WRITE, _CHAT, "user_denied")]
    [write] = calls_of(db, WriteFileAbility.NAME)
    assert write["id"] in {c.id for c in ToolCall.by_turn(_USER, asked_on)}
    assert write["state"] == ToolCall.ERROR
    assert _BLOCK.format(permission=_FILE_WRITE) in write["result"]
    assert execution(_USER, asked_on).state == TurnExecution.COMPLETED
    assert assistant_texts(db, _USER, asked_on)[-1] == _NOT_WRITTEN


def test_a_chat_turn_stopped_while_its_message_waits_for_a_working_search_drops_the_message(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """While a ``web_search`` is held mid-turn, a second chat turn messages
    it — the message waits behind the search's turn — and the person stops
    that chat turn. The stop drops the waiting message: no turn of the search
    opens for it, neither as the search's turn ends nor at a later drain, and
    the message's text stays in the call that sent it."""
    client, db, _store = authed_client
    held = Held(text(_DRAFT))
    asking = Held(text("Asked."))
    seen: dict[str, int] = {}

    def message_it() -> ProviderResponse:
        [agent] = Agent.running()
        seen["n"] = agent.agent_id
        return calls(_message(agent.agent_id, _MESSAGE))

    provider = ChannelProvider(
        chat=[_spawn(), text("Started."), _find_agent_tools(), message_it, asking, text("Noted.")],
        web_search=[_working_search(), held],
    )

    with listening(), patched(provider):
        begin(UserConfig(), "find the bridge news").result()
        wait(held.entered, "the search's second step")
        second = begin(UserConfig(), "only this year")
        wait(asking.entered, "the chat's step after its message")
        [spawn] = calls_of(db, _WEB)
        spawn_id = cast("int", spawn["id"])
        searcher, turn = child(db, _WEB_CHANNEL, spawn_id)
        before = _queued(searcher)
        stop = client.delete(f"/api/threads/{second.turn_id}")
        after = _queued(searcher)
        asking.gate.set()
        held.gate.set()
        drain()
        reopened = MessageProcessor.drain(searcher)

    assert stop.status_code == 200 and (stop.get_json() or {})["result"]["cancelled"] is True
    n = seen["n"]
    assert agent_number(searcher) == n
    [sent] = calls_of(db, MessageSubagentAbility.NAME)
    sent_id = cast("int", sent["id"])
    assert _answer(sent) == ("success", None, _sent_to(n))

    # The message waited behind the search's turn until the stop dropped it.
    assert (before, after) == ([(_MESSAGE, sent_id)], [])
    assert reopened is None
    assert QueuedMessage.channels() == set()
    assert _reached(db, _WEB_CHANNEL, sent_id) == 0
    assert _turns(db, searcher) == [turn]
    kept = ToolCall.get(sent_id)
    assert kept is not None and json.loads(kept.params)["message"] == _MESSAGE

    assert held.opened and asking.opened and provider.unused() == {}
    assert execution(_USER, second.turn_id).state == TurnExecution.CANCELLED
    agent = cast("Agent", Agent.by_id(n))
    assert (agent.status, agent.turn_id) == (Agent.DONE, turn)
    assert [r["content"] for r in replies(db, spawn_id)] == [_DRAFT]


def test_stopping_a_chat_turn_stops_the_search_turn_its_message_opened_minutes_after_its_call_ended(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """A second chat turn messages a ``web_search`` held mid-turn — the
    message waits behind the search's turn — and holds its own next step.
    Minutes after the message's call ended, the search's turn ends and the
    message opens its next turn. The person stops the chat turn: the search's
    turn the message opened stops with it, as a parent's stop, and the search
    ends cancelled with nothing sent back for that turn."""
    client, db, _store = authed_client
    held = Held(text(_DRAFT))
    asking = Held(text("Asked."))
    again = Held(text(_AGAIN))
    seen: dict[str, int] = {}

    def message_it() -> ProviderResponse:
        [agent] = Agent.running()
        seen["n"] = agent.agent_id
        return calls(_message(agent.agent_id, _MESSAGE))

    provider = ChannelProvider(
        chat=[_spawn(), text("Started."), _find_agent_tools(), message_it, asking, text("Noted.")],
        web_search=[_working_search(), held, again],
    )

    with listening() as listener, patched(provider):
        begin(UserConfig(), "find the bridge news").result()
        wait(held.entered, "the search's second step")
        second = begin(UserConfig(), "only this year")
        wait(asking.entered, "the chat's step after its message")
        # Every call of the second chat turn opened and ended five minutes
        # earlier than it did.
        earlier = timedelta(minutes=5)
        for c in ToolCall.by_turn(_USER, second.turn_id):
            assert c.ended_at is not None, c.tool_name
            assert ToolCall.filter("id", c.id).update(
                created_at=(parse_utc(c.created_at) - earlier).isoformat(),
                ended_at=(parse_utc(c.ended_at) - earlier).isoformat(),
            ) == 1
        held.gate.set()
        wait(again.entered, "the search's turn the message opened")
        stop = client.delete(f"/api/threads/{second.turn_id}")
        asking.gate.set()
        again.gate.set()

    assert stop.status_code == 200 and (stop.get_json() or {})["result"]["cancelled"] is True
    n = seen["n"]
    [spawn] = calls_of(db, _WEB)
    spawn_id = cast("int", spawn["id"])
    searcher, turn = child(db, _WEB_CHANNEL, spawn_id)
    assert agent_number(searcher) == n
    [sent] = calls_of(db, MessageSubagentAbility.NAME)
    sent_id = cast("int", sent["id"])
    opened = child(db, _WEB_CHANNEL, sent_id)
    assert opened.channel == searcher and opened.turn_id > turn

    # The message opened the search's turn more than a minute after its call ended.
    [row] = db.execute(
        "SELECT created_at FROM transcript WHERE channel = ? AND turn_id = ? AND tool_call_id = ?",
        (searcher, opened.turn_id, str(sent_id)),
    ).fetchall()
    assert parse_utc(row["created_at"]) - parse_utc(sent["ended_at"]) > _CHILD_ROW_LAG

    # The chat turn's stop stopped that turn, as a parent's stop.
    ended = execution(searcher, opened.turn_id)
    assert (ended.state, ended.stop_reason) == (TurnExecution.CANCELLED, STOP_REASON_PARENT_STOPPED)
    agent = cast("Agent", Agent.by_id(n))
    assert (agent.status, agent.turn_id) == (Agent.CANCELLED, opened.turn_id)
    assert _statuses(listener.frames, n) == [Agent.RUNNING, Agent.DONE, Agent.RUNNING, Agent.CANCELLED]
    assert _queued(searcher) == []
    assert execution(_USER, second.turn_id).state == TurnExecution.CANCELLED
    assert held.opened and asking.opened and again.opened and provider.unused() == {}
    assert [r["content"] for r in replies(db, spawn_id)] == [_DRAFT]


def test_a_search_that_ended_over_a_week_ago_is_archived_and_one_still_working_never_is(
    db: sqlite3.Connection,
) -> None:
    """The chat reads the status of a ``web_search`` while it works, before it
    has answered: it reads running, not archived. The search answers and ends.
    Once the row it ended on is more than a week old, a message, a stop and a
    status read are each answered that it is archived: the message opens no
    turn, and the stop leaves it done."""
    held = Held(text(_ANSWER))
    seen: dict[str, int] = {}

    def find_while_it_works() -> ProviderResponse:
        wait(held.entered, "the search's model call")
        return _find_agent_tools()

    def status_while_it_works() -> ProviderResponse:
        [agent] = Agent.running()
        seen["n"] = agent.agent_id
        return calls(tool(AgentStatusAbility.NAME, agent_id=agent.agent_id))

    def release_it() -> ProviderResponse:
        held.gate.set()
        return text("Started.")

    def reach_it() -> ProviderResponse:
        n = seen["n"]
        return calls(
            _message(n, _LATER),
            tool(StopSubagentAbility.NAME, agent_id=n),
            tool(AgentStatusAbility.NAME, agent_id=n),
        )

    provider = ChannelProvider(
        chat=[
            _spawn(),
            find_while_it_works,
            status_while_it_works,
            release_it,
            text("Noted."),
            _find_agent_tools(),
            reach_it,
            text("It was archived."),
        ],
        web_search=[held],
    )

    with listening(), patched(provider):
        begin(UserConfig(), "find the bridge news").result()

    n = seen["n"]
    [spawn] = calls_of(db, _WEB)
    searcher, turn = child(db, _WEB_CHANNEL, cast("int", spawn["id"]))
    answered = cast("Agent", Agent.by_id(n))
    assert (answered.status, answered.turn_id) == (Agent.DONE, turn)
    assert answered.response_id is not None
    aged = utc_now() - ARCHIVE_AFTER - timedelta(days=1)
    assert Transcript.filter("id", answered.response_id).update(created_at=aged.strftime("%Y-%m-%d %H:%M:%S")) == 1

    run_turn(provider, UserConfig(), "any news on the bridge?")

    assert held.opened and provider.unused() == {}
    # While it worked, it read running.
    working, status_later = calls_of(db, AgentStatusAbility.NAME)
    assert _answer(working)[:2] == ("success", None)
    assert json.loads(_answer(working)[2]) == {
        "agent_id": n, "type": _WEB, "status": Agent.RUNNING, "last_output": None,
    }
    # Ended over a week ago, it is archived to every tool.
    archived = ("error", "agent-archived", ARCHIVED_TEXT.format(type=_WEB, agent_id=n))
    [sent] = calls_of(db, MessageSubagentAbility.NAME)
    [stopped] = calls_of(db, StopSubagentAbility.NAME)
    assert [_answer(sent), _answer(stopped), _answer(status_later)] == [archived] * 3

    agent = cast("Agent", Agent.by_id(n))
    assert (agent.status, agent.turn_id) == (Agent.DONE, turn)
    assert _reached(db, _WEB_CHANNEL, cast("int", sent["id"])) == 0
    assert _turns(db, searcher) == [turn]
    assert QueuedMessage.channels() == set()
    assert [r["content"] for r in replies(db, cast("int", spawn["id"]))] == [_ANSWER]


def test_a_stop_of_a_finished_subtask_drops_the_message_waiting_on_its_memory_step(db: sqlite3.Connection) -> None:
    """The chat starts a subtask, which answers and ends done. While the
    memory step works on the subtask's own channel, the chat messages the
    subtask — the message waits behind that turn — and then stops it. The stop
    drops the message and ends the subtask cancelled: no turn of its opens
    for the message, neither as the memory step ends nor at a later drain or
    the boot's."""
    starting = ChannelProvider(
        chat=[
            calls(tool(SubtaskAbility.NAME, action="start", instructions=_TASK)),
            text("Started it."),
            text("Got the reply."),
        ],
        subtask=[text(_ANSWER)],
    )
    with listening(), patched(starting):
        begin(UserConfig(), "start a subtask on the bridge news").result()

    assert starting.unused() == {}
    [start] = calls_of(db, SubtaskAbility.NAME)
    start_id = cast("int", start["id"])
    channel, turn = child(db, _SUBTASK, start_id)
    n = agent_number(channel)
    assert cast("Agent", Agent.by_id(n)).status == Agent.DONE

    step = Held(text("nothing new"))
    waiting: list[list[tuple[str, int | None]]] = []

    def message_it() -> ProviderResponse:
        return calls(_message(_the_search().agent_id, _MESSAGE))

    def stop_it() -> ProviderResponse:
        waiting.append(_queued(channel))
        return calls(tool(StopSubagentAbility.NAME, agent_id=n))

    def after_stop() -> ProviderResponse:
        waiting.append(_queued(channel))
        return text("Stopped it.")

    stopping = ChannelProvider(chat=[_find_agent_tools(), message_it, stop_it, after_stop], background=[step])
    with listening() as listener, patched(stopping):
        memory = MessageProcessor.process(
            memory_step_config(UserSubtaskConfig().on_channel(channel), Transcript.ids_for_turn(channel, turn)),
        )
        wait(step.entered, "the memory step's model call")
        begin(UserConfig(), "stop that subtask").result()
        step.gate.set()
        memory.result()
        drain()
        reopened = MessageProcessor.drain(channel)
        booted = [MessageProcessor.drain(queued) for queued in QueuedMessage.channels()]

    assert step.opened and stopping.unused() == {}
    [sent] = calls_of(db, MessageSubagentAbility.NAME)
    sent_id = cast("int", sent["id"])
    assert _answer(sent) == _sent(n)

    # The message waited behind the memory step until the stop dropped it.
    assert waiting == [[(_MESSAGE, sent_id)], []]
    [stopped] = calls_of(db, StopSubagentAbility.NAME)
    assert _answer(stopped) == ("success", None, f"Subagent {label(cast('Agent', Agent.by_id(n)))} stopped.")
    assert reopened is None and booted == []
    assert QueuedMessage.channels() == set()
    assert _reached(db, _SUBTASK, sent_id) == 0
    [memory_turn] = _memory_turns(db, channel)
    assert _turns(db, channel) == [turn, memory_turn]

    agent = cast("Agent", Agent.by_id(n))
    assert (agent.status, agent.turn_id) == (Agent.CANCELLED, turn)
    assert _statuses(listener.frames, n) == [Agent.CANCELLED]
