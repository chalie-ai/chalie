"""Feature tests: a subagent the person's chat spawned sends the chat a
message, and the chat's reply opens the subagent's next turn.

* Every kind of subagent the chat spawns — a subtask, a ``web_search``, a
  ``web_browse``, a ``code_agent`` — reads who spawned it and that it may ask,
  and ``message_parent`` is offered to it.
* The call returns "Message sent." at once. On every later step of that turn
  the subagent reads that it is invoked again once the reply lands — never
  stored — and the turn, ending on that call, posts nothing back. The message
  opens a turn on the chat, labelled for the chat's model as a message from
  that subagent and followed by a nudge to reply, with ``message_subagent``
  offered on that turn. The chat's reply opens the subagent's next turn on its
  own channel, which reads its earlier turn and the reply bare; its answer
  comes back to the chat on a turn of its own. Once replied to, the nudge is
  gone. What the person sees — the stored row, the thread read over HTTP and
  every websocket frame — carries only the message's text and which subagent
  sent it.
* A message sent while the chat's turn is still working queues behind it and
  opens the chat's next turn once that turn ends; that turn reads it with the
  nudge and is offered ``message_subagent``.
* Stopping the subagent that waits for a reply — by the chat's
  ``stop_subagent`` or by the person's stop of its turn — ends it cancelled;
  the nudge is gone from the chat's next read. The chat's own stop sends
  nothing back; the person's stop tells the chat the session was cancelled.
* The person's stop of the chat turn the message opened leaves the subagent
  waiting: it is still named after the person's next message. Once the chat
  has replied in that turn, the same stop also stops the subagent turn the
  reply opened, which ends cancelled with its caller and sends nothing back.
* A ``web_search`` or a subtask a scheduled task or an outside agent spawned
  is waited for, so it holds no ``message_parent``: it reads that it should finish, is not offered
  the tool, and a call to it is refused as an unknown tool and posts nothing.
* A restart while the subagent posts its message leaves it, and the chat turn
  its message opened, for the next boot: the sweep ends both, the chat is told
  the session crashed, and no nudge is left behind.

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
from abilities.message_parent import PARENT_NUDGE, WAITING_NOTICE, MessageParentAbility
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
from models.queued_message import QueuedMessage
from models.tool_call import ToolCall
from models.turn_execution import TurnExecution
from run import _backfill_redesign_columns, _reply_to_swept_delegates
from services.agent_service import CANCELLED_TEXT, CRASHED_TEXT
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
    child,
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
from tests.test_agent_tools import _answer, _executions, _input_rows, _statuses
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
_WAITS = "I need to know which bridge first."
_REPLY = "the old stone bridge"
_ANSWER = "The old stone bridge reopened in March."

_ASKS = (
    f"You can ask for more context if needed by using `{MessageParentAbility.NAME}`. "
    "This will send a message to the agent that spawned you; you are invoked again when its reply lands. Else, finish"
)
_FINISH = " the task naturally and your final output will be sent to the agent that spawned you automatically."
_SENT = "Message sent."
_FOLLOW_UP = (
    "You'll be invoked again once the reply lands. "
    "If there is nothing else you can do without the answer just stop."
)


def _spawned(turn_id: int, *, asks: bool) -> str:
    """The task as a subagent of any kind reads it, spawned by turn *turn_id*
    — told it may ask, or only to finish."""
    return f"You have been spawned by agent {turn_id} to perform the task; {_TASK}. " + (
        _ASKS if asks else "Finish"
    ) + _FINISH


def _sent_to(n: int) -> str:
    """What ``message_subagent`` returns for a message to the search *n*."""
    return f'Message sent to subagent {_WEB} #{n}. Its reply will reach you as "Reply from {_WEB} #{n}".'


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


def _the_search() -> Agent:
    """The one subagent the chat spawned, whatever its status."""
    [agent] = Agent.filter("spawned_by_channel", _USER).get()
    return agent


def _reply_to_it(seen: dict[str, int]) -> ProviderResponse:
    """The chat's reply to the one subagent it spawned, whose number it notes."""
    agent = _the_search()
    seen["n"] = agent.agent_id
    return calls(tool(MessageSubagentAbility.NAME, agent_id=agent.agent_id, message=_REPLY))


def _stop_it(seen: dict[str, int]) -> ProviderResponse:
    """The chat's stop of the one subagent it spawned, whose number it notes."""
    agent = _the_search()
    seen["n"] = agent.agent_id
    return calls(tool(StopSubagentAbility.NAME, agent_id=agent.agent_id))


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


def test_a_subagent_asks_the_chat_and_the_chats_reply_opens_its_next_turn(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """The chat spawns a ``web_search`` and ends its turn. The search asks a
    question: the call returns at once, the search reads that it is invoked
    again once the reply lands, and its turn ends posting nothing back. The
    question opens a turn on the chat labelled as its message and nudging the
    chat to reply; the chat replies with ``message_subagent``, offered on that
    turn. The reply opens the search's next turn, which reads its first turn
    and the reply bare and answers; the answer comes back to the chat on a
    turn of its own."""
    client, db, _store = authed_client
    asking = Held(_ask())
    seen: dict[str, int] = {}
    provider = ChannelProvider(
        chat=[_spawn(), text("Started."), lambda: _reply_to_it(seen), text("Told it."), text("Noted.")],
        web_search=[asking, text(_WAITS), text(_ANSWER)],
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
    searcher, turn = child(db, _WEB_CHANNEL, spawn_id)
    assert agent_number(searcher) == n
    [asked] = calls_of(db, MessageParentAbility.NAME)
    assert _answer(asked) == ("success", None, _SENT)
    [sent] = calls_of(db, MessageSubagentAbility.NAME)
    sent_id = cast("int", sent["id"])
    assert _answer(sent) == ("success", None, _sent_to(n))

    # The search read who spawned it and that it may ask, with the tool to;
    # after asking, and only on that turn's later step, it read when it runs
    # again — never stored with the call's result.
    read = provider.read(_WEB)
    assert len(read) == 3
    assert _spawned(chat.turn_id, asks=True) in read[0]
    assert MessageParentAbility.NAME in provider.offered(_WEB)[0]
    assert [_FOLLOW_UP in r for r in read] == [False, True, False]
    assert _FOLLOW_UP not in cast("str", asked["result"])
    assert _input_rows(db, searcher, turn) == [(_TASK, False, str(spawn_id))]
    assert execution(searcher, turn).state == TurnExecution.COMPLETED

    # The reply opened the search's next turn on its own channel: the row
    # carries the chat's call and the reply alone, and the search read its
    # first turn and then the reply, bare.
    again = child(db, _WEB_CHANNEL, sent_id)
    assert again.channel == searcher and again.turn_id > turn
    assert _input_rows(db, searcher, again.turn_id) == [(_REPLY, False, str(sent_id))]
    assert _executions(searcher, again.turn_id) == 1
    assert execution(searcher, again.turn_id).state == TurnExecution.COMPLETED
    assert f"{_WEB}: {_TASK}\n" in read[2] and f"Assistant: {_WAITS}\n" in read[2]
    assert f"Research query:\n{_REPLY}\n" in read[2]
    assert "Message from Chalie" not in read[2] and "You have been spawned" not in read[2]
    agent = cast("Agent", Agent.by_id(n))
    assert (agent.status, agent.turn_id) == (Agent.DONE, again.turn_id)
    assert _statuses(listener.frames, n) == [Agent.RUNNING, Agent.DONE, Agent.RUNNING, Agent.DONE]

    # The message opened a turn of its own on the chat, carrying its own call.
    [message] = replies(db, cast("int", asked["id"]))
    assert (message["content"], message["joined"]) == (_QUESTION, 0)
    assert message["turn_id"] != chat.turn_id
    # The chat's model read it labelled, with the nudge, and was offered the
    # tool to reply on that turn — never on the turn that spawned it.
    chat_read = provider.read("chat")
    offered = provider.offered("chat")
    assert len(chat_read) == 5
    labelled = f"Message from {_WEB} #{n}: {_QUESTION}"
    assert f"{labelled}\n{PARENT_NUDGE}" in chat_read[2]
    assert [MessageSubagentAbility.NAME in tools for tools in offered] == [False, False, True, True, False]
    # Replied to, the label stays and the nudge is gone.
    assert labelled in chat_read[3] and _sent_to(n) in chat_read[3]
    assert sum(PARENT_NUDGE in r for r in chat_read) == 1

    # What the person sees carries only its text and which subagent sent it.
    opener = messages(block(client, f"/api/threads/{message['turn_id']}"))[0]
    assert opener["content"] == _QUESTION
    assert opener["reply_from"] == {"tool_name": _WEB, "summary": _SUMMARY, "kind": "message"}
    assert opener.get("speaker") is None
    frames = json.dumps(listener.frames)
    assert PARENT_NUDGE not in frames and f"Message from {_WEB}" not in frames
    assert _FOLLOW_UP not in frames

    # Only the second turn's answer came back, on a chat turn of its own: the
    # first turn ended on its question and posted nothing.
    [answer] = replies(db, spawn_id)
    assert (answer["content"], answer["joined"]) == (_ANSWER, 0)
    assert answer["turn_id"] not in (chat.turn_id, message["turn_id"])
    assert f"Reply from {_WEB} #{n}: {_ANSWER}" in chat_read[4]


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
    kind = _SUBTASK if tool_name == SubtaskAbility.NAME else delegate_channel(tool_name)
    channel, turn = child(db, kind, spawn_id)
    assert _input_rows(db, channel, turn) == [(_TASK, False, str(spawn_id))]
    assert [r["content"] for r in replies(db, spawn_id)] == [_ANSWER]


def test_a_message_sent_while_the_chats_turn_works_queues_and_opens_its_next_turn(db: sqlite3.Connection) -> None:
    """The search asks while the chat's turn is still working, so the message
    waits in the chat's queue and opens the chat's next turn once that turn
    ends. That turn reads it with the nudge and is offered
    ``message_subagent``; the chat replies, the nudge is gone, and the
    search's answer comes back on a turn of its own."""
    seen: dict[str, int] = {}
    asked_for = threading.Event()
    asked = threading.Event()
    queued: list[tuple[str, int | None]] = []

    def wait_for_the_message() -> ProviderResponse:
        asked_for.set()
        wait(asked, "the search's step after it asked")
        queued.extend((q.text, q.tool_call_id) for q in QueuedMessage.filter("channel", _USER).get())
        return text("Searching now.")

    def ask_once_the_chat_waits() -> ProviderResponse:
        wait(asked_for, "the chat's second step")
        return _ask()

    def having_asked() -> ProviderResponse:
        asked.set()
        return text(_WAITS)

    provider = ChannelProvider(
        chat=[_spawn(), wait_for_the_message, lambda: _reply_to_it(seen), text("Told it."), text("Noted.")],
        web_search=[ask_once_the_chat_waits, having_asked, text(_ANSWER)],
    )

    with listening() as listener, patched(provider):
        chat = begin(UserConfig(), "find the bridge news")
        chat.result()

    assert provider.unused() == {}
    assert [f for f in listener.frames if f.get("type") == "permission_request"] == []
    n = seen["n"]
    [spawn_call] = calls_of(db, _WEB)
    [asked_call] = calls_of(db, MessageParentAbility.NAME)
    asked_id = cast("int", asked_call["id"])
    assert _answer(asked_call) == ("success", None, _SENT)
    # The message waited in the chat's queue while its turn worked, and left
    # it to open the chat's next turn.
    assert queued == [(_QUESTION, asked_id)]
    assert QueuedMessage.filter("channel", _USER).get() == []
    [message] = replies(db, asked_id)
    assert (message["content"], message["joined"]) == (_QUESTION, 0)
    assert message["turn_id"] > chat.turn_id
    assert _executions(_USER, chat.turn_id) == 1
    [answer] = replies(db, cast("int", spawn_call["id"]))
    assert (answer["content"], answer["joined"]) == (_ANSWER, 0)
    assert answer["turn_id"] not in (chat.turn_id, message["turn_id"])

    chat_read = provider.read("chat")
    offered = provider.offered("chat")
    assert len(chat_read) == 5
    labelled = f"Message from {_WEB} #{n}: {_QUESTION}"
    assert all(labelled not in r for r in chat_read[:2])
    assert f"{labelled}\n{PARENT_NUDGE}" in chat_read[2]
    assert [MessageSubagentAbility.NAME in tools for tools in offered] == [False, False, True, True, False]
    assert labelled in chat_read[3] and PARENT_NUDGE not in chat_read[3]
    assert f"Reply from {_WEB} #{n}: {_ANSWER}" in chat_read[4]
    assert sum(PARENT_NUDGE in r for r in chat_read) == 1
    assert cast("Agent", Agent.by_id(n)).status == Agent.DONE


def test_stopping_a_subagent_that_waits_for_a_reply_ends_it_and_sends_nothing_back(
    db: sqlite3.Connection, real_memory_step: None,
) -> None:
    """The search asks and its turn ends; the chat's turn its message opened
    ends without a reply, so the search still waits and the chat's next turn
    reads the message, nudge and all, among its previous messages. The memory
    step that turn's settle ran while the search waited read the message but
    never the nudge, since what it reads goes into stored prose. The chat's
    next turn stops the search: it ends cancelled, nothing is sent back, and
    the chat's next read carries no nudge."""
    asking = Held(_ask())
    unanswered = Held(text("Let me think about that."))
    unanswered.gate.set()
    seen: dict[str, int] = {}

    provider = ChannelProvider(
        chat=[
            _spawn(),
            text("Started."),
            unanswered,
            calls(tool(FindToolsAbility.NAME, query=[StopSubagentAbility.NAME])),
            lambda: _stop_it(seen),
            text("Stopped it."),
        ],
        web_search=[asking, text(_WAITS)],
    )

    with listening() as listener, patched(provider):
        begin(UserConfig(), "find the bridge news").result()
        asking.gate.set()
        wait(unanswered.entered, "the chat's turn the message opened")
        cast("threading.Thread", unanswered.thread).join(timeout=WAIT_S)
        cast("threading.Thread", asking.thread).join(timeout=WAIT_S)
        for step in [t for t in turn_threads() if t.name.startswith("memory-step-")]:
            step.join(timeout=WAIT_S)
        begin(UserConfig(), "stop the search").result()

    assert asking.opened and provider.unused() == {}
    assert [f for f in listener.frames if f.get("type") == "permission_request"] == []
    n = seen["n"]
    [spawn] = calls_of(db, _WEB)
    searcher, turn = child(db, _WEB_CHANNEL, cast("int", spawn["id"]))
    [stopped] = calls_of(db, StopSubagentAbility.NAME)
    assert _answer(stopped) == ("success", None, f"Subagent {_WEB} #{n} stopped.")
    [asked] = calls_of(db, MessageParentAbility.NAME)
    assert _answer(asked) == ("success", None, _SENT)

    chat_read = provider.read("chat")
    # The chat's next turn read the unanswered message, nudge and all.
    assert f"Message from {_WEB} #{n}: {_QUESTION} {PARENT_NUDGE}" in chat_read[3]
    # Stopped, the message stays and the nudge is gone.
    assert f"Message from {_WEB} #{n}: {_QUESTION}" in chat_read[5]
    assert PARENT_NUDGE not in chat_read[5]
    remembered = provider.read("background")
    assert any(_QUESTION in r for r in remembered)
    assert all(PARENT_NUDGE not in r for r in remembered)

    assert len(provider.read(_WEB)) == 2
    assert cast("Agent", Agent.by_id(n)).status == Agent.CANCELLED
    assert _statuses(listener.frames, n) == [Agent.RUNNING, Agent.DONE, Agent.CANCELLED]
    assert execution(searcher, turn).state == TurnExecution.COMPLETED
    assert replies(db, cast("int", spawn["id"])) == []


def test_a_message_the_person_sends_while_a_subagent_waits_is_followed_by_the_waiting_line(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object], real_memory_step: None,
) -> None:
    """The search asks and the chat's turn its message opened ends without a
    reply, so the search still waits. The person then says "Hi", and adds a
    message that joins that turn while it works: the chat's model reads each
    followed by the line naming the waiting search, while the stored rows, the
    thread read over HTTP, every websocket frame and what the memory step
    reads keep only the person's own text. Before the search asked, and once
    it is stopped, the person's message reads with no line."""
    client, db, _store = authed_client
    asking = Held(_ask())
    unanswered = Held(text("Let me think about that."))
    unanswered.gate.set()
    greeting = Held(text("Hello!"))
    seen: dict[str, int] = {}

    provider = ChannelProvider(
        chat=[
            _spawn(),
            text("Started."),
            unanswered,
            greeting,
            text("Hello! Noted."),
            calls(tool(FindToolsAbility.NAME, query=[StopSubagentAbility.NAME])),
            lambda: _stop_it(seen),
            text("Stopped it."),
            text("Hi there."),
        ],
        web_search=[asking, text(_WAITS)],
    )

    with listening() as listener, patched(provider):
        begin(UserConfig(), "find the bridge news").result()
        asking.gate.set()
        wait(unanswered.entered, "the chat's turn the message opened")
        cast("threading.Thread", unanswered.thread).join(timeout=WAIT_S)
        cast("threading.Thread", asking.thread).join(timeout=WAIT_S)
        hi = begin(UserConfig(), "Hi")
        wait(greeting.entered, "the chat's turn the person said Hi in")
        joined = client.post(f"/api/threads/{hi.turn_id}", data={"text": "one more thing"})
        assert joined.status_code == 200, joined.get_data(as_text=True)
        greeting.gate.set()
        hi.result()
        begin(UserConfig(), "stop the search").result()
        begin(UserConfig(), "Hi again").result()
        drain()

    assert asking.opened and provider.unused() == {}
    n = seen["n"]
    line = WAITING_NOTICE.format(agents=f"{_WEB} #{n}")
    chat_read = provider.read("chat")
    # Nobody waited yet when the person asked for the news.
    assert "Subagents waiting" not in chat_read[0]
    # "Hi" reads followed by the line, ending the person's message; only the
    # turn's own blocks, such as recalled memory, come after it. The message
    # that joined the turn reads followed by it too, on the turn's next step.
    said_hi, read_joined = [i for i, r in enumerate(chat_read) if f"user: Hi\n{line}\n" in r]
    assert chat_read[said_hi].count(line) == 1
    assert _the_one(chat_read, f"user: one more thing\n{line}") == read_joined
    assert chat_read[read_joined].count(line) == 2
    # Every read of the stop turn's message carries it until the stop lands,
    # and none after; the person's next message reads with no line.
    stopped = _the_one(chat_read, f"Subagent {_WEB} #{n} stopped.")
    waited = [i for i, r in enumerate(chat_read) if "Subagents waiting" in r]
    assert waited == list(range(said_hi, stopped))
    assert chat_read[stopped - 1].count(f"user: stop the search\n{line}\n") == 1
    hi_again = _the_one(chat_read, "user: Hi again")
    assert "Subagents waiting" not in chat_read[hi_again]

    # What the person sees, and what the memory step stores, keep only "Hi".
    assert [r["content"] for r in db.execute(
        "SELECT content FROM transcript WHERE channel = ? AND turn_id = ? AND role = 'user' ORDER BY id",
        (_USER, hi.turn_id),
    )] == ["Hi", "one more thing"]
    thread = messages(block(client, f"/api/threads/{hi.turn_id}"))
    assert [m["content"] for m in thread if m["role"] == "user"] == ["Hi", "one more thing"]
    assert "Subagents waiting" not in json.dumps(listener.frames)
    assert all("Subagents waiting" not in r for r in provider.read("background"))
    assert cast("Agent", Agent.by_id(n)).status == Agent.CANCELLED


def test_the_waiting_line_is_gone_from_the_step_after_the_chat_replies(db: sqlite3.Connection) -> None:
    """The search asks and still waits. The person tells the chat what to
    answer: that message reads followed by the line, the chat replies with
    ``message_subagent``, and its very next step reads no line — the reply has
    opened the search's next turn. The search then answers with the reply."""
    asking = Held(_ask())
    unanswered = Held(text("Let me think about that."))
    unanswered.gate.set()
    seen: dict[str, int] = {}
    provider = ChannelProvider(
        chat=[_spawn(), text("Started."), unanswered, lambda: _reply_to_it(seen), text("Told it."), text("Noted.")],
        web_search=[asking, text(_WAITS), text(_ANSWER)],
    )

    with patched(provider):
        begin(UserConfig(), "find the bridge news").result()
        asking.gate.set()
        wait(unanswered.entered, "the chat's turn the message opened")
        cast("threading.Thread", unanswered.thread).join(timeout=WAIT_S)
        cast("threading.Thread", asking.thread).join(timeout=WAIT_S)
        begin(UserConfig(), "tell it the old stone bridge").result()
        drain()

    assert asking.opened and provider.unused() == {}
    n = seen["n"]
    chat_read = provider.read("chat")
    told = _the_one(chat_read, f"user: tell it the old stone bridge\n{WAITING_NOTICE.format(agents=f'{_WEB} #{n}')}\n")
    assert [i for i, r in enumerate(chat_read) if "Subagents waiting" in r] == [told]
    assert _sent_to(n) in chat_read[told + 1]
    [spawn] = calls_of(db, _WEB)
    assert [r["content"] for r in replies(db, cast("int", spawn["id"]))] == [_ANSWER]
    assert cast("Agent", Agent.by_id(n)).status == Agent.DONE


def test_the_persons_stop_of_the_waiting_subagent_tells_the_chat_it_was_cancelled(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """While the chat's turn the message opened is working, and the search's
    turn is still on its step after asking, the person stops the search. It
    ends cancelled, and the chat is told on its next turn that the session
    was cancelled, with no nudge."""
    client, db, _store = authed_client
    asking = Held(_ask())
    after_asking = Held(text(_WAITS))
    reading = Held(text("Let me check."))
    provider = ChannelProvider(
        chat=[_spawn(), text("Started."), reading, text("Okay.")],
        web_search=[asking, after_asking],
    )

    with listening(), patched(provider):
        chat = begin(UserConfig(), "find the bridge news")
        chat.result()
        asking.gate.set()
        wait(reading.entered, "the chat's turn the message opened")
        wait(after_asking.entered, "the search's step after it asked")
        [spawn] = calls_of(db, _WEB)
        searcher, turn = child(db, _WEB_CHANNEL, cast("int", spawn["id"]))
        stop = client.delete(f"/api/threads/{turn}?channel={searcher}")
        assert stop.status_code == 200 and (stop.get_json() or {})["result"]["cancelled"] is True
        after_asking.gate.set()
        cast("threading.Thread", after_asking.thread).join(timeout=WAIT_S)
        reading.gate.set()

    assert asking.opened and after_asking.opened and reading.opened and provider.unused() == {}
    n = agent_number(searcher)
    [asked] = calls_of(db, MessageParentAbility.NAME)
    assert _answer(asked) == ("success", None, _SENT)
    assert len(provider.read(_WEB)) == 2
    assert cast("Agent", Agent.by_id(n)).status == Agent.CANCELLED
    assert execution(searcher, turn).state == TurnExecution.CANCELLED

    [message] = replies(db, cast("int", asked["id"]))
    [told] = replies(db, cast("int", spawn["id"]))
    assert (told["content"], told["joined"]) == (CANCELLED_TEXT, 0)
    assert told["turn_id"] > message["turn_id"]
    chat_read = provider.read("chat")
    assert len(chat_read) == 4
    assert f"Message from {_WEB} #{n}: {_QUESTION}\n{PARENT_NUDGE}" in chat_read[2]
    assert f"Reply from {_WEB} #{n}: {CANCELLED_TEXT}" in chat_read[3]
    assert PARENT_NUDGE not in chat_read[3]


def test_the_persons_stop_of_the_chat_turn_a_message_opened_leaves_the_subagent_waiting(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """The chat's turn has ended when the search asks, so its message opens a
    chat turn of its own, and the search's turn ends. While that chat turn
    works, the person stops it. The search is left alone: it stays done,
    nothing is sent back, and the person's next message reads followed by the
    line naming it as waiting."""
    client, db, _store = authed_client
    asking = Held(_ask())
    reading = Held(text("Let me check."))
    provider = ChannelProvider(
        chat=[_spawn(), text("Started."), reading, text("Hello.")],
        web_search=[asking, text(_WAITS)],
    )

    with listening() as listener, patched(provider):
        chat = begin(UserConfig(), "find the bridge news")
        chat.result()
        asking.gate.set()
        wait(reading.entered, "the chat's turn the message opened")
        cast("threading.Thread", asking.thread).join(timeout=WAIT_S)
        [asked] = calls_of(db, MessageParentAbility.NAME)
        [message] = replies(db, cast("int", asked["id"]))
        opened = cast("int", message["turn_id"])
        stop = client.delete(f"/api/threads/{opened}")
        assert stop.status_code == 200 and (stop.get_json() or {})["result"]["cancelled"] is True
        reading.gate.set()
        cast("threading.Thread", reading.thread).join(timeout=WAIT_S)
        begin(UserConfig(), "Hi").result()

    assert asking.opened and reading.opened and provider.unused() == {}
    assert opened != chat.turn_id
    assert execution(_USER, opened).state == TurnExecution.CANCELLED
    [spawn] = calls_of(db, _WEB)
    searcher, turn = child(db, _WEB_CHANNEL, cast("int", spawn["id"]))
    n = agent_number(searcher)
    assert _answer(asked) == ("success", None, _SENT)
    assert cast("Agent", Agent.by_id(n)).status == Agent.DONE
    assert _statuses(listener.frames, n) == [Agent.RUNNING, Agent.DONE]
    assert execution(searcher, turn).state == TurnExecution.COMPLETED
    assert replies(db, cast("int", spawn["id"])) == []
    chat_read = provider.read("chat")
    assert len(chat_read) == 4
    assert f"Message from {_WEB} #{n}: {_QUESTION}\n{PARENT_NUDGE}" in chat_read[2]
    assert f"user: Hi\n{WAITING_NOTICE.format(agents=f'{_WEB} #{n}')}" in chat_read[3]


def test_the_persons_stop_of_the_chat_turn_after_it_replied_stops_the_turn_its_reply_opened(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """The search's message opens a chat turn, and the chat replies in it; the
    reply opens the search's next turn. While that chat turn and the search's
    new turn are both still working, the person stops the chat turn. The stop
    reaches the turn its reply opened: the search ends cancelled with its
    caller, and nothing is sent back."""
    client, db, _store = authed_client
    asking = Held(_ask())
    noting = Held(text("Noted."))
    finishing = Held(text(_ANSWER))
    seen: dict[str, int] = {}
    provider = ChannelProvider(
        chat=[_spawn(), text("Started."), lambda: _reply_to_it(seen), noting],
        web_search=[asking, text(_WAITS), finishing],
    )

    with listening() as listener, patched(provider):
        chat = begin(UserConfig(), "find the bridge news")
        chat.result()
        asking.gate.set()
        wait(noting.entered, "the chat's turn after its reply")
        wait(finishing.entered, "the search's turn the reply opened")
        [asked] = calls_of(db, MessageParentAbility.NAME)
        [message] = replies(db, cast("int", asked["id"]))
        opened = cast("int", message["turn_id"])
        n = seen["n"]
        assert cast("Agent", Agent.by_id(n)).status == Agent.RUNNING
        stop = client.delete(f"/api/threads/{opened}")
        assert stop.status_code == 200 and (stop.get_json() or {})["result"]["cancelled"] is True
        noting.gate.set()
        finishing.gate.set()
        cast("threading.Thread", finishing.thread).join(timeout=WAIT_S)

    assert asking.opened and noting.opened and finishing.opened and provider.unused() == {}
    assert opened != chat.turn_id
    assert execution(_USER, opened).state == TurnExecution.CANCELLED
    [spawn] = calls_of(db, _WEB)
    searcher, turn = child(db, _WEB_CHANNEL, cast("int", spawn["id"]))
    assert agent_number(searcher) == n
    assert execution(searcher, turn).state == TurnExecution.COMPLETED
    [sent] = calls_of(db, MessageSubagentAbility.NAME)
    again = child(db, _WEB_CHANNEL, cast("int", sent["id"]))
    assert again.channel == searcher and again.turn_id > turn
    ended = execution(searcher, again.turn_id)
    assert (ended.state, ended.stop_reason) == (TurnExecution.CANCELLED, STOP_REASON_PARENT_STOPPED)
    assert cast("Agent", Agent.by_id(n)).status == Agent.CANCELLED
    assert _statuses(listener.frames, n) == [Agent.RUNNING, Agent.DONE, Agent.RUNNING, Agent.CANCELLED]
    assert replies(db, cast("int", spawn["id"])) == []


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
    kind = _SUBTASK if tool_name == SubtaskAbility.NAME else delegate_channel(tool_name)
    agent = cast("Agent", Agent.by_id(agent_number(child(db, kind, cast("int", spawn["id"])).channel)))
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
    ``message_parent`` call, and tell the chat the session crashed; the
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
        chat=[_spawn(), text("Started."), text("It crashed.")],
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
    searcher, turn = child(db, _WEB_CHANNEL, cast("int", spawn["id"]))
    n = agent_number(searcher)
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
    cut_off = execution(searcher, turn)
    assert (cut_off.state, cut_off.stop_reason) == (TurnExecution.CRASHED, STOP_REASON_PROCESS_DEATH)
    assert execution(_USER, message["turn_id"]).state == TurnExecution.CRASHED
    [told] = replies(db, cast("int", spawn["id"]))
    assert told["content"] == CRASHED_TEXT
    chat_read = provider.read("chat")
    assert len(chat_read) == 3
    assert f"Reply from {_WEB} #{n}: {CRASHED_TEXT}" in chat_read[2]
    assert all(PARENT_NUDGE not in r for r in chat_read)
