"""Feature tests: a subagent the person's chat spawned waits for the chat's
reply only while the turn it asked in ended on that question.

* A turn that asks with ``message_parent`` and then goes on working ends its
  subagent done, and its answer reaches the chat; a ``message_parent`` the
  turn was refused sends nothing, so the subagent ends done and its answer is
  posted. A turn that asks and then crashes ends its subagent failed, and the
  chat is told the session crashed.
* A reply the chat sends while the subagent's asking turn still works waits
  behind it; that turn's end opens the subagent's next turn, moving it from
  waiting to running.
* The next boot's sweep leaves a waiting subagent waiting: no turn of its was
  cut off, so nothing ends it and nothing is posted, and the chat lists it as
  waiting and reads its status as waiting.
* A chat turn reads the reminder naming, oldest first, every subagent that
  waits and whose message has reached the chat — not one whose message still
  waits behind a working turn, and not the one whose message opened the turn.
* Once the row its turn ended on is older than the archive age, a waiting
  subagent drops out of the reminder and out of ``GET /api/agents/all``.
* The hand-over a turn writes when its request outgrows the window, and the
  compaction it sets off, never carry the reminder, though every step of
  that turn reads it.

Drives real turns against the real, fully migrated SQLite DB, observed through
the real socket registry and read back through the real Flask app. The only
substitution is the LLM network boundary
(``services.provider_service.build_client``), answered from one script per
channel. A step is held on an event while another turn acts, never timed with
a sleep.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import timedelta
from typing import cast

import pytest
from flask.testing import FlaskClient

from abilities._agents import label
from abilities._turn_handover_config import TurnHandoverConfig
from abilities.agent_status import AgentStatusAbility
from abilities.list_agents import ListAgentsAbility
from abilities.message_parent import WAITING_REMINDER, MessageParentAbility
from abilities.message_subagent import MessageSubagentAbility
from abilities.recall import Recall
from abilities.subtask import SubtaskAbility
from configs.channels import config_for
from configs.channels.user import UserConfig
from configs.enums.config_type import ConfigTypeEnum
from controllers.message_processor import _MAX_PROVIDER_ATTEMPTS, MessageProcessor
from exceptions import ContextLimit
from models.agent import Agent
from models.provider_response import ProviderResponse
from models.queued_message import QueuedMessage
from models.tool_call import ToolCall
from models.transcript import Transcript
from models.turn_execution import TurnExecution
from run import _backfill_redesign_columns, _reply_to_swept_delegates
from services.agent_service import ARCHIVE_AFTER, CRASHED_TEXT
from services.time_utils import utc_now
from tests._delegates import (
    WAIT_S,
    ChannelProvider,
    Held,
    begin,
    calls,
    calls_of,
    child,
    drain,
    execution,
    listening,
    patched,
    raises,
    replies,
    text,
    tool,
    wait,
)
from tests.test_agent_tools import _answer, _find_agent_tools, _input_rows, _statuses
from tests.test_agents_table import _empty_listing, _listing
from tests.test_message_parent import (
    _ANSWER,
    _QUESTION,
    _REPLY,
    _SENT,
    _SUMMARY,
    _USER,
    _WAITS,
    _WEB,
    _WEB_CHANNEL,
    _ask,
    _carrying,
    _reminds,
    _reply_to_it,
    _spawn,
    _the_one,
    _the_search,
)

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider", "earlier_agents")]

_FERRY = "Which ferry, the morning or the evening one?"


def _named(read: str) -> str | None:
    """Who the waiting reminder in *read* names, as the model reads it
    (``"web_search #101, subtask #102"``) — ``None`` when it carries none."""
    head, tail = WAITING_REMINDER.split("{agents}")
    if head not in read:
        return None
    return read.split(head, 1)[1].split(tail, 1)[0]


def _queued(channel: str) -> list[tuple[str, int | None]]:
    """Every message waiting on *channel*: its text and the call it carries."""
    return [(q.text, q.tool_call_id) for q in QueuedMessage.filter("channel", channel).get()]


def _left_waiting(asking: Held, unanswered: Held) -> int:
    """Run the chat's spawn turn, release the search's question, and let the
    chat turn it opens end without a reply — leaving the search waiting.
    Returns its number."""
    begin(UserConfig(), "find the bridge news").result()
    asking.gate.set()
    wait(unanswered.entered, "the chat's turn the message opened")
    cast("threading.Thread", unanswered.thread).join(timeout=WAIT_S)
    cast("threading.Thread", asking.thread).join(timeout=WAIT_S)
    n = _the_search().agent_id
    assert cast("Agent", Agent.by_id(n)).status == Agent.WAITING
    return n


def test_a_subagent_that_asks_then_works_on_and_answers_is_done_and_its_answer_reaches_the_chat(
    db: sqlite3.Connection,
) -> None:
    """The search asks the chat, then goes on in the same turn: it recalls and
    answers. Its turn did not end on the question, so nothing waits on a
    reply: the search is done, its answer comes back to the chat on a turn of
    its own, and no chat turn reads the reminder."""
    asking = Held(_ask())
    provider = ChannelProvider(
        chat=[_spawn(), text("Started."), text("I'll wait for its answer."), text("Noted.")],
        web_search=[asking, calls(tool(Recall.NAME, query=_REPLY)), text(_ANSWER)],
    )

    with listening() as listener, patched(provider):
        begin(UserConfig(), "find the bridge news").result()
        asking.gate.set()
        drain()

    assert asking.opened and provider.unused() == {}
    [spawn] = calls_of(db, _WEB)
    spawn_id = cast("int", spawn["id"])
    searcher, turn = child(db, _WEB_CHANNEL, spawn_id)
    [asked] = calls_of(db, MessageParentAbility.NAME)
    assert _answer(asked) == ("success", None, _SENT)
    assert [c.tool_name for c in ToolCall.by_turn(searcher, turn)][-2:] == [MessageParentAbility.NAME, Recall.NAME]
    n = _the_search().agent_id
    agent = cast("Agent", Agent.by_id(n))
    assert (agent.status, agent.turn_id) == (Agent.DONE, turn)
    assert _statuses(listener.frames, n) == [Agent.RUNNING, Agent.DONE]
    [answer] = replies(db, spawn_id)
    assert answer["content"] == _ANSWER
    chat_read = provider.read("chat")
    assert len(chat_read) == 4
    _the_one(chat_read, f"Reply from {_WEB} #{n}: {_ANSWER}")
    assert not any(_reminds(r) for r in chat_read)


def test_a_turn_that_ends_on_a_refused_message_parent_ends_its_subagent_done_and_posts_its_answer(
    db: sqlite3.Connection,
) -> None:
    """The search calls ``message_parent`` with no message, which is refused,
    and then answers. The refused call sent nothing — no row anywhere carries
    it — so there is no reply to wait for: the search is done and its answer
    comes back to the chat."""
    provider = ChannelProvider(
        chat=[_spawn(), text("Started."), text("Noted.")],
        web_search=[calls(tool(MessageParentAbility.NAME)), text(_ANSWER)],
    )

    with listening() as listener, patched(provider):
        begin(UserConfig(), "find the bridge news").result()
        drain()

    assert provider.unused() == {}
    [spawn] = calls_of(db, _WEB)
    spawn_id = cast("int", spawn["id"])
    searcher, turn = child(db, _WEB_CHANNEL, spawn_id)
    [refused] = calls_of(db, MessageParentAbility.NAME)
    assert refused["state"] == ToolCall.ERROR
    assert _answer(refused)[0] == "error"
    assert _carrying(db, cast("int", refused["id"])) == 0
    assert ToolCall.by_turn(searcher, turn)[-1].tool_name == MessageParentAbility.NAME
    n = _the_search().agent_id
    agent = cast("Agent", Agent.by_id(n))
    assert (agent.status, agent.turn_id) == (Agent.DONE, turn)
    assert _statuses(listener.frames, n) == [Agent.RUNNING, Agent.DONE]
    [answer] = replies(db, spawn_id)
    assert answer["content"] == _ANSWER
    _the_one(provider.read("chat"), f"Reply from {_WEB} #{n}: {_ANSWER}")


def test_a_turn_that_asks_and_then_crashes_ends_its_subagent_failed_and_tells_the_chat(
    db: sqlite3.Connection,
) -> None:
    """The search asks the chat, then its provider fails every attempt. The
    turn crashed, so the question it sent leaves nothing to wait for: the
    search ends failed, the chat is told the session crashed, and no chat turn
    reads the reminder."""
    asking = Held(_ask())
    provider = ChannelProvider(
        chat=[_spawn(), text("Started."), text("Let me check."), text("Okay.")],
        web_search=[asking, *(raises(RuntimeError("the provider is down")) for _ in range(_MAX_PROVIDER_ATTEMPTS))],
    )

    with listening() as listener, patched(provider):
        begin(UserConfig(), "find the bridge news").result()
        asking.gate.set()
        drain()

    assert asking.opened and provider.unused() == {}
    [spawn] = calls_of(db, _WEB)
    spawn_id = cast("int", spawn["id"])
    searcher, turn = child(db, _WEB_CHANNEL, spawn_id)
    [asked] = calls_of(db, MessageParentAbility.NAME)
    assert _answer(asked) == ("success", None, _SENT)
    assert execution(searcher, turn).state == TurnExecution.CRASHED
    n = _the_search().agent_id
    assert cast("Agent", Agent.by_id(n)).status == Agent.FAILED
    assert _statuses(listener.frames, n) == [Agent.RUNNING, Agent.FAILED]
    [told] = replies(db, spawn_id)
    assert told["content"] == CRASHED_TEXT
    chat_read = provider.read("chat")
    _the_one(chat_read, f"Reply from {_WEB} #{n}: {CRASHED_TEXT}")
    assert not any(_reminds(r) for r in chat_read)


def test_a_reply_sent_while_the_asking_turn_still_works_opens_the_waiting_subagents_next_turn(
    db: sqlite3.Connection,
) -> None:
    """The search asks and is held on its next step. The chat's turn the
    question opened replies at once: the search still runs, so the reply
    queues on its channel. Released, the search's turn ends on its question —
    waiting — and that end opens its next turn from the queue, moving it to
    running; it reads the reply, answers and is done."""
    asking = Held(_ask())
    after_asking = Held(text(_WAITS))
    seen: dict[str, int] = {}
    queued: list[tuple[str, int | None]] = []
    status_then: list[str] = []

    def told() -> ProviderResponse:
        search = _the_search()
        queued.extend(_queued(search.channel))
        status_then.append(search.status)
        after_asking.gate.set()
        return text("Told it.")

    provider = ChannelProvider(
        chat=[_spawn(), text("Started."), lambda: _reply_to_it(seen), told, text("Noted.")],
        web_search=[asking, after_asking, text(_ANSWER)],
    )

    with listening() as listener, patched(provider):
        begin(UserConfig(), "find the bridge news").result()
        asking.gate.set()
        drain()

    assert asking.opened and after_asking.opened and provider.unused() == {}
    n = seen["n"]
    [spawn] = calls_of(db, _WEB)
    spawn_id = cast("int", spawn["id"])
    searcher, turn = child(db, _WEB_CHANNEL, spawn_id)
    [sent] = calls_of(db, MessageSubagentAbility.NAME)
    sent_id = cast("int", sent["id"])
    assert (queued, status_then) == ([(_REPLY, sent_id)], [Agent.RUNNING])
    assert _queued(searcher) == []
    again = child(db, _WEB_CHANNEL, sent_id)
    assert again.channel == searcher and again.turn_id > turn
    assert _input_rows(db, searcher, again.turn_id) == [(_REPLY, False, str(sent_id))]
    agent = cast("Agent", Agent.by_id(n))
    assert (agent.status, agent.turn_id) == (Agent.DONE, again.turn_id)
    assert _statuses(listener.frames, n) == [Agent.RUNNING, Agent.WAITING, Agent.RUNNING, Agent.DONE]
    assert [r["content"] for r in replies(db, spawn_id)] == [_ANSWER]


def test_the_next_boot_leaves_a_waiting_subagent_waiting(db: sqlite3.Connection) -> None:
    """The search asks and waits. What the next boot runs — the backfill, the
    sweep and the notices — cuts off no turn of its: the search still waits,
    no frame moves it, nothing is posted for it, and the person's next message
    reads the reminder naming it. That turn finds the agent tools, lists the
    search as waiting, and reads its status as waiting, with the text its turn
    ended on."""
    asking = Held(_ask())
    unanswered = Held(text("Let me think about that."))
    unanswered.gate.set()

    def check_it() -> ProviderResponse:
        return calls(tool(ListAgentsAbility.NAME), tool(AgentStatusAbility.NAME, agent_id=_the_search().agent_id))

    provider = ChannelProvider(
        chat=[_spawn(), text("Started."), unanswered, _find_agent_tools(), check_it, text("Hello.")],
        web_search=[asking, text(_WAITS)],
    )

    with listening() as listener:
        with patched(provider):
            n = _left_waiting(asking, unanswered)
        with patched(provider):
            _backfill_redesign_columns()
            swept = MessageProcessor(config_for(ConfigTypeEnum.USER)).turn_execution_service.sweep_orphaned()
            _reply_to_swept_delegates(swept)
            drain()
            begin(UserConfig(), "Hi").result()

    assert asking.opened and provider.unused() == {}
    assert swept == []
    agent = cast("Agent", Agent.by_id(n))
    assert agent.status == Agent.WAITING
    assert _statuses(listener.frames, n) == [Agent.RUNNING, Agent.WAITING]
    [spawn] = calls_of(db, _WEB)
    assert replies(db, cast("int", spawn["id"])) == []
    assert [_named(r) for r in provider.read("chat")] == [None, None, None] + [label(agent)] * 3
    [listed] = calls_of(db, ListAgentsAbility.NAME)
    assert _answer(listed)[:2] == ("success", None)
    assert json.loads(_answer(listed)[2]) == [{"agent_id": n, "type": _WEB, "summary": _SUMMARY, "status": Agent.WAITING}]
    [status] = calls_of(db, AgentStatusAbility.NAME)
    assert _answer(status)[:2] == ("success", None)
    assert json.loads(_answer(status)[2]) == {"agent_id": n, "type": _WEB, "status": Agent.WAITING, "last_output": _WAITS}


def test_the_reminder_names_oldest_first_the_waiting_subagents_whose_message_reached_the_chat(
    db: sqlite3.Connection,
) -> None:
    """The chat spawns a search and then a subtask. The subtask asks first: its
    question opens a chat turn, which is held. The search asks next: its
    question queues behind that turn. Both now wait. The person's "Hi" opens a
    turn alongside, which names the subtask only — the search's message has
    not reached the chat. The held turn ends and the search's question opens
    the next: it names the subtask only, since its own opener needs no
    reminding. The person's next message names both, oldest first."""
    search_asks = Held(_ask())
    subtask_asks = Held(calls(tool(MessageParentAbility.NAME, message=_FERRY)))
    reading = Held(text("Let me look."))
    provider = ChannelProvider(
        chat=[
            _spawn(),
            calls(tool(SubtaskAbility.NAME, action="start", instructions="find the ferry times")),
            text("Started both."),
            reading,
            text("Hello."),
            text("Noted."),
            text("Both still wait."),
        ],
        web_search=[search_asks, text(_WAITS)],
        subtask=[subtask_asks, text("I need to know which ferry first.")],
    )

    with patched(provider):
        begin(UserConfig(), "find the bridge news and the ferry times").result()
        wait(subtask_asks.entered, "the subtask's first step")
        subtask_asks.gate.set()
        wait(reading.entered, "the chat's turn the subtask's question opened")
        cast("threading.Thread", subtask_asks.thread).join(timeout=WAIT_S)
        wait(search_asks.entered, "the search's first step")
        search_asks.gate.set()
        cast("threading.Thread", search_asks.thread).join(timeout=WAIT_S)
        queued = _queued(_USER)
        begin(UserConfig(), "Hi").result()
        reading.gate.set()
        drain()
        begin(UserConfig(), "And now?").result()

    assert subtask_asks.opened and search_asks.opened and reading.opened and provider.unused() == {}
    search, subtask = Agent.filter("spawned_by_channel", _USER).order_by("agent_id ASC").get()
    assert (search.type, subtask.type) == (_WEB, SubtaskAbility.NAME)
    assert (search.status, subtask.status) == (Agent.WAITING, Agent.WAITING)
    subtask_asked, search_asked = calls_of(db, MessageParentAbility.NAME)
    assert queued == [(_QUESTION, cast("int", search_asked["id"]))]
    assert _queued(_USER) == []
    assert _carrying(db, cast("int", subtask_asked["id"])) == 1
    assert [_named(r) for r in provider.read("chat")] == [
        None, None, None, None,
        label(subtask), label(subtask),
        f"{label(search)}, {label(subtask)}",
    ]


def test_an_archived_waiting_subagent_leaves_the_reminder_and_the_agents_listing(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """The search asks and waits: the listing holds it as waiting, and the
    person's "Hi" reads the reminder naming it. Once the row its turn ended on
    is older than the archive age, the listing is empty and the person's next
    message reads no reminder."""
    client, _db, _store = authed_client
    asking = Held(_ask())
    unanswered = Held(text("Let me think about that."))
    unanswered.gate.set()
    provider = ChannelProvider(
        chat=[_spawn(), text("Started."), unanswered, text("Hello."), text("Hello again.")],
        web_search=[asking, text(_WAITS)],
    )

    with patched(provider):
        n = _left_waiting(asking, unanswered)
        while_waiting = [(e["agent_id"], e["status"]) for e in cast("list[dict[str, object]]", _listing(client)["result"])]
        begin(UserConfig(), "Hi").result()
        agent = cast("Agent", Agent.by_id(n))
        assert agent.response_id is not None
        aged = utc_now() - ARCHIVE_AFTER - timedelta(days=1)
        assert Transcript.filter("id", agent.response_id).update(created_at=aged.strftime("%Y-%m-%d %H:%M:%S")) == 1
        archived = _listing(client)
        begin(UserConfig(), "Hi again").result()

    assert asking.opened and provider.unused() == {}
    assert while_waiting == [(n, Agent.WAITING)]
    assert archived == _empty_listing()
    assert cast("Agent", Agent.by_id(n)).status == Agent.WAITING
    assert [_named(r) for r in provider.read("chat")] == [None, None, None, label(agent), None]


def test_the_hand_over_of_a_turn_that_outgrows_the_window_never_carries_the_reminder(
    db: sqlite3.Connection,
) -> None:
    """The search asks and waits, and the person's "Hi" opens a turn that
    reads the reminder. Its first request outgrows the window: the turn
    writes its hand-over and sends again, the reminder still after its input
    line. The hand-over read the person's "Hi", but neither background read
    — the hand-over, nor the compaction it sets off — carries the reminder."""
    asking = Held(_ask())
    unanswered = Held(text("Let me think about that."))
    unanswered.gate.set()
    provider = ChannelProvider(
        chat=[
            _spawn(),
            text("Started."),
            unanswered,
            raises(ContextLimit("payload exceeds the model's maximum context length")),
            text("Hello."),
        ],
        web_search=[asking, text(_WAITS)],
    )

    with patched(provider):
        n = _left_waiting(asking, unanswered)
        begin(UserConfig(), "Hi").result()
        drain()

    assert asking.opened and provider.unused() == {}
    head = TurnHandoverConfig().system_prompt.splitlines()[0]
    background = list(zip(provider.systems("background"), provider.read("background"), strict=True))
    handed_over = [read for system, read in background if system.startswith(head)]
    assert handed_over and all("user: Hi" in read for read in handed_over)
    assert not any(_reminds(read) for _system, read in background)
    agent = cast("Agent", Agent.by_id(n))
    assert agent.status == Agent.WAITING
    assert [_named(r) for r in provider.read("chat")] == [None, None, None, label(agent), label(agent)]
