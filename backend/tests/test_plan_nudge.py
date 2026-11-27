"""Feature tests: a turn whose exchange would end with its plan still open is
sent the plan nudge as a message, the way a person's message joins a working
turn, and carries on from it.

* The chat, an outside agent, a discovery pass, a scheduled task, a subtask and
  a ``web_search`` subagent each plan their message. One that answers with a
  task still open keeps that answer unsettled, gets the nudge written into the
  turn after it as a joined message — the chat shows it as one — and reads it
  on its next step as a message like any other.
* A nudged exchange is nudged again only once a step after the nudge called a
  tool other than the plan. A step that does no other work — no tool, or only
  plan calls — leaves its next answer as the one the turn ends on.
* A stop during the nudged step ends the turn cancelled with nothing more sent.
* A subagent whose turn ended on a delivered ``message_parent`` waits for its
  reply and is never nudged, its plan still open; until it is archived, the
  chat turn that spawned it is not nudged either.
* A chat turn with a subagent it spawned, or sent a message, still running is
  never nudged, its plan still open: that subagent's answer opens a turn of
  its own.

These drive the real send endpoint (``POST /api/threads/<turn_id>``) or the
real ``MessageProcessor``, the real plan ability through real dispatch, the
real prompt, transcript, tool-call and ``plan_tasks`` tables against the real
migrated SQLite database. The only substitution is the LLM network boundary
(``services.provider_service.build_client``): a scripted double.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Iterator
from datetime import timedelta
from typing import cast

import pytest

from abilities.find_tools import FindToolsAbility
from abilities.message_parent import MessageParentAbility
from abilities.message_subagent import MessageSubagentAbility
from abilities.plan import PlanAbility
from abilities.recall import Recall
from abilities.web_search import WebSearchAbility
from configs.channels.discovery import DiscoveryConfig
from configs.channels.external_agent import EAMPConfig
from configs.channels.scheduled import ScheduledConfig
from configs.channels.subtask import UserSubtaskConfig
from configs.channels.user import UserConfig
from configs.channels.web_search import WebSearchConfig
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import _PLAN_NUDGE, MessageProcessor
from models.agent import Agent
from models.plan_task import PlanTask
from models.provider_request import ProviderRequest
from models.provider_response import ProviderResponse
from models.transcript import Transcript
from models.turn_execution import TurnExecution
from services import provider_service
from services.agent_service import ARCHIVE_AFTER, AgentService
from services.processor_config import ProcessorConfig
from services.time_utils import utc_now
from services.turn_serializer_service import TurnSerializerService
from tests._delegates import (
    EARLIER_AGENTS,
    USER,
    ChannelProvider,
    Entry,
    Held,
    begin,
    calls,
    calls_of,
    child,
    delegate_channel,
    drain,
    execution,
    patched,
    replies,
    text,
    tool,
    turn_threads,
)
from tests.helpers import is_planning_request
from tests.test_turn_join import (
    AuthedClient,
    _drain,
    _executions,
    _label_thread,
    _post,
    _result,
    _say,
)

pytestmark = pytest.mark.unit

_PLANNING = frozenset({Recall.NAME, PlanAbility.NAME, FindToolsAbility.NAME})
_PROSE = "Your flight leaves at 9."
#: Long enough for the first turn of a run, which loads the embedding model.
_SETTLE_S = 60.0


class _NudgeProvider:
    """Replays scripted responses to the turn's own requests — every request
    that offers the plan tool — and records each one's tool names and body.
    Any other call (the thread label, the memory step) gets a benign terminal
    reply and is not recorded, so the script is the turn's alone; so does a
    turn request past the end of the script, which the test then sees as one
    request too many. ``hold_at(n)`` parks the n-th recorded request until
    ``release`` is set — or a stop aborts it — signalling ``entered`` first."""

    _TERMINAL = ProviderResponse(text="done", model="scripted-terminal", tool_calls=None)

    def __init__(self) -> None:
        self._queue: list[ProviderResponse] = []
        self._lock = threading.Lock()
        self._hold_at: int | None = None
        self.requests: list[tuple[frozenset[str], str]] = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def script(self, *responses: ProviderResponse) -> None:
        with self._lock:
            self._queue.extend(responses)

    def hold_at(self, index: int) -> None:
        self._hold_at = index

    def get_context_limit(self) -> int:
        return 200000

    def abort(self) -> None:
        """A stopped turn's watcher aborts its call in flight."""
        self.release.set()

    def send(self, dto: ProviderRequest) -> ProviderResponse:
        names = frozenset(str(t["name"]) for t in dto.tools or [])
        if PlanAbility.NAME not in names:
            return self._TERMINAL
        with self._lock:
            index = len(self.requests)
            self.requests.append((names, str(dto.messages[0]["content"])))
            response = self._queue.pop(0) if self._queue else self._TERMINAL
        if index == self._hold_at:
            self.entered.set()
            assert self.release.wait(10), "the held provider call was never released"
        return response

    @property
    def tools(self) -> list[frozenset[str]]:
        return [names for names, _ in self.requests]

    @property
    def bodies(self) -> list[str]:
        return [body for _, body in self.requests]

    @property
    def nudges(self) -> list[int]:
        """How many times each recorded request carried the nudge."""
        return [body.count(_PLAN_NUDGE) for body in self.bodies]

    @property
    def unused(self) -> int:
        with self._lock:
            return len(self._queue)


@pytest.fixture
def provider(chat_provider: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch) -> Iterator[_NudgeProvider]:
    double = _NudgeProvider()
    monkeypatch.setattr(provider_service, "build_client", lambda *_a, **_k: double)
    yield double
    double.release.set()  # a failed test must not strand a parked drive thread
    _drain(_SETTLE_S)


def _task(title: str, task: str) -> dict[str, str]:
    return {"title": title, "task": task}


def _plan(**params: object) -> dict[str, object]:
    return {"name": PlanAbility.NAME, "input": params}


def _calls(*tool_calls: dict[str, object]) -> ProviderResponse:
    return ProviderResponse(text="", model="scripted", tool_calls=list(tool_calls))


_FLIGHT = _task("Find the flight", "Search the inbox for the booking email")
_ANSWER = _task("Answer", "Tell the user the departure time")
_CREATE = _calls(_plan(action="create", tasks=[_FLIGHT, _ANSWER]))


def _assistant_rows(channel: str, turn_id: int) -> list[tuple[object, object]]:
    return [(r["content"], r["settled"]) for r in Transcript.by_turn(channel, turn_id) if r["role"] == "assistant"]


def _statuses(channel: str, turn_id: int) -> list[str]:
    """The plan's task statuses, in order, on the turn's input — its first row."""
    message = Transcript.by_turn(channel, turn_id)[0]
    return [t.status for t in PlanTask.for_transcript(cast("int", message["id"]))]


def _rows(channel: str, turn_id: int) -> list[tuple[object, ...]]:
    """Every row of the turn after its first input, oldest first: role,
    content, settled, joined."""
    return [
        (r["role"], r["content"], r["settled"], r["joined"]) for r in Transcript.by_turn(channel, turn_id)[1:]
    ]


#: Answers while closing the plan's last open task, which ends the turn.
_CLOSE = ProviderResponse(
    text="It leaves at 9 from gate B12.", model="scripted",
    tool_calls=[_plan(action="update", task=2, status="done")],
)
_RECALL: dict[str, object] = {"name": Recall.NAME, "input": {"query": "flight booking"}}
_RESTARTED = _plan(action="update", task=1, status="in_progress")
_NUDGED = ("user", _PLAN_NUDGE, 0, 1)


# ── 1. the chat is nudged with a message it shows and reads like any other ──


def test_a_chat_turn_left_with_its_plan_open_is_sent_the_nudge_as_a_message_and_carries_on(
    authed_client: AuthedClient, provider: _NudgeProvider,
) -> None:
    """The model answers while the plan's last task is still open. That answer
    is kept unsettled and the nudge is written into the turn after it as a
    joined message, which the thread shows as the person's and the next step
    reads as a ``user`` line after the answer. That step answers while closing
    the plan, which ends the turn. The thread's next message plans nothing and
    is never nudged."""
    client, db, _store = authed_client
    provider.script(_CREATE, _calls(_plan(action="update", task=1, status="done")), _say(_PROSE), _CLOSE)
    turn_id = cast("int", _result(_post(client, -1, "when does my flight leave?"))["turn_id"])
    _drain(_SETTLE_S)

    loop = frozenset(UserConfig().always_available)
    assert provider.tools == [_PLANNING] + [loop] * 3
    assert provider.nudges == [0, 0, 0, 1]
    nudged = provider.bodies[3]
    assert nudged.index(f"[interim_response] {_PROSE}") < nudged.index(f" user: {_PLAN_NUDGE}")
    assert provider.unused == 0
    assert _rows("user", turn_id) == [
        ("assistant", "", 0, 0), ("assistant", _PROSE, 0, 0), _NUDGED, ("assistant", _CLOSE.text, 1, 0),
    ]
    assert _statuses("user", turn_id) == ["done", "done"]
    assert [state for _id, state in _executions(db, turn_id)] == [TurnExecution.COMPLETED]
    messages = cast("list[dict[str, object]]", _result(client.get(f"/api/threads/{turn_id}"))["messages"])
    assert [(m["role"], m["content"], m.get("joined")) for m in messages if m["content"]] == [
        ("user", "when does my flight leave?", False),
        ("assistant", _PROSE, None),
        ("user", _PLAN_NUDGE, True),
        ("assistant", _CLOSE.text, None),
    ]

    _label_thread(turn_id)
    provider.script(_say(""), _say("You're welcome."))
    _result(_post(client, turn_id, "thanks"))
    _drain(_SETTLE_S)

    assert provider.tools[4:] == [_PLANNING, loop]
    assert provider.unused == 0
    assert _rows("user", turn_id)[-1] == ("assistant", "You're welcome.", 1, 0)
    assert [row for row in _rows("user", turn_id) if row[1] == _PLAN_NUDGE] == [_NUDGED]


# ── 2. every other kind of turn that plans is nudged the same way ────────────


_SURFACES = [
    pytest.param(lambda: EAMPConfig("reviewer", "sample-project", False), -1, id="outside-agent"),
    pytest.param(DiscoveryConfig, -1, id="discovery"),
    pytest.param(ScheduledConfig, 7, id="scheduled-task"),
    pytest.param(UserSubtaskConfig, -1, id="subtask"),
    pytest.param(lambda: WebSearchConfig(PolicyChannel.CHAT), -1, id="web-search-subagent"),
]


@pytest.mark.parametrize(("make", "turn_id"), _SURFACES)
def test_every_kind_of_turn_that_plans_is_nudged_to_close_its_plan(
    db: sqlite3.Connection, provider: _NudgeProvider, make: Callable[[], ProcessorConfig], turn_id: int,
) -> None:
    """The kind of turn plans its message, answers with a task still open,
    gets the nudge as a joined message in its own role, which its thread
    shows as input, and on its next step answers while closing the plan; that
    answer is its result."""
    config = make()
    provider.script(_CREATE, _calls(_plan(action="update", task=1, status="done")), _say(_PROSE), _CLOSE)
    mp = MessageProcessor.process(config, raw_input="when does my flight leave?", turn_id=turn_id)
    answer = mp.result()
    _drain(_SETTLE_S)

    pinned = frozenset(config.always_available)
    assert provider.tools == [_PLANNING & pinned] + [pinned] * 3
    assert provider.nudges == [0, 0, 0, 1]
    assert provider.unused == 0
    assert answer == _CLOSE.text
    assert _rows(mp.channel, mp.turn_id) == [
        ("assistant", "", 0, 0), ("assistant", _PROSE, 0, 0), (config.role, _PLAN_NUDGE, 0, 1),
        ("assistant", _CLOSE.text, 1, 0),
    ]
    assert _statuses(mp.channel, mp.turn_id) == ["done", "done"]
    assert execution(mp.channel, mp.turn_id).state == TurnExecution.COMPLETED
    thread = TurnSerializerService().serialize(mp.channel, mp.turn_id)
    messages = cast("list[dict[str, object]]", thread["messages"])
    assert [(m["role"], m.get("joined")) for m in messages if m["content"] == _PLAN_NUDGE] == [("user", True)]


# ── 3. the loop limit: nudged again only after real work ─────────────────────


def test_a_nudge_that_gets_work_done_is_followed_by_another_while_the_plan_stays_open(
    authed_client: AuthedClient, provider: _NudgeProvider,
) -> None:
    """After the first nudge the model searches its memory — a tool other than
    the plan — and answers again with both tasks still open, so it is nudged
    again. Its answer to that second nudge calls no tool, so the turn ends on
    it with the tasks still open."""
    client, db, _store = authed_client
    provider.script(
        _CREATE, _say(_PROSE), _calls(_RECALL), _say("It still leaves at 9."), _say("Nothing else to do."),
    )
    turn_id = cast("int", _result(_post(client, -1, "when does my flight leave?"))["turn_id"])
    _drain(_SETTLE_S)

    assert provider.tools == [_PLANNING] + [frozenset(UserConfig().always_available)] * 4
    assert provider.nudges == [0, 0, 1, 1, 2]
    assert provider.unused == 0
    assert _rows("user", turn_id) == [
        ("assistant", _PROSE, 0, 0), _NUDGED, ("assistant", "", 0, 0),
        ("assistant", "It still leaves at 9.", 0, 0), _NUDGED, ("assistant", "Nothing else to do.", 1, 0),
    ]
    assert _statuses("user", turn_id) == ["todo", "todo"]
    assert [state for _id, state in _executions(db, turn_id)] == [TurnExecution.COMPLETED]


@pytest.mark.parametrize(
    ("nudged", "rows", "statuses"),
    [
        pytest.param(
            [_say("Nothing else to do.")], [("assistant", "Nothing else to do.", 1, 0)], ["todo", "todo"],
            id="answers-again",
        ),
        pytest.param(
            [_calls(_RESTARTED), _say("Nothing else to do.")],
            [("assistant", "", 0, 0), ("assistant", "Nothing else to do.", 1, 0)], ["in_progress", "todo"],
            id="only-updates-the-plan",
        ),
        pytest.param(
            [
                ProviderResponse(text="Checking the inbox again.", model="scripted", tool_calls=[_RESTARTED]),
                _say("Nothing else to do."),
            ],
            [("assistant", "Checking the inbox again.", 0, 0), ("assistant", "Nothing else to do.", 1, 0)],
            ["in_progress", "todo"],
            id="answers-while-updating-the-plan",
        ),
    ],
)
def test_a_nudge_that_gets_no_work_done_lets_the_next_answer_end_the_turn(
    authed_client: AuthedClient, provider: _NudgeProvider, nudged: list[ProviderResponse],
    rows: list[tuple[object, ...]], statuses: list[str],
) -> None:
    """After the nudge the model calls no tool other than the plan — it only
    marks a task in progress again, or nothing at all — and its next answer
    leaves the plan open, so the turn ends on that answer with no second
    nudge."""
    client, db, _store = authed_client
    provider.script(_CREATE, _say(_PROSE), *nudged)
    turn_id = cast("int", _result(_post(client, -1, "when does my flight leave?"))["turn_id"])
    _drain(_SETTLE_S)

    assert provider.tools == [_PLANNING] + [frozenset(UserConfig().always_available)] * (1 + len(nudged))
    assert provider.nudges == [0, 0] + [1] * len(nudged)
    assert provider.unused == 0
    assert _rows("user", turn_id) == [("assistant", _PROSE, 0, 0), _NUDGED, *rows]
    assert _statuses("user", turn_id) == statuses
    assert [state for _id, state in _executions(db, turn_id)] == [TurnExecution.COMPLETED]


# ── 4. a stop during the nudged step ends the turn ───────────────────────────


def test_a_stop_during_the_nudged_step_ends_the_turn_cancelled_and_sends_nothing_more(
    authed_client: AuthedClient, provider: _NudgeProvider,
) -> None:
    """The person stops the turn while its nudged step is in flight. The turn
    ends cancelled, no request follows, and the answer it gave before the
    nudge and the nudge itself are still in the thread."""
    client, db, _store = authed_client
    provider.script(_CREATE, _say(_PROSE), _calls(_plan(action="update", task=1, status="done")))
    provider.hold_at(2)
    turn_id = cast("int", _result(_post(client, -1, "when does my flight leave?"))["turn_id"])
    assert provider.entered.wait(10), "the nudged step never reached the provider"

    stop = cast("dict[str, object]", client.delete(f"/api/threads/{turn_id}").get_json())
    assert cast("dict[str, object]", stop["result"])["cancelled"] is True
    _drain(_SETTLE_S)

    assert provider.nudges == [0, 0, 1]
    assert [state for _id, state in _executions(db, turn_id)] == [TurnExecution.CANCELLED]
    assert _statuses("user", turn_id) == ["todo", "todo"]
    assert _rows("user", turn_id)[:2] == [("assistant", _PROSE, 0, 0), _NUDGED]
    messages = cast("list[dict[str, object]]", _result(client.get(f"/api/threads/{turn_id}"))["messages"])
    assert {_PROSE, _PLAN_NUDGE} <= {m["content"] for m in messages}


# ── 5. a subagent waiting on its question to the chat is never nudged ────────


_WEB = WebSearchAbility.NAME
_WEB_CHANNEL = delegate_channel(_WEB)
_BRIDGE = _task("Search", "Find the bridge news")


class _Plans(ChannelProvider):
    """The channel scripts, except that the ``web_search`` subagent's planning
    request plans ``search``, and the chat's ``n``-th planning request (counting
    from 1), for each ``n`` in ``chat_at``, plans the bridge news; every other
    planning request plans nothing. A chat turn's planning request offers no
    ``find_tools``, so it is told apart from the subagent's by its route alone."""

    def __init__(
        self, search: list[dict[str, str]], chat_at: frozenset[int] = frozenset(), **scripts: list[Entry],
    ) -> None:
        super().__init__(**scripts)
        self._search = search
        self._chat_at = chat_at
        self._planned = 0
        self._count = threading.Lock()

    def send(self, dto: ProviderRequest) -> ProviderResponse:
        if is_planning_request(dto):
            tasks = self._search if self.route(dto) == _WEB else self._chat_tasks()
            if tasks:
                return calls(tool(PlanAbility.NAME, action="create", tasks=tasks))
        return super().send(dto)

    def _chat_tasks(self) -> list[dict[str, str]]:
        with self._count:
            self._planned += 1
            return [_BRIDGE] if self._planned in self._chat_at else []


def _searcher(db: sqlite3.Connection, spawn_id: int) -> Agent:
    searcher, _turn = child(db, _WEB_CHANNEL, spawn_id)
    [agent] = [a for a in Agent.spawned_by(UserConfig().channel) if a.channel == searcher]
    return agent


@pytest.mark.usefixtures("chat_provider", "earlier_agents")
def test_a_subagent_waiting_on_its_question_to_the_chat_holds_back_both_nudges_until_it_is_archived(
    db: sqlite3.Connection,
) -> None:
    """The chat spawns a ``web_search``, which plans, asks the chat a question
    and ends its turn with its task still open. It waits for the reply instead
    of being nudged: its turn reads nothing more, ends completed, and the
    subagent is left waiting with its plan open. The person's next message on
    the chat turn that spawned it plans, says the search is still waiting and
    is not nudged either; once the waiting search is archived, the message
    after that is nudged and answers while closing its plan."""
    asking = Held(calls(tool(MessageParentAbility.NAME, message="Which bridge do you mean?")))
    provider = _Plans(
        [_BRIDGE],
        frozenset({3, 4}),
        chat=[
            calls(tool(_WEB, instructions="find this week's bridge news", act_summary="Searching the bridge news")),
            text("Started."),
            text("I'll ask which bridge they mean."),
            text("The search is still waiting on which bridge."),
            text("No bridge news yet."),
            ProviderResponse(
                text="Nothing more on the bridge.", model="m",
                tool_calls=[tool(PlanAbility.NAME, action="update", task=1, status="cancelled")],
            ),
        ],
        web_search=[asking, text("I need to know which bridge first.")],
    )

    with patched(provider):
        mp = begin(UserConfig(), "find the bridge news")
        mp.result()
        asking.gate.set()
        drain()
        [spawn] = calls_of(db, _WEB)
        agent = _searcher(db, cast("int", spawn["id"]))
        waiting = begin(UserConfig(), "any bridge news?", turn_id=mp.turn_id)
        waiting.result()
        drain()
        assert agent.response_id is not None
        aged = utc_now() - ARCHIVE_AFTER - timedelta(days=1)
        assert Transcript.filter("id", agent.response_id).update(created_at=aged.strftime("%Y-%m-%d %H:%M:%S")) == 1
        archived = begin(UserConfig(), "any bridge news now?", turn_id=mp.turn_id)
        archived.result()
        drain()

    assert asking.opened and provider.unused() == {}
    reads = provider.read(_WEB)
    assert len(reads) == 2
    assert all(_PLAN_NUDGE not in read for read in reads)
    searcher, turn = child(db, _WEB_CHANNEL, cast("int", spawn["id"]))
    assert execution(searcher, turn).state == TurnExecution.COMPLETED
    assert _statuses(searcher, turn) == ["todo"]
    assert cast("Agent", Agent.by_id(agent.agent_id)).status == Agent.WAITING
    assert [_PLAN_NUDGE in read for read in provider.read("chat")] == [False] * 5 + [True]
    assert [t.status for t in PlanTask.for_transcript(cast("int", waiting.uid))] == ["todo"]
    assert [t.status for t in PlanTask.for_transcript(cast("int", archived.uid))] == ["cancelled"]
    assert _assistant_rows(USER, mp.turn_id)[-3:] == [
        ("The search is still waiting on which bridge.", 1), ("No bridge news yet.", 0),
        ("Nothing more on the bridge.", 1),
    ]


@pytest.mark.usefixtures("chat_provider", "earlier_agents")
def test_a_spawned_subagent_nudged_to_close_its_plan_reports_the_answer_it_closes_with(
    db: sqlite3.Connection,
) -> None:
    """The chat spawns a ``web_search``, which plans, answers with its tasks
    still open and, on the nudge, answers again while closing them. The
    subagent ends done, its output is that closing answer, and that answer is
    the report the chat receives."""
    provider = _Plans(
        [_FLIGHT, _ANSWER],
        chat=[
            calls(tool(_WEB, instructions="find when my flight leaves", act_summary="Searching for the flight")),
            text("Started."),
            text("It leaves at 9."),
        ],
        web_search=[
            text(_PROSE),
            ProviderResponse(
                text=_CLOSE.text, model="m",
                tool_calls=[
                    tool(PlanAbility.NAME, action="update", task=1, status="done"),
                    tool(PlanAbility.NAME, action="update", task=2, status="done"),
                ],
            ),
        ],
    )

    with patched(provider):
        begin(UserConfig(), "when does my flight leave?").result()
        drain()

    assert provider.unused() == {}
    reads = provider.read(_WEB)
    assert [_PLAN_NUDGE in read for read in reads] == [False, True]
    [spawn] = calls_of(db, _WEB)
    searcher, turn = child(db, _WEB_CHANNEL, cast("int", spawn["id"]))
    assert execution(searcher, turn).state == TurnExecution.COMPLETED
    assert _statuses(searcher, turn) == ["done", "done"]
    [agent] = [a for a in Agent.spawned_by(UserConfig().channel) if a.channel == searcher]
    assert agent.status == Agent.DONE
    assert AgentService.last_output(agent) == _CLOSE.text
    [reply] = replies(db, cast("int", spawn["id"]))
    assert _CLOSE.text in cast("str", reply["content"])


# ── 6. a chat turn whose subagent's answer is still to come is never nudged ──


@pytest.mark.usefixtures("chat_provider", "earlier_agents")
def test_a_chat_turn_whose_subagent_is_still_running_is_not_nudged(db: sqlite3.Connection) -> None:
    """The chat plans, starts a ``web_search`` and says so with its task still
    open. The search is still running, so the turn ends there instead of being
    nudged. The search's answer opens a turn of its own, which — starting
    nothing — plans, sends the search a message and says so with its task
    still open: the search is running again on the message, so that turn ends
    there too. The search's second answer opens yet another turn, which
    answers the person, and both earlier plans are left as they were."""
    searching = Held(text("The bridge reopened on Monday."))
    narrowing = Held(text("It reopened at 6am."))
    provider = _Plans(
        [],
        frozenset({1, 2}),
        chat=[
            calls(tool(_WEB, instructions="find this week's bridge news", act_summary="Searching the bridge news")),
            text("Started."),
            calls(tool(FindToolsAbility.NAME, query=[MessageSubagentAbility.NAME])),
            calls(tool(MessageSubagentAbility.NAME, agent_id=EARLIER_AGENTS + 1, message="Only this morning's news.")),
            text("Asked it."),
            text("It reopened at 6am."),
        ],
        web_search=[searching, narrowing],
    )

    with patched(provider):
        mp = begin(UserConfig(), "find the bridge news")
        mp.result()
        assert searching.entered.wait(10), "the search never reached the provider"
        [spawn] = calls_of(db, _WEB)
        assert all(_PLAN_NUDGE not in read for read in provider.read("chat"))
        assert _searcher(db, cast("int", spawn["id"])).status == Agent.RUNNING
        searching.gate.set()
        assert narrowing.entered.wait(10), "the message never reached the search"
        for thread in turn_threads():
            if thread is not narrowing.thread:
                thread.join(10)
        [first] = replies(db, cast("int", spawn["id"]))
        asked = cast("int", first["turn_id"])
        assert ("Asked it.", 1) in _assistant_rows(USER, asked)
        assert all(_PLAN_NUDGE not in read for read in provider.read("chat"))
        assert _searcher(db, cast("int", spawn["id"])).status == Agent.RUNNING
        narrowing.gate.set()
        drain()

    assert searching.opened and narrowing.opened and provider.unused() == {}
    assert all(_PLAN_NUDGE not in read for read in provider.read("chat"))
    assert ("Started.", 1) in _assistant_rows(USER, mp.turn_id)
    assert _statuses(USER, mp.turn_id) == ["todo"]
    assert _statuses(USER, asked) == ["todo"]
    assert execution(USER, mp.turn_id).state == TurnExecution.COMPLETED
    assert execution(USER, asked).state == TurnExecution.COMPLETED
    assert _searcher(db, cast("int", spawn["id"])).status == Agent.DONE
    [_first, second] = replies(db, cast("int", spawn["id"]))
    answered = cast("int", second["turn_id"])
    assert answered not in (mp.turn_id, asked)
    assert _assistant_rows(USER, answered)[-1] == ("It reopened at 6am.", 1)
