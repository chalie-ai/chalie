"""Feature tests: every message the agent works on is planned first, then
worked with its plan in view.

A new message on a config that pins the ``plan`` tool gets a planning pre-pass:
one or more provider calls that may call only ``recall`` and ``plan`` and that
show the other tools as a catalogue. The loop that follows sees the plan as a
``[plan]`` block that tracks every update, keyed by the message's input row;
once the exchange settles the plan leaves the context and stays on its message,
where the thread view returns it. A message joined into a working turn adds
tasks to the running plan and is never planned on its own. A response that
answers while its only calls close the plan's last open tasks is the turn's
settled answer.

These drive the real send endpoint (``POST /api/threads/<turn_id>``) or the
real ``MessageProcessor`` for a delegate, the real plan ability through real
dispatch, the real prompt, transcript, tool-call and ``plan_tasks`` tables, the
real turn serializer and the real migration runner, against the real migrated
SQLite database. The only substitution is the LLM network boundary
(``services.provider_service.build_client``): a scripted double.
"""

import logging
import sqlite3
import threading
from collections.abc import Iterator
from typing import cast

import pytest
from flask.testing import FlaskClient

import services.provider_service as provider_service
from abilities.find_tools import FindToolsAbility
from abilities.plan import PlanAbility
from abilities.recall import Recall
from configs.channels.user import UserConfig
from configs.channels.web_search import WebSearchConfig
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from migrations.runner import run_all
from models.plan_task import PlanTask
from models.provider_request import ProviderRequest
from models.provider_response import ProviderResponse
from models.tool_call import ToolCall
from models.transcript import Transcript
from services.file_mapper_service import FileMapperService
from services.prompt_service import _NO_PLAN, _PLAN_INSTRUCTION, _PLANNING_INSTRUCTION
from tests.test_turn_join import AuthedClient, _drain, _label_thread, _post, _result, _say

pytestmark = pytest.mark.unit

_CHANNEL = "user"
_PLANNING = frozenset({Recall.NAME, PlanAbility.NAME})
_MIGRATION_KEY = "plan-tasks-table-v1"


class _PlanningProvider:
    """Replays scripted responses to the turn's own requests — every request
    that offers the plan tool — and records each one's tool names and body.
    Any other call (the thread label, the post-turn suggestion) gets a benign
    terminal reply and is not recorded, so the script is the turn's alone.
    ``hold_at(n)`` parks the n-th recorded request until ``release`` is set,
    signalling ``entered`` first."""

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

    def send(self, dto: ProviderRequest) -> ProviderResponse:
        names = frozenset(str(tool["name"]) for tool in dto.tools or [])
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


@pytest.fixture
def provider(chat_provider: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch) -> Iterator[_PlanningProvider]:
    double = _PlanningProvider()
    monkeypatch.setattr(provider_service, "build_client", lambda *_a, **_k: double)
    yield double
    double.release.set()  # a failed test must not strand a parked drive thread
    _drain()


def _task(title: str, task: str) -> dict[str, str]:
    return {"title": title, "task": task}


def _plan(**params: object) -> dict[str, object]:
    return {"name": PlanAbility.NAME, "input": params}


def _calls(*calls: dict[str, object]) -> ProviderResponse:
    return ProviderResponse(text="", model="scripted", tool_calls=list(calls))


def _input_rows(turn_id: int) -> list[int]:
    """The ids of the turn's user rows, in order."""
    return [cast("int", r["id"]) for r in Transcript.by_turn(_CHANNEL, turn_id) if r["role"] == "user"]


def _assistant_rows(turn_id: int) -> list[tuple[object, object]]:
    return [(r["content"], r["settled"]) for r in Transcript.by_turn(_CHANNEL, turn_id) if r["role"] == "assistant"]


def _plan_rows(message_id: int) -> list[tuple[int, str, str, str]]:
    return [(t.position, t.title, t.task, t.status) for t in PlanTask.for_transcript(message_id)]


def _thread_users(client: FlaskClient, turn_id: int) -> list[dict[str, object]]:
    """The thread view's user messages, as the client receives them."""
    messages = cast("list[dict[str, object]]", _result(client.get(f"/api/threads/{turn_id}"))["messages"])
    return [m for m in messages if m["role"] == "user"]


_FLIGHT = _task("Find the flight", "Search the inbox for the booking email")
_ANSWER = _task("Answer", "Tell the user the departure time")


# ── 1. planned first, then worked with the plan in view ──────────────────────


def test_a_new_message_is_planned_first_then_worked_with_its_plan_in_view(
    authed_client: AuthedClient, provider: _PlanningProvider,
) -> None:
    """The first call may only recall and plan, and sees the planning ask and
    the other tools as a catalogue. Every loop step after it offers the full
    toolset and shows the plan as it stands — each update is in the very next
    request. The pre-pass writes no assistant row and its call anchors to the
    message; the plan is stored on the message and returned with it."""
    client, _db, _store = authed_client
    provider.script(
        _calls(_plan(action="create", tasks=[_FLIGHT, _ANSWER])),
        _calls(_plan(action="update", task=1, status="in_progress")),
        _calls(_plan(action="update", task=1, status="done"), _plan(action="update", task=2, status="in_progress")),
        _calls(_plan(action="update", task=2, status="done")),
        _say("Your flight leaves at 9."),
    )
    turn_id = cast("int", _result(_post(client, -1, "when does my flight leave?"))["turn_id"])
    _drain()

    assert provider.tools == [_PLANNING] + [frozenset(UserConfig().always_available)] * 4
    prepass = provider.bodies[0]
    assert "when does my flight leave?" in prepass
    assert _PLANNING_INSTRUCTION in prepass
    assert f"- `{FindToolsAbility.NAME}`:" in prepass.split("[tool_catalogue]", 1)[1]
    assert "[plan]\n" not in prepass

    first_step = provider.bodies[1]
    assert (
        "[plan]\n"
        "1. [todo] Find the flight — Search the inbox for the booking email\n"
        "2. [todo] Answer — Tell the user the departure time\n"
        f"[end:plan]\n{_PLAN_INSTRUCTION}"
    ) in first_step
    assert "Plan saved: 2 tasks." in first_step
    assert _PLANNING_INSTRUCTION not in first_step
    assert "1. [in_progress] Find the flight" in provider.bodies[2]
    assert "1. [done] Find the flight" in provider.bodies[3]
    assert "2. [in_progress] Answer" in provider.bodies[3]
    assert "2. [done] Answer" in provider.bodies[4]

    [message_id] = _input_rows(turn_id)
    assert _plan_rows(message_id) == [
        (1, "Find the flight", "Search the inbox for the booking email", "done"),
        (2, "Answer", "Tell the user the departure time", "done"),
    ]
    assert _assistant_rows(turn_id) == [("", 0), ("", 0), ("", 0), ("Your flight leaves at 9.", 1)]
    # On the message: the turn-zero memory recall, then the pre-pass's plan.
    anchored = ToolCall.by_transcript(message_id)
    assert [(c.tool_name, '"create"' in c.params) for c in anchored] == [(Recall.NAME, False), (PlanAbility.NAME, True)]
    [user] = _thread_users(client, turn_id)
    assert user["plan"] == [
        {"number": 1, "title": "Find the flight", "task": "Search the inbox for the booking email", "status": "done"},
        {"number": 2, "title": "Answer", "task": "Tell the user the departure time", "status": "done"},
    ]

    # A reply is a new message: it is planned afresh, and the settled plan
    # above it never reaches its context.
    _label_thread(turn_id)
    provider.script(_say(""), _say("Anything else?"))
    _result(_post(client, turn_id, "thanks"))
    _drain()

    assert provider.tools[5:] == [_PLANNING, frozenset(UserConfig().always_available)]
    reply_step = provider.bodies[6]
    assert _NO_PLAN in reply_step
    assert "[plan]\n" not in reply_step
    assert "Find the flight" not in reply_step.rsplit("thanks", 1)[1]
    opener, reply = _thread_users(client, turn_id)
    assert len(cast("list[object]", opener["plan"])) == 2
    assert reply["plan"] is None


# ── 2. the pre-pass runs nothing but recall and plan ─────────────────────────


def test_a_planning_call_to_any_other_tool_is_never_run(
    authed_client: AuthedClient, provider: _PlanningProvider, caplog: pytest.LogCaptureFixture,
) -> None:
    """The model reaches for a catalogue tool while planning. Dispatch would
    run any registered tool, so the call is dropped and logged, never run;
    with nothing run, planning ends and the loop asks for the plan instead."""
    client, _db, _store = authed_client
    caplog.set_level(logging.WARNING)
    provider.script(
        _calls({"name": FindToolsAbility.NAME, "input": {"query": "weather"}}),
        _say("It is sunny."),
    )
    turn_id = cast("int", _result(_post(client, -1, "what is the weather?"))["turn_id"])
    _drain()

    assert provider.tools == [_PLANNING, frozenset(UserConfig().always_available)]
    assert _NO_PLAN in provider.bodies[1]
    assert [c.tool_name for c in ToolCall.by_turn(_CHANNEL, turn_id)] == [Recall.NAME]  # the memory seed alone
    assert f"planning skipped calls to ['{FindToolsAbility.NAME}']" in caplog.text
    assert _assistant_rows(turn_id) == [("It is sunny.", 1)]


# ── 3. refusals reach the model, and planning gives up after three sends ─────


def test_plan_refusals_reach_the_model_and_planning_gives_up_after_three_sends(
    authed_client: AuthedClient, provider: _PlanningProvider,
) -> None:
    """Three planning calls that each fail: a task added before any plan, a
    plan over the ten-task cap, a task updated that does not exist. Each
    refusal is in the next request. Planning stops at its third send and the
    loop asks for the plan; there, a second plan and an unknown task number
    are refused too, naming the numbers that exist, and nothing but the one
    good plan is stored."""
    client, _db, _store = authed_client
    eleven = [_task(f"Step {n}", f"Do step {n}") for n in range(1, 12)]
    provider.script(
        _calls(_plan(action="add", tasks=[_FLIGHT])),
        _calls(_plan(action="create", tasks=eleven)),
        _calls(_plan(action="update", task=1, status="done")),
        _calls(_plan(action="create", tasks=[_FLIGHT, _ANSWER])),
        _calls(_plan(action="create", tasks=[_ANSWER]), _plan(action="update", task=5, status="done")),
        _say("Your flight leaves at 9."),
    )
    turn_id = cast("int", _result(_post(client, -1, "when does my flight leave?"))["turn_id"])
    _drain()

    loop = frozenset(UserConfig().always_available)
    assert provider.tools == [_PLANNING] * 3 + [loop] * 3
    assert "code=no-plan" in provider.bodies[1]
    assert "This message has no plan yet." in provider.bodies[1]
    assert "code=too-many-tasks" in provider.bodies[2]
    assert "A plan holds at most 10 tasks; got 11." in provider.bodies[2]
    assert "code=unknown-task" in provider.bodies[3]
    assert "There is no task 1." in provider.bodies[3]
    assert _NO_PLAN in provider.bodies[3]
    assert "Plan saved: 2 tasks." in provider.bodies[4]
    refused = provider.bodies[5]
    assert "code=plan-exists" in refused
    assert "This message already has a plan." in refused
    assert "There is no task 5." in refused
    assert "valid: 1 | 2" in refused

    [message_id] = _input_rows(turn_id)
    assert [(position, status) for position, _t, _k, status in _plan_rows(message_id)] == [(1, "todo"), (2, "todo")]


# ── 4. a joined message extends the running plan ────────────────────────────


def test_a_joined_message_adds_to_the_running_plan_and_is_never_planned_itself(
    authed_client: AuthedClient, provider: _PlanningProvider,
) -> None:
    """The user adds to the question while the loop works it. The addition
    gets no pre-pass of its own: the loop reads it and adds a task to the plan
    it is already keeping, on the message that opened the turn. The thread
    view returns the plan, three tasks long, on that message alone."""
    client, _db, _store = authed_client
    provider.script(
        _calls(_plan(action="create", tasks=[_FLIGHT, _ANSWER])),
        _calls(_plan(action="update", task=1, status="in_progress")),
        _calls(_plan(action="add", tasks=[_task("Check the hotel", "Look up the hotel booking too")])),
        _say("Flight at 9, hotel from 3."),
    )
    provider.hold_at(1)
    opened = _result(_post(client, -1, "when does my flight leave?"))
    turn_id = cast("int", opened["turn_id"])
    assert provider.entered.wait(10), "the loop never reached its first step"
    assert _result(_post(client, turn_id, "and my hotel?"))["id"] == opened["id"]
    provider.release.set()
    _drain()

    assert provider.tools == [_PLANNING] + [frozenset(UserConfig().always_available)] * 3
    assert "and my hotel?" in provider.bodies[2]
    assert "Added task 3." in provider.bodies[3]
    assert "3. [todo] Check the hotel — Look up the hotel booking too" in provider.bodies[3]

    opener_id, joined_id = _input_rows(turn_id)
    assert [position for position, *_ in _plan_rows(opener_id)] == [1, 2, 3]
    assert _plan_rows(joined_id) == []
    opener, joined = _thread_users(client, turn_id)
    assert [task["title"] for task in cast("list[dict[str, object]]", opener["plan"])] == [
        "Find the flight", "Answer", "Check the hotel",
    ]
    assert joined["plan"] is None


# ── 5. a delegate plans its own brief ────────────────────────────────────────


def test_a_delegate_plans_its_brief_before_working_it(provider: _PlanningProvider) -> None:
    """A subagent is handed a brief: memory is recalled for it, it plans it
    with only recall and plan callable, then works it with its own toolset and
    the plan in view, and the plan is stored on the brief's input row."""
    config = WebSearchConfig(PolicyChannel.CHAT)
    provider.script(
        _calls(_plan(action="create", tasks=[_task("Search", "Find the final score")])),
        _calls(_plan(action="update", task=1, status="done")),
        _say("Team A won 2-1."),
    )
    mp = MessageProcessor(config, raw_input="who won the match")
    mp.begin()
    mp.result()
    _drain()

    assert provider.tools == [_PLANNING] + [frozenset(config.always_available)] * 2
    assert "1. [todo] Search — Find the final score" in provider.bodies[1]
    assert mp.uid is not None
    assert _plan_rows(mp.uid) == [(1, "Search", "Find the final score", "done")]
    assert [c.tool_name for c in ToolCall.by_transcript(mp.uid)] == [Recall.NAME, PlanAbility.NAME]


# ── 6. the answer that closes the plan ends the turn ─────────────────────────


def _answer(text: str, *calls: dict[str, object]) -> ProviderResponse:
    return ProviderResponse(text=text, model="scripted", tool_calls=list(calls))


def test_the_answer_that_closes_the_plan_ends_the_turn(
    authed_client: AuthedClient, provider: _PlanningProvider,
) -> None:
    """The model answers in the same response as the updates that close its
    plan — one task done, the other no longer needed. That response is the
    turn's last: the updates ran, its row is the settled answer, and the model
    is never sent another step."""
    client, _db, _store = authed_client
    provider.script(
        _calls(_plan(action="create", tasks=[_FLIGHT, _ANSWER])),
        _calls(_plan(action="update", task=1, status="in_progress")),
        _answer(
            "Your flight leaves at 9.",
            _plan(action="update", task=1, status="done"),
            _plan(action="update", task=2, status="cancelled"),
        ),
    )
    turn_id = cast("int", _result(_post(client, -1, "when does my flight leave?"))["turn_id"])
    _drain()

    assert provider.tools == [_PLANNING] + [frozenset(UserConfig().always_available)] * 2
    [message_id] = _input_rows(turn_id)
    assert [status for *_, status in _plan_rows(message_id)] == ["done", "cancelled"]
    assert _assistant_rows(turn_id) == [("", 0), ("Your flight leaves at 9.", 1)]


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(
            _answer("Your flight leaves at 9.", _plan(action="update", task=1, status="done")),
            id="a-task-left-open",
        ),
        pytest.param(
            _answer(
                "Your flight leaves at 9.",
                _plan(action="update", task=1, status="done"),
                _plan(action="update", task=2, status="done"),
                {"name": Recall.NAME, "input": {"query": "flight booking"}},
            ),
            id="another-tool-called",
        ),
        pytest.param(
            _answer(
                "",
                _plan(action="update", task=1, status="done"),
                _plan(action="update", task=2, status="done"),
            ),
            id="nothing-said",
        ),
    ],
)
def test_a_response_that_does_not_both_answer_and_close_the_plan_is_a_step(
    authed_client: AuthedClient, provider: _PlanningProvider, answer: ProviderResponse,
) -> None:
    """Prose beside plan updates is a step, not the answer, unless the updates
    are all it called and they close the plan: a task still open, another
    tool's call, or no prose at all, and the loop asks again. The reply after
    it is the turn's settled row."""
    client, _db, _store = authed_client
    provider.script(_calls(_plan(action="create", tasks=[_FLIGHT, _ANSWER])), answer, _say("Anything else?"))
    turn_id = cast("int", _result(_post(client, -1, "when does my flight leave?"))["turn_id"])
    _drain()

    assert provider.tools == [_PLANNING] + [frozenset(UserConfig().always_available)] * 2
    assert _assistant_rows(turn_id) == [(answer.text, 0), ("Anything else?", 1)]


_ECHO = f'[{PlanAbility.NAME}] {{"action": "update", "task": 2, "status": "done"}}'


@pytest.mark.parametrize(
    ("text", "rows"),
    [
        pytest.param(_ECHO, [("", 0), ("Anything else?", 1)], id="only-the-echo"),
        pytest.param(f"Your flight leaves at 9.\n{_ECHO}", [("Your flight leaves at 9.", 1)], id="echo-beside-prose"),
    ],
)
def test_a_line_echoing_the_responses_own_call_is_never_its_prose(
    authed_client: AuthedClient, provider: _PlanningProvider, text: str, rows: list[tuple[str, int]],
) -> None:
    """The model writes its own plan update out in the act trail's call
    notation beside the real updates that close the plan. That line is never
    stored or taken for the answer: alone, the response is a silent step and
    the reply after it settles; beside prose, the prose is the settled answer."""
    client, _db, _store = authed_client
    provider.script(
        _calls(_plan(action="create", tasks=[_FLIGHT, _ANSWER])),
        _answer(text, _plan(action="update", task=1, status="done"), _plan(action="update", task=2, status="done")),
        _say("Anything else?"),
    )
    turn_id = cast("int", _result(_post(client, -1, "when does my flight leave?"))["turn_id"])
    _drain()

    assert provider.tools == [_PLANNING] + [frozenset(UserConfig().always_available)] * len(rows)
    assert _assistant_rows(turn_id) == rows


def test_a_message_joined_while_the_plan_closes_keeps_the_turn_going(
    authed_client: AuthedClient, provider: _PlanningProvider,
) -> None:
    """The user adds to the question while the closing answer is being
    written. That answer is no longer the turn's last: it stays a step, the
    loop takes the new message in, and the reply to it is the settled row."""
    client, _db, _store = authed_client
    provider.script(
        _calls(_plan(action="create", tasks=[_FLIGHT])),
        _answer("Your flight leaves at 9.", _plan(action="update", task=1, status="done")),
        _say("Your hotel is from 3."),
    )
    provider.hold_at(1)
    opened = _result(_post(client, -1, "when does my flight leave?"))
    turn_id = cast("int", opened["turn_id"])
    assert provider.entered.wait(10), "the loop never reached its first step"
    assert _result(_post(client, turn_id, "and my hotel?"))["id"] == opened["id"]
    provider.release.set()
    _drain()

    assert provider.tools == [_PLANNING] + [frozenset(UserConfig().always_available)] * 2
    assert "and my hotel?" in provider.bodies[2]
    assert _assistant_rows(turn_id) == [("Your flight leaves at 9.", 0), ("Your hotel is from 3.", 1)]


# ── 7. the upgrade path ──────────────────────────────────────────────────────


def _outcome(db: sqlite3.Connection) -> object:
    row = db.execute("SELECT outcome FROM schema_migrations WHERE key = ?", (_MIGRATION_KEY,)).fetchone()
    return None if row is None else row[0]


def test_an_install_without_the_plan_table_gets_it_on_boot(db: sqlite3.Connection) -> None:
    """A database provisioned before plans existed has no ``plan_tasks``; the
    boot migration creates it with its guards — one task per number on a
    message, and only the four statuses."""
    db.execute("DROP TABLE plan_tasks")
    db.commit()
    run_all(str(FileMapperService.get_db_path()))

    assert _outcome(db) == "applied"
    message_id = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES ('user', 'user', 'hi')"
    ).lastrowid
    insert = "INSERT INTO plan_tasks (transcript_id, position, title, task, status) VALUES (?, ?, 't', 'k', ?)"
    db.execute(insert, (message_id, 1, "todo"))
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(insert, (message_id, 1, "todo"))
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(insert, (message_id, 2, "finished"))


def test_a_fresh_install_records_the_plan_migration_as_not_needed(db: sqlite3.Connection) -> None:
    run_all(str(FileMapperService.get_db_path()))
    assert _outcome(db) == "noop"
