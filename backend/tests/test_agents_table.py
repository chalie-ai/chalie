"""Feature test: every subagent spawned is one ``agents`` row, numbered by it.

Each spawn tool — ``web_search``, ``web_browse``, ``code_agent``, ``pim`` and
``subtask`` — writes one row the moment its subagent's turn opens, whoever
spawned it: the person's chat, a scheduled task, or another subagent. The row's
``agent_id`` is the number the model knows the subagent by, never its turn id.
The row ends with its turn, once: ``done`` with the id of its final answer,
``cancelled`` when the person or its caller's stop ends it, ``failed`` when it
crashed or a restart cut it off. Every write goes out on the socket as an
``agent_changed`` frame carrying the row as the listing returns it, and
``GET /api/agents/all`` lists the agents still running to a person signed in on
the page — never to a bearer-token client.

Drives the real production entry point (construct inertly, ``begin()``,
``result()``), the real boot sweep and the real Flask routes against the real,
fully-migrated SQLite DB, observing frames through the REAL socket registry.
The only substitution is the LLM network boundary
(``services.provider_service.build_client``); each subagent is held inside its
own model call on a ``threading.Event`` so the test reads it while it runs.
Every test starts past ``EARLIER_AGENTS`` ended agents, so an agent's number and
its turn id always differ.
"""

import sqlite3
from contextlib import AbstractContextManager, nullcontext
from typing import cast

import pytest
from flask.testing import FlaskClient

from abilities.code_agent import CodeAgentAbility
from abilities.pim import PimAbility
from abilities.subtask import SubtaskAbility
from abilities.web_browse import WebBrowseAbility
from abilities.web_search import WebSearchAbility
from configs.channels import config_for
from configs.channels.scheduled import ScheduledConfig
from configs.channels.user import UserConfig
from configs.enums.channels import Channel
from configs.enums.config_type import ConfigTypeEnum
from controllers.message_processor import _MAX_PROVIDER_ATTEMPTS, MessageProcessor
from migrations.runner import run_all
from models.turn_execution import TurnExecution
from models.agent import Agent
from run import _reply_to_swept_delegates
from services.database import Database
from services.file_mapper_service import FileMapperService
from services.garbage_collector_service import GarbageCollectorService
from services.wrapper_auth_service import WrapperAuthService
from tests._delegates import (
    EARLIER_AGENTS,
    USER,
    ChannelProvider,
    Entry,
    Held,
    RecordingClient,
    begin,
    calls,
    calls_of,
    child_turn,
    connected_mail,
    delegate_channel,
    dies,
    drain,
    execution,
    listening,
    patched,
    process_deaths_expected,
    raises,
    run_turn,
    text,
    tool,
    wait,
)

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider", "earlier_agents")]

_Client = tuple[FlaskClient, sqlite3.Connection, object]

_QUESTION = "what is the capital of France?"
_INSTRUCTIONS = "find the capital of France"
_SUMMARY = "Finding the capital of France"
_ANSWER = "Paris is the capital of France."
_STARTED = "I have started it."
_ANSWERED = "Here is what it found."
_SUBTASK = Channel.DELEGATE_SUBTASK.value
_LISTING = "/api/agents/all"
_SPAWNERS = (
    WebSearchAbility.NAME, WebBrowseAbility.NAME, CodeAgentAbility.NAME, PimAbility.NAME, SubtaskAbility.NAME,
)


def _spawn(name: str, instructions: str = _INSTRUCTIONS, **extra: object) -> dict[str, object]:
    """A call that spawns the *name* subagent — a subtask is started by action."""
    if name == SubtaskAbility.NAME:
        extra = {"action": "start", **extra}
    return tool(name, instructions=instructions, **extra)


def _route(name: str) -> str:
    """The script the *name* subagent's own turn is answered from."""
    return "subtask" if name == SubtaskAbility.NAME else name


def _channel(name: str) -> str:
    """The channel the *name* subagent's own turn runs on."""
    return _SUBTASK if name == SubtaskAbility.NAME else delegate_channel(name)


def _mailbox(client: FlaskClient, name: str) -> AbstractContextManager[None]:
    """A connected mailbox for ``pim``, which refuses without one; nothing for
    any other subagent."""
    return connected_mail(client) if name == PimAbility.NAME else nullcontext()


def _frames(listener: RecordingClient) -> list[dict[str, object]]:
    """Every ``agent_changed`` frame the listener saw, whole, in order."""
    return [f for f in listener.frames if f.get("type") == "agent_changed"]


def _listing(client: FlaskClient, query: str = "") -> dict[str, object]:
    response = client.get(f"{_LISTING}{query}")
    assert response.status_code == 200, response.data
    return cast("dict[str, object]", response.get_json())


def _empty_listing() -> dict[str, object]:
    return {"success": True, "result": [], "pagination": {"page": 1, "limit": 20, "total": 0}}


def _row(db: sqlite3.Connection, name: str, turn_id: int) -> sqlite3.Row:
    row = db.execute("SELECT * FROM agents WHERE type = ? AND turn_id = ?", (name, turn_id)).fetchone()
    assert row is not None, f"no agents row for {name} turn {turn_id}"
    return cast("sqlite3.Row", row)


def _final_answer(db: sqlite3.Connection, channel: str, turn_id: int) -> sqlite3.Row:
    """The last assistant row of one turn — the answer it ended on."""
    row = db.execute(
        "SELECT id, content FROM transcript WHERE channel = ? AND turn_id = ? AND role = 'assistant' "
        "ORDER BY id DESC LIMIT 1",
        (channel, turn_id),
    ).fetchone()
    assert row is not None, f"no answer on {channel} turn {turn_id}"
    return cast("sqlite3.Row", row)


def _running(
    agent_id: int, name: str, turn_id: int, *, by: str, in_turn: int, prompt: str = _INSTRUCTIONS,
    summary: str = _SUMMARY,
) -> dict[str, object]:
    """One running agent exactly as the listing and its frame carry it."""
    return {
        "agent_id": agent_id, "spawned_by_channel": by, "spawned_in_turn": in_turn, "type": name,
        "prompt": prompt, "status": "running", "response_id": None, "turn_id": turn_id,
        "channel": _channel(name), "summary": summary,
    }


def _changed(agent: dict[str, object]) -> dict[str, object]:
    return {"type": "agent_changed", "agent": agent}


@pytest.mark.parametrize("name", _SPAWNERS)
def test_each_subagent_the_chat_spawns_is_one_row_listed_while_it_runs_and_done_with_its_answer(
    authed_client: _Client, name: str,
) -> None:
    """The chat spawns one subagent of each kind. While it runs, the listing
    holds exactly its row — numbered past every earlier agent, never by its
    turn id — with who spawned it, its task, its own turn and channel, and the
    call's summary; the call told the model that number. Once it answers, its
    row is ``done`` with the id of that answer, the socket carried the row
    twice — running, then done — and the listing is empty."""
    client, db, _store = authed_client
    held = Held(text(_ANSWER))
    script: list[Entry] = [held]
    provider = ChannelProvider(
        chat=[calls(_spawn(name, act_summary=_SUMMARY)), text(_STARTED), text(_ANSWERED)],
        **{_route(name): script},
    )

    with _mailbox(client, name), listening() as listener, patched(provider):
        parent = begin(UserConfig(), _QUESTION)
        parent.result()
        wait(held.entered, f"the {name} model call")
        while_running = _listing(client)
        held.gate.set()
        drain()
    after = _listing(client)

    assert held.opened and provider.unused() == {}
    [call] = calls_of(db, name)
    turn = child_turn(db, _channel(name), call["id"])
    agent_id = EARLIER_AGENTS + 1
    assert agent_id != turn
    running = _running(agent_id, name, turn, by=USER, in_turn=parent.turn_id)
    assert while_running == {"success": True, "result": [running], "pagination": {"page": 1, "limit": 20, "total": 1}}
    answer = _final_answer(db, _channel(name), turn)
    assert answer["content"] == _ANSWER
    done = {**running, "status": "done", "response_id": answer["id"]}
    assert dict(_row(db, name, turn)) == {k: v for k, v in done.items() if k not in ("channel", "summary")}
    assert _frames(listener) == [_changed(running), _changed(done)]
    assert f"Subagent {name} #{agent_id} has been spawned." in call["result"]
    assert f'"Reply from {name} #{agent_id}"' in call["result"]
    assert after == _empty_listing()


def test_a_subagent_a_scheduled_task_spawns_names_the_schedule_and_its_turn(authed_client: _Client) -> None:
    """A scheduled task spawns a search and waits for it. The search's row
    names the schedule's channel and turn as its spawner, carries an empty
    summary since the call gave none, and ends ``done`` with its answer."""
    client, db, _store = authed_client
    held = Held(text(_ANSWER))
    provider = ChannelProvider(chat=[calls(_spawn(WebSearchAbility.NAME)), text("Done.")], web_search=[held])
    schedule_channel = ScheduledConfig().channel

    with listening() as listener, patched(provider):
        schedule = MessageProcessor.process(ScheduledConfig(), raw_input="write a note", turn_id=7)
        wait(held.entered, "the search's model call")
        while_running = _listing(client)
        held.gate.set()
        schedule.result()
        drain()

    assert held.opened and provider.unused() == {}
    [call] = calls_of(db, WebSearchAbility.NAME)
    turn = child_turn(db, _channel(WebSearchAbility.NAME), call["id"])
    running = _running(EARLIER_AGENTS + 1, WebSearchAbility.NAME, turn, by=schedule_channel, in_turn=7, summary="")
    assert while_running["result"] == [running]
    done = {**running, "status": "done", "response_id": _final_answer(db, _channel(WebSearchAbility.NAME), turn)["id"]}
    assert _frames(listener) == [_changed(running), _changed(done)]


def test_a_subagent_another_subagent_spawns_names_that_subagent_and_its_turn(authed_client: _Client) -> None:
    """The chat starts a subtask, and the subtask spawns a search it waits
    for. While the search runs both are listed, oldest first; the search's row
    names the subtask's channel and turn as its spawner. Both end ``done``,
    the search first, each with its own answer."""
    client, db, _store = authed_client
    held = Held(text("Paris."))
    provider = ChannelProvider(
        chat=[calls(_spawn(SubtaskAbility.NAME, "find out the capital of France")), text(_STARTED), text(_ANSWERED)],
        subtask=[calls(_spawn(WebSearchAbility.NAME)), text(_ANSWER)],
        web_search=[held],
    )

    with listening() as listener, patched(provider):
        parent = begin(UserConfig(), _QUESTION)
        parent.result()
        wait(held.entered, "the search's model call")
        while_running = _listing(client)
        held.gate.set()
        drain()

    assert held.opened and provider.unused() == {}
    sub_turn = child_turn(db, _SUBTASK, calls_of(db, SubtaskAbility.NAME)[0]["id"])
    search_turn = child_turn(db, _channel(WebSearchAbility.NAME), calls_of(db, WebSearchAbility.NAME)[0]["id"])
    subtask = _running(
        EARLIER_AGENTS + 1, SubtaskAbility.NAME, sub_turn, by=USER, in_turn=parent.turn_id,
        prompt="find out the capital of France", summary="",
    )
    search = _running(EARLIER_AGENTS + 2, WebSearchAbility.NAME, search_turn, by=_SUBTASK, in_turn=sub_turn, summary="")
    assert while_running["result"] == [subtask, search]
    search_done = {
        **search, "status": "done", "response_id": _final_answer(db, _channel(WebSearchAbility.NAME), search_turn)["id"],
    }
    subtask_answer = _final_answer(db, _SUBTASK, sub_turn)
    assert subtask_answer["content"] == _ANSWER
    subtask_done = {**subtask, "status": "done", "response_id": subtask_answer["id"]}
    assert _frames(listener) == [_changed(subtask), _changed(search), _changed(search_done), _changed(subtask_done)]


def test_a_subagent_the_person_stops_is_cancelled_once_with_no_answer(authed_client: _Client) -> None:
    """The person stops a running search from its panel. Its row is
    ``cancelled`` with no answer, and the socket carries the row twice —
    running, then cancelled — even though the search's own turn ends after."""
    client, db, _store = authed_client
    channel = _channel(WebSearchAbility.NAME)
    held = Held(text(_ANSWER))
    provider = ChannelProvider(
        chat=[calls(_spawn(WebSearchAbility.NAME)), text(_STARTED), text("It was stopped.")], web_search=[held],
    )

    with listening() as listener, patched(provider):
        parent = begin(UserConfig(), _QUESTION)
        parent.result()
        wait(held.entered, "the search's model call")
        turn = child_turn(db, channel, calls_of(db, WebSearchAbility.NAME)[0]["id"])
        stop = client.delete(f"/api/threads/{turn}?channel={channel}")
        held.gate.set()
        drain()

    assert stop.status_code == 200 and (stop.get_json() or {})["result"]["cancelled"] is True
    assert held.opened and provider.unused() == {}
    running = _running(EARLIER_AGENTS + 1, WebSearchAbility.NAME, turn, by=USER, in_turn=parent.turn_id, summary="")
    cancelled = {**running, "status": "cancelled"}
    assert (_row(db, WebSearchAbility.NAME, turn)["status"], _row(db, WebSearchAbility.NAME, turn)["response_id"]) == (
        "cancelled", None,
    )
    assert _frames(listener) == [_changed(running), _changed(cancelled)]
    assert _listing(client) == _empty_listing()


def test_stopping_the_caller_cancels_the_subtask_it_started_once(authed_client: _Client) -> None:
    """The person stops the chat while the subtask it started is mid-call.
    The subtask's row is ``cancelled`` with no answer, and the socket carries
    exactly one cancelled frame for it."""
    client, db, _store = authed_client
    caller_held = Held(text("Done for now."))
    subtask_held = Held(text(_ANSWER))
    provider = ChannelProvider(chat=[calls(_spawn(SubtaskAbility.NAME)), caller_held], subtask=[subtask_held])

    with listening() as listener, patched(provider):
        parent = begin(UserConfig(), "find the capital of France in the background")
        wait(caller_held.entered, "the caller's second model call")
        wait(subtask_held.entered, "the subtask's model call")
        stop = client.delete(f"/api/threads/{parent.turn_id}")
        caller_held.gate.set()
        subtask_held.gate.set()
        parent.result()
        drain()

    assert stop.status_code == 200 and (stop.get_json() or {})["result"]["cancelled"] is True
    assert caller_held.opened and subtask_held.opened
    turn = child_turn(db, _SUBTASK, calls_of(db, SubtaskAbility.NAME)[0]["id"])
    assert execution(_SUBTASK, turn).stop_reason == "parent stopped"
    running = _running(EARLIER_AGENTS + 1, SubtaskAbility.NAME, turn, by=USER, in_turn=parent.turn_id, summary="")
    assert (_row(db, SubtaskAbility.NAME, turn)["status"], _row(db, SubtaskAbility.NAME, turn)["response_id"]) == (
        "cancelled", None,
    )
    assert _frames(listener) == [_changed(running), _changed({**running, "status": "cancelled"})]


def test_a_subagent_whose_provider_keeps_failing_ends_failed(authed_client: _Client) -> None:
    """The search's provider fails on every attempt, so the search crashes:
    its row is ``failed`` with no answer, sent once."""
    client, db, _store = authed_client
    boom = RuntimeError("boom")
    script: list[Entry] = [Held(boom), *(raises(boom) for _ in range(_MAX_PROVIDER_ATTEMPTS - 1))]
    provider = ChannelProvider(
        chat=[calls(_spawn(WebSearchAbility.NAME)), text(_STARTED), text("It failed.")], web_search=script,
    )

    parent, frames = run_turn(provider, UserConfig(), _QUESTION)

    assert provider.unused() == {}
    turn = child_turn(db, _channel(WebSearchAbility.NAME), calls_of(db, WebSearchAbility.NAME)[0]["id"])
    assert execution(_channel(WebSearchAbility.NAME), turn).state == TurnExecution.CRASHED
    running = _running(EARLIER_AGENTS + 1, WebSearchAbility.NAME, turn, by=USER, in_turn=parent.turn_id, summary="")
    assert (_row(db, WebSearchAbility.NAME, turn)["status"], _row(db, WebSearchAbility.NAME, turn)["response_id"]) == (
        "failed", None,
    )
    assert [f for f in frames if f.get("type") == "agent_changed"] == [
        _changed(running), _changed({**running, "status": "failed"}),
    ]
    assert _listing(client) == _empty_listing()


def test_a_subagent_a_restart_cut_off_stays_listed_until_boot_ends_it_failed(authed_client: _Client) -> None:
    """The process dies while the search is mid-call, so its row is still
    ``running`` — and still listed. The next boot's sweep ends it ``failed``,
    sends that row once, and the listing is empty."""
    client, db, _store = authed_client
    script: list[Entry] = [dies]
    provider = ChannelProvider(
        chat=[calls(_spawn(WebSearchAbility.NAME)), text(_STARTED), text("It was cut off.")], web_search=script,
    )

    with process_deaths_expected():
        parent, _frames_before = run_turn(provider, UserConfig(), _QUESTION)
    turn = child_turn(db, _channel(WebSearchAbility.NAME), calls_of(db, WebSearchAbility.NAME)[0]["id"])
    running = _running(EARLIER_AGENTS + 1, WebSearchAbility.NAME, turn, by=USER, in_turn=parent.turn_id, summary="")
    assert _listing(client)["result"] == [running]

    with listening() as listener, patched(provider):
        swept = MessageProcessor(config_for(ConfigTypeEnum.USER)).turn_execution_service.sweep_orphaned()
        _reply_to_swept_delegates(swept)
        drain()

    assert provider.unused() == {}
    assert (_row(db, WebSearchAbility.NAME, turn)["status"], _row(db, WebSearchAbility.NAME, turn)["response_id"]) == (
        "failed", None,
    )
    assert _frames(listener) == [_changed({**running, "status": "failed"})]
    assert _listing(client) == _empty_listing()


def test_boot_ends_failed_an_agent_left_running_with_no_turn_to_sweep(authed_client: _Client) -> None:
    """A spawn recorded its agent, but the process died before the agent's
    turn wrote a row the sweep could close. The next boot still ends that
    agent ``failed``, sends its row once, and the listing is empty."""
    client, db, _store = authed_client
    with Database.transaction():
        stranded = Agent(
            type=WebSearchAbility.NAME, prompt=_INSTRUCTIONS, spawned_by_channel=USER, spawned_in_turn=1, turn_id=1,
        ).insert()
    assert stranded.agent_id == EARLIER_AGENTS + 1
    assert db.execute("SELECT COUNT(*) FROM turn_executions WHERE ended_at IS NULL").fetchone()[0] == 0
    running = _running(stranded.agent_id, WebSearchAbility.NAME, 1, by=USER, in_turn=1, summary="")
    assert _listing(client)["result"] == [running]

    with listening() as listener:
        swept = MessageProcessor(config_for(ConfigTypeEnum.USER)).turn_execution_service.sweep_orphaned()

    assert swept == []
    ended = _row(db, WebSearchAbility.NAME, 1)
    assert (ended["status"], ended["response_id"]) == ("failed", None)
    assert _frames(listener) == [_changed({**running, "status": "failed"})]
    assert _listing(client) == _empty_listing()


def test_a_subagent_spawned_after_retention_purged_its_channel_is_a_new_agent_on_the_reused_turn(
    authed_client: _Client,
) -> None:
    """The chat's first search answers; retention then purges the search
    channel's old turns, so the next search runs as the same turn id again. It
    is spawned all the same, as a new agent with the next number: its row, its
    frames, the number its call names and the label its reply carries are its
    own, and the first agent keeps its outcome, its purged answer cleared."""
    client, db, _store = authed_client
    name, channel = WebSearchAbility.NAME, _channel(WebSearchAbility.NAME)
    first = ChannelProvider(
        chat=[calls(_spawn(name)), text(_STARTED), text(_ANSWERED)], web_search=[text(_ANSWER)],
    )
    run_turn(first, UserConfig(), _QUESTION)
    turn = child_turn(db, channel, calls_of(db, name)[0]["id"])

    db.execute("UPDATE transcript SET created_at = datetime('now', '-200 days') WHERE channel = ?", (channel,))
    db.commit()
    GarbageCollectorService().run()
    assert db.execute("SELECT COUNT(*) FROM transcript WHERE channel = ?", (channel,)).fetchone()[0] == 0

    task = "find the capital of Spain"
    second = ChannelProvider(
        chat=[calls(_spawn(name, task)), text(_STARTED), text(_ANSWERED)],
        web_search=[text("Madrid is the capital of Spain.")],
    )
    parent, frames = run_turn(second, UserConfig(), "what is the capital of Spain?")

    assert first.unused() == {} and second.unused() == {}
    call = calls_of(db, name)[-1]
    assert child_turn(db, channel, call["id"]) == turn
    first_id, second_id = EARLIER_AGENTS + 1, EARLIER_AGENTS + 2
    rows = db.execute(
        "SELECT agent_id, prompt, status, response_id FROM agents WHERE type = ? AND turn_id = ? ORDER BY agent_id",
        (name, turn),
    ).fetchall()
    answer = _final_answer(db, channel, turn)
    assert [tuple(r) for r in rows] == [
        (first_id, _INSTRUCTIONS, "done", None), (second_id, task, "done", answer["id"]),
    ]
    running = _running(second_id, name, turn, by=USER, in_turn=parent.turn_id, prompt=task, summary="")
    assert [f for f in frames if f.get("type") == "agent_changed"] == [
        _changed(running), _changed({**running, "status": "done", "response_id": answer["id"]}),
    ]
    assert f"Subagent {name} #{second_id} has been spawned." in call["result"]
    assert f"Reply from {name} #{second_id}" in second.read("chat")[-1]
    assert _listing(client) == _empty_listing()


def test_the_listing_pages_the_running_agents_oldest_first(authed_client: _Client) -> None:
    """Three searches run at once; the earlier agents that ended are never
    listed. Two to a page, the first page holds the two oldest and the second
    the newest, each page counting all three."""
    client, db, _store = authed_client
    held = [Held(text(_ANSWER)) for _ in range(3)]
    script: list[Entry] = list(held)
    tasks = ("find the capital of France", "find the capital of Spain", "find the capital of Italy")
    provider = ChannelProvider(
        chat=[calls(*(_spawn(WebSearchAbility.NAME, task) for task in tasks)), text(_STARTED)],
        web_search=script,
    )

    with patched(provider):
        parent = begin(UserConfig(), _QUESTION)
        parent.result()
        for h in held:
            wait(h.entered, "a search's model call")
        first = _listing(client, "?page=1&limit=2")
        second = _listing(client, "?page=2&limit=2")
        for h in held:
            h.gate.set()
        drain()

    numbers = [EARLIER_AGENTS + 1, EARLIER_AGENTS + 2, EARLIER_AGENTS + 3]
    rows = cast("list[dict[str, object]]", first["result"]) + cast("list[dict[str, object]]", second["result"])
    assert [r["agent_id"] for r in rows] == numbers
    assert {r["prompt"] for r in rows} == set(tasks)
    assert {r["status"] for r in rows} == {"running"}
    assert first["pagination"] == {"page": 1, "limit": 2, "total": 3}
    assert second["pagination"] == {"page": 2, "limit": 2, "total": 3}
    by_number = {
        cast("int", r["agent_id"]): r for r in db.execute(
            "SELECT agent_id, turn_id FROM agents WHERE agent_id > ?", (EARLIER_AGENTS,),
        ).fetchall()
    }
    assert [r["turn_id"] for r in rows] == [by_number[n]["turn_id"] for n in numbers]


def test_the_listing_refuses_a_bearer_token_and_a_caller_with_no_session(db: sqlite3.Connection) -> None:
    """The listing serves the person's page only: a valid bearer token is
    refused with the cookie-only error, and no credentials at all is the
    usual sign-in refusal."""
    from api import create_app

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    raw_token, _wrapper_id = WrapperAuthService().create_token(name="an outside agent")

    bearer = client.get(_LISTING, headers={"Authorization": f"Bearer {raw_token}"})
    nobody = client.get(_LISTING)

    assert bearer.status_code == 403
    assert bearer.get_json() == {"success": False, "result": [], "error": "This action requires cookie session auth"}
    assert nobody.status_code == 401
    assert nobody.get_json() == {"error": "Authentication required"}


def _migration_outcome(db: sqlite3.Connection) -> object:
    row = db.execute("SELECT outcome FROM schema_migrations WHERE key = 'agents-table-v1'").fetchone()
    return None if row is None else row[0]


def test_an_install_without_the_agents_table_gets_it_on_boot(db: sqlite3.Connection) -> None:
    """A database provisioned before agents were recorded has no ``agents``;
    the boot migration creates it with its guards and its lookup index — only
    the four statuses, numbers never reused, a turn id a purged channel reuses
    taking a second row, and an answer that is deleted leaves its agent
    without one."""
    db.execute("DROP TABLE agents")
    db.commit()
    run_all(str(FileMapperService.get_db_path()))

    assert _migration_outcome(db) == "applied"
    index = db.execute("SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'idx_agents_type_turn'").fetchone()
    assert index is not None and "agents(type, turn_id)" in index[0]
    answer_id = db.execute("INSERT INTO transcript (channel, role, content) VALUES ('user', 'assistant', 'hi')").lastrowid
    insert = (
        "INSERT INTO agents (spawned_by_channel, spawned_in_turn, type, prompt, turn_id, status, response_id) "
        "VALUES ('user', 1, 'web_search', 'p', ?, ?, ?)"
    )
    first = db.execute(insert, (1, "done", answer_id)).lastrowid
    reused = db.execute(insert, (1, "running", None)).lastrowid
    assert reused == cast("int", first) + 1
    db.execute("DELETE FROM agents WHERE agent_id = ?", (reused,))
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(insert, (2, "finished", None))
    db.execute("DELETE FROM agents WHERE agent_id = ?", (first,))
    second = db.execute(insert, (3, "done", answer_id)).lastrowid
    assert second == cast("int", first) + 2
    db.execute("DELETE FROM transcript WHERE id = ?", (answer_id,))
    assert tuple(db.execute("SELECT status, response_id FROM agents WHERE agent_id = ?", (second,)).fetchone()) == (
        "done", None,
    )


def test_a_fresh_install_records_the_agents_migration_as_not_needed(db: sqlite3.Connection) -> None:
    run_all(str(FileMapperService.get_db_path()))
    assert _migration_outcome(db) == "noop"
