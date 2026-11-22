"""Feature tests: what a tool call's row can show, live and read back.

A call's live frames and its history chips carry the params it was made with,
never its result. The full record, result included, is read on demand by the
call's id, and only by the signed-in person. A subagent call the person's chat
makes ends at once, its result the notice that the subagent started; its record
names the subagent's own turn and whether that turn is still working, and the
subagent's answer arrives later as a reply. A finished ``browser`` call lists
only the page its stored result names: none, quietly, when the result names no
page, even though the call was given a url; none, with one warning naming the
call, when the stored result cannot be read.

Drives real user turns (construct inertly, ``begin()``, ``result()``) against the
real, fully migrated SQLite DB, observed through the real socket registry and
read back through the real Flask app with a real signed-in session. The only
substitution is the LLM network boundary (``services.provider_service.build_client``),
answered from one script per channel, every script used up by the end of a test.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import cast

import pytest
from flask import Flask
from flask.testing import FlaskClient

from abilities.browser import BrowserAbility
from abilities.find_tools import FindToolsAbility
from abilities.web_browse import WebBrowseAbility
from abilities.web_search import WebSearchAbility
from configs.channels.user import UserConfig
from models.provider_response import ProviderResponse
from models.tool_call import ToolCall
from services.turn_serializer_service import TurnSerializerService
from tests._browser_live import bearer_header, signed_in, use_own_data_dir
from tests._delegates import (
    ChannelProvider, Held, agent_number, child, delegate_channel, kind_rows, replies, run_turn,
)

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider", "earlier_agents")]

_SERIALIZER_LOG = TurnSerializerService.__module__
_WEB_SEARCH_CHANNEL = delegate_channel(WebSearchAbility.NAME)
_WEB_BROWSE_CHANNEL = delegate_channel(WebBrowseAbility.NAME)
_SUMMARY = "Looking for the weather tools"
_ANSWER = "Paris is the capital of France."

Frame = dict[str, object]


def _says(text: str) -> ProviderResponse:
    return ProviderResponse(text=text, model="m", tool_calls=None)


def _calls(name: str, **params: object) -> ProviderResponse:
    return ProviderResponse(text="", model="m", tool_calls=[{"name": name, "input": params}])


@pytest.fixture
def app(db: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Flask:
    """The real app on the test's database, keeping its data under *tmp_path*."""
    use_own_data_dir(tmp_path, monkeypatch)
    from api import create_app  # noqa: PLC0415 — built per test, on the test's database

    built = create_app()
    built.config["TESTING"] = True
    return built


@pytest.fixture
def person(app: Flask) -> FlaskClient:
    """The signed-in person's client."""
    client, _token = signed_in(app)
    return client


def _ok(client: FlaskClient, path: str) -> dict[str, object]:
    response = client.get(path)
    assert response.status_code == 200, response.data
    return cast("dict[str, object]", cast("dict[str, object]", response.get_json())["result"])


def _row(db: sqlite3.Connection, tool_name: str) -> sqlite3.Row:
    rows: list[sqlite3.Row] = db.execute(
        "SELECT * FROM tool_calls WHERE tool_name = ? ORDER BY id", (tool_name,),
    ).fetchall()
    [row] = rows
    return row


def _serializer_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == _SERIALIZER_LOG and r.levelno >= logging.WARNING]


def test_a_calls_live_frames_and_history_chips_carry_its_params_but_never_its_result(
    db: sqlite3.Connection, person: FlaskClient,
) -> None:
    provider = ChannelProvider(chat=[_calls(FindToolsAbility.NAME, query=["weather"]), _says("Here they are.")])

    mp, frames = run_turn(provider, UserConfig(), "which weather tools do you have?")

    assert provider.unused() == {}
    row = _row(db, FindToolsAbility.NAME)
    assert row["result"], "the call stored a result, so leaving it off is a choice"
    sent = [f for f in frames if f.get("tool_name") == FindToolsAbility.NAME]
    assert {f["state"] for f in sent} == {"started", "done"}
    assert all(f["id"] == row["id"] and f["params"] == {"query": ["weather"]} for f in sent), sent
    assert all("result" not in f for f in sent)
    block = _ok(person, f"/api/threads/{mp.turn_id}")  # the read validates the whole block
    messages = cast("list[dict[str, object]]", block["messages"])
    [chip] = [
        c for m in messages for c in cast("list[dict[str, object]]", m.get("tool_calls") or [])
        if c["id"] == row["id"]
    ]
    assert chip["params"] == {"query": ["weather"]}
    assert "result" not in chip and "pages" not in chip


def test_a_running_call_reads_back_with_an_empty_result_and_a_finished_one_with_exactly_what_was_stored(
    db: sqlite3.Connection, person: FlaskClient,
) -> None:
    while_running: list[tuple[int, object]] = []

    def read_while_running(frame: Frame) -> None:
        if frame.get("tool_name") == FindToolsAbility.NAME and frame.get("state") == "started":
            response = person.get(f"/api/tool-calls/{frame['id']}")
            while_running.append((response.status_code, response.get_json()))

    provider = ChannelProvider(chat=[
        _calls(FindToolsAbility.NAME, query=["weather"], act_summary=_SUMMARY), _says("Found them."),
    ])

    run_turn(provider, UserConfig(), "which weather tools do you have?", read_while_running)

    assert provider.unused() == {}
    row = _row(db, FindToolsAbility.NAME)
    common = {
        "id": row["id"],
        "tool_name": FindToolsAbility.NAME,
        "params": {"query": ["weather"]},
        "summary": _SUMMARY,
        "created_at": row["created_at"],
        "delegate": None,
        "pages": [],
    }
    [(status, body)] = while_running
    assert status == 200, body
    assert cast("dict[str, object]", body)["result"] == {
        **common, "result": "", "state": "started", "ended_at": None,
    }
    assert row["result"] and row["ended_at"]
    assert _ok(person, f"/api/tool-calls/{row['id']}") == {
        **common, "result": row["result"], "state": "done", "ended_at": row["ended_at"],
    }


def test_a_subagent_call_from_the_chat_names_its_turn_while_it_works_and_after_and_its_answer_arrives_as_a_reply(
    db: sqlite3.Connection, person: FlaskClient,
) -> None:
    """The chat starts a search alongside an ordinary call. The search call
    ends at once, its result the notice that the subagent started. Read while
    the subagent is still working, and again once it has answered, its record
    is that stored notice and names the subagent's own turn, working and then
    not; the answer is never the call's result but arrives as a reply of its
    own. The ordinary call names no subagent turn."""
    search_id: list[object] = []
    mid_run: list[tuple[int, object]] = []

    def note_the_search_call(frame: Frame) -> None:
        if frame.get("tool_name") == WebSearchAbility.NAME and not search_id:
            search_id.append(frame["id"])

    def read_the_search_call() -> None:
        response = person.get(f"/api/tool-calls/{search_id[0]}")
        mid_run.append((response.status_code, response.get_json()))

    search = Held(_says(_ANSWER), then=read_the_search_call)
    provider = ChannelProvider(
        chat=[
            ProviderResponse(text="", model="m", tool_calls=[
                {"name": FindToolsAbility.NAME, "input": {"query": ["weather"]}},
                {"name": WebSearchAbility.NAME, "input": {"instructions": "find the capital of France"}},
            ]),
            _says("I have started a search."),
            _says("It is Paris."),
        ],
        web_search=[search],
    )

    run_turn(provider, UserConfig(), "what is the capital of France?", note_the_search_call)

    assert search.opened and provider.unused() == {}
    row = _row(db, WebSearchAbility.NAME)
    channel, n = child(db, _WEB_SEARCH_CHANNEL, row["id"])
    assert f"Reply from {WebSearchAbility.NAME} #{agent_number(channel)}" in row["result"]
    assert _ANSWER not in row["result"]
    [(status, body)] = mid_run
    assert status == 200, body
    while_working = cast("dict[str, object]", cast("dict[str, object]", body)["result"])
    assert (while_working["state"], while_working["result"], while_working["delegate"]) == (
        "done", row["result"], {
            "channel": channel, "turn_id": n,
            "working": True, "crashed": False, "cancelled": False,
        },
    )
    after = _ok(person, f"/api/tool-calls/{row['id']}")
    assert (after["state"], after["result"], after["delegate"]) == (
        "done", row["result"], {
            "channel": channel, "turn_id": n,
            "working": False, "crashed": False, "cancelled": False,
        },
    )
    assert after["pages"] == []
    [reply] = replies(db, row["id"])
    assert reply["content"] == _ANSWER
    ordinary = _ok(person, f"/api/tool-calls/{_row(db, FindToolsAbility.NAME)['id']}")
    assert ordinary["delegate"] is None


def test_only_the_signed_in_person_reads_a_call_and_only_by_its_numeric_id(
    app: Flask, db: sqlite3.Connection, person: FlaskClient,
) -> None:
    provider = ChannelProvider(chat=[_calls(FindToolsAbility.NAME, query=["weather"]), _says("Done.")])
    run_turn(provider, UserConfig(), "find weather tools")
    assert provider.unused() == {}
    path = f"/api/tool-calls/{_row(db, FindToolsAbility.NAME)['id']}"
    stranger = app.test_client()

    assert person.get(path).status_code == 200
    assert person.get("/api/tool-calls/987654").status_code == 404
    assert person.get("/api/tool-calls/abc").status_code == 400
    not_allowed = [
        person.get("/api/tool-calls/all"),
        person.post("/api/tool-calls/all"),
        person.put("/api/tool-calls/all"),
        person.delete("/api/tool-calls/all"),
        person.post(path),
        person.put(path),
        person.delete(path),
    ]
    assert [r.status_code for r in not_allowed] == [405] * 7, [r.data for r in not_allowed]
    assert stranger.get(path).status_code == 401
    assert stranger.get(path, headers=bearer_header()).status_code == 403


def test_a_finished_browser_call_whose_result_names_no_page_lists_none_though_it_was_given_a_url(
    db: sqlite3.Connection, person: FlaskClient, caplog: pytest.LogCaptureFixture,
) -> None:
    provider = ChannelProvider(chat=[_says("Hello.")])
    mp, _frames = run_turn(provider, UserConfig(), "hello")
    assert provider.unused() == {}
    failed = (
        f"[{BrowserAbility.NAME}(status=error, code=browser-action-failed)]\n"
        f"The browser could not start.\n[end:{BrowserAbility.NAME}]"
    )
    mp.tool_call_service.record(
        BrowserAbility.NAME, {"action": "open", "url": "https://example.com/"}, failed, state=ToolCall.ERROR,
    )
    caplog.set_level(logging.WARNING, logger=_SERIALIZER_LOG)

    call = _ok(person, f"/api/tool-calls/{_row(db, BrowserAbility.NAME)['id']}")

    assert (call["result"], call["pages"]) == (failed, [])
    assert _serializer_warnings(caplog) == []


def test_a_browser_result_that_cannot_be_read_lists_no_page_and_warns_once_naming_the_call(
    db: sqlite3.Connection, person: FlaskClient, caplog: pytest.LogCaptureFixture,
) -> None:
    provider = ChannelProvider(chat=[_says("Hello.")])
    mp, _frames = run_turn(provider, UserConfig(), "hello")
    assert provider.unused() == {}
    unreadable = f'[{BrowserAbility.NAME}(status=success)]\n{{"ok":true}}\n[end:{BrowserAbility.NAME}]'
    mp.tool_call_service.record(BrowserAbility.NAME, {"action": "open", "url": "https://example.com/"}, unreadable)
    call_id = _row(db, BrowserAbility.NAME)["id"]
    caplog.set_level(logging.WARNING, logger=_SERIALIZER_LOG)

    call = _ok(person, f"/api/tool-calls/{call_id}")

    assert call["pages"] == []
    [warning] = _serializer_warnings(caplog)
    assert str(call_id) in warning


def test_a_web_browse_call_refused_before_it_starts_opens_no_run_and_lists_no_pages(
    db: sqlite3.Connection, person: FlaskClient,
) -> None:
    """The call names a parameter web_browse does not declare, so dispatch
    refuses it whole before any subagent turn opens."""
    provider = ChannelProvider(chat=[
        _calls(WebBrowseAbility.NAME, instructions="read the front page", url="https://example.com/"),
        _says("I could not browse."),
    ])

    run_turn(provider, UserConfig(), "what is on example.com?")

    assert provider.unused() == {}
    row = _row(db, WebBrowseAbility.NAME)
    call = _ok(person, f"/api/tool-calls/{row['id']}")
    assert (call["state"], call["result"]) == ("error", row["result"])
    assert (call["delegate"], call["pages"]) == (None, [])
    assert kind_rows(db, _WEB_BROWSE_CHANNEL) == 0
