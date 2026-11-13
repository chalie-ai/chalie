"""Feature tests: what a tool call's row can show, live and read back.

A call's live frames and its history chips carry the params it was made with,
never its result. The full record, result included, is read on demand by the
call's id, and only by the signed-in person. A finished ``browser`` call lists
only the page its stored result names: none, quietly, when the result names no
page, even though the call was given a url; none, with one warning naming the
call, when the stored result cannot be read.

Drives real user turns (construct inertly, ``begin()``, ``result()``) against the
real, fully migrated SQLite DB, observed through the real socket registry and
read back through the real Flask app with a real signed-in session. The only
substitution is the LLM network boundary (``services.provider_service.build_client``).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from flask import Flask
from flask.testing import FlaskClient

from abilities.browser import BrowserAbility
from abilities.find_tools import FindToolsAbility
from abilities.recall import Recall
from abilities.web_browse import WebBrowseAbility
from abilities.web_search import WebSearchAbility
from configs.channels.user import UserConfig
from configs.channels.web_browse import WebBrowseConfig
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from models.policy import Policy
from models.provider_response import ProviderResponse
from models.tool_call import ToolCall
from services.turn_serializer_service import TurnSerializerService
from services.websocket import Websocket
from tests._browser_live import bearer_header, signed_in, use_own_data_dir
from tests.helpers import NO_PLAN, drain_background_turns, is_planning_request

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

_BUILD_CLIENT = "services.provider_service.build_client"
_SERIALIZER_LOG = TurnSerializerService.__module__
_WEB_SEARCH_CHANNEL = "delegate:web_search"
_WEB_BROWSE_CHANNEL = WebBrowseConfig(PolicyChannel.CHAT).channel
_SUMMARY = "Looking for the weather tools"

Frame = dict[str, object]


def _says(text: str) -> ProviderResponse:
    return ProviderResponse(text=text, model="m", tool_calls=None)


def _calls(name: str, **params: object) -> ProviderResponse:
    return ProviderResponse(text="", model="m", tool_calls=[{"name": name, "input": params}])


class _ScriptedProvider:
    """Replays one scripted response per ``send()``, in order; past the script it
    answers with an empty terminal response, which the post-turn background turns
    this module does not script for receive."""

    _TERMINAL = ProviderResponse(text="", model="scripted-overflow", tool_calls=None)

    def __init__(self, *responses: ProviderResponse) -> None:
        self._responses = list(responses)
        self._lock = threading.Lock()

    def get_context_limit(self) -> int:
        return 200000

    def send(self, dto: object) -> ProviderResponse:
        if is_planning_request(dto):
            return NO_PLAN
        with self._lock:
            return self._responses.pop(0) if self._responses else self._TERMINAL


class _RecordingClient:
    """A real socket-registry subscriber that records every frame broadcast to it
    and hands each to *on_frame* on the thread that broadcast it."""

    def __init__(self, on_frame: Callable[[Frame], None] | None) -> None:
        self.frames: list[Frame] = []
        self._on_frame = on_frame

    def close(self) -> None:
        """Nothing to release: the recorded frames stay readable."""

    def send(self, data: str) -> None:
        frame = cast("Frame", json.loads(data))
        self.frames.append(frame)
        if self._on_frame is not None:
            self._on_frame(frame)


@contextmanager
def _listening(on_frame: Callable[[Frame], None] | None) -> Iterator[_RecordingClient]:
    listener = _RecordingClient(on_frame)
    Websocket._connect(listener)
    try:
        yield listener
    finally:
        Websocket._disconnect(listener)


def _run(
    provider: _ScriptedProvider, raw_input: str, on_frame: Callable[[Frame], None] | None = None,
) -> tuple[MessageProcessor, list[Frame]]:
    """Drive one real user turn to its end; the processor and every frame sent meanwhile."""
    with _listening(on_frame) as listener:
        mp = MessageProcessor(UserConfig(), raw_input=raw_input)
        with patch(_BUILD_CLIENT, return_value=provider):
            mp.begin()
            mp.result()
            drain_background_turns()
    return mp, listener.frames


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
    provider = _ScriptedProvider(_calls(FindToolsAbility.NAME, query=["weather"]), _says("Here they are."))

    mp, frames = _run(provider, "which weather tools do you have?")

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

    provider = _ScriptedProvider(
        _calls(FindToolsAbility.NAME, query=["weather"], act_summary=_SUMMARY), _says("Found them."),
    )

    _run(provider, "which weather tools do you have?", read_while_running)

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


def test_a_delegate_calls_record_names_the_child_turn_it_spawned_and_an_ordinary_calls_names_none(
    db: sqlite3.Connection, person: FlaskClient,
) -> None:
    provider = _ScriptedProvider(
        ProviderResponse(text="", model="m", tool_calls=[
            {"name": FindToolsAbility.NAME, "input": {"query": ["weather"]}},
            {"name": WebSearchAbility.NAME, "input": {"instructions": "find the capital of France"}},
        ]),
        _calls(Recall.NAME, query="capital of France"),
        _says("Paris is the capital of France."),
        _says("It is Paris."),
    )

    _run(provider, "what is the capital of France?")

    child = db.execute(
        "SELECT turn_id FROM transcript WHERE channel = ? AND role = ?", (_WEB_SEARCH_CHANNEL, WebSearchAbility.NAME),
    ).fetchone()
    delegate_row = _row(db, WebSearchAbility.NAME)
    delegate_call = _ok(person, f"/api/tool-calls/{delegate_row['id']}")
    assert delegate_call["delegate"] == {"channel": _WEB_SEARCH_CHANNEL, "turn_id": child["turn_id"], "working": False}
    assert delegate_call["result"] == delegate_row["result"] and "Paris" in delegate_row["result"]
    assert delegate_call["pages"] == []
    ordinary = _ok(person, f"/api/tool-calls/{_row(db, FindToolsAbility.NAME)['id']}")
    assert ordinary["delegate"] is None


def test_only_the_signed_in_person_reads_a_call_and_only_by_its_numeric_id(
    app: Flask, db: sqlite3.Connection, person: FlaskClient,
) -> None:
    _run(_ScriptedProvider(_calls(FindToolsAbility.NAME, query=["weather"]), _says("Done.")), "find weather tools")
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
    mp, _frames = _run(_ScriptedProvider(_says("Hello.")), "hello")
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
    mp, _frames = _run(_ScriptedProvider(_says("Hello.")), "hello")
    unreadable = f'[{BrowserAbility.NAME}(status=success)]\n{{"ok":true}}\n[end:{BrowserAbility.NAME}]'
    mp.tool_call_service.record(BrowserAbility.NAME, {"action": "open", "url": "https://example.com/"}, unreadable)
    call_id = _row(db, BrowserAbility.NAME)["id"]
    caplog.set_level(logging.WARNING, logger=_SERIALIZER_LOG)

    call = _ok(person, f"/api/tool-calls/{call_id}")

    assert call["pages"] == []
    [warning] = _serializer_warnings(caplog)
    assert str(call_id) in warning


def test_a_web_browse_call_the_person_has_denied_starts_no_run_and_lists_no_pages(
    db: sqlite3.Connection, person: FlaskClient,
) -> None:
    Policy.upsert(PolicyChannel.CHAT.value, WebBrowseAbility.NAME, "deny")
    provider = _ScriptedProvider(
        _calls(WebBrowseAbility.NAME, instructions="read the front page of example.com"),
        _says("I am not allowed to browse."),
    )

    _run(provider, "what is on example.com?")

    row = _row(db, WebBrowseAbility.NAME)
    call = _ok(person, f"/api/tool-calls/{row['id']}")
    assert (call["state"], call["result"]) == ("error", row["result"])
    assert (call["delegate"], call["pages"]) == (None, [])
    assert db.execute("SELECT COUNT(*) FROM transcript WHERE channel = ?", (_WEB_BROWSE_CHANNEL,)).fetchone()[0] == 0
