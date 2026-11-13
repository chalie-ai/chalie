"""Feature tests: the pages a browsing call's record lists.

A ``browser`` call lists the page it ended on: the page a success reports, or
the page an error names when the browser had one open. A call still opening a
page lists the address it was asked for; one that failed before any page loaded
lists none, and neither does one made while the browser shows its own error
page where a page did not load. A ``web_browse`` call lists the pages its run
went through, in order, a page reached twice in a row listed once under the
title it first had.

A real user turn calls the real ``web_browse`` delegate; its child turn drives
the real ``browser`` ability in the real headless shell against a loopback
site, and every record is read back through the real Flask app with a real
signed-in session. The only substitution is the LLM network boundary
(``services.provider_service.build_client``), answered from a script chosen by
the tools each request offers.
"""

from __future__ import annotations

import json
import logging
import socket
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import cast
from unittest.mock import patch

import pytest
from flask import Flask
from flask.testing import FlaskClient

from abilities.browser import BrowserAbility
from abilities.web_browse import WebBrowseAbility
from abilities.web_fetch import WebFetchAbility
from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.provider_request import ProviderRequest
from models.provider_response import ProviderResponse
from services.turn_serializer_service import TurnSerializerService
from services.websocket import Websocket
from tests._browser_live import Site, signed_in
from tests.helpers import NO_PLAN, drain_background_turns, is_planning_request

pytest_plugins = ("tests._browser_live",)
pytestmark = [pytest.mark.e2e, pytest.mark.usefixtures("chat_provider")]

_BUILD_CLIENT = "services.provider_service.build_client"
_SERIALIZER_LOG = TurnSerializerService.__module__
_TERMINAL = ProviderResponse(text="", model="scripted-overflow", tool_calls=None)
_MISSING = "Nothing like this"

Frame = dict[str, object]
Detail = dict[str, object]


def _says(text: str) -> ProviderResponse:
    return ProviderResponse(text=text, model="m", tool_calls=None)


def _calls(name: str, **params: object) -> ProviderResponse:
    return ProviderResponse(text="", model="m", tool_calls=[{"name": name, "input": params}])


def _page(url: str, title: str) -> dict[str, str]:
    return {"url": url, "title": title}


class _RoutedProvider:
    """Answers the browsing run from *child* and the turn that started it from
    *parent*, told apart by whether the request offers the raw browser tool.
    Past either script it answers with an empty terminal response, which the
    post-turn background turns this module does not script for receive."""

    def __init__(self, parent: list[ProviderResponse], child: list[ProviderResponse]) -> None:
        self._parent = parent
        self._child = child
        self._lock = threading.Lock()

    def get_context_limit(self) -> int:
        return 200000

    def send(self, dto: object) -> ProviderResponse:
        if is_planning_request(dto):
            return NO_PLAN
        assert isinstance(dto, ProviderRequest)
        offered = {str(tool["name"]) for tool in dto.tools or []}
        with self._lock:
            script = self._child if BrowserAbility.NAME in offered else self._parent
            return script.pop(0) if script else _TERMINAL


class _RecordingClient:
    """A real socket-registry subscriber that hands every frame broadcast to it
    to *on_frame*, on the thread that broadcast it."""

    def __init__(self, on_frame: Callable[[Frame], None]) -> None:
        self._on_frame = on_frame

    def close(self) -> None:
        """Nothing to release."""

    def send(self, data: str) -> None:
        self._on_frame(cast("Frame", json.loads(data)))


@contextmanager
def _listening(on_frame: Callable[[Frame], None]) -> Iterator[None]:
    listener = _RecordingClient(on_frame)
    Websocket._connect(listener)
    try:
        yield
    finally:
        Websocket._disconnect(listener)


def _browse(provider: _RoutedProvider, raw_input: str, on_frame: Callable[[Frame], None]) -> None:
    """Drive one real user turn, and the browsing run it starts, to their end."""
    with _listening(on_frame):
        turn = MessageProcessor(UserConfig(), raw_input=raw_input)
        with patch(_BUILD_CLIENT, return_value=provider):
            turn.begin()
            turn.result()
            drain_background_turns()


def _detail(client: FlaskClient, call_id: object) -> Detail:
    response = client.get(f"/api/tool-calls/{call_id}")
    assert response.status_code == 200, response.data
    return cast("Detail", cast("dict[str, object]", response.get_json())["result"])


def _closed_port() -> int:
    """A loopback port nothing listens on: bound to learn a free one, then let go."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return cast("int", probe.getsockname()[1])


def test_each_browser_call_lists_the_page_it_ended_on_or_while_opening_the_address_it_was_asked_for(
    app: Flask, site: Site, db: sqlite3.Connection, caplog: pytest.LogCaptureFixture,
) -> None:
    client, _token = signed_in(app)
    caplog.set_level(logging.WARNING, logger=_SERIALIZER_LOG)
    form = f"{site.url}/form"
    sent = f"{site.url}/echo?q="
    # A page whose title reads like a second url in the record's header.
    tricky = f"{site.url}/echo?q=x%2C%20url%3Dhttps%3A%2F%2Fevil.example"
    refused = f"http://127.0.0.1:{_closed_port()}/"
    while_opening: list[Detail] = []

    def read_the_first_open_while_it_runs(frame: Frame) -> None:
        if frame.get("tool_name") == BrowserAbility.NAME and frame.get("state") == "started" and not while_opening:
            while_opening.append(_detail(client, frame["id"]))

    provider = _RoutedProvider(
        parent=[_calls(WebBrowseAbility.NAME, instructions="send the form and report"), _says("Done.")],
        child=[
            _calls(BrowserAbility.NAME, action="open", url=form),
            _calls(BrowserAbility.NAME, action="click", target="Send"),
            _calls(BrowserAbility.NAME, action="click", target=_MISSING),
            _calls(BrowserAbility.NAME, action="open", url=tricky),
            _calls(BrowserAbility.NAME, action="click", target=_MISSING),
            _calls(BrowserAbility.NAME, action="open", url=refused),
            _says("The form was sent."),
        ],
    )

    _browse(provider, "send the form on my site", read_the_first_open_while_it_runs)

    [running] = while_opening
    assert (running["state"], running["result"], running["pages"]) == ("started", "", [_page(form, "")])
    rows = db.execute("SELECT id, state FROM tool_calls WHERE tool_name = ? ORDER BY id", (BrowserAbility.NAME,))
    calls = [(row["state"], _detail(client, row["id"])["pages"]) for row in rows.fetchall()]
    assert calls == [
        ("done", [_page(form, "Form")]),
        ("done", [_page(sent, "Echo:")]),  # the page the click navigated to
        ("error", [_page(sent, "Echo:")]),  # the page open when the click found nothing
        ("done", [_page(tricky, "Echo: x, url=https://evil.example")]),
        ("error", [_page(tricky, "Echo: x, url=https://evil.example")]),
        ("error", []),  # nothing loaded: no page reached, though it was asked for one
    ]
    assert [r.getMessage() for r in caplog.records if r.name == _SERIALIZER_LOG] == []


def test_a_browsing_run_lists_the_pages_it_went_through_in_order_while_it_runs_and_after(
    app: Flask, site: Site, db: sqlite3.Connection, caplog: pytest.LogCaptureFixture,
) -> None:
    client, _token = signed_in(app)
    caplog.set_level(logging.WARNING, logger=_SERIALIZER_LOG)
    count, a, b = f"{site.url}/count", f"{site.url}/a", f"{site.url}/b"
    run_id: list[object] = []
    mid_run: list[Detail] = []

    def read_the_run_while_it_fetches(frame: Frame) -> None:
        if frame.get("state") != "started":
            return
        if frame.get("tool_name") == WebBrowseAbility.NAME and not run_id:
            run_id.append(frame["id"])
        elif frame.get("tool_name") == WebFetchAbility.NAME and run_id:
            mid_run.append(_detail(client, run_id[0]))

    provider = _RoutedProvider(
        parent=[_calls(WebBrowseAbility.NAME, instructions="visit my site's pages"), _says("Visited.")],
        child=[
            _calls(BrowserAbility.NAME, action="open", url=count),
            _calls(BrowserAbility.NAME, action="open", url=count),  # the same page again, retitled
            _calls(BrowserAbility.NAME, action="open"),  # no address: refused before it runs
            _calls(WebFetchAbility.NAME, url=b),  # not the browser: lists nothing, even while it runs
            _calls(BrowserAbility.NAME, action="open", url=a),
            _calls(BrowserAbility.NAME, action="open", url=b),
            _calls(BrowserAbility.NAME, action="open", url=a),
            _calls(BrowserAbility.NAME, action="click", target=_MISSING),  # still on page A
            _says("Visited every page."),
        ],
    )

    _browse(provider, "visit the pages on my site", read_the_run_while_it_fetches)

    [run] = run_id
    [fetching] = mid_run
    delegate = cast("dict[str, object]", fetching["delegate"])
    assert (fetching["state"], delegate["working"]) == ("started", True)
    assert fetching["pages"] == [_page(count, "Visit 1")]
    finished = _detail(client, run)
    assert (finished["state"], cast("dict[str, object]", finished["delegate"])["working"]) == ("done", False)
    assert finished["pages"] == [_page(count, "Visit 1"), _page(a, "Page A"), _page(b, "Page B"), _page(a, "Page A")]
    assert [r.getMessage() for r in caplog.records if r.name == _SERIALIZER_LOG] == []
    states = db.execute("SELECT state FROM tool_calls WHERE tool_name = ? ORDER BY id", (BrowserAbility.NAME,))
    assert [row["state"] for row in states.fetchall()] == ["done", "done", "error", "done", "done", "done", "error"]


def test_a_page_that_did_not_load_is_no_page_reached_by_a_call_made_on_its_error_page_nor_by_the_run(
    app: Flask, site: Site, db: sqlite3.Connection, caplog: pytest.LogCaptureFixture,
) -> None:
    client, _token = signed_in(app)
    caplog.set_level(logging.INFO, logger=BrowserAbility.__module__)
    a, b = f"{site.url}/a", f"{site.url}/b"
    refused = f"http://127.0.0.1:{_closed_port()}/"
    provider = _RoutedProvider(
        parent=[_calls(WebBrowseAbility.NAME, instructions="visit my site's pages"), _says("Visited.")],
        child=[
            _calls(BrowserAbility.NAME, action="open", url=a),
            _calls(BrowserAbility.NAME, action="open", url=refused),  # the browser shows its own error page
            _calls(BrowserAbility.NAME, action="read"),  # made on that error page
            _calls(BrowserAbility.NAME, action="open", url=b),
            _says("Visited what loaded."),
        ],
    )

    _browse(provider, "visit the pages on my site", lambda _frame: None)

    rows = db.execute("SELECT id, state, result FROM tool_calls WHERE tool_name = ? ORDER BY id",
                      (BrowserAbility.NAME,)).fetchall()
    assert [(row["state"], _detail(client, row["id"])["pages"]) for row in rows] == [
        ("done", [_page(a, "Page A")]),
        ("error", []),
        ("error", []),
        ("done", [_page(b, "Page B")]),
    ]
    # What the model was told about the error page: no page is open, and which address did not load.
    assert rows[2]["result"].split("\n")[:2] == [
        f"[{BrowserAbility.NAME}(status=error, code=no-open-page)]",
        f"No page is open: {refused} did not load.",
    ]
    [run] = db.execute("SELECT id FROM tool_calls WHERE tool_name = ?", (WebBrowseAbility.NAME,)).fetchall()
    assert _detail(client, run["id"])["pages"] == [_page(a, "Page A"), _page(b, "Page B")]
    # The audit trail names the address that did not load, never the error page.
    verbs = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[BROWSER] verb read ")]
    assert [line.split(" url=")[1] for line in verbs] == [f"{refused} outcome=no-open-page"]
