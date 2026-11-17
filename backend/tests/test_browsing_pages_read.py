"""Feature tests: the pages a browsing call's record lists.

A ``browser`` call lists the page it ended on: the page a success reports, or
the page an error names when the browser had one open. A call still opening a
page lists the address it was asked for; one that failed before any page loaded
lists none, and neither does one made while the browser shows its own error
page where a page did not load. A ``web_browse`` call lists the pages its run
went through, in order, a page reached twice in a row listed once under the
title it first had — while the run is still working, though the chat's call
that started it ended at once, and after the run has answered.

A real chat turn calls the real ``web_browse`` subagent, which starts its run
and returns at once; the run drives the real ``browser`` ability in the real
headless shell against a loopback site, and its answer arrives on the chat as
a reply. Every record is read back through the real Flask app with a real
signed-in session. The only substitution is the LLM network boundary
(``services.provider_service.build_client``), answered from one script per
channel, every script used up by the end of a test. The run's first step is
held until the chat's turn has ended, so the run always works after its call
has ended, as it does for a person.
"""

from __future__ import annotations

import logging
import socket
import sqlite3
from collections.abc import Callable
from typing import cast

import pytest
from flask import Flask
from flask.testing import FlaskClient

from abilities.browser import BrowserAbility
from abilities.web_browse import WebBrowseAbility
from abilities.web_fetch import WebFetchAbility
from configs.channels.user import UserConfig
from models.provider_response import ProviderResponse
from services.turn_serializer_service import TurnSerializerService
from tests._browser_live import Site, signed_in
from tests._delegates import (
    ChannelProvider,
    Entry,
    Held,
    agent_number,
    calls_of,
    child_turn,
    delegate_channel,
    replies,
    run_turn,
)

pytest_plugins = ("tests._browser_live",)
pytestmark = [pytest.mark.e2e, pytest.mark.usefixtures("chat_provider", "earlier_agents")]

_SERIALIZER_LOG = TurnSerializerService.__module__
_RUN_CHANNEL = delegate_channel(WebBrowseAbility.NAME)
_MISSING = "Nothing like this"

Frame = dict[str, object]
Detail = dict[str, object]


def _says(text: str) -> ProviderResponse:
    return ProviderResponse(text=text, model="m", tool_calls=None)


def _calls(name: str, **params: object) -> ProviderResponse:
    return ProviderResponse(text="", model="m", tool_calls=[{"name": name, "input": params}])


def _page(url: str, title: str) -> dict[str, str]:
    return {"url": url, "title": title}


def _browse(
    instructions: str, run: list[ProviderResponse], answer: str, raw_input: str,
    on_frame: Callable[[Frame], None] | None = None,
) -> None:
    """Drive one real chat turn that starts a browsing run with *instructions*,
    the run taking the steps in *run*, and the chat answering the run's reply
    with *answer*; wait for all of it to end, and check every scripted step was
    taken and the run's first step was let go by the chat's turn ending."""
    first = Held(run[0])
    steps: list[Entry] = [first, *run[1:]]
    provider = ChannelProvider(
        chat=[_calls(WebBrowseAbility.NAME, instructions=instructions), _says("I have started browsing."),
              _says(answer)],
        web_browse=steps,
    )
    run_turn(provider, UserConfig(), raw_input, on_frame)
    assert first.opened and provider.unused() == {}


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

    _browse(
        "send the form and report",
        [
            _calls(BrowserAbility.NAME, action="open", url=form),
            _calls(BrowserAbility.NAME, action="click", target="Send"),
            _calls(BrowserAbility.NAME, action="click", target=_MISSING),
            _calls(BrowserAbility.NAME, action="open", url=tricky),
            _calls(BrowserAbility.NAME, action="click", target=_MISSING),
            _calls(BrowserAbility.NAME, action="open", url=refused),
            _says("The form was sent."),
        ],
        "Done.",
        "send the form on my site",
        read_the_first_open_while_it_runs,
    )

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

    _browse(
        "visit my site's pages",
        [
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
        "Visited.",
        "visit the pages on my site",
        read_the_run_while_it_fetches,
    )

    [call] = calls_of(db, WebBrowseAbility.NAME)
    assert run_id == [call["id"]]
    n = child_turn(db, _RUN_CHANNEL, call["id"])
    # The chat's call ended at once: its result is the notice that the run
    # started, never the run's answer, which arrives as a reply of its own.
    assert f"Reply from {WebBrowseAbility.NAME} #{agent_number(WebBrowseAbility.NAME, n)}" in call["result"]
    assert "Visited every page." not in call["result"]
    [reply] = replies(db, call["id"])
    assert reply["content"] == "Visited every page."
    [fetching] = mid_run
    assert (fetching["state"], fetching["result"], fetching["delegate"]) == (
        "done", call["result"], {
            "channel": _RUN_CHANNEL, "turn_id": n,
            "working": True, "crashed": False, "cancelled": False,
        },
    )
    assert fetching["pages"] == [_page(count, "Visit 1")]
    finished = _detail(client, call["id"])
    assert (finished["state"], finished["result"], finished["delegate"]) == (
        "done", call["result"], {
            "channel": _RUN_CHANNEL, "turn_id": n,
            "working": False, "crashed": False, "cancelled": False,
        },
    )
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
    _browse(
        "visit my site's pages",
        [
            _calls(BrowserAbility.NAME, action="open", url=a),
            _calls(BrowserAbility.NAME, action="open", url=refused),  # the browser shows its own error page
            _calls(BrowserAbility.NAME, action="read"),  # made on that error page
            _calls(BrowserAbility.NAME, action="open", url=b),
            _says("Visited what loaded."),
        ],
        "Visited.",
        "visit the pages on my site",
    )

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
