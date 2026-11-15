"""Feature tests: a browsing run's tab lives exactly as long as the run.

A real chat turn, or a real scheduled task, calls the real ``web_browse``
subagent; the subagent's own turn drives the real ``browser`` ability, which opens
a real page in the real headless shell. The person's chat gets the call back at
once and the run's answer later as a reply; a scheduled task waits for the run.
Whichever way the run ends — it finishes, the person stops it from its own
panel while it waits on them, or the person stops the scheduled task waiting on
it — its tab closes in the browser itself, and only the chat ever hears back.

The only substitution is the LLM network boundary, answered per channel by the
shared :class:`tests._delegates.ChannelProvider`.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import cast

import pytest
from flask import Flask
from flask.testing import FlaskClient

from abilities.browser import BrowserAbility
from abilities.web_browse import WebBrowseAbility
from configs.channels.scheduled import ScheduledConfig
from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.provider_response import ProviderResponse
from models.turn_execution import TurnExecution
from services.turn_execution_service import STOP_REASON_PARENT_STOPPED
from tests._browser_live import (
    Site,
    as_frame,
    close_sessions,
    closed_in_browser,
    launches,
    listed,
    live_server,
    main_socket,
    receive_until,
    signed_in,
    wait_for,
)
from tests._delegates import (
    USER,
    ChannelProvider,
    Entry,
    Held,
    assistant_texts,
    begin,
    calls,
    calls_of,
    child_turn,
    delegate_channel,
    drain,
    execution,
    linked_rows,
    patched,
    replies,
    text,
    tool,
)
from tools.browser import tabs
from tools.browser.tabs import Tab

pytest_plugins = ("tests._browser_live",)
pytestmark = [pytest.mark.e2e, pytest.mark.usefixtures("chat_provider")]

_RUN_CHANNEL = delegate_channel(WebBrowseAbility.NAME)
_SCHEDULE_ID = 7
_RUN_WAIT_S = 60.0  # a first page load launches the browser
_ANSWER = "The page is titled Page A."
_STARTED = "I have started it."


class _RunTab:
    """The tab a browsing run has open, noted when the run's next model call
    arrives — its page is open by then."""

    def __init__(self) -> None:
        self.tab: Tab | None = None

    def noted_then(self, outcome: ProviderResponse) -> Entry:
        def entry() -> ProviderResponse:
            [row] = tabs.listing()
            self.tab = tabs.get(str(row["id"]))
            return outcome

        return entry

    def open(self) -> Tab:
        assert self.tab is not None, "the run never had a page open"
        return self.tab


def _signs_in(site: Site, run: _RunTab) -> list[Entry]:
    """A run that opens page A, then hands its tab to the person to sign in."""
    return [
        calls(tool(BrowserAbility.NAME, action="open", url=f"{site.url}/a")),
        run.noted_then(calls(tool(BrowserAbility.NAME, action="handoff", reason="Please sign in"))),
    ]


def _waits_on_the_person(client: FlaskClient) -> Callable[[str | bytes], bool]:
    """A tab-list change after which a listed tab is handed to the person."""
    return lambda m: (
        as_frame(m, "browser_tabs_changed") is not None
        and any(t["handoff"] is not None for t in listed(client))
    )


def _assert_owned_by(client: FlaskClient, origin: dict[str, object]) -> None:
    """The one listed tab, and the person's hand-off on it, belong to *origin*."""
    [row] = listed(client)
    assert row["origin"] == origin
    assert cast("dict[str, object]", row["handoff"])["origin"] == origin


def _assert_closed(client: FlaskClient, run: Tab) -> None:
    """The run's tab is gone from the browser itself — the same browser, not a
    fresh one — and from the tab list."""
    browsers = launches()
    assert closed_in_browser(run.id)
    assert launches() == browsers
    assert not run.live
    assert listed(client) == []


def _hand_off_ended(db: sqlite3.Connection) -> bool:
    """The run's second browser call — its hand-off — failed as ended."""
    _opened, waited = calls_of(db, BrowserAbility.NAME)
    return waited["state"] == "error" and "handoff-ended" in waited["result"]


def test_a_browsing_run_started_from_the_chat_answers_later_and_its_tab_closes_when_it_finishes(
    app: Flask, site: Site, db: sqlite3.Connection
) -> None:
    """The chat's call comes back at once under the run's number and reply
    label, with no answer in it, and the chat's turn ends before the run has
    even opened its page. The run then reads the page and finishes: its answer
    comes back to the chat as one reply the chat answers, and its tab is closed
    in the same browser and no longer listed."""
    client, _token = signed_in(app)
    run = _RunTab()
    held = Held(calls(tool(BrowserAbility.NAME, action="open", url=f"{site.url}/a")))
    provider = ChannelProvider(
        chat=[
            calls(tool(WebBrowseAbility.NAME, instructions="read the title of page A")),
            text(_STARTED),
            text("Page A is called Page A."),
        ],
        web_browse=[held, run.noted_then(text(_ANSWER))],
    )

    with patched(provider):
        parent = begin(UserConfig(), "what is page A called?")
        parent.result()
        provider.release_all()
        drain(_RUN_WAIT_S)

    assert held.opened and provider.unused() == {}
    [call] = calls_of(db, WebBrowseAbility.NAME)
    n = child_turn(db, _RUN_CHANNEL, call["id"])
    assert call["state"] == "done", call["result"]
    assert f"Reply from web_browse #{n}" in call["result"]
    assert _ANSWER not in call["result"]
    assert assistant_texts(db, USER, parent.turn_id)[-1] == _STARTED

    [reply] = replies(db, call["id"])
    assert reply["content"] == _ANSWER
    assert reply["turn_id"] != parent.turn_id
    assert execution(USER, reply["turn_id"]).state == TurnExecution.COMPLETED
    assert assistant_texts(db, USER, reply["turn_id"]) == ["Page A is called Page A."]
    assert f"Reply from web_browse #{n}" in provider.texts("chat")[-1]

    _assert_closed(client, run.open())


def test_the_person_stops_a_browsing_run_waiting_on_them_from_its_panel_and_the_chat_hears_once(
    app: Flask, site: Site, db: sqlite3.Connection
) -> None:
    """The chat's turn has ended and the run it started has handed its tab to
    the person, both carrying the chat turn as their origin. The person stops
    the run from the run's own panel: the run ends cancelled, its hand-off
    fails as ended, its tab is closed in the browser and no longer listed, and
    the chat gets exactly one reply — read under the label its call promised —
    which it answers."""
    client, token = signed_in(app)
    run = _RunTab()
    provider = ChannelProvider(
        chat=[
            calls(tool(WebBrowseAbility.NAME, instructions="sign in to the site")),
            text(_STARTED),
            text("It was stopped before signing in."),
        ],
        web_browse=_signs_in(site, run),
    )

    with patched(provider), live_server(app) as host, main_socket(host, token) as ui:
        parent = begin(UserConfig(), "sign me in to the site")
        try:
            parent.result()
            receive_until(ui, _waits_on_the_person(client), timeout=_RUN_WAIT_S)
            assert execution(USER, parent.turn_id).state == TurnExecution.COMPLETED
            _assert_owned_by(client, {"type": "user", "turn_id": parent.turn_id, "forked": False})

            [call] = calls_of(db, WebBrowseAbility.NAME)
            n = child_turn(db, _RUN_CHANNEL, call["id"])
            stopped = client.delete(f"/api/threads/{n}?channel={_RUN_CHANNEL}")
            assert stopped.status_code == 200, stopped.data
            assert (stopped.get_json() or {})["result"]["cancelled"] is True
            tab = run.open()
            assert wait_for(tab, lambda: not tab.live, timeout=30), "the run's page stayed open after the stop"
        finally:
            close_sessions()  # frees a run still waiting, so its turn can end

    assert provider.unused() == {}
    assert execution(_RUN_CHANNEL, n).state == TurnExecution.CANCELLED
    assert _hand_off_ended(db)
    _assert_closed(client, tab)

    [reply] = replies(db, call["id"])
    assert reply["content"] == "This web_browse was stopped before it finished."
    assert execution(USER, reply["turn_id"]).state == TurnExecution.COMPLETED
    assert assistant_texts(db, USER, reply["turn_id"]) == ["It was stopped before signing in."]
    assert f"Reply from web_browse #{n}" in provider.texts("chat")[-1]


def test_stopping_a_scheduled_task_stops_the_browsing_run_it_waits_on_and_nobody_is_replied_to(
    app: Flask, site: Site, db: sqlite3.Connection
) -> None:
    """A scheduled task calls ``web_browse`` and waits on it; the run hands
    its tab to the person, both carrying the scheduled task's run as their
    origin. The person stops the scheduled task, which is still open: the stop
    reaches the run, which ends cancelled because its caller stopped, its
    hand-off fails as ended, and its tab is closed in the browser and no longer
    listed. Nothing is sent back as a reply — not to the schedule, not to the
    person's chat."""
    client, token = signed_in(app)
    run = _RunTab()
    provider = ChannelProvider(
        chat=[calls(tool(WebBrowseAbility.NAME, instructions="sign in to the site"))],
        web_browse=_signs_in(site, run),
    )
    schedule = ScheduledConfig().channel

    with patched(provider), live_server(app) as host, main_socket(host, token) as ui:
        task = MessageProcessor.process(ScheduledConfig(), raw_input="sign in to the site", turn_id=_SCHEDULE_ID)
        try:
            receive_until(ui, _waits_on_the_person(client), timeout=_RUN_WAIT_S)
            _assert_owned_by(client, {"type": "scheduled", "turn_id": _SCHEDULE_ID, "forked": False})

            stopped = client.delete(f"/api/threads/{_SCHEDULE_ID}?type=scheduled")
            assert stopped.status_code == 200, stopped.data
            assert (stopped.get_json() or {})["result"]["cancelled"] is True
            tab = run.open()
            assert wait_for(tab, lambda: not tab.live, timeout=30), "the run's page stayed open after the stop"
        finally:
            close_sessions()  # frees a run still waiting, so its turn can end
            task.result()

    assert provider.unused() == {}
    assert execution(schedule, _SCHEDULE_ID).state == TurnExecution.CANCELLED
    [call] = calls_of(db, WebBrowseAbility.NAME)
    n = child_turn(db, _RUN_CHANNEL, call["id"])
    ended = execution(_RUN_CHANNEL, n)
    assert (ended.state, ended.stop_reason) == (TurnExecution.CANCELLED, STOP_REASON_PARENT_STOPPED)
    assert _hand_off_ended(db)
    _assert_closed(client, tab)

    assert replies(db, call["id"]) == []
    assert replies(db, call["id"], schedule) == []
    assert linked_rows(db, USER) == []
