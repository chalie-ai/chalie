"""Feature tests: a browsing run's tab lives exactly as long as the run.

A real user turn calls the real ``web_browse`` delegate; its child turn drives
the real ``browser`` ability, which opens a real page in the real headless
shell. When the child turn ends, its tab closes in the browser itself, whether
the run finished or the user stopped the turn that started it while the run
was waiting on the user. The only substitution is the LLM network boundary
(``services.provider_service.build_client``), answered from a script chosen by
the tools each request offers.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import cast
from unittest.mock import patch

import pytest
from flask import Flask

from abilities.browser import BrowserAbility
from abilities.web_browse import WebBrowseAbility
from configs.channels.user import UserConfig
from configs.channels.web_browse import WebBrowseConfig
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from models.provider_request import ProviderRequest
from models.provider_response import ProviderResponse
from models.turn_execution import TurnExecution
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
from tests.helpers import NO_PLAN, drain_background_turns, is_planning_request
from tools.browser import tabs
from tools.browser.tabs import Tab

pytest_plugins = ("tests._browser_live",)
pytestmark = [pytest.mark.e2e, pytest.mark.usefixtures("chat_provider")]

_BUILD_CLIENT = "services.provider_service.build_client"
_CHILD_CHANNEL = WebBrowseConfig(PolicyChannel.CHAT).channel
_TERMINAL = ProviderResponse(text="", model="scripted-overflow", tool_calls=None)


def _says(text: str) -> ProviderResponse:
    return ProviderResponse(text=text, model="m", tool_calls=None)


def _calls(name: str, **params: object) -> ProviderResponse:
    return ProviderResponse(
        text="", model="m", tool_calls=[{"name": name, "input": params}]
    )


class _RoutedProvider:
    """Answers the browsing run from *child* and the turn that started it from
    *parent*, told apart by whether the request offers the raw browser tool.

    Past either script it answers with an empty terminal response, which the
    post-turn background turns this test does not script for receive. The tab
    the run has open is noted when its second request arrives, the page being
    open by then."""

    def __init__(
        self, parent: list[ProviderResponse], child: list[ProviderResponse]
    ) -> None:
        self._parent = parent
        self._child = child
        self._lock = threading.Lock()
        self._child_sends = 0
        self.run_tab: Tab | None = None

    def get_context_limit(self) -> int:
        return 200000

    def send(self, dto: object) -> ProviderResponse:
        if is_planning_request(dto):
            return NO_PLAN
        assert isinstance(dto, ProviderRequest)
        offered = {str(tool["name"]) for tool in dto.tools or []}
        with self._lock:
            if BrowserAbility.NAME not in offered:
                return self._parent.pop(0) if self._parent else _TERMINAL
            self._child_sends += 1
            if self._child_sends == 2:
                [row] = tabs.listing()
                self.run_tab = tabs.get(str(row["id"]))
            return self._child.pop(0) if self._child else _TERMINAL


def _call_result(db: sqlite3.Connection, tool: str, needle: str) -> sqlite3.Row:
    """The one recorded call of *tool* whose params mention *needle*."""
    rows: list[sqlite3.Row] = db.execute(
        "SELECT state, result FROM tool_calls WHERE tool_name = ? AND params LIKE ?",
        (tool, f"%{needle}%"),
    ).fetchall()
    [row] = rows
    return row


def test_a_browsing_runs_tab_closes_in_the_browser_when_the_run_finishes(
    app: Flask, site: Site, db: sqlite3.Connection
) -> None:
    client, _token = signed_in(app)
    provider = _RoutedProvider(
        parent=[
            _calls(WebBrowseAbility.NAME, instructions="read the title of page A"),
            _says("Page A is called Page A."),
        ],
        child=[
            _calls(BrowserAbility.NAME, action="open", url=f"{site.url}/a"),
            _says("The page is titled Page A."),
        ],
    )
    turn = MessageProcessor(UserConfig(), raw_input="what is page A called?")
    with patch(_BUILD_CLIENT, return_value=provider):
        turn.begin()
        turn.result()
        drain_background_turns()

    browsed = _call_result(db, WebBrowseAbility.NAME, "page A")
    assert browsed["state"] == "done" and "titled Page A" in browsed["result"]
    run = provider.run_tab
    assert run is not None, "the run never had a page open"
    browsers = launches()
    assert closed_in_browser(run.id)
    assert launches() == browsers  # the same browser answered, not a fresh one
    assert not run.live
    assert listed(client) == []


def test_stopping_the_turn_ends_the_browsing_run_waiting_on_the_user_and_closes_its_tab(
    app: Flask, site: Site, db: sqlite3.Connection
) -> None:
    client, token = signed_in(app)
    provider = _RoutedProvider(
        parent=[_calls(WebBrowseAbility.NAME, instructions="sign in to the site")],
        child=[
            _calls(BrowserAbility.NAME, action="open", url=f"{site.url}/a"),
            _calls(BrowserAbility.NAME, action="handoff", reason="Please sign in"),
        ],
    )
    turn = MessageProcessor(UserConfig(), raw_input="sign me in to the site")
    with (
        patch(_BUILD_CLIENT, return_value=provider),
        live_server(app) as host,
        main_socket(host, token) as ui,
    ):
        turn.begin()
        try:
            receive_until(
                ui,
                lambda m: (
                    as_frame(m, "browser_tabs_changed") is not None
                    and any(t["handoff"] is not None for t in listed(client))
                ),
                timeout=60,
            )
            # The run's tab, and its ask, belong to the turn that started the run.
            started = {"type": "user", "turn_id": turn.turn_id, "forked": False}
            [row] = listed(client)
            assert row["origin"] == started
            assert cast("dict[str, object]", row["handoff"])["origin"] == started
            stopped = client.delete(f"/api/threads/{turn.turn_id}?type=user")
            assert stopped.status_code == 200, stopped.data
            run = provider.run_tab
            assert run is not None, "the run never had a page open"
            assert wait_for(run, lambda: not run.live, timeout=30), (
                "the run's page stayed open after the stop"
            )
        finally:
            close_sessions()  # frees a run still waiting, so the turn can end
            turn.result()
            drain_background_turns()

    [child_turn] = db.execute(
        "SELECT DISTINCT turn_id FROM transcript WHERE channel = ?", (_CHILD_CHANNEL,)
    ).fetchall()
    child = TurnExecution.latest(_CHILD_CHANNEL, int(child_turn["turn_id"]))
    assert child is not None
    assert (child.state, child.stop_reason) == (
        TurnExecution.CANCELLED,
        "parent stopped",
    )
    waited = _call_result(db, BrowserAbility.NAME, "handoff")
    assert waited["state"] == "error" and "handoff-ended" in waited["result"]
    browsers = launches()
    assert closed_in_browser(run.id)
    assert launches() == browsers  # the same browser answered, not a fresh one
    assert listed(client) == []
