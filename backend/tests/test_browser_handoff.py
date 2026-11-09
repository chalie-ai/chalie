"""Feature tests: a browsing run hands its page to the user for a step only they can take.

A real browsing run (the real browser ability, dispatched by a real processor)
asks the user to sign in; the user's own interface socket hears the tab list
change, the list shows the ask with its reason and the conversation it belongs
to, the user signs in on the real page through the address bar and presses
Done, and the run carries on from the page as they left it. Real Flask app over
a real socket, real headless shell, loopback site.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import cast

import pytest
from flask import Flask
from flask.testing import FlaskClient

from controllers.message_processor import MessageProcessor
from tests._browser_live import (
    Site,
    StreamSocket,
    as_frame,
    close_sessions,
    listed,
    live_server,
    main_socket,
    receive_until,
    run_opens,
    signed_in,
    wait_for,
)
from tools.browser import session, tabs

pytest_plugins = ("tests._browser_live",)
pytestmark = pytest.mark.e2e

_ORIGIN: dict[str, object] = {"type": "user", "turn_id": 1, "forked": False}
_HANDOFF_WAIT = 30.0  # seconds a refused hand-off gets to answer before the test fails


def _heard(ws: StreamSocket, frames: list[str], want: Callable[[], bool]) -> None:
    """Read *ws* until a tab-list change arrives after which *want* holds, keeping
    every text frame read on the way in *frames*."""

    def note(message: str | bytes) -> bool:
        if isinstance(message, str):
            frames.append(message)
        return as_frame(message, "browser_tabs_changed") is not None and want()

    receive_until(ws, note)


def _asking(client: FlaskClient) -> Callable[[], bool]:
    return lambda: any(t["handoff"] is not None for t in listed(client))


def _browse(mp: MessageProcessor, **params: str) -> tuple[str, dict[str, object]]:
    """Dispatch one browser action; its result header line and its page envelope."""
    rendered = mp.dispatch_service.dispatch("browser", dict[str, object](params))
    header, _, body = rendered.partition("\n")
    return header, cast("dict[str, object]", json.loads(body.splitlines()[0]))


def test_a_run_waits_on_the_user_to_sign_in_and_carries_on_from_the_page_they_left(
    app: Flask, site: Site
) -> None:
    client, token = signed_in(app)
    run, tab = run_opens(f"{site.url}/a", _ORIGIN)

    results: list[tuple[str, dict[str, object]]] = []
    waiting = threading.Thread(
        target=lambda: results.append(
            _browse(run, action="handoff", reason="Please sign in")
        ),
        name="handoff-run",
        daemon=True,
    )
    frames: list[str] = []
    with live_server(app) as host, main_socket(host, token) as ui:
        waiting.start()
        try:
            _heard(ui, frames, _asking(client))
            [row] = listed(client)
            asked = cast("dict[str, object]", row["handoff"])
            assert (row["id"], row["origin"]) == (tab.id, _ORIGIN)
            assert (asked["reason"], asked["origin"]) == ("Please sign in", _ORIGIN)

            moved = client.post(
                f"/api/browser/navigate/{tab.id}", json={"url": f"{site.url}/login"}
            )
            assert moved.status_code == 204
            assert wait_for(tab, lambda: tab.title == "Signed in"), tab.title
            done = client.post(f"/api/browser/handoffs/{asked['id']}")
            assert done.status_code == 204, done.data
            waiting.join(timeout=60)
            assert not waiting.is_alive(), "the run did not carry on after Done"
            tabs.announce()  # one more change frame; any frame the ask sent comes before it
            _heard(ui, frames, lambda: True)
        finally:
            if waiting.is_alive():  # free the run before the browser goes away
                session.close_session(run.uid or 0)
                waiting.join(timeout=30)

    [(header, envelope)] = results
    assert "status=success" in header, header
    page = cast("dict[str, object]", envelope["page"])
    assert (page["url"], page["title"]) == (f"{site.url}/login", "Signed in")
    assert client.post(f"/api/browser/handoffs/{asked['id']}").status_code == 404
    assert [t["handoff"] for t in listed(client)] == [None]
    # The tab list is the one announcement: no frame of its own names the ask.
    assert not [f for f in frames if str(asked["id"]) in f], frames


def test_a_run_no_one_is_watching_cannot_hand_its_page_over(
    app: Flask, site: Site
) -> None:
    client, _token = signed_in(app)
    run, _tab = run_opens(f"{site.url}/a")  # a run on no conversation the user sees

    # Should the run wait for a user after all, it waits here, not on the test.
    answers: list[str] = []
    asking = threading.Thread(
        target=lambda: answers.append(
            run.dispatch_service.dispatch(
                "browser", {"action": "handoff", "reason": "Please sign in"}
            )
        ),
        name="handoff-run",
        daemon=True,
    )
    asking.start()
    try:
        asking.join(timeout=_HANDOFF_WAIT)
        assert not asking.is_alive(), "the run waited for a user no one can reach"
        [refused] = answers
        assert "code=handoff-unavailable" in refused.splitlines()[0], refused
        assert [t["handoff"] for t in listed(client)] == [None]
    finally:
        close_sessions()  # frees a run still waiting
        asking.join(timeout=_HANDOFF_WAIT)
