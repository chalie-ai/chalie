"""Feature tests: a browsing run hands its page to the user for a step only they can take.

A real browsing run (the real browser ability, dispatched by a real processor)
asks the user to sign in; the user's own interface socket hears the ask open
with its tab, reason and the conversation it belongs to, the user signs in on
the real page through the address bar and says they are done, and the run
carries on from the page as they left it. A run on no conversation the user
sees still asks; the user dismissing it ends the run's wait. Real Flask app
over a real socket, real headless shell, loopback site.
"""

from __future__ import annotations

import json
import threading
from typing import cast

import pytest
from flask import Flask
from flask.testing import FlaskClient

from controllers.message_processor import MessageProcessor
from tests._browser_live import (
    StreamSocket,
    Site,
    as_frame,
    close_sessions,
    live_server,
    main_socket,
    receive_until,
    run_opens,
    signed_in,
    wait_for,
)
from tools.browser import session

pytestmark = pytest.mark.e2e

_ORIGIN: dict[str, object] = {"type": "user", "turn_id": 1, "forked": False}
_HANDOFF_WAIT = 30.0  # seconds a dismissed hand-off gets to answer before the test fails


def _frame(ws: StreamSocket, kind: str) -> dict[str, object]:
    """The next frame of type *kind* the socket *ws* hears."""
    frame = as_frame(receive_until(ws, lambda m: as_frame(m, kind) is not None), kind)
    assert frame is not None
    return frame


def _open_asks(client: FlaskClient) -> list[dict[str, object]]:
    response = client.get("/api/asks/open")
    assert response.status_code == 200, response.data
    return cast("list[dict[str, object]]", cast("dict[str, object]", response.get_json())["result"])


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
    with live_server(app) as host, main_socket(host, token) as ui:
        waiting.start()
        try:
            opened = _frame(ui, "ask_opened")
            ask_id = opened["ask_id"]
            assert (opened["kind"], opened["data"]) == (
                "handoff",
                {"tab_id": tab.id, "reason": "Please sign in", "origin": _ORIGIN},
            )
            # A screen that connects now rebuilds the same ask from the listing.
            assert _open_asks(client) == [{k: v for k, v in opened.items() if k != "type"}]

            moved = client.post(
                f"/api/browser/navigate/{tab.id}", json={"url": f"{site.url}/login"}
            )
            assert moved.status_code == 204
            assert wait_for(tab, lambda: tab.title == "Signed in"), tab.title
            done = client.post(f"/api/asks/reply/{ask_id}", json={"value": None})
            assert done.status_code == 204, done.data
            assert _frame(ui, "ask_closed") == {"type": "ask_closed", "ask_id": ask_id}
            waiting.join(timeout=60)
            assert not waiting.is_alive(), "the run did not carry on after Done"
        finally:
            if waiting.is_alive():  # free the run before the browser goes away
                session.close_session(run.uid or 0)
                waiting.join(timeout=30)

    [(header, envelope)] = results
    assert "status=success" in header, header
    page = cast("dict[str, object]", envelope["page"])
    assert (page["url"], page["title"]) == (f"{site.url}/login", "Signed in")
    again = client.post(f"/api/asks/reply/{ask_id}", json={"value": None})
    assert again.status_code == 404, again.data
    assert _open_asks(client) == []


def test_a_run_no_one_is_watching_still_asks_and_a_dismissal_ends_its_wait(
    app: Flask, site: Site
) -> None:
    client, token = signed_in(app)
    run, tab = run_opens(f"{site.url}/a")  # a run on no conversation the user sees

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
    with live_server(app) as host, main_socket(host, token) as ui:
        asking.start()
        try:
            opened = _frame(ui, "ask_opened")
            assert (opened["kind"], opened["data"]) == (
                "handoff",
                {"tab_id": tab.id, "reason": "Please sign in", "origin": None},
            )
            dismissed = client.post(f"/api/asks/dismiss/{opened['ask_id']}")
            assert dismissed.status_code == 204, dismissed.data
            assert _frame(ui, "ask_closed") == {"type": "ask_closed", "ask_id": opened["ask_id"]}
            asking.join(timeout=_HANDOFF_WAIT)
            assert not asking.is_alive(), "the run kept waiting after the user dismissed the ask"
        finally:
            close_sessions()  # frees a run still waiting
            asking.join(timeout=_HANDOFF_WAIT)

    [ended] = answers
    assert "code=handoff-ended" in ended.splitlines()[0], ended
    assert _open_asks(client) == []
