"""Feature tests: the tab list names the conversation each browsing run's tab belongs to.

A browsing run's tab carries the interactive turn the run belongs to, so the
interface can show the tab where the user is. A popup the user opens from that
tab while the run waits on them is kept as the user's own tab, and still belongs
to the run's conversation. The user's own tabs, and a run on no conversation the
user sees, belong to none. Real Flask app, real tab registry, real headless
shell, loopback site; the popup opens from the user's real click over the
live-page socket.
"""

from __future__ import annotations

import json
import threading
from typing import cast

import pytest
from flask import Flask
from flask.testing import FlaskClient

from tests._browser_live import (
    FORM_BUTTON,
    Site,
    as_frame,
    cookie_header,
    listed,
    live_server,
    main_socket,
    open_user_tab,
    receive_until,
    run_opens,
    signed_in,
    stream_socket,
    wait_for,
)
from tools.browser import session, tabs

pytest_plugins = ("tests._browser_live",)
pytestmark = pytest.mark.e2e

# A thread's turn, so a tab can only list it by carrying the run's own origin.
_ORIGIN: dict[str, object] = {"type": "user", "turn_id": 7, "forked": True}


def test_each_tab_lists_the_conversation_of_the_run_that_opened_it_and_the_users_own_none(
    site: Site, user: tuple[FlaskClient, str]
) -> None:
    client, _token = user
    mine = open_user_tab(client, f"{site.url}/a", "Page A")
    watched, run = run_opens(f"{site.url}/b", _ORIGIN)
    assert [(t["id"], t["opener"], t["origin"]) for t in listed(client)] == [
        (mine.id, "user", None),
        (run.id, "chalie", _ORIGIN),
    ]

    session.close_session(watched.uid or 0)  # that run ends; one no one sees starts
    _unseen, hidden = run_opens(f"{site.url}/a")
    assert [(t["id"], t["opener"], t["origin"]) for t in listed(client)] == [
        (mine.id, "user", None),
        (hidden.id, "chalie", None),
    ]


def test_a_popup_the_user_opens_while_a_run_waits_on_them_stays_in_the_runs_conversation(
    app: Flask, site: Site
) -> None:
    client, token = signed_in(app)
    run, tab = run_opens(f"{site.url}/opener", _ORIGIN)

    answers: list[str] = []
    waiting = threading.Thread(
        target=lambda: answers.append(
            run.dispatch_service.dispatch(
                "browser", {"action": "handoff", "reason": "Please sign in"}
            )
        ),
        name="handoff-run",
        daemon=True,
    )
    with live_server(app) as host, main_socket(host, token) as ui:
        waiting.start()
        try:
            receive_until(
                ui,
                lambda m: (
                    as_frame(m, "browser_tabs_changed") is not None
                    and any(t["handoff"] is not None for t in listed(client))
                ),
            )
            with stream_socket(host, tab.id, cookie_header(token)) as ws:
                receive_until(ws, lambda m: as_frame(m, "meta") is not None)
                for event in ("mousePressed", "mouseReleased"):
                    ws.send(
                        json.dumps(
                            {
                                "type": "mouse",
                                "event": event,
                                "x": FORM_BUTTON[0],
                                "y": FORM_BUTTON[1],
                                "button": "left",
                                "clickCount": 1,
                                "modifiers": 0,
                            }
                        )
                    )
                receive_until(
                    ui,
                    lambda m: (
                        as_frame(m, "browser_tabs_changed") is not None
                        and len(listed(client)) == 2
                    ),
                )
            popup = tabs.get(str(listed(client)[1]["id"]))
            assert popup is not None, "the popup closed"
            assert wait_for(popup, lambda: popup.title == "Page B"), popup.title

            [asking, kept] = listed(client)
            assert (asking["id"], asking["opener"], asking["origin"]) == (
                tab.id,
                "chalie",
                _ORIGIN,
            )
            assert (kept["url"], kept["opener"], kept["origin"], kept["handoff"]) == (
                f"{site.url}/b",
                "user",
                _ORIGIN,
                None,
            )

            handoff = cast("dict[str, object]", asking["handoff"])
            done = client.post(f"/api/browser/handoffs/{handoff['id']}")
            assert done.status_code == 204, done.data
            waiting.join(timeout=60)
            assert not waiting.is_alive(), "the run did not carry on after Done"
        finally:
            if waiting.is_alive():  # free the run before the browser goes away
                session.close_session(run.uid or 0)
                waiting.join(timeout=30)
