"""Feature tests: a browsing run hands its page to the user for a step only they can take.

A real browsing run (the real browser ability, dispatched by a real processor)
asks the user to sign in; the user's own interface socket hears the ask, the
tab list shows it with its reason, the user signs in on the real page through
the address bar and presses Done, and the run carries on from the page as they
left it. Real Flask app over a real socket, real headless shell, loopback site.
"""

from __future__ import annotations

import json
import threading
from typing import cast

import pytest
from flask import Flask

from controllers.message_processor import MessageProcessor
from tests._browser_live import (
    Site,
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
from tools.browser import session

pytest_plugins = ("tests._browser_live",)
pytestmark = pytest.mark.e2e

_ORIGIN: dict[str, object] = {"type": "user", "turn_id": 1, "forked": False}
_HANDOFF_WAIT = 30.0  # seconds a refused hand-off gets to answer before the test fails


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
            asked = as_frame(
                receive_until(ui, lambda m: as_frame(m, "browser_handoff") is not None),
                "browser_handoff",
            )
            assert asked is not None
            assert (asked["tab"], asked["reason"], asked["origin"]) == (
                tab.id,
                "Please sign in",
                _ORIGIN,
            )
            [row] = listed(client)
            shown = cast("dict[str, object]", row["handoff"])
            assert (shown["id"], shown["reason"]) == (
                asked["handoff"],
                "Please sign in",
            )

            moved = client.post(
                f"/api/browser/navigate/{tab.id}", json={"url": f"{site.url}/login"}
            )
            assert moved.status_code == 204
            assert wait_for(tab, lambda: tab.title == "Signed in"), tab.title
            done = client.post(f"/api/browser/handoffs/{asked['handoff']}")
            assert done.status_code == 204, done.data
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
    assert client.post(f"/api/browser/handoffs/{asked['handoff']}").status_code == 404
    assert [t["handoff"] for t in listed(client)] == [None]


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
