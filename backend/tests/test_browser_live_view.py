"""Feature tests: the browser's live view — its tab list, address bar and live page.

Drives the real Flask app (cookie-authenticated through a real signed-in
session), the real tab registry and the real headless shell against a loopback
site, and the live page over a real WebSocket upgrade on a real socket. The
browser profile lives under the test's tmp dir (see ``_browser_live``).
"""

from __future__ import annotations

import io
import json
from typing import cast

import pytest
from flask import Flask
from flask.testing import FlaskClient
from PIL import Image

from tests._browser_live import (
    FORM_BACKGROUND,
    FORM_BUTTON,
    FORM_INPUT,
    Site,
    as_frame,
    closed_in_browser,
    cookie_header,
    launches,
    listed,
    live_server,
    open_user_tab,
    page_targets,
    receive_until,
    run_opens,
    stream_socket,
    wait_for,
)
from tools.browser import session
from tools.browser.tabs import Tab

pytest_plugins = ("tests._browser_live",)
pytestmark = pytest.mark.e2e


def _shows(tab: Tab, url: str, title: str) -> bool:
    return wait_for(tab, lambda: (tab.url, tab.title) == (url, title))


def test_a_tab_the_user_opens_outlives_a_browsing_run_and_closes_only_when_they_close_it(
    site: Site, user: tuple[FlaskClient, str]
) -> None:
    client, _token = user
    mine = open_user_tab(client, f"{site.url}/a", "Page A")
    assert listed(client) == [
        {
            "id": mine.id,
            "url": f"{site.url}/a",
            "title": "Page A",
            "opener": "user",
            "origin": None,
            "handoff": None,
        }
    ]

    browsing, run = run_opens(f"{site.url}/b")
    assert [t["id"] for t in listed(client)] == [mine.id, run.id]
    session.close_session(browsing.uid or 0)  # the run ends
    assert closed_in_browser(run.id)
    assert [t["id"] for t in listed(client)] == [mine.id]
    assert mine.live

    browsers = launches()
    assert client.delete(f"/api/browser/tabs/{mine.id}").status_code == 204
    assert listed(client) == []
    assert closed_in_browser(mine.id)
    assert launches() == browsers
    assert client.delete(f"/api/browser/tabs/{mine.id}").status_code == 404


def test_the_user_cannot_close_a_browsing_runs_tab_or_one_that_does_not_exist(
    site: Site, user: tuple[FlaskClient, str]
) -> None:
    client, _token = user
    _browsing, run = run_opens(f"{site.url}/a")

    assert client.delete(f"/api/browser/tabs/{run.id}").status_code == 409
    assert run.id in page_targets()
    assert [t["id"] for t in listed(client)] == [run.id]
    assert client.delete("/api/browser/tabs/no-such-tab").status_code == 404
    assert run.live


def test_the_address_bar_back_forward_and_reload_steer_the_real_page(
    site: Site, user: tuple[FlaskClient, str]
) -> None:
    client, _token = user
    tab = open_user_tab(client, f"{site.url}/a", "Page A")

    def steer(verb: str, body: dict[str, str]) -> None:
        assert (
            client.post(f"/api/browser/{verb}/{tab.id}", json=body).status_code == 204
        )

    steer("navigate", {"url": f"{site.url}/b"})
    assert _shows(tab, f"{site.url}/b", "Page B")
    steer("history", {"action": "back"})
    assert _shows(tab, f"{site.url}/a", "Page A")
    steer("history", {"action": "forward"})
    assert _shows(tab, f"{site.url}/b", "Page B")
    steer("navigate", {"url": f"{site.url}/count"})
    assert _shows(tab, f"{site.url}/count", "Visit 1")
    steer("history", {"action": "reload"})
    assert _shows(tab, f"{site.url}/count", "Visit 2")
    assert site.visits == 2
    assert [(t["url"], t["title"]) for t in listed(client)] == [
        (f"{site.url}/count", "Visit 2")
    ]

    assert (
        client.post(
            "/api/browser/navigate/no-such-tab", json={"url": f"{site.url}/a"}
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/api/browser/history/no-such-tab", json={"action": "back"}
        ).status_code
        == 404
    )


def _near(
    pixel: tuple[int, ...], colour: tuple[int, int, int], tolerance: int = 12
) -> bool:
    return all(
        abs(got - want) <= tolerance for got, want in zip(pixel, colour, strict=False)
    )


def test_the_live_page_shows_the_real_page_and_the_users_clicks_and_typing_reach_it(
    app: Flask, site: Site, user: tuple[FlaskClient, str]
) -> None:
    client, token = user
    tab = open_user_tab(client, f"{site.url}/form", "Form")

    with (
        live_server(app) as host,
        stream_socket(host, tab.id, cookie_header(token)) as ws,
    ):
        first = as_frame(
            receive_until(ws, lambda m: as_frame(m, "meta") is not None), "meta"
        )
        assert first is not None and (first["url"], first["title"]) == (
            f"{site.url}/form",
            "Form",
        )
        frame = receive_until(ws, lambda m: isinstance(m, bytes))
        picture = Image.open(io.BytesIO(cast(bytes, frame))).convert("RGB")
        width, height = picture.size
        pixel = cast(
            "tuple[int, int, int]",
            picture.getpixel((width * 7 // 10, height * 7 // 10)),
        )
        assert _near(pixel, FORM_BACKGROUND), pixel

        def press(at: tuple[int, int]) -> None:
            for event in ("mousePressed", "mouseReleased"):
                ws.send(
                    json.dumps(
                        {
                            "type": "mouse",
                            "event": event,
                            "x": at[0],
                            "y": at[1],
                            "button": "left",
                            "clickCount": 1,
                            "modifiers": 0,
                        }
                    )
                )

        press(FORM_INPUT)
        ws.send(json.dumps({"type": "text", "text": "hello world"}))
        press(FORM_BUTTON)

        echoed = f"{site.url}/echo?q=hello+world"
        arrived = receive_until(
            ws,
            lambda m: (as_frame(m, "meta") or {}).get("title") == "Echo: hello world",
        )
        assert (as_frame(arrived, "meta") or {}).get("url") == echoed
    assert site.echoed == ["hello world"]
    assert (tab.url, tab.title) == (echoed, "Echo: hello world")


def test_the_page_is_streamed_only_while_someone_watches_it(
    app: Flask, site: Site, user: tuple[FlaskClient, str]
) -> None:
    client, token = user
    tab = open_user_tab(client, f"{site.url}/anim", "Moving")
    unwatched = tab.frames
    assert site.ticked(3), "the page stopped painting"
    assert tab.frames == unwatched, "frames arrived with nobody watching"

    with (
        live_server(app) as host,
        stream_socket(host, tab.id, cookie_header(token)) as ws,
    ):
        assert wait_for(tab, lambda: tab.frames >= unwatched + 4), (
            tab.frames
        )  # the seed and a stream behind it
        receive_until(ws, lambda m: isinstance(m, bytes))
        receive_until(ws, lambda m: isinstance(m, bytes))
    assert wait_for(tab, lambda: tab.viewers == 0), tab.viewers

    # Frames already in flight may land while the page paints a few more; after
    # that it keeps painting, and nothing more is streamed.
    assert site.ticked(3), "the page stopped painting"
    quiet = tab.frames
    assert site.ticked(3), "the page stopped painting"
    assert tab.frames == quiet, "the page kept streaming after its last viewer left"
    assert tab.live
