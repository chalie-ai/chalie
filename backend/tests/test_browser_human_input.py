"""Feature tests: the built-in browser as a person uses it through the live panel.

A person never talks to the page directly: the panel sends the size of its canvas
and, as they use the mouse and keyboard, mouse, key and text frames over the live
page socket (``/ws/browser/<tab id>``). These tests send exactly those frames, over
the real app on a real port, to the real headless shell on a throwaway profile, and
watch what loopback sites receive. A page counts as "reached" only when a site saw
the request its script or form made.

Two sites are used: the *main* site on ``127.0.0.1`` and the *frame* site on
``localhost``. Different hosts are different origins (and different sites), so a
page of the one embedding a page of the other is cross-origin.
"""

from __future__ import annotations

import html
import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast
from urllib.parse import parse_qs, urlsplit

import pytest
from flask import Flask
from flask.testing import FlaskClient

from tests._browser_live import (
    StreamSocket,
    as_frame,
    cookie_header,
    live_server,
    open_user_tab,
    receive_until,
    run_opens,
    stream_socket,
    wait_for,
)
from tools.browser.tabs import Tab

pytest_plugins = ("tests._browser_live",)
pytestmark = pytest.mark.e2e

_WAIT = 15.0  # seconds a site gets to see a request, or a page to show a title
_QUIET = 2.0  # seconds a site is watched for a request that must not come

VIEW_WIDTH, VIEW_HEIGHT = 900, 700

# A button every page below places at the same rect: left 200, top 150, 120 x 40.
BUTTON_RECT = (200, 150, 120, 40)
BUTTON_CENTRE = (260, 170)
_BUTTON_STYLE = (
    "position:absolute; left:200px; top:150px; width:120px; height:40px; margin:0"
)
_RESET = "<style>html,body{margin:0;padding:0}</style>"


# --------------------------------------------------------------------------- sites


@dataclass(frozen=True)
class Req:
    """One request a site received."""

    method: str
    target: str  # path and query
    cookie: str | None
    body: str

    @property
    def path(self) -> str:
        return urlsplit(self.target).path

    @property
    def query(self) -> dict[str, list[str]]:
        return parse_qs(urlsplit(self.target).query)


Reply = tuple[int, list[tuple[str, str]], str]
Route = Callable[[Req], Reply]


def page(body: str, *headers: tuple[str, str], status: int = 200) -> Reply:
    return status, [("Content-Type", "text/html; charset=utf-8"), *headers], body


def cookies_of(req: Req) -> dict[str, str]:
    jar: SimpleCookie = SimpleCookie()
    jar.load(req.cookie or "")
    return {name: morsel.value for name, morsel in jar.items()}


class Origin(ThreadingHTTPServer):
    """A loopback site that records every request it gets and answers from its routes."""

    def __init__(self, name: str) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.url = f"http://{name}:{self.server_address[1]}"
        self.routes: dict[str, Route] = {}
        self.changed = threading.Condition()
        self.requests: list[Req] = []

    def hits(self, path: str) -> list[Req]:
        with self.changed:
            return [r for r in self.requests if r.path == path]

    def wait_hits(self, path: str, count: int = 1) -> list[Req]:
        """The requests for *path* once there are *count*; whatever there are on timeout."""
        with self.changed:
            self.changed.wait_for(
                lambda: sum(r.path == path for r in self.requests) >= count, _WAIT
            )
            return [r for r in self.requests if r.path == path]

    def stays_quiet(self, path: str, count: int) -> bool:
        """Whether *path* is NOT requested more than *count* times over ``_QUIET`` s."""
        with self.changed:
            return not self.changed.wait_for(
                lambda: sum(r.path == path for r in self.requests) > count, _QUIET
            )

    def seen(self) -> list[str]:
        """Every request received, as ``path [cookie]``, for a failure message."""
        with self.changed:
            return [f"{r.method} {r.target} [{r.cookie}]" for r in self.requests]


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._serve("")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self._serve(self.rfile.read(length).decode())

    def _serve(self, body: str) -> None:
        origin = cast(Origin, self.server)
        req = Req(self.command, self.path, self.headers.get("Cookie"), body)
        with origin.changed:
            origin.requests.append(req)
            origin.changed.notify_all()
        route = origin.routes.get(req.path)
        status, headers, text = (
            route(req) if route else page("<title>Missing</title>", status=404)
        )
        self.send_response(status)
        self.send_header("Cache-Control", "no-store")
        for name, value in headers:
            self.send_header(name, value)
        data = text.encode()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 — the base class's name
        pass  # the suite's output is not a request log


@contextmanager
def _serving(name: str) -> Iterator[Origin]:
    origin = Origin(name)
    thread = threading.Thread(
        target=origin.serve_forever, name=f"site-{name}", daemon=True
    )
    thread.start()
    try:
        yield origin
    finally:
        origin.shutdown()
        origin.server_close()
        thread.join(timeout=5)


@pytest.fixture
def main() -> Iterator[Origin]:
    """The site a person browses to, on 127.0.0.1."""
    with _serving("127.0.0.1") as origin:
        yield origin


@pytest.fixture
def other() -> Iterator[Origin]:
    """A second site, on ``localhost``: another origin from *main*."""
    with _serving("localhost") as origin:
        yield origin


# ------------------------------------------------------------------ the person's side


def _mouse(ws: StreamSocket, event: str, at: tuple[int, int], button: str) -> None:
    ws.send(
        json.dumps(
            {
                "type": "mouse",
                "event": event,
                "x": at[0],
                "y": at[1],
                "button": button,
                "clickCount": 0 if event == "mouseMoved" else 1,
                "modifiers": 0,
            }
        )
    )


def click(ws: StreamSocket, at: tuple[int, int]) -> None:
    """A left click at page point *at*, as the panel sends it: the pointer arrives, then goes down and up."""
    _mouse(ws, "mouseMoved", at, "none")
    _mouse(ws, "mousePressed", at, "left")
    _mouse(ws, "mouseReleased", at, "left")


def press_enter(ws: StreamSocket) -> None:
    """The Enter key, down and up, as the panel's key frames carry it (it types a carriage return)."""
    for event in ("keyDown", "keyUp"):
        ws.send(
            json.dumps(
                {
                    "type": "key",
                    "event": event,
                    "key": "Enter",
                    "code": "Enter",
                    "text": "\r" if event == "keyDown" else None,
                    "keyCode": 13,
                    "modifiers": 0,
                }
            )
        )


def type_text(ws: StreamSocket, text: str) -> None:
    ws.send(json.dumps({"type": "text", "text": text}))


@contextmanager
def watching(
    app: Flask, token: str, tab: Tab, scale: float = 1
) -> Iterator[StreamSocket]:
    """The panel open on *tab*: a live-page socket on which the panel's size frame
    has been sent and the page has taken that size."""
    with (
        live_server(app) as host,
        stream_socket(host, tab.id, cookie_header(token)) as ws,
    ):
        ws.send(
            json.dumps(
                {
                    "type": "size",
                    "width": VIEW_WIDTH,
                    "height": VIEW_HEIGHT,
                    "scale": scale,
                }
            )
        )
        receive_until(
            ws,
            lambda m: (
                (meta := as_frame(m, "meta")) is not None
                and (meta.get("width"), meta.get("height")) == (VIEW_WIDTH, VIEW_HEIGHT)
            ),
        )
        yield ws


def _title_is(tab: Tab, *titles: str) -> bool:
    return wait_for(tab, lambda: tab.title in titles, _WAIT)


def _address_bar(client: FlaskClient, tab: Tab, verb: str, body: dict[str, str]) -> None:
    response = client.post(f"/api/browser/{verb}/{tab.id}", json=body)
    assert response.status_code == 204, response.data


# --------------------------------------------------------------------------- tests


@pytest.mark.parametrize("scale", [1, 2])
def test_a_click_at_a_buttons_centre_in_the_panel_lands_on_that_button_at_scale(
    app: Flask, user: tuple[FlaskClient, str], main: Origin, scale: int
) -> None:
    client, token = user
    main.routes["/p"] = lambda _r: page(
        f"<title>Button</title>{_RESET}"
        f'<button id="b" style="{_BUTTON_STYLE}" '
        "onclick=\"fetch('/clicked?x=' + event.clientX + '&y=' + event.clientY)\">Go</button>"
    )
    tab = open_user_tab(client, f"{main.url}/p", "Button")

    with watching(app, token, tab, scale) as ws:
        click(ws, BUTTON_CENTRE)
        hits = main.wait_hits("/clicked")

    left, top, width, height = BUTTON_RECT
    assert hits, (
        f"at scale {scale} the button never received a click; site saw {main.seen()}"
    )
    got = parse_qs(urlsplit(hits[0].target).query)
    x, y = float(got["x"][0]), float(got["y"][0])
    assert left <= x <= left + width and top <= y <= top + height, (
        f"at scale {scale} the page received a click at ({x}, {y}); the button is "
        f"x {left}..{left + width}, y {top}..{top + height}; the click was sent at {BUTTON_CENTRE}"
    )


def test_a_click_inside_a_cross_origin_frame_reaches_the_frames_own_script(
    app: Flask, user: tuple[FlaskClient, str], main: Origin, other: Origin
) -> None:
    client, token = user
    main.routes["/parent"] = lambda _r: page(
        f"<title>Parent</title>{_RESET}"
        f'<iframe src="{other.url}/frame" style="position:absolute; left:100px; '
        'top:100px; width:400px; height:300px; border:0"></iframe>'
    )
    # The frame's button sits at (50, 50) in the frame, 120 x 40: page point (210, 170).
    other.routes["/frame"] = lambda _r: page(
        f"<title>Frame</title>{_RESET}"
        '<button style="position:absolute; left:50px; top:50px; width:120px; height:40px; margin:0" '
        "onclick=\"fetch('/frame-clicked?x=' + event.clientX + '&y=' + event.clientY)\">In frame</button>"
        "<script>fetch('/frame-loaded')</script>"
    )
    tab = open_user_tab(client, f"{main.url}/parent", "Parent")

    with watching(app, token, tab) as ws:
        assert other.wait_hits("/frame-loaded"), (
            f"the frame's script never ran; frame site saw {other.seen()}"
        )
        click(ws, (210, 170))
        hits = other.wait_hits("/frame-clicked")

    assert hits, (
        "the click at page point (210, 170) never reached the frame's script; "
        f"frame site saw {other.seen()}; main site saw {main.seen()}"
    )
    got = parse_qs(urlsplit(hits[0].target).query)
    assert (float(got["x"][0]), float(got["y"][0])) == (110.0, 70.0), (
        f"the frame's script saw the click at {got}; expected (110, 70) in the frame"
    )


def _cookie_site(main: Origin) -> None:
    main.routes["/set-cookie"] = lambda _r: page(
        "<title>Set cookie</title>"
        f'<a id="go" href="/whoami" style="display:block; {_BUTTON_STYLE}">Who am I</a>',
        ("Set-Cookie", "sid=abc; Path=/"),
    )
    main.routes["/whoami"] = lambda r: page(
        f"<title>{html.escape(r.cookie or 'No cookie')}</title>"
    )


def test_a_cookie_a_page_sets_is_sent_on_the_next_page_the_person_clicks_through_to(
    app: Flask, user: tuple[FlaskClient, str], main: Origin
) -> None:
    client, token = user
    _cookie_site(main)
    tab = open_user_tab(client, f"{main.url}/set-cookie", "Set cookie")

    with watching(app, token, tab) as ws:
        click(ws, BUTTON_CENTRE)
        main.wait_hits("/whoami")
        shown = _title_is(tab, "sid=abc", "No cookie")

    assert shown, f"the link never led to /whoami; tab is {tab.url!r} {tab.title!r}"
    assert tab.title == "sid=abc", (
        f"/whoami showed {tab.title!r}; the server saw cookie "
        f"{[h.cookie for h in main.hits('/whoami')]} on it"
    )


def test_a_cookie_a_page_sets_is_sent_on_a_page_opened_from_the_address_bar(
    user: tuple[FlaskClient, str], main: Origin
) -> None:
    client, _token = user
    _cookie_site(main)
    tab = open_user_tab(client, f"{main.url}/set-cookie", "Set cookie")

    _address_bar(client, tab, "navigate", {"url": f"{main.url}/whoami"})
    shown = _title_is(tab, "sid=abc", "No cookie")

    assert shown, f"the address bar never reached /whoami; tab is {tab.url!r} {tab.title!r}"
    assert tab.title == "sid=abc", (
        f"/whoami showed {tab.title!r}; the server saw cookie "
        f"{[h.cookie for h in main.hits('/whoami')]} on it"
    )


def test_a_cookie_a_cross_origin_frame_sets_is_sent_on_the_frames_next_request(
    user: tuple[FlaskClient, str], main: Origin, other: Origin
) -> None:
    client, _token = user
    main.routes["/parent"] = lambda _r: page(
        f'<title>Parent</title><iframe src="{other.url}/frame"></iframe>'
    )
    other.routes["/frame"] = lambda _r: page(
        "<title>Frame</title><script>fetch('/frame-ping')</script>",
        ("Set-Cookie", "fid=xyz; Path=/; SameSite=None; Secure"),
    )
    tab = open_user_tab(client, f"{main.url}/parent", "Parent")
    first = other.wait_hits("/frame-ping", 1)
    assert first, f"the frame never loaded; frame site saw {other.seen()}"

    _address_bar(client, tab, "history", {"action": "reload"})
    pings = other.wait_hits("/frame-ping", 2)

    assert len(pings) == 2, (
        f"the frame did not load again after the parent reloaded; frame site saw {other.seen()}"
    )
    assert "fid=xyz" in (pings[1].cookie or ""), (
        "the frame's second /frame-ping carried cookie "
        f"{pings[1].cookie!r}, not the fid=xyz the frame set; frame site saw {other.seen()}"
    )


def test_a_login_form_filled_and_submitted_from_the_panel_signs_the_person_in_and_keeps_them_signed_in(
    app: Flask, user: tuple[FlaskClient, str], main: Origin
) -> None:
    client, token = user
    main.routes["/login"] = lambda r: (
        (
            303,
            [
                ("Location", "/account"),
                ("Set-Cookie", f"session={parse_qs(r.body)['user'][0]}; Path=/"),
            ],
            "",
        )
        if r.method == "POST"
        else page(
            f"<title>Login</title>{_RESET}"
            '<form method="post" action="/login">'
            '<input name="user" style="position:absolute; left:20px; top:20px; width:300px; height:40px; box-sizing:border-box">'
            '<input name="pass" type="password" style="position:absolute; left:20px; top:80px; width:300px; height:40px; box-sizing:border-box">'
            '<button style="position:absolute; left:20px; top:140px; width:120px; height:40px">Sign in</button>'
            "</form>"
        )
    )
    main.routes["/account"] = lambda r: page(
        f"<title>Signed in as {html.escape(cookies_of(r)['session'])}</title>"
        if "session" in cookies_of(r)
        else "<title>Signed out</title>"
    )
    tab = open_user_tab(client, f"{main.url}/login", "Login")

    with watching(app, token, tab) as ws:
        click(ws, (170, 40))
        type_text(ws, "alice")
        click(ws, (170, 100))
        type_text(ws, "hunter2")
        press_enter(ws)
        shown = _title_is(tab, "Signed in as alice", "Signed out")

    assert shown, (
        f"the form never reached /account; tab is {tab.url!r} {tab.title!r}; "
        f"site saw {main.seen()}"
    )
    assert tab.title == "Signed in as alice", (
        f"after submitting, the page showed {tab.title!r}; the login POST body was "
        f"{[h.body for h in main.hits('/login') if h.method == 'POST']}; site saw {main.seen()}"
    )

    _address_bar(client, tab, "navigate", {"url": f"{main.url}/account?again=1"})
    assert wait_for(tab, lambda: tab.url.endswith("again=1"), _WAIT), tab.url
    assert _title_is(tab, "Signed in as alice", "Signed out")
    assert tab.title == "Signed in as alice", (
        f"on the next page load the page showed {tab.title!r}; "
        f"/account was sent cookies {[h.cookie for h in main.hits('/account')]}"
    )


def test_a_page_that_sets_a_cookie_and_reloads_itself_when_clicked_ends_up_past_its_gate(
    app: Flask, user: tuple[FlaskClient, str], main: Origin
) -> None:
    client, token = user
    main.routes["/gate"] = lambda r: page(
        "<title>Verified</title>"
        if "ok" in cookies_of(r)
        else (
            f"<title>Gate</title>{_RESET}"
            f'<button style="{_BUTTON_STYLE}" '
            "onclick=\"document.cookie='ok=1; path=/'; location.reload()\">Verify</button>"
        )
    )
    tab = open_user_tab(client, f"{main.url}/gate", "Gate")

    with watching(app, token, tab) as ws:
        click(ws, BUTTON_CENTRE)
        shown = _title_is(tab, "Verified")

    loads = main.hits("/gate")
    assert shown, (
        f"after the click the page still showed {tab.title!r}; /gate was requested "
        f"{len(loads)} time(s) with cookies {[h.cookie for h in loads]}"
    )
    assert main.stays_quiet("/gate", 2), (
        f"the page kept reloading; /gate cookies seen {[h.cookie for h in main.hits('/gate')]}"
    )
    assert len(main.hits("/gate")) == 2, (
        f"expected the first load and one reload, /gate cookies seen {[h.cookie for h in main.hits('/gate')]}"
    )


def test_a_pages_script_runs_and_chalie_reads_what_it_wrote(main: Origin) -> None:
    main.routes["/script"] = lambda _r: page(
        "<title>Script</title><h1 id='h'>Loading</h1>"
        "<script>document.getElementById('h').textContent = 'Written by the page script'</script>"
    )
    run, _tab = run_opens(f"{main.url}/script")

    rendered = run.dispatch_service.dispatch("browser", {"action": "read"})
    header, body = rendered.splitlines()[:2]
    assert "status=success" in header, rendered
    text = str(
        cast("dict[str, dict[str, object]]", json.loads(body))["data"]["text"]
    )
    assert "Written by the page script" in text, f"Chalie read {text!r}"
