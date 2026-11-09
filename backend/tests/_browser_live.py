"""Shared harness for the browser live-view feature tests.

Everything here is real: a loopback site the headless shell browses, the real
Flask app served over a real socket (the live-page stream needs a real HTTP
upgrade), real cookie sessions and bearer tokens minted by the services that
mint them, and the real socket registry. The one redirection is the data
directory, pointed at the test's tmp dir so the browser profile a test signs in
to is its own and never the install's.

A test module loads the fixtures below with ``pytest_plugins = ("tests._browser_live",)``.
That module imports this one before pytest can mark it for assertion rewriting, so
it opts out (PYTEST_DONT_REWRITE) and every assert here carries its own message.
"""

from __future__ import annotations

import html
import json
import socket
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import parse_qs, urlsplit

import pytest
from flask import Flask, Response
from flask.testing import FlaskClient
from simple_websocket import Client, ConnectionClosed
from werkzeug.serving import make_server

from configs.channels.web_browse import WebBrowseConfig
from configs.enums.policy_channel import PolicyChannel
from contracts.constants.auth import SESSION_COOKIE_NAME
from controllers.message_processor import MessageProcessor
from services.auth_session_service import create_session
from services.file_mapper_service import FileMapperService
from services.wrapper_auth_service import WrapperAuthService
from tools.browser import pool, session, tabs
from tools.browser.cdp import Browser, get_objects
from tools.browser.tabs import Tab

FORM_INPUT = (170, 40)  # centre of the form's text field, page CSS px
FORM_BUTTON = (120, 125)  # centre of its submit button
FORM_BACKGROUND = (0, 128, 255)

_CLOSE_WAIT = 10.0  # seconds a page or a socket gets to close before the test fails
_TICK_WAIT = 15.0  # seconds the animated page gets to report the ticks a test waits on
_PAINTS_PER_TICK = 10

_PAGES = {
    "/a": "<title>Page A</title><p>first page</p>",
    "/b": "<title>Page B</title><p>second page</p>",
    "/form": """<title>Form</title>
<style>
  body { margin: 0; background: rgb(0, 128, 255); }
  input, button { position: fixed; left: 20px; margin: 0; box-sizing: border-box; }
</style>
<form action="/echo">
  <input name="q" style="top: 20px; width: 300px; height: 40px">
  <button style="top: 100px; width: 200px; height: 50px">Send</button>
</form>""",
    # A button where the form's is, that opens page B in a popup.
    "/opener": """<title>Opener</title>
<button style="position: fixed; left: 20px; top: 100px; width: 200px; height: 50px; margin: 0"
        onclick="window.open('/b')">Open page B</button>""",
    # Repaints on every animation frame, and tells the site each time it has painted
    # a few more, so a test can see the page painting without looking at it.
    "/anim": f"""<title>Moving</title>
<canvas id="c" width="1280" height="720"></canvas>
<script>
  const pen = document.getElementById('c').getContext('2d');
  let n = 0;
  (function paint() {{
    n += 1;
    pen.fillStyle = `rgb(${{n % 256}}, ${{(n * 7) % 256}}, 90)`;
    pen.fillRect(0, 0, 1280, 720);
    if (n % {_PAINTS_PER_TICK} === 0) fetch('/tick');
    requestAnimationFrame(paint);
  }})();
</script>""",
}


class Site(ThreadingHTTPServer):
    """A loopback site that counts its visits and its animated page's ticks, and
    remembers what its form was sent."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _SiteHandler)
        self.url = f"http://127.0.0.1:{self.server_address[1]}"
        # Guards the counts; notified on every tick.
        self.changed = threading.Condition()
        self.visits = 0
        self.ticks = 0
        self.echoed: list[str] = []

    def ticked(self, count: int) -> bool:
        """Whether the animated page reports *count* more ticks; False once it has
        been quiet for ``_TICK_WAIT`` s."""
        with self.changed:
            goal = self.ticks + count
            return self.changed.wait_for(lambda: self.ticks >= goal, _TICK_WAIT)


class _SiteHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        site = cast(Site, self.server)
        url = urlsplit(self.path)
        extra: list[tuple[str, str]] = []
        page = _PAGES.get(url.path)
        if url.path == "/tick":
            with site.changed:
                site.ticks += 1
                site.changed.notify_all()
            self.send_response(204)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        if url.path == "/count":
            with site.changed:
                site.visits += 1
                page = f"<title>Visit {site.visits}</title>"
        elif url.path == "/echo":
            sent = parse_qs(url.query).get("q", [""])[0]
            with site.changed:
                site.echoed.append(sent)
            page = f"<title>Echo: {html.escape(sent)}</title>"
        elif url.path == "/login":
            extra.append(("Set-Cookie", "sid=signed-in; Max-Age=3600; Path=/"))
            page = "<title>Signed in</title>"
        elif url.path == "/whoami":
            cookie = self.headers.get("Cookie")
            page = f"<title>{html.escape(cookie) if cookie else 'No cookie'}</title>"
        self.send_response(200 if page else 404)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        for name, value in extra:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write((page or "<title>Missing</title>").encode())

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 — the base class's name
        pass  # the suite's output is not a request log


@contextmanager
def running_site() -> Iterator[Site]:
    site = Site()
    thread = threading.Thread(
        target=site.serve_forever, name="browser-site", daemon=True
    )
    thread.start()
    try:
        yield site
    finally:
        site.shutdown()
        site.server_close()
        thread.join(timeout=5)


def use_own_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the data directory at *tmp_path*, so a browser launched from here on keeps
    its profile there and never in the install's data."""
    monkeypatch.setattr(FileMapperService, "_DATA_DIR", tmp_path / "data")


def close_sessions() -> None:
    """End every browsing run's page. The pool closes them in order, behind any job
    already queued, so a pool call made after this one sees them closed."""
    for key in list(session._SESSIONS):
        session.close_session(key)


def close_all() -> None:
    """End every browsing run's page, then the browser behind them."""
    close_sessions()
    pool.get_pool().execute(pool._close)


@contextmanager
def isolated_browser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Run the block on a browser whose profile lives under *tmp_path*; yields that profile.

    Any browser already up was launched on another profile, so it is closed first and
    the next operation launches on this one. On the way out every run's tab is closed
    and the browser with it, so no page and no process outlives the test."""
    use_own_data_dir(tmp_path, monkeypatch)
    try:
        pool.get_pool().execute(pool._close)
        yield FileMapperService.get_browser_profile_path()
    finally:
        close_all()


@pytest.fixture
def browser_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The test's own browser profile; the browser and every page close with the test."""
    with isolated_browser(tmp_path, monkeypatch) as profile:
        yield profile


@pytest.fixture
def site() -> Iterator[Site]:
    with running_site() as running:
        yield running


@pytest.fixture
def app(db: sqlite3.Connection, browser_data: Path) -> Flask:
    """The real app, on the test's database and the test's own browser."""
    from api import create_app  # noqa: PLC0415 — built per test, on the test's database

    built = create_app()
    built.config["TESTING"] = True
    return built


@pytest.fixture
def user(app: Flask) -> tuple[FlaskClient, str]:
    """The signed-in user's client of the app, and their session cookie."""
    return signed_in(app)


def launches() -> int:
    """How many browsers the pool has started so far."""
    return cast(int, pool.get_pool().status()["browser_launches"])


def page_targets() -> set[str]:
    """The ids of every page the running browser has open, asked of the browser itself.

    Runs behind any close already queued on the pool. Call it only while a browser is
    up (check :func:`launches` is unchanged), or it launches one with no pages."""

    def ids(browser: Browser) -> set[str]:
        infos = get_objects(browser.send("Target.getTargets"), "targetInfos")
        return {str(info["targetId"]) for info in infos if info.get("type") == "page"}

    return cast("set[str]", pool.get_pool().execute(ids))


def closed_in_browser(target_id: str) -> bool:
    """Whether the browser itself drops the page *target_id* within ``_CLOSE_WAIT`` s.

    The browser answers a close before the page is gone, so this re-asks it, paced by
    each round trip, until the page is no longer listed."""
    deadline = time.monotonic() + _CLOSE_WAIT
    while time.monotonic() < deadline:
        if target_id not in page_targets():
            return True
    return False


@contextmanager
def live_server(app: Flask) -> Iterator[str]:
    """Serve *app* on a real loopback port; yields its ``host:port``."""
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, name="live-app", daemon=True)
    thread.start()
    try:
        yield f"127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def cookie_header(token: str) -> dict[str, str]:
    return {"Cookie": f"{SESSION_COOKIE_NAME}={token}"}


def signed_in(app: Flask) -> tuple[FlaskClient, str]:
    """A test client of *app* carrying a real signed-in session's cookie, and that
    cookie's value."""
    token = create_session(Response())
    client = app.test_client()
    client.set_cookie(SESSION_COOKIE_NAME, token)
    return client, token


def listed(client: FlaskClient) -> list[dict[str, object]]:
    """The tab list as the live view reads it."""
    response = client.get("/api/browser/tabs")
    assert response.status_code == 200, response.data
    body = cast("dict[str, dict[str, list[dict[str, object]]]]", response.get_json())
    return body["result"]["tabs"]


def open_user_tab(client: FlaskClient, url: str, title: str) -> Tab:
    """Open a tab the way the live view's address bar does; the tab once its page
    shows *title*."""
    response = client.post("/api/browser/tabs", json={"url": url})
    assert response.status_code == 201, response.data
    created = cast("dict[str, dict[str, dict[str, object]]]", response.get_json())
    tab = tabs.get(str(created["result"]["tab"]["id"]))
    assert tab is not None, "the new tab is not open"
    assert wait_for(tab, lambda: tab.title == title), (tab.url, tab.title)
    return tab


def run_opens(
    url: str, origin: dict[str, object] | None = None
) -> tuple[MessageProcessor, Tab]:
    """A browsing run, attached to the conversation *origin* names (or to none), opens
    *url* with the real browser ability; the run, and its tab."""
    metadata: dict[str, object] = {} if origin is None else {"origin": origin}
    run = MessageProcessor(
        WebBrowseConfig(PolicyChannel.CHAT), raw_input="browse", metadata=metadata
    )
    run.active_tools = list(run.config.always_available or [])
    rendered = run.dispatch_service.dispatch("browser", {"action": "open", "url": url})
    assert "status=success" in rendered.splitlines()[0], rendered
    tab = session.run_tab(run.uid or 0)
    assert tab is not None, "the run's page is not open"
    return run, tab


def bearer_header() -> dict[str, str]:
    """A real external-agent bearer token, as its ``Authorization`` header."""
    raw, _wrapper_id = WrapperAuthService().create_token(name="external agent")
    return {"Authorization": f"Bearer {raw}"}


class StreamSocket(Protocol):
    """The client side of a live-page socket, as these tests use it."""

    sock: socket.socket

    def send(self, data: str | bytes) -> None: ...

    def receive(self, timeout: float | None = None) -> str | bytes | None: ...

    def close(self, reason: int | None = None, message: str | None = None) -> None: ...


@contextmanager
def stream_socket(
    host: str, tab_id: str, headers: dict[str, str]
) -> Iterator[StreamSocket]:
    """A connected live-page socket for *tab_id*, closed on the way out whatever state it is in."""
    ws = cast(
        StreamSocket,
        Client.connect(f"ws://{host}/ws/browser/{tab_id}", headers=headers),
    )
    try:
        yield ws
    finally:
        with suppress(ConnectionClosed):
            ws.close()
        ws.sock.close()


def closing_code(ws: StreamSocket) -> tuple[list[str | bytes], int]:
    """Every message *ws* receives until the server closes it, and the close code it
    sent; fails the test if the server has not closed it within ``_CLOSE_WAIT`` s."""
    received: list[str | bytes] = []
    deadline = time.monotonic() + _CLOSE_WAIT
    while time.monotonic() < deadline:
        try:
            message = ws.receive(timeout=max(0.0, deadline - time.monotonic()))
        except ConnectionClosed as closed:
            return received, int(closed.reason)
        if message is not None:
            received.append(message)
    raise AssertionError(f"the server never closed the socket; it sent {received!r}")


def receive_until(
    ws: StreamSocket, want: Callable[[str | bytes], bool], timeout: float = 15.0
) -> str | bytes:
    """The first message *ws* receives that *want* accepts; fails the test if none comes in time."""
    skipped: list[str | bytes] = []
    deadline = time.monotonic() + timeout
    while (left := deadline - time.monotonic()) > 0:
        message = ws.receive(timeout=left)
        if message is not None:
            if want(message):
                return message
            skipped.append(
                message if isinstance(message, str) else f"<{len(message)} bytes>"
            )
    raise AssertionError(f"no wanted message within {timeout}s; got {skipped!r}")


def as_frame(message: str | bytes, kind: str) -> dict[str, object] | None:
    """*message* as a frame of type *kind*, or ``None`` when it is anything else."""
    if isinstance(message, bytes):
        return None
    payload = cast("dict[str, object]", json.loads(message))
    return payload if payload.get("type") == kind else None


@contextmanager
def main_socket(host: str, token: str) -> Iterator[StreamSocket]:
    """The signed-in user's main interface socket, once the server is broadcasting to it.

    The server registers the socket just after the handshake; announcing a tab-list
    change until one reaches it proves the registration without guessing a delay."""
    ws = cast(
        StreamSocket, Client.connect(f"ws://{host}/ws", headers=cookie_header(token))
    )
    try:
        deadline = time.monotonic() + 10.0
        while as_frame(ws.receive(timeout=0.1) or b"", "browser_tabs_changed") is None:
            if time.monotonic() > deadline:
                raise AssertionError("the main socket never received a broadcast")
            tabs.announce()
        yield ws
    finally:
        with suppress(ConnectionClosed):
            ws.close()
        ws.sock.close()


def wait_for(tab: Tab, done: Callable[[], bool], timeout: float = 15.0) -> bool:
    """Wait, on the tab's own change signal, until *done* holds; False on timeout."""
    with tab.changed:
        return tab.changed.wait_for(done, timeout)
