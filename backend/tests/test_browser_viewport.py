"""Feature test: the browser panel is the page's window.

Sizing the panel resizes the tab the way resizing a browser window does: a
responsive page lays itself out again for that size, and Chalie never scales the
picture. The view's size reaches the tab over the real live-page socket, the
page reports its window through the real ``browser`` ability's ``read``, and
Chalie's own screenshot is the run's real ``grab`` verb, the capture the
ability's ``screenshot`` stores. Frames and screenshots are measured by
decoding them. Nothing is substituted: a real browsing run, the real app on a
real port, and the real headless shell on the test's own throwaway profile.
"""

from __future__ import annotations

import io
import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

import pytest
from flask import Flask
from flask.testing import FlaskClient
from PIL import Image

from controllers.message_processor import MessageProcessor
from tests._browser_live import (
    StreamSocket,
    as_frame,
    cookie_header,
    live_server,
    run_opens,
    stream_socket,
    wait_for,
)
from tools.browser import session

pytest_plugins = ("tests._browser_live",)
pytestmark = pytest.mark.e2e

_SIZED_WAIT = 15.0  # seconds a new size gets to reach the socket's meta and frames
_PHONE = "Laid out for a phone"
_PAGE = f"""<title>Responsive</title>
<style>
  body {{ margin: 0; }}
  #m {{ display: none; }}
  @media (max-width: 600px) {{ #m {{ display: block; }} }}
</style>
<p id="window"></p>
<p id="m">{_PHONE}</p>
<script>
  const report = () => {{
    document.getElementById('window').textContent = JSON.stringify(
      [innerWidth, innerHeight, devicePixelRatio, matchMedia('(max-width: 600px)').matches]);
  }};
  report();
  addEventListener('resize', report);
</script>"""

Size = tuple[int, int]


class _Page(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(_PAGE.encode())

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 — the base class's name
        pass  # the suite's output is not a request log


@pytest.fixture
def responsive_page() -> Iterator[str]:
    """The URL of a loopback page that shows a phone layout only in a window 600 CSS px wide or less."""
    site = ThreadingHTTPServer(("127.0.0.1", 0), _Page)
    thread = threading.Thread(target=site.serve_forever, name="responsive-site", daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{site.server_address[1]}/"
    finally:
        site.shutdown()
        site.server_close()
        thread.join(timeout=5)


def _window(run: MessageProcessor) -> tuple[list[object], bool]:
    """What the run reads on its page: the window the page reports
    (``[innerWidth, innerHeight, devicePixelRatio, phone query matches]``) and
    whether its phone layout shows."""
    rendered = run.dispatch_service.dispatch("browser", {"action": "read"})
    header, body = rendered.splitlines()[:2]
    assert "status=success" in header, rendered
    text = cast(str, cast("dict[str, dict[str, object]]", json.loads(body))["data"]["text"])
    return cast("list[object]", json.loads(text.splitlines()[0])), _PHONE in text


def _screenshot(run: MessageProcessor) -> Size:
    """The pixel size of the screenshot Chalie takes of the run's page."""
    grabbed = session.run_verb(run.uid or 0, "grab", {})
    assert "png" in grabbed, grabbed
    return Image.open(io.BytesIO(cast(bytes, grabbed["png"]))).size


def _resize(ws: StreamSocket, width: int, height: int, scale: float) -> tuple[list[Size], list[Size]]:
    """Size the view as the panel does; the sizes of every meta and frame the socket then
    receives, until its latest meta and one of its frames are *width* x *height* or
    ``_SIZED_WAIT`` s pass.

    It waits for a frame of that size rather than taking the next one: the picture a
    joining viewer is seeded with is in device px, not CSS px."""
    ws.send(json.dumps({"type": "size", "width": width, "height": height, "scale": scale}))
    want = (width, height)
    metas: list[Size] = []
    frames: list[Size] = []
    deadline = time.monotonic() + _SIZED_WAIT
    while not (metas[-1:] == [want] and want in frames) and (left := deadline - time.monotonic()) > 0:
        message = ws.receive(timeout=left)
        if isinstance(message, bytes):
            frames.append(Image.open(io.BytesIO(message)).size)
        elif message is not None and (meta := as_frame(message, "meta")) is not None:
            metas.append((cast(int, meta["width"]), cast(int, meta["height"])))
    return metas, frames


def test_the_browser_panel_is_the_pages_window_so_a_page_reflows_to_its_size_and_chalie_sees_the_same_page(
    app: Flask, user: tuple[FlaskClient, str], responsive_page: str
) -> None:
    _client, token = user
    run, tab = run_opens(responsive_page)
    # A new tab is a 1280x720 desktop window.
    assert _window(run) == ([1280, 720, 1, False], False)
    assert _screenshot(run) == (1280, 720)

    with live_server(app) as host, stream_socket(host, tab.id, cookie_header(token)) as ws:
        # A phone-sized panel on a 3x screen: the page gets a phone's window and its
        # phone layout; it is offered at most 2 device px per CSS px.
        metas, frames = _resize(ws, 390, 844, 3)
        assert _window(run) == ([390, 844, 2, True], True)
        assert metas[-1:] == [(390, 844)], metas
        assert (390, 844) in frames, frames
        assert _screenshot(run) == (390, 844)

        # Widened on a 1x screen: the page goes back to its desktop layout.
        metas, frames = _resize(ws, 1280, 800, 1)
        assert _window(run) == ([1280, 800, 1, False], False)
        assert metas[-1:] == [(1280, 800)], metas
        assert (1280, 800) in frames, frames
        assert _screenshot(run) == (1280, 800)

    # Closing the panel leaves the window as it last was.
    assert wait_for(tab, lambda: tab.viewers == 0), tab.viewers
    assert _window(run) == ([1280, 800, 1, False], False)
    assert _screenshot(run) == (1280, 800)
