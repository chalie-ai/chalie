"""Feature tests: the rebuilt 10-verb browser ability.

Drives the REAL production hot path — ``ToolDispatcher.dispatch()`` on a real
``MessageProcessor`` bound to ``WebBrowseConfig`` (where ``browser`` is
always-available; ``browser`` is in ``PolicyManager.INTERNAL`` so no policy rows
are needed) — with zero mocks, asserted on the dispatcher WIRE envelope. The
schema and the no-open-page guard need no browser; every other verb drives the
real headless shell against a loopback site.

The parameter contract (unknown-action / missing-params / per-verb field
validation) is enforced by ``BrowserParamsBag`` at the dispatch seam and needs
no tests here.
"""

import html
import json
import sqlite3
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast
from urllib.parse import parse_qs, urlsplit

import pytest

from configs.channels.web_browse import WebBrowseConfig
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from tools.browser.cdp import Browser
from tools.browser.pool import get_pool
from tools.browser.session import close_session

_PAGES = {
    "/": """<title>Home</title><h1>Welcome</h1>
<button>Sign in</button><button>Sign up</button><button style="display:none">Sign hidden</button>
<a href="/two">Go to page two</a>
<button onclick="alert('hi there')">Alert me</button>
<button onclick="window.open('/two')">Open popup</button>
<button onclick="setTimeout(() => document.getElementById('late').disabled = false, 500)">Arm</button>
<button id="late" disabled onclick="document.title = 'Late clicked'">Late button</button>
<form action="/echo">
  <label>Email <input name="email"></label>
  <input name="city" placeholder="Your city">
  <select name="colour" aria-label="Colour"><option value="r">Red</option><option value="g">Green</option></select>
  <button>Send form</button>
</form>
<h2 style="color: rgb(255, 0, 0)">Red heading</h2>
<div style="height: 3000px">tall</div>""",
    "/two": "<title>Two</title><p>second page</p>",
    "/hop": "<script>location.replace('/two')</script>",
    "/lazy": """<title>Lazy</title><div style="height: 3000px">tall</div>
<div id="slot"><button>Lazy button</button></div>
<script>addEventListener('scroll', () => {
  document.getElementById('slot').innerHTML = `<button onclick="document.title = 'Lazy clicked'">Lazy button</button>`;
}, {once: true})</script>""",
}

Browse = Callable[..., str]


class _Site(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        url = urlsplit(self.path)
        page = _PAGES.get(url.path)
        if url.path == "/echo":
            fields = "; ".join(f"{key}: {values[0]}" for key, values in parse_qs(url.query).items())
            page = f"<title>Echo</title><p>{html.escape(fields)}</p>"
        self.send_response(200 if page else 404)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write((page or "<title>Missing</title><p>no such page</p>").encode())

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 — the base class's name
        pass  # the suite's output is not a request log


@pytest.fixture(scope="module")
def site() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Site)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    get_pool().execute(Browser.close)  # no headless shell outlives the module; the pool relaunches on demand


@pytest.fixture
def browse(db: sqlite3.Connection) -> Iterator[Browse]:
    """One delegate run: consecutive calls drive the same live page."""
    mp = _browse_mp()
    yield lambda **params: mp.dispatch_service.dispatch("browser", params)
    close_session(mp.uid or 0)


def _browse_mp() -> MessageProcessor:
    mp = MessageProcessor(WebBrowseConfig(PolicyChannel.CHAT), raw_input="drive a web page")
    mp.active_tools = list(mp.config.always_available or [])
    return mp


def _dispatch(params: dict[str, object]) -> str:
    return _browse_mp().dispatch_service.dispatch("browser", params)


@pytest.mark.unit
def test_verbs_demand_an_open_page_first(db: sqlite3.Connection) -> None:
    """No session for this key → mechanical guidance, no browser launch."""
    rendered = _dispatch({"action": "click", "target": "Sign in"})
    assert "status=error" in rendered.splitlines()[0], rendered
    assert "code=no-open-page" in rendered.splitlines()[0], rendered
    assert "No page is open" in rendered, rendered


def _envelope(rendered: str) -> dict[str, object]:
    """The JSON envelope of a successful call — the wire body's first line."""
    header, body = rendered.splitlines()[:2]
    assert "status=success" in header, rendered
    return cast("dict[str, object]", json.loads(body))


def _part(envelope: dict[str, object], key: str) -> dict[str, object]:
    return cast("dict[str, object]", envelope[key])


@pytest.mark.e2e
def test_open_click_and_back_walk_a_real_site(site: str, browse: Browse) -> None:
    home = _envelope(browse(action="open", url=f"{site}/"))
    assert _part(home, "page") == {"url": f"{site}/", "title": "Home", "status": 200}
    assert "Welcome" in str(_part(home, "data")["text"])

    # Accessible names match case-insensitively on any part of the name.
    two = _envelope(browse(action="click", target="go to PAGE"))
    assert _part(two, "changed")["navigated"] is True, two
    assert _part(two, "page")["url"] == f"{site}/two"
    assert _part(two, "data") == {"text": "second page"}

    back = _envelope(browse(action="back"))
    assert _part(back, "page")["url"] == f"{site}/"
    assert "Welcome" in str(_part(back, "data")["text"])

    assert _part(_envelope(browse(action="open", url=f"{site}/missing")), "page")["status"] == 404

    # A page that redirects itself by script opens as the page it sent the browser to.
    hop = _envelope(browse(action="open", url=f"{site}/hop"))
    assert _part(hop, "page") == {"url": f"{site}/two", "title": "Two", "status": 200}, hop


@pytest.mark.e2e
def test_targets_resolve_to_one_visible_element_or_say_why_not(site: str, browse: Browse) -> None:
    browse(action="open", url=f"{site}/")

    # "Sign hidden" is not displayed, so only the two visible buttons compete.
    ambiguous = browse(action="click", target="Sign")
    assert "code=ambiguous-target" in ambiguous.splitlines()[0], ambiguous
    assert "matches 2 elements" in ambiguous, ambiguous

    missing = browse(action="click", target="Nothing like this")
    assert "code=element-not-found" in missing.splitlines()[0], missing


@pytest.mark.e2e
def test_a_form_filled_by_label_placeholder_and_option_submits_what_was_typed(site: str, browse: Browse) -> None:
    browse(action="open", url=f"{site}/")
    browse(action="fill", target="Email", value="draft")
    for target, value in (("email", "a@b.c"), ("your city", "Valletta")):
        assert _part(_envelope(browse(action="fill", target=target, value=value)), "changed")["navigated"] is False
    _envelope(browse(action="select", target="Colour", value="Green"))

    sent = _envelope(browse(action="click", target="Send form"))
    # A second fill replaces the field's text instead of appending to it.
    assert _part(sent, "data") == {"text": "email: a@b.c; city: Valletta; colour: g"}, sent


@pytest.mark.e2e
def test_click_waits_out_a_disabled_or_re_rendered_target(site: str, browse: Browse) -> None:
    browse(action="open", url=f"{site}/")
    browse(action="click", target="Arm")
    assert _part(_envelope(browse(action="click", target="Late button")), "page")["title"] == "Late clicked"

    # The page replaces the button as it scrolls into view; the click finds the new one.
    browse(action="open", url=f"{site}/lazy")
    assert _part(_envelope(browse(action="click", target="Lazy button")), "page")["title"] == "Lazy clicked"


@pytest.mark.e2e
def test_dialogs_and_popups_are_reported_and_the_page_stays_put(site: str, browse: Browse) -> None:
    browse(action="open", url=f"{site}/")

    alert = _part(_envelope(browse(action="click", target="Alert me")), "changed")
    assert alert["dialog"] == "alert: hi there", alert

    popup = _envelope(browse(action="click", target="Open popup"))
    assert _part(popup, "changed")["popup"] == f"{site}/two", popup
    assert _part(popup, "changed")["navigated"] is False
    assert _part(popup, "page")["url"] == f"{site}/"


@pytest.mark.e2e
def test_style_scroll_and_find_report_what_the_page_holds(site: str, browse: Browse) -> None:
    browse(action="open", url=f"{site}/")
    assert _part(_envelope(browse(action="style", target="Red heading")), "data")["color"] == "rgb(255, 0, 0)"
    assert cast(int, _part(_envelope(browse(action="scroll", direction="down")), "data")["y"]) > 0
    found = _part(_envelope(browse(action="find", query="welcome")), "data")
    assert cast("list[str]", found["matches"])[0].startswith("Welcome Sign in"), found
