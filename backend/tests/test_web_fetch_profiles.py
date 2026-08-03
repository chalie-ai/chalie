"""Feature tests for the shared outbound fetch stack (services/web_fetch.py).

Two tiers, kept strictly separate:

* ``@pytest.mark.unit`` — a host on the local network is reachable. Chalie is a
  local-first assistant, so internal sites are ordinary targets: the stack
  carries no private-address blocklist and refuses no URL by policy.

* ``@pytest.mark.integration`` — real fetch + streamed download against
  httpbin.org, skipped when unreachable.

No mocks anywhere — every tier runs a real socket against a real server.
"""

import http.server
import os
import socket
import tempfile
import threading

import pytest
import requests

from services import web_fetch

_HTTPBIN_HOST = "httpbin.org"


def _httpbin_reachable() -> bool:
    try:
        with socket.create_connection((_HTTPBIN_HOST, 443), timeout=3):
            return True
    except OSError:
        return False


@pytest.fixture
def httpbin() -> None:
    if not _httpbin_reachable():
        pytest.skip(f"{_HTTPBIN_HOST} unreachable — skip network-dependent fetch tests.")


# ── Unit tier: the local network is reachable ───────────────────────────────


@pytest.mark.unit
def test_fetch_text_reaches_a_host_on_the_local_network() -> None:
    """A real server on loopback is fetched, not refused.

    The stack used to resolve every destination against a private-IP blocklist,
    which put the user's own LAN out of reach. A live server proves the whole
    path — socket included — now reaches it.
    """

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — http.server contract
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>internal site</h1>")

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = int(server.server_address[1])
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        body = web_fetch.fetch_text(f"http://127.0.0.1:{port}/")
    finally:
        server.shutdown()
        server.server_close()

    assert "<h1>internal site</h1>" in body


# ── Integration tier: real fetch + streamed download ────────────────────────


@pytest.mark.integration
def test_fetch_text_browser_profile_returns_body(httpbin: None) -> None:
    """A real GET with the browser profile returns the decoded body."""
    body = web_fetch.fetch_text(
        "https://httpbin.org/user-agent", profile=web_fetch.BROWSER
    )
    assert "Chrome/131" in body  # httpbin echoes the UA we sent


@pytest.mark.integration
def test_fetch_text_api_profile_sends_bot_user_agent(httpbin: None) -> None:
    """The API profile presents the identified bot UA over the wire."""
    body = web_fetch.fetch_text(
        "https://httpbin.org/user-agent", profile=web_fetch.API
    )
    assert "ChalieBot" in body


@pytest.mark.integration
def test_fetch_text_raises_on_http_error(httpbin: None) -> None:
    """HTTP failures bubble (raise_for_status) — never swallowed."""
    with pytest.raises(requests.RequestException):
        web_fetch.fetch_text("https://httpbin.org/status/404")


@pytest.mark.integration
def test_stream_to_file_writes_full_body(httpbin: None) -> None:
    """A real streamed download writes the file under the dest path."""
    dest = os.path.join(tempfile.gettempdir(), "chalie_webfetch_test", "robots.txt")
    if os.path.exists(dest):
        os.remove(dest)

    web_fetch.stream_to_file("https://httpbin.org/robots.txt", dest)

    assert os.path.exists(dest)
    assert os.path.getsize(dest) > 0
    os.remove(dest)
