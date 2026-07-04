"""Feature tests for the shared outbound fetch stack (services/web_fetch.py).

Two tiers, kept strictly separate (mirrors test_web_download.py):

* ``@pytest.mark.unit`` — SSRF gate and named-profile contract. Never touch
   the network: a private/internal URL is blocked BEFORE any socket opens.

* ``@pytest.mark.integration`` — real fetch + streamed download against httpbin.org,
   skipped when unreachable.

No mocks anywhere. The SSRF guard under test is production's own
``services.ssrf.is_private_url``, and profiles are the same objects read / web_download
consume.
"""

import os
import socket
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import requests

from services import web_fetch
from services.ssrf import is_private_url

if TYPE_CHECKING:
    from typing import Protocol

    class _WebFetchModule(Protocol):
        is_private_url: Callable[[str], bool]

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


# ── Unit tier: profiles + SSRF gate, no network ─────────────────────────────


@pytest.mark.unit
def test_fetch_text_uses_the_real_ssrf_guard() -> None:
    """The fetch service shares ONE SSRF guard with the abilities."""
    # Identity, not equality: web_fetch must call the same production guard.
    from typing import cast
    assert cast("_WebFetchModule", web_fetch).is_private_url is is_private_url


@pytest.mark.unit
def test_fetch_text_refuses_private_host_before_socket() -> None:
    """A loopback URL is blocked by the guard, raising FetchBlocked (no network)."""
    with pytest.raises(web_fetch.FetchBlocked):
        web_fetch.fetch_text("http://127.0.0.1/secret", profile=web_fetch.API)


@pytest.mark.unit
def test_stream_to_file_refuses_private_host_and_writes_nothing(tmp_path: Path) -> None:
    """A blocked host raises before any file is created."""
    dest = os.path.join(str(tmp_path), "should_not_exist", "f.bin")
    with pytest.raises(web_fetch.FetchBlocked):
        web_fetch.stream_to_file("http://169.254.169.254/latest/meta-data/", dest)
    assert not os.path.exists(dest)


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
