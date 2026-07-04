"""Boot screen holding server — binds instantly, answers 503, releases the port.

Guards the cold-start contract: from the first moment of boot the public port
must answer (holding page for browsers, JSON for fetches) instead of resetting
connections, and ``stop()`` must free the port for the real server's bind.
"""

import socket
from collections.abc import Iterator
import urllib.error
import urllib.request

import pytest

from boot_screen import BootScreen

pytestmark = pytest.mark.unit


@pytest.fixture
def boot_screen() -> Iterator[tuple[BootScreen, int]]:
    bs = BootScreen("127.0.0.1", 0)  # OS-assigned free port
    bs.start()
    assert bs._server is not None, "boot screen failed to bind"
    yield bs, bs._server.server_address[1]
    bs.stop()


def _get(port: int, path: str, accept: str) -> "urllib.error.HTTPError":
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", headers={"Accept": accept}
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=5)
    return exc_info.value


def test_browser_navigation_gets_holding_page(boot_screen: tuple[BootScreen, int]) -> None:
    _, port = boot_screen
    resp = _get(port, "/brain/", "text/html,application/xhtml+xml")
    assert resp.code == 503
    assert resp.headers["Content-Type"].startswith("text/html")
    body = resp.read().decode()
    assert "Chalie is starting" in body
    assert "/ready" in body  # self-refresh poll target


def test_fetches_get_starting_json(boot_screen: tuple[BootScreen, int]) -> None:
    _, port = boot_screen
    resp = _get(port, "/auth/status", "*/*")
    assert resp.code == 503
    assert resp.headers["Content-Type"] == "application/json"
    assert b'"ready": false' in resp.read()


def test_stop_releases_port_for_real_server(boot_screen: tuple[BootScreen, int]) -> None:
    bs, port = boot_screen
    bs.stop()
    # The real server must be able to bind the same port immediately.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))


def test_busy_port_is_survivable() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        taken = s.getsockname()[1]
        bs = BootScreen("127.0.0.1", taken)
        bs.start()  # must not raise — real bind surfaces the error later
        assert bs._server is None
        bs.stop()  # no-op, must not raise
