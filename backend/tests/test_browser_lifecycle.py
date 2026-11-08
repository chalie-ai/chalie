"""Feature tests: how long the browser lives, and what it keeps between launches.

The browser idles out when nothing needs it, but never while the user has a
tab open or is watching one; and a site the user signed in to stays signed in
after the browser closes and launches again, because every launch uses the
same profile. Real Flask app, real tab registry, real pool thread and real
headless shell against a loopback site; the profile lives under the test's tmp
dir. The idle timer is shortened (its two module settings, nothing else) so
the test runs in seconds rather than minutes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from flask import Flask

from tests._browser_live import (
    Site,
    cookie_header,
    launches,
    listed,
    live_server,
    open_user_tab,
    receive_until,
    run_opens,
    signed_in,
    stream_socket,
    wait_for,
)
from tools.browser import pool
from tools.browser.tabs import Tab

pytest_plugins = ("tests._browser_live",)
pytestmark = pytest.mark.e2e


def _titled(tab: Tab, title: str) -> bool:
    return wait_for(tab, lambda: tab.title == title)


def test_the_browser_idles_out_only_once_the_user_has_nothing_open_or_watched_in_it(
    app: Flask, site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pool, "_IDLE_TTL", 0.2)
    monkeypatch.setattr(pool, "_QUEUE_POLL", 0.05)
    client, token = signed_in(app)
    mine = open_user_tab(client, f"{site.url}/a", "Page A")
    _browsing, run = run_opens(f"{site.url}/b")
    browsers = launches()

    def idles_out(within: float) -> bool:
        return wait_for(run, lambda: not run.live, timeout=within)

    assert not idles_out(1.5), "the browser closed under the user's open tab"

    with (
        live_server(app) as host,
        stream_socket(host, run.id, cookie_header(token)) as ws,
    ):
        receive_until(ws, lambda m: isinstance(m, bytes))  # the user is watching
        assert client.delete(f"/api/browser/tabs/{mine.id}").status_code == 204
        assert not idles_out(1.5), "the browser closed under the page being watched"

    assert idles_out(10.0), "the browser never idled out"
    assert listed(client) == []
    assert launches() == browsers


def test_a_site_the_user_signed_in_to_stays_signed_in_after_the_browser_relaunches(
    app: Flask, site: Site, browser_data: Path
) -> None:
    client, _token = signed_in(app)
    before = open_user_tab(client, f"{site.url}/whoami", "No cookie")
    moved = client.post(
        f"/api/browser/navigate/{before.id}", json={"url": f"{site.url}/login"}
    )
    assert moved.status_code == 204, moved.data
    assert _titled(before, "Signed in"), before.title
    back = client.post(
        f"/api/browser/navigate/{before.id}", json={"url": f"{site.url}/whoami"}
    )
    assert back.status_code == 204, back.data
    assert _titled(before, "sid=signed-in"), before.title

    assert client.delete(f"/api/browser/tabs/{before.id}").status_code == 204
    pool.get_pool().execute(pool._close)  # the browser closes, as it does when idle
    browsers = launches()

    open_user_tab(client, f"{site.url}/whoami", "sid=signed-in")
    assert launches() == browsers + 1  # a fresh browser process, not the old one
    assert any(browser_data.iterdir())
