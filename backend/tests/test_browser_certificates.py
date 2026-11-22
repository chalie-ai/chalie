"""Feature tests: the browser enforces certificates like a normal browser.

A real browsing run drives the real ``browser`` ability, through the real
dispatcher, at a real loopback HTTPS site whose certificate is self-signed. The
real headless shell refuses the connection: the site's page never reaches the
browser, and the run is told the certificate is why and that retrying will not
help, so it can say so instead of reading a blank page. Until a page does load,
every read of the tab says nothing loaded rather than reporting the refused
page as a page. A site that refuses the connection outright is reported the
same way, under its own reason.

As in any browser, the refusal is a warning the person can get past. The tab
list the panel reads names the refused site, the browser's reason and the url,
for the person's tabs and Chalie's alike, and the person's main socket hears it
change. Only the person's Proceed on that warning (the panel's request, on
their signed-in session) lets the browser open the site: then in every tab of
the profile, Chalie's included, until the browser restarts, and for that host
and port alone. Nothing a run sends gets it there.

Nothing is substituted: the real Flask app on a real port, the real headless
shell on the test's own throwaway profile, loopback sites only.
"""

from __future__ import annotations

import datetime
import ipaddress
import logging
import socket
import sqlite3
import ssl
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from flask import Flask
from flask.testing import FlaskClient

from configs.channels.web_browse import WebBrowseConfig
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from tests._browser_live import (
    Site,
    StreamSocket,
    as_frame,
    close_all,
    listed,
    live_server,
    main_socket,
    open_user_tab,
    receive_until,
    signed_in,
    wait_for,
)
from tools.browser import session, tabs
from tools.browser.tabs import Tab

pytest_plugins = ("tests._browser_live",)
pytestmark = pytest.mark.e2e

_MARKER = "only a trusted connection ever reads this line"
_ORIGIN: dict[str, object] = {"type": "user", "turn_id": 1, "forked": False}
_RUN_WAIT = 60.0  # seconds a run parked on a hand-off gets to carry on once the person is done


def _self_signed(where: Path) -> ssl.SSLContext:
    """A server context presenting a certificate for 127.0.0.1 that signs itself,
    so the only thing wrong with it is that no authority vouches for it."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_file, key_file = where / "site.crt", where / "site.key"
    cert_file.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_file, key_file)
    return context


class _UntrustedSite(ThreadingHTTPServer):
    """A loopback HTTPS site on a self-signed certificate that remembers every page it served."""

    def __init__(self, context: ssl.SSLContext) -> None:
        super().__init__(("127.0.0.1", 0), _Page)
        self.url = f"https://127.0.0.1:{self.server_address[1]}/"
        self.served: list[str] = []
        self._context = context

    def get_request(self) -> tuple[socket.socket, object]:
        # The handshake runs on the request's own thread, so a browser refusing the
        # certificate fails that thread and never stalls the accepting one.
        connection, address = self.socket.accept()
        return self._context.wrap_socket(connection, server_side=True, do_handshake_on_connect=False), address

    def handle_error(self, request: object, client_address: object) -> None:
        pass  # a refused handshake is what the test expects, not a server fault


class _Page(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        cast(_UntrustedSite, self.server).served.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(f"<title>Untrusted</title><p>{_MARKER}</p>".encode())

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 — the base class's name
        pass  # the suite's output is not a request log


@contextmanager
def _serving_untrusted(keys: Path) -> Iterator[_UntrustedSite]:
    keys.mkdir()
    site = _UntrustedSite(_self_signed(keys))
    thread = threading.Thread(target=site.serve_forever, name="untrusted-site", daemon=True)
    thread.start()
    try:
        yield site
    finally:
        site.shutdown()
        site.server_close()
        thread.join(timeout=5)


@pytest.fixture
def untrusted_site(tmp_path: Path) -> Iterator[_UntrustedSite]:
    with _serving_untrusted(tmp_path / "keys") as site:
        yield site


@pytest.fixture
def other_untrusted_site(tmp_path: Path) -> Iterator[_UntrustedSite]:
    """A second self-signed site on its own port, with its own key."""
    with _serving_untrusted(tmp_path / "other-keys") as site:
        yield site


def _browsing_run(origin: dict[str, object] | None = None) -> MessageProcessor:
    """A browsing run, attached to the conversation *origin* names (or to none)."""
    metadata: dict[str, object] = {} if origin is None else {"origin": origin}
    run = MessageProcessor(WebBrowseConfig(PolicyChannel.CHAT), raw_input="read the page", metadata=metadata)
    run.active_tools = list(run.config.always_available or [])
    return run


def _browse(run: MessageProcessor, **params: object) -> str:
    """Dispatch one browser action for *run*; its rendered result."""
    return run.dispatch_service.dispatch("browser", dict[str, object](params))


def _heard(ui: StreamSocket, want: Callable[[], bool]) -> None:
    """Read the person's main socket until a tab-list change arrives after which *want* holds."""
    receive_until(ui, lambda m: as_frame(m, "browser_tabs_changed") is not None and want())


def _refusal(client: FlaskClient, tab_id: str) -> object:
    """The certificate warning the panel's tab list shows for the tab *tab_id*."""
    [row] = [t for t in listed(client) if t["id"] == tab_id]
    return row["certificate_error"]


def _user_tab(client: FlaskClient, url: str) -> Tab:
    """A tab the person opens on *url* from the panel's address bar, before its page loads."""
    response = client.post("/api/browser/tabs", json={"url": url})
    assert response.status_code == 201, response.data
    created = cast("dict[str, dict[str, dict[str, object]]]", response.get_json())
    tab = tabs.get(str(created["result"]["tab"]["id"]))
    assert tab is not None, "the new tab is not open"
    return tab


def _warning(site: _UntrustedSite, host: str = "127.0.0.1") -> dict[str, object]:
    """The warning the panel lists for *site* reached by *host*: the site, the browser's reason, the url."""
    port = site.server_address[1]
    url = site.url if host == "127.0.0.1" else f"https://{host}:{port}/"
    return {"host": host, "port": port, "error": "ERR_CERT_AUTHORITY_INVALID", "url": url}


def _proceed(client: FlaskClient, tab_id: str, host: str, port: int) -> int:
    """The panel's Proceed on the tab *tab_id*'s warning for *host*:*port*; the status it answers."""
    return client.post(f"/api/browser/proceed/{tab_id}", json={"host": host, "port": port}).status_code


def test_a_page_chalie_opens_on_a_self_signed_site_never_loads_and_chalie_is_told_the_certificate_is_why(
    db: sqlite3.Connection, browser_data: Path, untrusted_site: _UntrustedSite, site: Site,
    caplog: pytest.LogCaptureFixture,
) -> None:
    run = _browsing_run()

    with caplog.at_level(logging.INFO):
        opened = run.dispatch_service.dispatch("browser", {"action": "open", "url": untrusted_site.url})
    assert "status=error" in opened.splitlines()[0], opened
    assert "code=certificate-error" in opened.splitlines()[0], opened
    assert f"ERR_CERT_AUTHORITY_INVALID at {untrusted_site.url}" in opened, opened
    assert "Re-open the page" not in opened, opened  # retrying cannot fix a certificate
    assert _MARKER not in opened, opened
    # A site that does not load is the browser's verdict, not a fault: no traceback is logged for it.
    assert [r.getMessage() for r in caplog.records if r.exc_info] == []

    # The tab now shows the refusal: every look at it says nothing loaded.
    for action in ("read", "screenshot"):
        looked = run.dispatch_service.dispatch("browser", {"action": action})
        assert "code=no-open-page" in looked.splitlines()[0], looked
        assert f"No page is open: {untrusted_site.url} did not load." in looked, looked
        assert _MARKER not in looked, looked
    assert untrusted_site.served == []  # the browser never even asked for the page

    # Once a page does load, the same run reads it as usual.
    recovered = run.dispatch_service.dispatch("browser", {"action": "open", "url": f"{site.url}/count"})
    assert "status=success" in recovered.splitlines()[0], recovered
    read = run.dispatch_service.dispatch("browser", {"action": "read"})
    assert "status=success" in read.splitlines()[0], read
    assert '"title":"Visit 1"' in read, read


def test_a_site_that_refuses_the_connection_is_reported_as_not_loading_with_the_browsers_reason(
    db: sqlite3.Connection, browser_data: Path
) -> None:
    with socket.socket() as probe:  # a port nothing listens on once the probe lets it go
        probe.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{probe.getsockname()[1]}/"
    run = _browsing_run()

    opened = run.dispatch_service.dispatch("browser", {"action": "open", "url": url})
    assert "code=page-load-failed" in opened.splitlines()[0], opened
    assert f"ERR_CONNECTION_REFUSED at {url}" in opened, opened

    read = run.dispatch_service.dispatch("browser", {"action": "read"})
    assert "code=no-open-page" in read.splitlines()[0], read
    assert f"No page is open: {url} did not load." in read, read


def test_the_person_is_warned_of_a_self_signed_site_and_their_proceed_opens_it_in_every_tab_until_the_browser_restarts(
    app: Flask, untrusted_site: _UntrustedSite,
) -> None:
    client, token = signed_in(app)
    port = untrusted_site.server_address[1]
    with live_server(app) as host, main_socket(host, token) as ui:
        tab = _user_tab(client, untrusted_site.url)
        _heard(ui, lambda: _refusal(client, tab.id) == _warning(untrusted_site))
        assert untrusted_site.served == []

        # Reloading the warning asks the site again, and the browser refuses it again.
        first = tab.blocked.loader if tab.blocked is not None else ""
        assert client.post(f"/api/browser/history/{tab.id}", json={"action": "reload"}).status_code == 204
        _heard(ui, lambda: (refused := tab.blocked) is not None and refused.loader != first)
        assert _refusal(client, tab.id) == _warning(untrusted_site)

        # A Proceed for any other site than the one the warning names allows nothing.
        assert _proceed(client, tab.id, "localhost", port) == 409
        assert _refusal(client, tab.id) == _warning(untrusted_site)
        assert untrusted_site.served == []

        assert _proceed(client, tab.id, "127.0.0.1", port) == 204
        assert _refusal(client, tab.id) is None  # the warning goes at once, before the page loads
        assert wait_for(tab, lambda: tab.title == "Untrusted"), (tab.url, tab.title)
        assert untrusted_site.served == ["/"]

        # The site now opens in the person's other tabs, and in Chalie's, with no warning.
        second = open_user_tab(client, untrusted_site.url, "Untrusted")
        assert _refusal(client, second.id) is None
        opened = _browse(_browsing_run(_ORIGIN), action="open", url=untrusted_site.url)
        assert "status=success" in opened.splitlines()[0], opened
        assert _MARKER in opened, opened
        assert untrusted_site.served == ["/", "/", "/"]

        # There is nothing left to proceed past, and no tab to proceed in.
        assert _proceed(client, tab.id, "127.0.0.1", port) == 409
        assert _proceed(client, "no-such-tab", "127.0.0.1", port) == 404

        # A browser that restarts has forgotten the choice: the site is refused again.
        close_all()
        again = _user_tab(client, untrusted_site.url)
        _heard(ui, lambda: _refusal(client, again.id) == _warning(untrusted_site))
    assert untrusted_site.served == ["/", "/", "/"]


def test_going_back_from_the_warning_shows_the_page_the_tab_stayed_on_and_allows_nothing(
    app: Flask, site: Site, untrusted_site: _UntrustedSite,
) -> None:
    client, token = signed_in(app)
    with live_server(app) as host, main_socket(host, token) as ui:
        tab = open_user_tab(client, f"{site.url}/a", "Page A")
        assert client.post(f"/api/browser/navigate/{tab.id}", json={"url": untrusted_site.url}).status_code == 204
        _heard(ui, lambda: _refusal(client, tab.id) == _warning(untrusted_site))

        assert client.post(f"/api/browser/history/{tab.id}", json={"action": "back"}).status_code == 204
        [row] = listed(client)
        assert (row["certificate_error"], row["url"], row["title"]) == (None, f"{site.url}/a", "Page A")

        # Going back chose nothing for the site: asking for it again is refused again.
        assert client.post(f"/api/browser/navigate/{tab.id}", json={"url": untrusted_site.url}).status_code == 204
        _heard(ui, lambda: _refusal(client, tab.id) == _warning(untrusted_site))
    assert untrusted_site.served == []


def test_nothing_a_run_sends_gets_its_page_past_a_certificate_warning(
    app: Flask, untrusted_site: _UntrustedSite,
) -> None:
    client, token = signed_in(app)
    run = _browsing_run(_ORIGIN)
    opened = _browse(run, action="open", url=untrusted_site.url)
    assert "code=certificate-error" in opened.splitlines()[0], opened
    tab = session.run_tab(run.uid or 0)
    assert tab is not None, "the run's page is not open"
    [row] = listed(client)
    assert (row["id"], row["opener"], row["certificate_error"]) == (tab.id, "chalie", _warning(untrusted_site))

    # Every verb, with what it needs, and every way of asking again: none of them proceeds.
    looks: list[dict[str, object]] = [
        {"action": "read"},
        {"action": "find", "query": "trusted"},
        {"action": "click", "target": "only a trusted connection"},
        {"action": "fill", "target": "input", "value": "proceed"},
        {"action": "select", "target": "select", "value": "proceed"},
        {"action": "scroll", "direction": "down"},
        {"action": "style", "target": "p"},
        {"action": "screenshot"},
    ]
    for params in looks:
        answer = _browse(run, **params)
        assert "code=no-open-page" in answer.splitlines()[0], (params, answer)
        assert f"No page is open: {untrusted_site.url} did not load." in answer, (params, answer)
    # Opening again is refused again; no parameter asks the browser to proceed — any other is turned away.
    retries: list[tuple[dict[str, object], str]] = [
        ({"action": "open", "url": untrusted_site.url}, "certificate-error"),
        ({"action": "open", "url": untrusted_site.url, "proceed": True}, "unknown-params"),
        ({"action": "open", "url": untrusted_site.url, "ignore_certificate_errors": True}, "unknown-params"),
    ]
    for params, code in retries:
        answer = _browse(run, **params)
        assert f"code={code}" in answer.splitlines()[0], (params, answer)
        assert _MARKER not in answer, (params, answer)
    assert _refusal(client, tab.id) == _warning(untrusted_site)
    assert untrusted_site.served == []

    # A hand-off the person finishes without proceeding leaves the run where it was.
    answers: list[str] = []
    waiting = threading.Thread(
        target=lambda: answers.append(_browse(run, action="handoff", reason="Please check this site")),
        name="handoff-run", daemon=True,
    )
    with live_server(app) as host, main_socket(host, token) as ui:
        waiting.start()
        try:
            _heard(ui, lambda: listed(client)[0]["handoff"] is not None)
            asked = cast("dict[str, object]", listed(client)[0]["handoff"])
            assert client.post(f"/api/browser/handoffs/{asked['id']}").status_code == 204
            waiting.join(timeout=_RUN_WAIT)
            assert not waiting.is_alive(), "the run did not carry on after Done"
        finally:
            if waiting.is_alive():  # free the run before the browser goes away
                session.close_session(run.uid or 0)
                waiting.join(timeout=30)
    [handed] = answers
    assert "code=no-open-page" in handed.splitlines()[0], handed

    # Going back leaves the warning, as in any browser, and allows the site nothing.
    back = _browse(run, action="back")
    assert "status=success" in back.splitlines()[0], back
    assert _refusal(client, tab.id) is None
    with live_server(app) as host, main_socket(host, token) as ui:
        mine = _user_tab(client, untrusted_site.url)
        _heard(ui, lambda: _refusal(client, mine.id) == _warning(untrusted_site))
    assert untrusted_site.served == []


def test_a_run_the_person_proceeds_for_during_a_hand_off_carries_on_reading_the_site(
    app: Flask, untrusted_site: _UntrustedSite,
) -> None:
    client, token = signed_in(app)
    run = _browsing_run(_ORIGIN)
    opened = _browse(run, action="open", url=untrusted_site.url)
    assert "code=certificate-error" in opened.splitlines()[0], opened
    tab = session.run_tab(run.uid or 0)
    assert tab is not None, "the run's page is not open"

    answers: list[str] = []
    waiting = threading.Thread(
        target=lambda: answers.append(_browse(run, action="handoff", reason="Please check this site")),
        name="handoff-run", daemon=True,
    )
    with live_server(app) as host, main_socket(host, token) as ui:
        waiting.start()
        try:
            _heard(ui, lambda: listed(client)[0]["handoff"] is not None)
            [row] = listed(client)
            assert row["certificate_error"] == _warning(untrusted_site)
            asked = cast("dict[str, object]", row["handoff"])

            assert _proceed(client, tab.id, "127.0.0.1", untrusted_site.server_address[1]) == 204
            assert wait_for(tab, lambda: tab.title == "Untrusted"), (tab.url, tab.title)
            assert client.post(f"/api/browser/handoffs/{asked['id']}").status_code == 204
            waiting.join(timeout=_RUN_WAIT)
            assert not waiting.is_alive(), "the run did not carry on after Done"
        finally:
            if waiting.is_alive():  # free the run before the browser goes away
                session.close_session(run.uid or 0)
                waiting.join(timeout=30)

    [answer] = answers
    assert "status=success" in answer.splitlines()[0], answer
    assert _MARKER in answer, answer
    assert f'"url":"{untrusted_site.url}"' in answer, answer
    assert untrusted_site.served == ["/"]


def test_proceeding_to_one_self_signed_site_leaves_every_other_site_refused(
    app: Flask, untrusted_site: _UntrustedSite, other_untrusted_site: _UntrustedSite,
) -> None:
    client, token = signed_in(app)
    with live_server(app) as host, main_socket(host, token) as ui:
        tab = _user_tab(client, untrusted_site.url)
        _heard(ui, lambda: _refusal(client, tab.id) == _warning(untrusted_site))
        assert _proceed(client, tab.id, "127.0.0.1", untrusted_site.server_address[1]) == 204
        assert wait_for(tab, lambda: tab.title == "Untrusted"), (tab.url, tab.title)

        # Another port on the same host, and the same port under another name, are other sites.
        for other, name in ((other_untrusted_site, "127.0.0.1"), (untrusted_site, "localhost")):
            refused = _user_tab(client, str(_warning(other, name)["url"]))
            _heard(ui, lambda: _refusal(client, refused.id) == _warning(other, name))
        opened = _browse(_browsing_run(_ORIGIN), action="open", url=other_untrusted_site.url)
        assert "code=certificate-error" in opened.splitlines()[0], opened
    assert untrusted_site.served == ["/"]
    assert other_untrusted_site.served == []


def test_a_site_with_nothing_wrong_loads_as_before_and_shows_no_warning(app: Flask, site: Site) -> None:
    client, _token = signed_in(app)
    mine = open_user_tab(client, f"{site.url}/a", "Page A")
    opened = _browse(_browsing_run(_ORIGIN), action="open", url=f"{site.url}/b")
    assert "status=success" in opened.splitlines()[0], opened
    assert '"title":"Page B"' in opened, opened
    assert {str(t["id"]): t["certificate_error"] for t in listed(client)}[mine.id] is None
    assert [t["certificate_error"] for t in listed(client)] == [None, None]
