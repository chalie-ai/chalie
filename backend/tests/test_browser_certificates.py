"""Feature test: the browser enforces certificates like a normal browser.

A real browsing run drives the real ``browser`` ability, through the real
dispatcher, at a real loopback HTTPS site whose certificate is self-signed. The
real headless shell refuses the connection: the site's page never reaches the
browser, and the run is told the certificate is why and that retrying will not
help, so it can say so instead of reading a blank page. Until a page does load,
every read of the tab says nothing loaded rather than reporting the browser's
blank error page as a page. A site that refuses the connection outright is
reported the same way, under its own reason. Nothing is substituted; the
browser runs on the test's own throwaway profile.
"""

from __future__ import annotations

import datetime
import ipaddress
import logging
import socket
import sqlite3
import ssl
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from configs.channels.web_browse import WebBrowseConfig
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from tests._browser_live import Site

pytest_plugins = ("tests._browser_live",)
pytestmark = pytest.mark.e2e

_MARKER = "only a trusted connection ever reads this line"


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


@pytest.fixture
def untrusted_site(tmp_path: Path) -> Iterator[_UntrustedSite]:
    keys = tmp_path / "keys"
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


def _browsing_run() -> MessageProcessor:
    run = MessageProcessor(WebBrowseConfig(PolicyChannel.CHAT), raw_input="read the page")
    run.active_tools = list(run.config.always_available or [])
    return run


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

    # The tab now shows the browser's own blank error page: every look at it says nothing loaded.
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
