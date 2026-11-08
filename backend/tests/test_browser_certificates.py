"""Feature test: the browser enforces certificates like a normal browser.

A real browsing run drives the real ``browser`` ability, through the real
dispatcher, at a real loopback HTTPS site whose certificate is self-signed. The
real headless shell refuses the connection: the site's page never reaches the
browser, and the run is told the certificate is why, so it can say so instead
of reading a blank page. Nothing is substituted; the browser runs on the test's
own throwaway profile.
"""

from __future__ import annotations

import datetime
import ipaddress
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


def test_a_page_chalie_opens_on_a_self_signed_site_never_loads_and_chalie_is_told_the_certificate_is_why(
    db: sqlite3.Connection, browser_data: Path, untrusted_site: _UntrustedSite
) -> None:
    run = MessageProcessor(WebBrowseConfig(PolicyChannel.CHAT), raw_input="read the page")
    run.active_tools = list(run.config.always_available or [])

    opened = run.dispatch_service.dispatch("browser", {"action": "open", "url": untrusted_site.url})
    assert "status=error" in opened.splitlines()[0], opened
    # Chromium names every certificate failure ERR_CERT_*; a reworded message must still name the certificate.
    assert "ERR_CERT_" in opened or "certificate" in opened.lower(), opened
    assert _MARKER not in opened, opened

    # Whatever the tab now shows, the site's page is not in it.
    read = run.dispatch_service.dispatch("browser", {"action": "read"})
    assert _MARKER not in read, read
    assert untrusted_site.served == []  # the browser never even asked for the page
