"""Feature tests: setting up a capability against a console that serves HTTPS.

The UniFi console is a real HTTPS server on loopback that answers the UniFi OS
sign-in the way a console does: the right username and password get a session
cookie and a CSRF token, any other pair gets 403 with the console's own
"invalid username or password" body, and the API answers only a request that
carries the session cookie. It presents a certificate this machine does not
trust and records every request it receives.

The setup, status and detail endpoints, the vault and the database are the real
ones, so each test drives the same path the Brain's capability form does:
save the settings, then read them back to draw the form again.
"""

from __future__ import annotations

import json
import ssl
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import pytest
from flask.testing import FlaskClient

import capabilities
import services.vault_service as vault_module
from capabilities.ubiquiti_capability import unifi_rest_handler
from services.memory_store import MemoryStore
from services.vault_service import _vault_state, get_vault_service

pytestmark = pytest.mark.unit

_VAULT_PASSWORD = "test-password"

_USERNAME = "chalie-local"
_PASSWORD = "correct horse battery staple"
_SESSION_TOKEN = "console-session-1"
_CSRF_TOKEN = "console-csrf-1"
_REJECTED = {"code": "AUTHENTICATION_FAILED_INVALID_CREDENTIALS", "message": "Invalid username or password"}

@dataclass(frozen=True)
class _Seen:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes


@dataclass
class _Console:
    """A running HTTPS server and every request it has received."""

    url: str = ""
    seen: list[_Seen] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def requests(self) -> list[_Seen]:
        with self.lock:
            return list(self.seen)


class _Handler(BaseHTTPRequestHandler):
    console: _Console

    def _record(self) -> bytes:
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        with self.console.lock:
            self.console.seen.append(_Seen(self.command, self.path, dict(self.headers.items()), body))
        return body

    def _send(self, status: int, payload: object, headers: tuple[tuple[str, str], ...] = ()) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 — http.server contract
        return


class _UnifiHandler(_Handler):
    def do_POST(self) -> None:  # noqa: N802 — http.server contract
        body = self._record()
        if self.path != "/api/auth/login":
            self._send(404, {})
            return
        if json.loads(body or b"{}") == {"username": _USERNAME, "password": _PASSWORD, "rememberMe": True}:
            self._send(200, {"username": _USERNAME}, (
                ("Set-Cookie", f"TOKEN={_SESSION_TOKEN}; path=/; samesite=none; secure; httponly"),
                ("X-CSRF-Token", _CSRF_TOKEN),
            ))
            return
        self._send(403, _REJECTED)

    def do_GET(self) -> None:  # noqa: N802 — http.server contract
        self._record()
        if self.path != "/api/":
            self._send(404, {})
        elif f"TOKEN={_SESSION_TOKEN}" not in (self.headers.get("Cookie") or ""):
            self._send(401, {"message": "Unauthorized"})
        else:
            self._send(200, {"hardware": {"shortname": "UDMPRO"}})


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def certificate(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A self-signed certificate for 127.0.0.1 and its key."""
    folder = tmp_path_factory.mktemp("console-certificate")
    cert, key = folder / "cert.pem", folder / "key.pem"
    subprocess.run(  # noqa: S603 — fixed arguments
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
            "-keyout", str(key), "-out", str(cert),
            "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1",
        ],
        check=True, capture_output=True,
    )
    return cert, key


@contextmanager
def _serve(handler: type[_Handler], certificate: tuple[Path, Path]) -> Iterator[_Console]:
    console = _Console()
    bound = cast("type[_Handler]", type("Bound", (handler,), {"console": console}))
    server = ThreadingHTTPServer(("127.0.0.1", 0), bound)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(*certificate)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    console.url = f"https://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield console
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture()
def unifi(certificate: tuple[Path, Path]) -> Iterator[_Console]:
    with _serve(_UnifiHandler, certificate) as console:
        yield console
    unifi_rest_handler._session_cache.clear()


@pytest.fixture()
def client(authed_client: tuple[FlaskClient, object, MemoryStore]) -> Iterator[FlaskClient]:
    """The real app with an unlocked vault and a capability registry of its own."""
    _vault_state.dek = None
    vault_module._vault_service_instance = None
    vault = get_vault_service()
    vault.initialize(_VAULT_PASSWORD)
    vault.unlock(_VAULT_PASSWORD)
    saved_registry = capabilities._capabilities_cache
    capabilities._capabilities_cache = None
    try:
        yield authed_client[0]
    finally:
        capabilities._capabilities_cache = saved_registry
        _vault_state.dek = None
        vault_module._vault_service_instance = None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _set_up_unifi(client: FlaskClient, url: str, *, password: str = _PASSWORD, verify_ssl: bool = False) -> int:
    return client.post("/api/capabilities/setup/ubiquiti", json={
        "url": url, "auth_method": "credentials", "username": _USERNAME,
        "password": password, "site": "default", "verify_ssl": verify_ssl,
    }).status_code


def _connected(client: FlaskClient, cap_id: str) -> bool:
    resp = client.get(f"/api/capabilities/status/{cap_id}")
    assert resp.status_code == 200
    return cast("dict[str, dict[str, object]]", resp.get_json())["result"]["connected"] is True


def _saved_settings(client: FlaskClient, cap_id: str) -> dict[str, object]:
    """The settings the Brain reads back to draw the capability's form."""
    resp = client.get(f"/api/capabilities/{cap_id}")
    assert resp.status_code == 200
    return cast("dict[str, dict[str, dict[str, object]]]", resp.get_json())["result"]["config"]


# ── UniFi sign-in ────────────────────────────────────────────────────────────


def test_a_console_user_signs_in_and_unifi_connects(client: FlaskClient, unifi: _Console) -> None:
    assert _set_up_unifi(client, unifi.url) == 204
    assert _connected(client, "ubiquiti")

    login, probe = unifi.requests()[:2]
    assert (login.method, login.path) == ("POST", "/api/auth/login")
    assert json.loads(login.body) == {"username": _USERNAME, "password": _PASSWORD, "rememberMe": True}
    assert (probe.method, probe.path) == ("GET", "/api/")
    assert f"TOKEN={_SESSION_TOKEN}" in probe.headers.get("Cookie", "")


def test_a_wrong_password_is_refused_and_unifi_stays_unconnected(client: FlaskClient, unifi: _Console) -> None:
    assert _set_up_unifi(client, unifi.url, password="not the password") == 422
    assert not _connected(client, "ubiquiti")
    assert [(r.method, r.path) for r in unifi.requests()] == [("POST", "/api/auth/login")]
    assert _saved_settings(client, "ubiquiti") == {}


def test_with_the_certificate_check_on_an_untrusted_console_never_gets_the_password(
    client: FlaskClient, unifi: _Console,
) -> None:
    assert _set_up_unifi(client, unifi.url, verify_ssl=True) == 422
    assert not _connected(client, "ubiquiti")
    assert unifi.requests() == []
