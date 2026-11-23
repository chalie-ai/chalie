"""Feature tests: setting up a capability against a server that serves HTTPS.

The UniFi console is a real server on loopback that answers the UniFi OS
sign-in the way a console does for a UI.com account: the right username and
password get a session cookie and a CSRF token, or, for an account with 2FA
on, a 499 and a half-made sign-in that only the account's current code
finishes; any other pair gets 403 with the console's own "invalid username or
password" body, and the API answers only a request that carries a session it
handed out. A test can set the code the account wants, force a refusal, and
have the console refuse a session it still knows. The Home Assistant server
answers its API only for the right access token. Both present a certificate
this machine does not trust unless a test says it does, and both record the
method, path and status of every request they answer, nothing more.

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
from werkzeug.test import TestResponse

import capabilities
import services.vault_service as vault_module
from services.memory_store import MemoryStore
from services.vault_service import _vault_state, get_vault_service

pytestmark = pytest.mark.unit

_VAULT_PASSWORD = "test-password"

USERNAME = "chalie-local"
# The spaces at its ends are part of the password: the console takes it only exactly as typed.
PASSWORD = "  correct horse battery staple "
LOGIN_PATH = "/api/auth/login"
DEVICES_PATH = "/proxy/network/api/s/default/stat/device"
# The half-made sign-in a console hands back with its 499, as the cookie that finishes it.
MFA_COOKIE = "UBIC_2FA=half-made-sign-in-5Qv8ZtR2"
DEVICE = {
    "mac": "f4:92:bf:00:00:01",
    "name": "Office AP",
    "model": "U6-Pro",
    "type": "uap",
    "ip": "192.0.2.20",
    "state": 1,
    "uptime": 3600,
    "version": "6.6.55",
}
_ANSWERS: dict[int, dict[str, str]] = {
    401: {"message": "Unauthorized"},
    403: {
        "code": "AUTHENTICATION_FAILED_INVALID_CREDENTIALS",
        "message": "Invalid username or password",
    },
    429: {
        "code": "AUTHENTICATION_FAILED_LIMIT_REACHED",
        "message": "You've reached the login attempt limit",
    },
}

_HA_TOKEN = "home-assistant-long-lived-token"


@dataclass
class Console:
    """A running server, how it answers, and every request it has answered.

    For UniFi: ``code`` is the 2FA code the UI.com account wants with its
    password ("" for an account without 2FA); ``sign_in_status`` and
    ``code_status`` force the answer to the password sign-in and to the code
    sign-in; ``refuse_next`` is how many coming data calls are answered as
    signed out whatever session they carry.
    """

    url: str = ""
    code: str = ""
    sign_in_status: int | None = None
    code_status: int | None = None
    refuse_next: int = 0
    sessions: int = 0
    seen: list[tuple[str, str, int]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def requests(self) -> list[tuple[str, str, int]]:
        """Every request answered so far, oldest first, as (method, path, status)."""
        with self.lock:
            return list(self.seen)


class _Handler(BaseHTTPRequestHandler):
    console: Console

    def _body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def _send(
        self, status: int, payload: object, headers: tuple[tuple[str, str], ...] = ()
    ) -> None:
        with self.console.lock:
            self.console.seen.append((self.command, self.path, status))
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


class UnifiHandler(_Handler):
    """A UniFi OS console signed in to with a UI.com account."""

    def do_POST(self) -> None:  # noqa: N802 — http.server contract
        sign_in = json.loads(self._body() or b"{}")
        if self.path != LOGIN_PATH:
            self._send(404, {})
            return
        console = self.console
        code = sign_in.pop("token", None)
        forced = console.sign_in_status if code is None else console.code_status
        if forced is not None:
            self._send(forced, _ANSWERS[forced])
        elif sign_in != {
            "username": USERNAME,
            "password": PASSWORD,
            "rememberMe": True,
        }:
            self._send(403, _ANSWERS[403])
        elif console.code and (
            code != console.code or self.headers.get("Cookie") != MFA_COOKIE
        ):
            self._send(
                499,
                {
                    "code": "MFA_AUTH_REQUIRED",
                    "message": "MFA token required to authenticate to SSO",
                    "data": {"mfaCookie": f"{MFA_COOKIE}; path=/; secure; httponly"},
                },
            )
        else:
            with console.lock:
                console.sessions += 1
                n = console.sessions
            self._send(
                200,
                {"username": USERNAME},
                (
                    (
                        "Set-Cookie",
                        f"TOKEN=session-{n}; path=/; samesite=none; httponly",
                    ),
                    ("X-CSRF-Token", f"csrf-{n}"),
                ),
            )

    def do_GET(self) -> None:  # noqa: N802 — http.server contract
        console = self.console
        with console.lock:
            refused = console.refuse_next > 0
            console.refuse_next -= refused
            handed_out = {
                (f"TOKEN=session-{n}", f"csrf-{n}")
                for n in range(1, console.sessions + 1)
            }
        if self.path != DEVICES_PATH:
            self._send(404, {})
        elif (
            refused
            or (self.headers.get("Cookie"), self.headers.get("X-CSRF-Token"))
            not in handed_out
        ):
            self._send(401, _ANSWERS[401])
        else:
            self._send(200, {"data": [DEVICE]})


class _HomeAssistantHandler(_Handler):
    def do_GET(self) -> None:  # noqa: N802 — http.server contract
        if self.path != "/api/":
            self._send(404, {})
        elif self.headers.get("Authorization") != f"Bearer {_HA_TOKEN}":
            self._send(401, _ANSWERS[401])
        else:
            self._send(200, {"message": "API running."})


@contextmanager
def serve(
    handler: type[_Handler], certificate: tuple[Path, Path] | None
) -> Iterator[Console]:
    """Serve ``handler`` over HTTPS with ``certificate``, or over plain HTTP without one."""
    console = Console()
    bound = cast("type[_Handler]", type("Bound", (handler,), {"console": console}))
    server = ThreadingHTTPServer(("127.0.0.1", 0), bound)
    if certificate is not None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(*certificate)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    console.url = (
        f"{'https' if certificate else 'http'}://127.0.0.1:{server.server_address[1]}"
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield console
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def first_sync_done(cap_id: str) -> None:
    """Wait out the first sync a setup starts on its own thread."""
    for first_sync in [
        t for t in threading.enumerate() if t.name == f"cap-init-{cap_id}"
    ]:
        first_sync.join(timeout=15)
        assert not first_sync.is_alive(), f"the first sync of {cap_id} did not end"


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def certificate(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A self-signed certificate for 127.0.0.1 and its key."""
    folder = tmp_path_factory.mktemp("console-certificate")
    cert, key = folder / "cert.pem", folder / "key.pem"
    subprocess.run(  # noqa: S603 — fixed arguments
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-subj",
            "/CN=127.0.0.1",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


@pytest.fixture()
def unifi(certificate: tuple[Path, Path]) -> Iterator[Console]:
    with serve(UnifiHandler, certificate) as console:
        yield console


@pytest.fixture()
def home_assistant(certificate: tuple[Path, Path]) -> Iterator[Console]:
    with serve(_HomeAssistantHandler, certificate) as console:
        yield console


@pytest.fixture()
def trusted(certificate: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """This machine trusts the servers' certificate, as if its owner installed it."""
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(certificate[0]))


@pytest.fixture()
def client(
    authed_client: tuple[FlaskClient, object, MemoryStore],
) -> Iterator[FlaskClient]:
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


def set_up_unifi(
    client: FlaskClient, url: str, *, verify_ssl: bool = False
) -> TestResponse:
    """Save the UniFi form as the Brain does, and wait out the first sync it starts."""
    resp = client.post(
        "/api/capabilities/setup/ubiquiti",
        json={
            "url": url,
            "username": USERNAME,
            "password": PASSWORD,
            "site": "default",
            "verify_ssl": verify_ssl,
        },
    )
    first_sync_done("ubiquiti")
    return resp


def _set_up_home(
    client: FlaskClient, url: str, *, token: str = _HA_TOKEN, verify_ssl: bool
) -> TestResponse:
    return client.post(
        "/api/capabilities/setup/home",
        json={
            "url": url,
            "token": token,
            "verify_ssl": verify_ssl,
        },
    )


def _refusal(resp: TestResponse) -> object:
    """The plain reason a refused setup gives, which the Brain words for the reader."""
    assert resp.status_code == 422
    return cast("dict[str, object]", resp.get_json()).get("reason")


def _connected(client: FlaskClient, cap_id: str) -> bool:
    resp = client.get(f"/api/capabilities/status/{cap_id}")
    assert resp.status_code == 200
    return (
        cast("dict[str, dict[str, object]]", resp.get_json())["result"]["connected"]
        is True
    )


def _saved_settings(client: FlaskClient, cap_id: str) -> dict[str, object]:
    """The settings the Brain reads back to draw the capability's form."""
    resp = client.get(f"/api/capabilities/{cap_id}")
    assert resp.status_code == 200
    return cast("dict[str, dict[str, dict[str, object]]]", resp.get_json())["result"][
        "config"
    ]


# ── UniFi setup ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("checked", [False, True])
def test_unifi_is_set_up_and_reconnected_without_reaching_the_console(
    client: FlaskClient,
    unifi: Console,
    checked: bool,
) -> None:
    # A sign-in may wait on the user's 2FA code, so saving the form must not sign in.
    unifi.code = "048217"
    assert set_up_unifi(client, unifi.url, verify_ssl=checked).status_code == 204
    assert _connected(client, "ubiquiti")
    assert _saved_settings(client, "ubiquiti")["verify_ssl"] is checked

    capabilities._capabilities_cache = None  # as after a restart
    assert capabilities.load_capabilities()["ubiquiti"].connect()
    assert unifi.requests() == []


# ── The certificate check reads back as it was saved ────────────────────────


@pytest.mark.usefixtures("trusted")
@pytest.mark.parametrize("checked", [False, True])
def test_home_certificate_check_reads_back_as_saved(
    client: FlaskClient,
    home_assistant: Console,
    checked: bool,
) -> None:
    assert (
        _set_up_home(client, home_assistant.url, verify_ssl=checked).status_code == 204
    )
    assert _saved_settings(client, "home")["verify_ssl"] is checked


# ── Home Assistant sign-in ───────────────────────────────────────────────────


@pytest.mark.usefixtures("trusted")
def test_a_wrong_home_assistant_token_is_refused_as_rejected(
    client: FlaskClient, home_assistant: Console
) -> None:
    resp = _set_up_home(
        client, home_assistant.url, token="not the token", verify_ssl=True
    )
    assert _refusal(resp) == "credentials_rejected"
    assert not _connected(client, "home")
