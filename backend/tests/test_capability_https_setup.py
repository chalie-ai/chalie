"""Feature tests: setting up a capability against a console that serves HTTPS.

The UniFi console is a real HTTPS server on loopback that answers the UniFi OS
sign-in the way a console does: the right username and password, compared
exactly as typed, get a session cookie and a CSRF token, any other pair gets 403 with the console's own
"invalid username or password" body, and the API answers only a request that
carries the session cookie. A console signed in to with a UI.com account that has
an authenticator app first answers the right pair with 499 and a half-made
sign-in, and finishes it only for a code that account's own setup key makes at
that moment. The Home Assistant server answers its API only for
the right access token. Both present a certificate this machine does not trust
unless a test says it does, and both record every request they receive.

The setup, status and detail endpoints, the vault and the database are the real
ones, so each test drives the same path the Brain's capability form does:
save the settings, then read them back to draw the form again.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import re
import socket
import ssl
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import pytest
from flask.testing import FlaskClient
from werkzeug.test import TestResponse

import capabilities
import services.vault_service as vault_module
from capabilities.ubiquiti_capability import authenticator, unifi_rest_handler
from services.memory_store import MemoryStore
from services.vault_service import _vault_state, get_vault_service

pytestmark = pytest.mark.unit

_VAULT_PASSWORD = "test-password"

_USERNAME = "chalie-local"
_PASSWORD = "correct horse battery staple"
_SESSION_TOKEN = "console-session-1"
_CSRF_TOKEN = "console-csrf-1"
_REJECTED = {"code": "AUTHENTICATION_FAILED_INVALID_CREDENTIALS", "message": "Invalid username or password"}

_HA_TOKEN = "home-assistant-long-lived-token"

# The RFC 6238 test secret, the setup key UI.com would show for it, and a second
# account's secret whose key is shown with padding.
_SECRET = b"12345678901234567890"
_SETUP_KEY = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"
_OTHER_SECRET = b"another account!"
_OTHER_SETUP_KEY = base64.b32encode(_OTHER_SECRET).decode()
_HALF_SIGNED_IN = "UBIC_2FA=half-made-sign-in"


@dataclass(frozen=True)
class _Seen:
    method: str
    path: str
    headers: Message[str, str]  # looked up by name in any case, as HTTP header names are
    body: bytes


@dataclass
class _Console:
    """A running HTTPS server and every request it has received."""

    url: str = ""
    password: str = _PASSWORD
    # The authenticator secret of the UI.com account signed in with, and whether
    # UI.com has blocked its sign-ins after too many attempts.
    secret: bytes = _SECRET
    blocked: bool = False
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
            self.console.seen.append(_Seen(self.command, self.path, self.headers, body))
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
        if json.loads(body or b"{}") == {"username": _USERNAME, "password": self.console.password, "rememberMe": True}:
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


def _code(secret: bytes, step: int) -> str:
    """The RFC 6238 code for ``secret`` in the 30-second ``step``, worked out as UI.com does."""
    mac = hmac.new(secret, step.to_bytes(8, "big"), hashlib.sha1).digest()
    offset = mac[-1] & 0x0F
    return f"{(int.from_bytes(mac[offset:offset + 4], 'big') & 0x7FFFFFFF) % 1_000_000:06d}"


def _codes_now(secret: bytes) -> set[str]:
    """The codes accepted now: this step's and its neighbours', for clocks a little apart."""
    step = int(time.time()) // 30
    return {_code(secret, step + drift) for drift in (-1, 0, 1)}


class _UiAccountHandler(_UnifiHandler):
    """A console signed in to with a UI.com account that has an authenticator app."""

    def do_POST(self) -> None:  # noqa: N802 — http.server contract
        body = self._record()
        if self.path != "/api/auth/login":
            self._send(404, {})
            return
        if self.console.blocked:
            self._send(429, {"code": "AUTHENTICATION_FAILED_LIMIT_REACHED", "message": "You've reached the login attempt limit"})
            return
        sign_in = json.loads(body or b"{}")
        token = sign_in.pop("token", None)
        if sign_in != {"username": _USERNAME, "password": self.console.password, "rememberMe": True}:
            self._send(403, _REJECTED)
            return
        if (
            _HALF_SIGNED_IN not in (self.headers.get("Cookie") or "")
            or token not in _codes_now(self.console.secret)
        ):
            self._send(499, {
                "code": "MFA_AUTH_REQUIRED",
                "message": "MFA token required to authenticate to SSO",
                "data": {"mfaCookie": f"{_HALF_SIGNED_IN}; path=/; secure; httponly"},
            })
            return
        self._send(200, {"username": _USERNAME}, (
            ("Set-Cookie", f"TOKEN={_SESSION_TOKEN}; path=/; samesite=none; secure; httponly"),
            ("X-CSRF-Token", _CSRF_TOKEN),
        ))


class _HomeAssistantHandler(_Handler):
    def do_GET(self) -> None:  # noqa: N802 — http.server contract
        self._record()
        if self.path != "/api/":
            self._send(404, {})
        elif self.headers.get("Authorization") != f"Bearer {_HA_TOKEN}":
            self._send(401, {"message": "Unauthorized"})
        else:
            self._send(200, {"message": "API running."})


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
def _serve(handler: type[_Handler], certificate: tuple[Path, Path] | None) -> Iterator[_Console]:
    """Serve ``handler`` over HTTPS with ``certificate``, or over plain HTTP without one."""
    console = _Console()
    bound = cast("type[_Handler]", type("Bound", (handler,), {"console": console}))
    server = ThreadingHTTPServer(("127.0.0.1", 0), bound)
    if certificate is not None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(*certificate)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    console.url = f"{'https' if certificate else 'http'}://127.0.0.1:{server.server_address[1]}"
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
def ui_account(certificate: tuple[Path, Path]) -> Iterator[_Console]:
    with _serve(_UiAccountHandler, certificate) as console:
        yield console
    unifi_rest_handler._session_cache.clear()


@pytest.fixture()
def home_assistant(certificate: tuple[Path, Path]) -> Iterator[_Console]:
    with _serve(_HomeAssistantHandler, certificate) as console:
        yield console


@pytest.fixture()
def trusted(certificate: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """This machine trusts the consoles' certificate, as if its owner installed it."""
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(certificate[0]))


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


def _set_up_unifi(
    client: FlaskClient, url: str, *, password: str = _PASSWORD, verify_ssl: bool = False,
) -> TestResponse:
    return client.post("/api/capabilities/setup/ubiquiti", json={
        "url": url, "auth_method": "credentials", "username": _USERNAME,
        "password": password, "site": "default", "verify_ssl": verify_ssl,
    })


def _set_up_ui_account(
    client: FlaskClient, url: str, setup_key: str, *, auth_method: str = "ui_account",
) -> TestResponse:
    return client.post("/api/capabilities/setup/ubiquiti", json={
        "url": url, "auth_method": auth_method, "username": _USERNAME, "password": _PASSWORD,
        "totp_key": setup_key, "site": "default", "verify_ssl": False,
    })


def _sign_ins(console: _Console) -> list[_Seen]:
    return [r for r in console.requests() if r.path == "/api/auth/login"]


def _set_up_home(client: FlaskClient, url: str, *, token: str = _HA_TOKEN, verify_ssl: bool) -> TestResponse:
    return client.post("/api/capabilities/setup/home", json={
        "url": url, "token": token, "verify_ssl": verify_ssl,
    })


def _refusal(resp: TestResponse) -> object:
    """The plain reason a refused setup gives, which the Brain words for the reader."""
    assert resp.status_code == 422
    return cast("dict[str, object]", resp.get_json()).get("reason")


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
    assert _set_up_unifi(client, unifi.url).status_code == 204
    assert _connected(client, "ubiquiti")

    login, probe = unifi.requests()[:2]
    assert (login.method, login.path) == ("POST", "/api/auth/login")
    assert json.loads(login.body) == {"username": _USERNAME, "password": _PASSWORD, "rememberMe": True}
    assert (probe.method, probe.path) == ("GET", "/api/")
    assert f"TOKEN={_SESSION_TOKEN}" in probe.headers.get("Cookie", "")


def test_a_password_with_spaces_at_its_ends_is_sent_as_typed(client: FlaskClient, unifi: _Console) -> None:
    unifi.password = f"  {_PASSWORD} "
    assert _set_up_unifi(client, unifi.url, password=unifi.password).status_code == 204
    assert _connected(client, "ubiquiti")
    assert json.loads(unifi.requests()[0].body)["password"] == unifi.password


def test_a_wrong_password_is_refused_and_unifi_stays_unconnected(client: FlaskClient, unifi: _Console) -> None:
    assert _refusal(_set_up_unifi(client, unifi.url, password="not the password")) == "credentials_rejected"
    assert not _connected(client, "ubiquiti")
    assert [(r.method, r.path) for r in unifi.requests()] == [("POST", "/api/auth/login")]
    assert _saved_settings(client, "ubiquiti") == {}


def test_every_request_after_sign_in_carries_the_csrf_token(client: FlaskClient, unifi: _Console) -> None:
    assert _set_up_unifi(client, unifi.url).status_code == 204
    after_sign_in = [r for r in unifi.requests() if r.path != "/api/auth/login"]
    assert after_sign_in
    assert [r.headers.get("X-CSRF-Token") for r in after_sign_in] == [_CSRF_TOKEN] * len(after_sign_in)


def test_saving_a_changed_password_signs_in_again_with_it(client: FlaskClient, unifi: _Console) -> None:
    assert _set_up_unifi(client, unifi.url).status_code == 204
    assert _refusal(_set_up_unifi(client, unifi.url, password="not the password")) == "credentials_rejected"
    sign_ins = [json.loads(r.body)["password"] for r in unifi.requests() if r.path == "/api/auth/login"]
    assert sign_ins == [_PASSWORD, "not the password"]


def test_with_the_certificate_check_on_an_untrusted_console_never_gets_the_password(
    client: FlaskClient, unifi: _Console,
) -> None:
    assert _refusal(_set_up_unifi(client, unifi.url, verify_ssl=True)) == "certificate_untrusted"
    assert not _connected(client, "ubiquiti")
    assert unifi.requests() == []


def test_a_console_that_does_not_answer_is_unreachable(client: FlaskClient) -> None:
    with socket.socket() as spare:
        spare.bind(("127.0.0.1", 0))
        closed_port = spare.getsockname()[1]
    assert _refusal(_set_up_unifi(client, f"https://127.0.0.1:{closed_port}")) == "unreachable"
    assert not _connected(client, "ubiquiti")


def test_a_console_that_never_answers_is_unreachable(client: FlaskClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(unifi_rest_handler, "_TIMEOUT", 0.5)
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen()
        assert _refusal(_set_up_unifi(client, f"http://127.0.0.1:{silent.getsockname()[1]}")) == "unreachable"
    assert not _connected(client, "ubiquiti")


def test_an_https_address_for_a_plain_http_console_is_not_a_certificate_problem(client: FlaskClient) -> None:
    with _serve(_UnifiHandler, None) as plain:
        https_url = plain.url.replace("http://", "https://")
        assert _refusal(_set_up_unifi(client, https_url)) == "unreachable"
    assert not _connected(client, "ubiquiti")


def test_a_server_that_is_not_a_console_is_not_a_credentials_problem(
    client: FlaskClient, home_assistant: _Console,
) -> None:
    assert _refusal(_set_up_unifi(client, home_assistant.url)) == "invalid_configuration"
    assert not _connected(client, "ubiquiti")


# ── The certificate check reads back as it was saved ────────────────────────


@pytest.mark.usefixtures("trusted")
@pytest.mark.parametrize("checked", [False, True])
def test_unifi_certificate_check_reads_back_as_saved(client: FlaskClient, unifi: _Console, checked: bool) -> None:
    assert _set_up_unifi(client, unifi.url, verify_ssl=checked).status_code == 204
    assert _saved_settings(client, "ubiquiti")["verify_ssl"] is checked


@pytest.mark.usefixtures("trusted")
@pytest.mark.parametrize("checked", [False, True])
def test_home_certificate_check_reads_back_as_saved(
    client: FlaskClient, home_assistant: _Console, checked: bool,
) -> None:
    assert _set_up_home(client, home_assistant.url, verify_ssl=checked).status_code == 204
    assert _saved_settings(client, "home")["verify_ssl"] is checked


# ── Home Assistant sign-in ───────────────────────────────────────────────────


@pytest.mark.usefixtures("trusted")
def test_a_wrong_home_assistant_token_is_refused_as_rejected(client: FlaskClient, home_assistant: _Console) -> None:
    resp = _set_up_home(client, home_assistant.url, token="not the token", verify_ssl=True)
    assert _refusal(resp) == "credentials_rejected"
    assert not _connected(client, "home")


# ── UI.com account sign-in ───────────────────────────────────────────────────


@pytest.mark.parametrize(("at", "expected"), [
    (59, "287082"), (1111111109, "081804"), (1111111111, "050471"),
    (1234567890, "005924"), (2000000000, "279037"), (20000000000, "353130"),
])
def test_authenticator_codes_are_the_rfc_6238_ones(at: int, expected: str) -> None:
    assert _code(_SECRET, at // 30) == expected  # the stand-in console's own codes are right too
    assert authenticator.code(authenticator.setup_key(_SETUP_KEY), at) == expected


def test_a_ui_account_signs_in_with_the_code_its_setup_key_makes(client: FlaskClient, ui_account: _Console) -> None:
    assert _set_up_ui_account(client, ui_account.url, _SETUP_KEY).status_code == 204
    assert _connected(client, "ubiquiti")

    first, with_code, probe = ui_account.requests()[:3]
    assert json.loads(first.body) == {"username": _USERNAME, "password": _PASSWORD, "rememberMe": True}
    sent = json.loads(with_code.body)
    assert sent.pop("token") in _codes_now(_SECRET)
    assert sent == {"username": _USERNAME, "password": _PASSWORD, "rememberMe": True}
    assert with_code.headers.get("Cookie") == _HALF_SIGNED_IN
    assert (probe.method, probe.path) == ("GET", "/api/")
    assert probe.headers.get("X-CSRF-Token") == _CSRF_TOKEN
    assert "totp_key" not in _saved_settings(client, "ubiquiti")


@pytest.mark.parametrize(("secret", "typed"), [
    (_SECRET, "gezd gnbv gy3t qojq gezd gnbv gy3t qojq"),
    (_SECRET, f"  {_SETUP_KEY}\n"),
    (_SECRET, f"otpauth://totp/UI:{_USERNAME}?secret={_SETUP_KEY}&issuer=UI"),
    (_SECRET, f"otpauth://totp/UI?secret={_SETUP_KEY.lower()}&algorithm=SHA1&digits=6&period=30"),
    (_OTHER_SECRET, _OTHER_SETUP_KEY),
    (_OTHER_SECRET, _OTHER_SETUP_KEY.rstrip("=")),
])
def test_a_setup_key_is_read_however_it_was_copied(
    client: FlaskClient, ui_account: _Console, secret: bytes, typed: str,
) -> None:
    ui_account.secret = secret
    assert _set_up_ui_account(client, ui_account.url, typed).status_code == 204
    assert _connected(client, "ubiquiti")


@pytest.mark.parametrize("typed", [
    "not a key!",
    "GEZDGNBVGY3TQOJ1",
    "========",
    f"otpauth://hotp/UI?secret={_SETUP_KEY}&counter=0",
    f"otpauth://totp/UI?secret={_SETUP_KEY}&digits=8",
    f"otpauth://totp/UI?secret={_SETUP_KEY}&algorithm=SHA256",
    f"otpauth://totp/UI?secret={_SETUP_KEY}&period=60",
    "otpauth://totp/UI?issuer=UI",
])
def test_a_setup_key_that_is_not_one_is_refused_before_the_console_is_asked(
    client: FlaskClient, ui_account: _Console, typed: str,
) -> None:
    assert _refusal(_set_up_ui_account(client, ui_account.url, typed)) == "authenticator_key_invalid"
    assert ui_account.requests() == []
    assert not _connected(client, "ubiquiti")


@pytest.mark.parametrize("auth_method", ["ui_account", "credentials"])
def test_an_account_that_asks_for_a_code_with_no_setup_key_says_so(
    client: FlaskClient, ui_account: _Console, auth_method: str,
) -> None:
    assert _refusal(_set_up_ui_account(client, ui_account.url, "", auth_method=auth_method)) == "authenticator_key_missing"
    assert not _connected(client, "ubiquiti")
    assert ["token" in json.loads(r.body) for r in _sign_ins(ui_account)] == [False]
    assert _saved_settings(client, "ubiquiti") == {}


def test_a_code_the_account_does_not_accept_says_so_and_is_not_sent_twice(
    client: FlaskClient, ui_account: _Console,
) -> None:
    ui_account.secret = _OTHER_SECRET
    assert _refusal(_set_up_ui_account(client, ui_account.url, _SETUP_KEY)) == "authenticator_code_refused"
    assert not _connected(client, "ubiquiti")
    assert ["token" in json.loads(r.body) for r in _sign_ins(ui_account)] == [False, True]


def test_an_account_blocked_after_too_many_attempts_says_so(client: FlaskClient, ui_account: _Console) -> None:
    ui_account.blocked = True
    assert _refusal(_set_up_ui_account(client, ui_account.url, _SETUP_KEY)) == "too_many_attempts"
    assert not _connected(client, "ubiquiti")
    assert len(ui_account.requests()) == 1


def test_saving_again_with_the_setup_key_left_empty_keeps_the_stored_one(
    client: FlaskClient, ui_account: _Console,
) -> None:
    assert _set_up_ui_account(client, ui_account.url, _SETUP_KEY).status_code == 204
    unifi_rest_handler._session_cache.clear()  # as after a restart, so the save signs in afresh
    before = len(ui_account.requests())

    assert _set_up_ui_account(client, ui_account.url, "").status_code == 204
    assert _connected(client, "ubiquiti")
    resent = [json.loads(r.body) for r in ui_account.requests()[before:] if r.path == "/api/auth/login"]
    assert ["token" in r for r in resent] == [False, True]
    assert resent[-1]["token"] in _codes_now(_SECRET)
    assert "totp_key" not in _saved_settings(client, "ubiquiti")


def test_no_setup_key_code_password_or_half_made_sign_in_reaches_a_log_or_a_reply(
    client: FlaskClient, ui_account: _Console, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    typed = "gezd gnbv gy3t qojq gezd gnbv gy3t qojq"
    replies = [_set_up_ui_account(client, ui_account.url, typed)]
    unifi_rest_handler._session_cache.clear()
    ui_account.secret = _OTHER_SECRET
    replies.append(_set_up_ui_account(client, ui_account.url, _SETUP_KEY))
    replies.append(_set_up_ui_account(client, ui_account.url, "", auth_method="credentials"))
    replies.append(_set_up_ui_account(client, ui_account.url, f"otpauth://totp/UI?secret={_SETUP_KEY}&digits=8"))
    assert [r.status_code for r in replies] == [204, 422, 422, 422]

    shown = caplog.text + "".join(r.get_data(as_text=True) for r in replies)
    assert "AuthenticatorCodeRefused raised" in shown  # the log was read, and the refusals were in it
    for secret in (_SETUP_KEY, _SETUP_KEY.lower(), typed, "gezd gnbv", _PASSWORD, _HALF_SIGNED_IN.split("=")[1]):
        assert secret not in shown
    codes = [json.loads(r.body)["token"] for r in _sign_ins(ui_account) if "token" in json.loads(r.body)]
    assert len(codes) == 2
    for code in codes:
        assert not re.search(rf"(?<!\d){code}(?!\d)", shown)
