"""Feature test: setting up mail against a real IMAP server on loopback.

The server speaks just enough IMAP4rev1 for a sign-in: it greets, lists its
capabilities, accepts LOGIN only for the one address and password it was given,
compared exactly as typed, and signs out. It records every LOGIN it receives.
The setup and status endpoints, the vault and the database are the real ones,
so the test drives the same path the Brain's capability form does.
"""

from __future__ import annotations

import socketserver
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import cast

import pytest
from flask.testing import FlaskClient

import capabilities
import services.vault_service as vault_module
from services.memory_store import MemoryStore
from services.vault_service import _vault_state, get_vault_service

pytestmark = pytest.mark.unit

_VAULT_PASSWORD = "test-password"

_ADDRESS = "chalie@mail.test"
_APP_PASSWORD = " app password with spaces at its ends "


@dataclass
class _Mailbox:
    """A running IMAP server and every LOGIN it has received."""

    port: int = 0
    logins: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


class _ImapHandler(socketserver.StreamRequestHandler):
    mailbox: _Mailbox

    def _say(self, line: str) -> None:
        self.wfile.write(f"{line}\r\n".encode())

    def handle(self) -> None:
        self._say("* OK IMAP4rev1 server ready")
        for raw in self.rfile:
            tag, _, rest = raw.decode().rstrip("\r\n").partition(" ")
            command, _, args = rest.partition(" ")
            if command == "CAPABILITY":
                self._say("* CAPABILITY IMAP4rev1")
                self._say(f"{tag} OK CAPABILITY completed")
            elif command == "LOGIN":
                with self.mailbox.lock:
                    self.mailbox.logins.append(args)
                if args == f'{_ADDRESS} "{_APP_PASSWORD}"':
                    self._say(f"{tag} OK LOGIN completed")
                else:
                    self._say(f"{tag} NO [AUTHENTICATIONFAILED] Invalid credentials")
            elif command == "LOGOUT":
                self._say("* BYE")
                self._say(f"{tag} OK LOGOUT completed")
                return
            else:
                self._say(f"{tag} BAD unknown command")


@pytest.fixture()
def mailbox() -> Iterator[_Mailbox]:
    box = _Mailbox()
    bound = cast("type[_ImapHandler]", type("Bound", (_ImapHandler,), {"mailbox": box}))
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), bound)
    server.daemon_threads = True
    box.port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield box
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


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


def test_an_app_password_with_spaces_at_its_ends_signs_in_as_typed(client: FlaskClient, mailbox: _Mailbox) -> None:
    resp = client.post("/api/capabilities/setup/mail", json={
        "email": _ADDRESS, "password": _APP_PASSWORD,
        "imap_host": "127.0.0.1", "imap_port": mailbox.port, "imap_tls": "0",
    })
    assert resp.status_code == 204

    status = client.get("/api/capabilities/status/mail")
    assert cast("dict[str, dict[str, object]]", status.get_json())["result"]["connected"] is True
    assert mailbox.logins[0] == f'{_ADDRESS} "{_APP_PASSWORD}"'
