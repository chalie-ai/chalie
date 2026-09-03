# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests — the inbound MCP server follows its settings live and needs no token.

What an external agent and the dashboard can observe, and what these tests pin:

- With the server disabled nothing listens on its port; enabled, a real MCP
  ``initialize`` handshake succeeds over streamable HTTP with no
  ``Authorization`` header at all (and a stale token from an old client is
  simply ignored, not rejected).
- Turning the server off or moving its port through the settings endpoint
  takes effect before the request is answered — no backend restart — and the
  settings record reports the live listener state next to the stored intent.
- The routine reconcile tick must not restart a healthy server: an agent's
  session survives it.
- A port somebody else holds is reported on the record (``error``) instead of
  crashing anything, and the listener recovers as soon as the port frees up.
- Upgrading deletes the stored token rows and revokes the REST credential they
  minted, once, without touching anything else.

Real listener (real uvicorn thread on an ephemeral port), real settings
endpoint on a throwaway app behind a real cookie session, real per-test SQLite,
real migration module. Zero mocks.
"""

import http.client
import json
import socket
import sqlite3
import threading
from collections.abc import Iterator

import pytest
from flask import Response
from flask.testing import FlaskClient
from mcp.types import LATEST_PROTOCOL_VERSION

from api.endpoints.mcp_settings import McpSettingsEndpoint
from contracts.constants.auth import SESSION_COOKIE_NAME
from mcp_server.server import McpListener, listener
from migrations import migration_021_drop_mcp_server_token as mig
from migrations.runner import run_all
from models.setting import Setting
from services.auth_session_service import create_session
from services.file_mapper_service import FileMapperService
from tests.restx_test_app import mount_namespace

pytestmark = pytest.mark.unit

_ENABLED = "mcp_server_enabled"
_PORT = "mcp_server_port"
_SETTINGS_URL = "/api/mcp-server/-1"
_RECORD_URL = "/api/mcp-server/all"
_MIGRATION_KEY = "mcp-server-token-removal-v1"
_MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


# ── helpers ───────────────────────────────────────────────────────────────


def _free_ports(count: int) -> list[int]:
    """``count`` distinct ports the OS considers free right now.

    Every socket stays bound until all are picked, so no two picks collide.
    """
    sockets = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(count)]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        ports = [int(sock.getsockname()[1]) for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()
    return ports


def _port_refused(port: int) -> bool:
    """True when nothing accepts a TCP connection on the loopback port."""
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1.0).close()
    except OSError:
        return True
    return False


def _rpc(
    port: int,
    payload: dict[str, object],
    *,
    session_id: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, str | None, dict[str, object] | None]:
    """POST one JSON-RPC message to the real ``/mcp`` transport.

    Returns ``(status, session id header, result)``. Streamable HTTP answers a
    request on an SSE stream, so the JSON-RPC reply rides a ``data:`` line.
    """
    headers = dict(_MCP_HEADERS)
    if session_id is not None:
        headers["mcp-session-id"] = session_id
    if extra_headers:
        headers.update(extra_headers)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("POST", "/mcp", body=json.dumps(payload), headers=headers)
        response = conn.getresponse()
        status = response.status
        returned_session = response.getheader("mcp-session-id")
        raw = response.read().decode()
    finally:
        conn.close()
    for line in raw.splitlines():
        if line.startswith("data:"):
            message: dict[str, object] = json.loads(line[len("data:"):])
            result = message.get("result")
            return status, returned_session, result if isinstance(result, dict) else None
    return status, returned_session, None


def _initialize(
    port: int, extra_headers: dict[str, str] | None = None
) -> tuple[int, str | None, str | None]:
    """The real MCP ``initialize`` handshake: ``(status, session id, server name)``."""
    status, session_id, result = _rpc(
        port,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "feature-test", "version": "0"},
            },
        },
        extra_headers=extra_headers,
    )
    server_info = result.get("serverInfo") if result is not None else None
    name = server_info.get("name") if isinstance(server_info, dict) else None
    return status, session_id, name if isinstance(name, str) else None


def _tool_names(port: int, session_id: str) -> tuple[int, list[str]]:
    """``tools/list`` on an established session: ``(status, tool names)``."""
    status, _, result = _rpc(
        port, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, session_id=session_id
    )
    tools = result.get("tools") if result is not None else None
    names = [
        tool["name"]
        for tool in (tools if isinstance(tools, list) else [])
        if isinstance(tool, dict) and isinstance(tool.get("name"), str)
    ]
    return status, names


def _record(api: FlaskClient) -> dict[str, object]:
    """The singleton settings record as the dashboard reads it."""
    response = api.get(_RECORD_URL)
    assert response.status_code == 200, response.get_json()
    record: dict[str, object] = response.get_json()["result"][0]
    return record


def _db_path() -> str:
    return str(FileMapperService.get_db_path())


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def quiet_listener(db: sqlite3.Connection) -> Iterator[McpListener]:
    """The process-wide listener, parked off before the test and off again after.

    Absent settings rows mean "enabled, on the default port", so the first
    reconcile must already see an explicit off — otherwise a test would bind
    the real default port on the host. Teardown parks it the same way and
    proves the serving thread is gone, so nothing outlives the test.
    """
    Setting.set(_ENABLED, "false")
    listener.reconcile()
    assert listener.listening is False
    try:
        yield listener
    finally:
        Setting.set(_ENABLED, "false")
        listener.reconcile()
        assert listener.listening is False
        leaked = [t.name for t in threading.enumerate() if t.name.startswith("mcp-listener")]
        assert leaked == [], f"listener thread outlived the test: {leaked}"


@pytest.fixture
def api(quiet_listener: McpListener) -> FlaskClient:
    """The real settings endpoint on a throwaway app, behind a real cookie session."""
    app = mount_namespace(McpSettingsEndpoint("mcp-server").namespace())
    token = create_session(Response())
    client = app.test_client()
    client.set_cookie(SESSION_COOKIE_NAME, token)
    return client


# ── listener follows its settings ─────────────────────────────────────────


def test_disabled_server_listens_on_nothing(quiet_listener: McpListener) -> None:
    """Off means off: no socket is bound and the record says so."""
    (port,) = _free_ports(1)
    Setting.set(_PORT, str(port))
    Setting.set(_ENABLED, "false")

    quiet_listener.reconcile()

    assert quiet_listener.listening is False
    assert quiet_listener.listening_port is None
    assert _port_refused(port)


def test_enabled_server_answers_initialize_without_any_token(
    quiet_listener: McpListener,
) -> None:
    """An external agent connects with no credential at all — the transport
    carries no authentication; a stale token from an old client is ignored,
    not rejected."""
    (port,) = _free_ports(1)
    Setting.set(_PORT, str(port))
    Setting.set(_ENABLED, "true")

    quiet_listener.reconcile()

    status, _, server_name = _initialize(port)
    assert (status, server_name) == (200, "chalie")
    status, _, server_name = _initialize(
        port, extra_headers={"Authorization": "Bearer token-from-before-the-upgrade"}
    )
    assert (status, server_name) == (200, "chalie")
    assert quiet_listener.listening is True
    assert quiet_listener.listening_port == port
    assert quiet_listener.error is None


def test_routine_reconcile_keeps_a_live_agent_session(
    quiet_listener: McpListener,
) -> None:
    """The worker re-reconciles on a fixed tick; with nothing changed it must
    leave the running server alone — an agent mid-conversation keeps its
    session instead of being cut off every few seconds."""
    (port,) = _free_ports(1)
    Setting.set(_PORT, str(port))
    Setting.set(_ENABLED, "true")
    quiet_listener.reconcile()
    status, session_id, _ = _initialize(port)
    assert status == 200 and session_id is not None
    accepted, _, _ = _rpc(
        port, {"jsonrpc": "2.0", "method": "notifications/initialized"}, session_id=session_id
    )
    assert accepted == 202

    quiet_listener.reconcile()

    status, names = _tool_names(port, session_id)
    assert status == 200
    assert "talk_to_chalie" in names


def test_turning_the_server_off_closes_the_port_before_answering(
    api: FlaskClient, quiet_listener: McpListener
) -> None:
    """The dashboard toggle: by the time the 204 arrives the port is already
    closed, and the record read straight after reports not listening."""
    (port,) = _free_ports(1)
    assert api.post(_SETTINGS_URL, json={"enabled": True, "port": port}).status_code == 204
    assert _initialize(port)[0] == 200

    assert api.post(_SETTINGS_URL, json={"enabled": False}).status_code == 204

    assert _port_refused(port)
    record = _record(api)
    assert record["enabled"] is False
    assert record["listening"] is False
    assert record["listening_port"] is None


def test_changing_the_port_moves_the_listener(
    api: FlaskClient, quiet_listener: McpListener
) -> None:
    """A port change through the endpoint: the old port goes dark, the new
    one serves the handshake, and the record names the new port."""
    old_port, new_port = _free_ports(2)
    assert api.post(_SETTINGS_URL, json={"enabled": True, "port": old_port}).status_code == 204
    assert _initialize(old_port)[0] == 200

    assert api.post(_SETTINGS_URL, json={"port": new_port}).status_code == 204

    assert _port_refused(old_port)
    status, _, server_name = _initialize(new_port)
    assert (status, server_name) == (200, "chalie")
    record = _record(api)
    assert record["port"] == new_port
    assert record["listening"] is True
    assert record["listening_port"] == new_port


def test_settings_record_reports_live_state_and_exposes_no_token(
    api: FlaskClient, quiet_listener: McpListener, db: sqlite3.Connection
) -> None:
    """The record the dashboard renders: stored intent plus live state, and no
    token — neither on the wire nor left behind in the settings table after
    an enable/disable/port cycle."""
    first_port, second_port = _free_ports(2)

    assert api.post(_SETTINGS_URL, json={"enabled": True, "port": first_port}).status_code == 204
    assert _record(api) == {
        "enabled": True,
        "port": first_port,
        "listening": True,
        "listening_port": first_port,
        "error": None,
    }

    assert api.post(_SETTINGS_URL, json={"port": second_port}).status_code == 204
    assert api.post(_SETTINGS_URL, json={"enabled": False}).status_code == 204
    assert _record(api) == {
        "enabled": False,
        "port": second_port,
        "listening": False,
        "listening_port": None,
        "error": None,
    }

    leftover = db.execute(
        "SELECT key FROM settings WHERE key LIKE '%token%' AND key LIKE 'mcp_server%'"
    ).fetchall()
    assert [row[0] for row in leftover] == []


def test_taken_port_is_reported_and_the_listener_recovers_once_it_frees(
    api: FlaskClient, quiet_listener: McpListener
) -> None:
    """Somebody else holds the port: the reconcile neither raises nor leaves a
    half-started server — the record carries the reason — and the next
    reconcile after the port frees up serves normally."""
    (port,) = _free_ports(1)
    occupant = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupant.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    occupant.bind(("0.0.0.0", port))
    occupant.listen(1)
    try:
        Setting.set(_PORT, str(port))
        Setting.set(_ENABLED, "true")

        quiet_listener.reconcile()

        assert quiet_listener.listening is False
        assert quiet_listener.listening_port is None
        assert isinstance(quiet_listener.error, str) and quiet_listener.error
        record = _record(api)
        assert record["listening"] is False
        assert record["error"] == quiet_listener.error
    finally:
        occupant.close()

    quiet_listener.reconcile()

    status, _, server_name = _initialize(port)
    assert (status, server_name) == (200, "chalie")
    assert quiet_listener.error is None
    assert _record(api)["error"] is None


def test_unbindable_port_row_is_reported_and_a_valid_port_recovers(
    api: FlaskClient, quiet_listener: McpListener
) -> None:
    """Only a hand-edited settings row can hold a port no socket can bind at
    all. The re-check must record that on the settings record rather than
    raise — an escape here would take the worker down on every tick — and
    the next valid port serves as usual."""
    Setting.set(_PORT, "70000")
    Setting.set(_ENABLED, "true")

    quiet_listener.reconcile()

    assert quiet_listener.listening is False
    assert isinstance(quiet_listener.error, str) and quiet_listener.error
    assert _record(api)["error"] == quiet_listener.error

    (port,) = _free_ports(1)
    Setting.set(_PORT, str(port))
    quiet_listener.reconcile()

    assert _initialize(port)[0] == 200
    assert quiet_listener.error is None


# ── upgrade: the stored token and its credential go away ─────────────────


def _seed_token_residue(db: sqlite3.Connection) -> None:
    """The shape an upgrading install carries: the two token settings rows, the
    wrapper credential the server minted for itself, plus a bystander wrapper
    and an unrelated MCP setting that must both survive."""
    db.execute(
        "INSERT INTO settings (key, value) VALUES "
        "('mcp_server_token', 'raw-secret-in-clear-text'), "
        "('mcp_server_token_wrapper_id', '__mcp_server_abc__'), "
        "('mcp_server_port', '8462')"
    )
    db.execute(
        "INSERT INTO wrapper_tokens (id, name, token_hash, wrapper_id, created_at) VALUES "
        "('w-mcp', 'MCP server', 'hash-mcp', '__mcp_server_abc__', '2026-01-01T00:00:00+00:00'), "
        "('w-other', 'Some wrapper', 'hash-other', 'some-other-wrapper', '2026-01-01T00:00:00+00:00')"
    )
    db.commit()


def _revocations(db: sqlite3.Connection) -> dict[str, str | None]:
    rows = db.execute("SELECT wrapper_id, revoked_at FROM wrapper_tokens").fetchall()
    return {str(row[0]): row[1] for row in rows}


def test_upgrade_removes_the_stored_token_and_revokes_its_credential_once(
    db: sqlite3.Connection,
) -> None:
    """Boot-time migration on an upgrading install: the token rows are gone,
    the wrapper the server minted is revoked, the bystander wrapper and the
    unrelated setting are untouched — and a second pass changes nothing."""
    _seed_token_residue(db)
    assert mig.needed(db) is True

    run_all(_db_path())

    ledger = db.execute(
        "SELECT outcome FROM schema_migrations WHERE key = ?", (_MIGRATION_KEY,)
    ).fetchone()
    assert ledger is not None and ledger[0] == "applied"
    remaining = [
        row[0] for row in db.execute(
            "SELECT key FROM settings WHERE key LIKE 'mcp_server%' ORDER BY key"
        )
    ]
    assert remaining == ["mcp_server_port"]
    revocations = _revocations(db)
    assert revocations["__mcp_server_abc__"] is not None
    assert revocations["some-other-wrapper"] is None

    assert mig.needed(db) is False
    mig.apply(_db_path())
    assert _revocations(db) == revocations
    assert mig.needed(db) is False


def test_upgrade_is_a_no_op_on_a_fresh_install(db: sqlite3.Connection) -> None:
    """Nothing to remove: the step records itself as not needed and executes
    nothing — no spurious revocations on a database that never had a token."""
    db.execute(
        "INSERT INTO wrapper_tokens (id, name, token_hash, wrapper_id, created_at) VALUES "
        "('w-other', 'Some wrapper', 'hash-other', 'some-other-wrapper', '2026-01-01T00:00:00+00:00')"
    )
    db.commit()
    assert mig.needed(db) is False

    run_all(_db_path())

    ledger = db.execute(
        "SELECT outcome FROM schema_migrations WHERE key = ?", (_MIGRATION_KEY,)
    ).fetchone()
    assert ledger is not None and ledger[0] == "noop"
    assert _revocations(db) == {"some-other-wrapper": None}
