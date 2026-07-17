import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from api.endpoints.policies import PoliciesEndpoint
from tests.restx_test_app import mount_namespace

if TYPE_CHECKING:
    from flask.testing import FlaskClient

pytestmark = pytest.mark.unit


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> "Generator[FlaskClient, None, None]":
    import services.database as _db_gateway
    from services.file_mapper_service import FileMapperService

    conn = sqlite3.connect(":memory:")
    conn.isolation_level = None  # autocommit — matches the Database gateway's connections
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE policy (id INTEGER PRIMARY KEY AUTOINCREMENT, channel TEXT, "
        "permission TEXT, setting TEXT CHECK(setting IN ('internal','allow','ask','deny')), "
        "UNIQUE(channel, permission));"
        "CREATE TABLE policy_blocked_log (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "action_id TEXT, context TEXT, reason TEXT, params_json TEXT, created_at TEXT);"
    )
    conn.execute("INSERT INTO policy (channel, permission, setting) VALUES "
                 "('chat','email.search','allow'),('chat','memory.recall','internal')")

    # PolicyManager + McpClientService (both consulted by the GET handler) reach the
    # DB through the Database gateway → FileMapperService.get_db_path(). An in-memory
    # db is per-connection, so point the gateway at THIS handle. No mcp_client_servers
    # table here → list_servers() raises and MCP tagging is skipped (native rows only).
    sentinel = Path(":memory:policies-api-test")
    monkeypatch.setattr(FileMapperService, "get_db_path", lambda *_: sentinel)
    # require_session decorates at def-time, so patching it on the module is too late;
    # require_auth.decorated calls validate_session(request) at request-time via a
    # function-local import, so patch THAT to bypass auth (same approach as conftest).
    monkeypatch.setattr("services.auth_session_service.validate_session", lambda *a, **k: True)

    _db_gateway._local.conns = {str(sentinel): conn}
    _db_gateway._local.depths = {}
    # Bind the Model connection getter — same boot step run.py runs once at
    # startup (``Database().bind()``). Repeated per test because each Database()
    # captures this test's patched path, so the getter must point at the current
    # test's in-memory handle.
    _db_gateway.Database().bind()
    try:
        app = mount_namespace(PoliciesEndpoint().namespace())
        yield app.test_client()
    finally:
        _db_gateway._local.conns = {}
        _db_gateway._local.depths = {}
        conn.close()


def test_get_returns_flat_rows_excluding_internal(client: "FlaskClient") -> None:
    r = client.get("/api/policies/all?limit=100")
    assert r.status_code == 200
    body = r.get_json()
    assert body["success"] is True and "pagination" in body
    rows = body["result"]
    row = next(x for x in rows if x["permission"] == "email.search")
    # Native rows keep the flat triple and gain an action-only display label:
    # `email.search` -> `Search` (the `tool.` prefix is dropped + humanized).
    assert row["channel"] == "chat" and row["setting"] == "allow"
    assert row["label"] == "Search"
    # group is MCP-only — native rows carry it as null (the DTO field is nullable).
    assert row["group"] is None
    assert all(x["setting"] != "internal" for x in rows)


def test_put_single_upsert(client: "FlaskClient") -> None:
    r = client.post("/api/policies/-1", json={"channel": "chat", "permission": "email.send", "setting": "deny"})
    # Single-cell upsert is a non-CRUD action → 204 no-body (rowcount no longer surfaced).
    assert r.status_code == 204
    assert r.data == b""


@pytest.fixture()
def mcp_client(monkeypatch: pytest.MonkeyPatch) -> "Generator[FlaskClient, None, None]":
    import services.database as _db_gateway
    from services.file_mapper_service import FileMapperService

    conn = sqlite3.connect(":memory:")
    conn.isolation_level = None  # autocommit — matches the Database gateway's connections
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE policy (id INTEGER PRIMARY KEY AUTOINCREMENT, channel TEXT, "
        "permission TEXT, setting TEXT CHECK(setting IN ('internal','allow','ask','deny')), "
        "UNIQUE(channel, permission));"
        "CREATE TABLE policy_blocked_log (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "action_id TEXT, context TEXT, reason TEXT, params_json TEXT, created_at TEXT);"
        "CREATE TABLE mcp_client_servers (id TEXT PRIMARY KEY, name TEXT NOT NULL, "
        "host TEXT NOT NULL, headers TEXT NOT NULL DEFAULT '{}', enabled INTEGER NOT NULL DEFAULT 1, "
        "status TEXT NOT NULL DEFAULT 'unknown', last_pinged_at TEXT, "
        "created_at TEXT NOT NULL, updated_at TEXT NOT NULL);"
    )

    # PolicyManager (policy rows) and McpClientService (mcp_client_servers) both reach
    # the DB through the Database gateway → get_db_path(); the GET reads both tables
    # from this one in-memory handle.
    sentinel = Path(":memory:policies-api-mcp-test")
    monkeypatch.setattr(FileMapperService, "get_db_path", lambda *_: sentinel)
    monkeypatch.setattr("services.auth_session_service.validate_session", lambda *a, **k: True)

    _db_gateway._local.conns = {str(sentinel): conn}
    _db_gateway._local.depths = {}
    # Bind the Model connection getter — same boot step run.py runs once at
    # startup (``Database().bind()``). Repeated per test because each Database()
    # captures this test's patched path, so the getter must point at the current
    # test's in-memory handle.
    _db_gateway.Database().bind()
    try:
        app = mount_namespace(PoliciesEndpoint().namespace())
        yield app.test_client()
    finally:
        _db_gateway._local.conns = {}
        _db_gateway._local.depths = {}
        conn.close()


def test_get_groups_and_humanizes_mcp_rows(mcp_client: "FlaskClient") -> None:
    from services.mcp_client_service import McpClientService
    from services.policy_manager import PolicyManager

    # Production factories: register the server and provision the policy row the
    # same way the add-server endpoint and the policy gate do.
    McpClientService().add_server(
        name="GitHub", host="https://gh.example.com/mcp", headers={}, enabled=True)
    PolicyManager().upsert("chat", "_mcp_github_add_comment_to_pending_review", "ask")

    resp = mcp_client.get("/api/policies/all?limit=100")
    assert resp.status_code == 200
    rows = resp.get_json()["result"]
    row = next(r for r in rows if r["permission"] == "_mcp_github_add_comment_to_pending_review")
    assert row["group"] == "GitHub"
    assert row["label"] == "Add Comment to Pending Review"
    assert row["channel"] == "chat" and row["setting"] == "ask"
