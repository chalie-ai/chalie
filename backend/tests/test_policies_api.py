"""api/policies.py flat surface (D4): GET rows excl internal, single-upsert PUT."""
import contextlib
import sqlite3

import pytest

import api.policies as mod

pytestmark = pytest.mark.unit


@pytest.fixture()
def client(monkeypatch):
    from flask import Flask
    conn = sqlite3.connect(":memory:")
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
    conn.commit()

    class _FakeDB:
        def connection(self):
            @contextlib.contextmanager
            def _ctx(): yield conn
            return _ctx()

    monkeypatch.setattr("services.database_service.get_shared_db_service", lambda: _FakeDB())
    # require_session decorates at def-time, so patching it on the module is too late;
    # require_auth.decorated calls validate_session(request) at request-time via a
    # function-local import, so patch THAT to bypass auth (same approach as conftest).
    monkeypatch.setattr("services.auth_session_service.validate_session", lambda *a, **k: True)

    app = Flask(__name__)
    app.register_blueprint(mod.policies_bp)
    return app.test_client()


def test_get_returns_flat_rows_excluding_internal(client):
    r = client.get("/api/policies")
    assert r.status_code == 200
    rows = r.get_json()["policies"]
    row = next(x for x in rows if x["permission"] == "email.search")
    # Native rows keep the flat triple and gain an action-only display label:
    # `email.search` -> `Search` (the `tool.` prefix is dropped + humanized).
    assert row["channel"] == "chat" and row["setting"] == "allow"
    assert row["label"] == "Search"
    assert "group" not in row  # group is MCP-only — drives the cyan pill in the UI
    assert all(x["setting"] != "internal" for x in rows)


def test_put_single_upsert(client):
    r = client.put("/api/policies", json={"channel": "chat", "permission": "email.send", "setting": "deny"})
    assert r.status_code == 200 and r.get_json()["updated"] == 1


@pytest.fixture()
def mcp_client(monkeypatch):
    """Same flat policy surface, but with an mcp_client_servers table present so
    MCP rows get tagged through the real McpClientService.label_mcp_permissions."""
    from flask import Flask
    conn = sqlite3.connect(":memory:")
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
    conn.commit()

    class _FakeDB:
        def connection(self):
            @contextlib.contextmanager
            def _ctx(): yield conn
            return _ctx()

    monkeypatch.setattr("services.database_service.get_shared_db_service", lambda: _FakeDB())
    # McpClientService binds get_shared_db_service at import time, so patch its
    # module reference too — otherwise it reaches the real on-disk database.
    monkeypatch.setattr("services.mcp_client_service.get_shared_db_service", lambda: _FakeDB())
    monkeypatch.setattr("services.auth_session_service.validate_session", lambda *a, **k: True)

    app = Flask(__name__)
    app.register_blueprint(mod.policies_bp)
    return app.test_client()


def test_get_groups_and_humanizes_mcp_rows(mcp_client):
    """An _mcp_<server>_<tool> policy row comes back grouped by the server's
    configured title and labeled with the humanized tool name — exercised end
    to end through the real endpoint, McpClientService, and PolicyManager."""
    from services.database_service import get_shared_db_service
    from services.mcp_client_service import McpClientService
    from services.policy_manager import PolicyManager

    db = get_shared_db_service()
    # Production factories: register the server and provision the policy row the
    # same way the add-server endpoint and the policy gate do.
    McpClientService().add_server(
        name="GitHub", host="https://gh.example.com/mcp", headers={}, enabled=True)
    PolicyManager(db).upsert("chat", "_mcp_github_add_comment_to_pending_review", "ask")

    resp = mcp_client.get("/api/policies")
    assert resp.status_code == 200
    rows = resp.get_json()["policies"]
    row = next(r for r in rows if r["permission"] == "_mcp_github_add_comment_to_pending_review")
    assert row["group"] == "GitHub"
    assert row["label"] == "Add Comment to Pending Review"
    assert row["channel"] == "chat" and row["setting"] == "ask"
