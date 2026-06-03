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
    assert {"channel": "chat", "permission": "email.search", "setting": "allow"} in rows
    assert all(row["setting"] != "internal" for row in rows)


def test_put_single_upsert(client):
    r = client.put("/api/policies", json={"channel": "chat", "permission": "email.send", "setting": "deny"})
    assert r.status_code == 200 and r.get_json()["updated"] == 1
