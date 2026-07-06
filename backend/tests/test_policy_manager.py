import sqlite3
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

import services.database as _db_gateway
from configs.enums.policy_channel import PolicyChannel
from services.file_mapper_service import FileMapperService
from services.policy_manager import PolicyManager

pytestmark = pytest.mark.unit

CH = PolicyChannel


@pytest.fixture()
def db() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(":memory:")
    conn.isolation_level = None  # autocommit — matches the Database gateway's connections
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE policy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel TEXT NOT NULL, permission TEXT NOT NULL,
            setting TEXT NOT NULL CHECK (setting IN ('internal','allow','ask','deny')),
            UNIQUE (channel, permission)
        );
        CREATE TABLE policy_blocked_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, action_id TEXT NOT NULL,
            context TEXT NOT NULL, reason TEXT NOT NULL, params_json TEXT,
            created_at TEXT NOT NULL
        );
    """)
    # PolicyManager reaches the DB through the Database gateway
    # (Database.conn()/transaction() → FileMapperService.get_db_path()). An in-memory
    # db is per-connection, so point the gateway at THIS exact handle — the only way
    # the manager's writes and the test's seed/asserts share one database.
    sentinel = Path(":memory:policy-manager-test")
    with patch.object(FileMapperService, "get_db_path", return_value=sentinel):
        _db_gateway._local.conns = {str(sentinel): conn}
        _db_gateway._local.depths = {}
        try:
            yield conn
        finally:
            _db_gateway._local.conns = {}
            _db_gateway._local.depths = {}
    conn.close()


@pytest.fixture()
def mgr(db: sqlite3.Connection) -> PolicyManager:
    return PolicyManager()


def _ran() -> str:
    # Mirrors the real callback contract: Ability.execute returns a STRING, so
    # authorize() returns that string verbatim on allow.
    return "RAN"


def _seed(db: sqlite3.Connection, channel: str, permission: str, setting: str) -> None:
    db.execute("INSERT INTO policy (channel, permission, setting) VALUES (?,?,?)",
               (channel, permission, setting))
    db.commit()


# 1. allow + internal both run the callback (internal == allow at the gate)
@pytest.mark.parametrize("setting", ["allow", "internal"])
def test_allow_and_internal_run_callback(mgr: PolicyManager, db: sqlite3.Connection, setting: str) -> None:
    _seed(db, "chat", "email.search", setting)
    assert mgr.authorize(CH.CHAT, "email.search", _ran) == _ran()


# 1b. INTERNAL tools ALWAYS bypass — every channel, no row, even over a deny row
@pytest.mark.parametrize("channel", [CH.CHAT, CH.SUBCONSCIOUS, CH.EXTERNAL_AGENT])
@pytest.mark.parametrize("permission", ["read", "search", "browser.open", "memory.store", "save_graph"])
def test_internal_tools_always_bypass(mgr: PolicyManager, db: sqlite3.Connection, channel: PolicyChannel, permission: str) -> None:
    # a deny row for the same key must be ignored — INTERNAL wins, no DB lookup
    _seed(db, channel.value, permission, "deny")
    assert mgr.authorize(channel, permission, _ran) == _ran()
    assert db.execute("SELECT count(*) FROM policy_blocked_log").fetchone()[0] == 0


# 2. deny blocks with the shared message and logs reason 'deny'
def test_deny_blocks_and_logs(mgr: PolicyManager, db: sqlite3.Connection) -> None:
    _seed(db, "chat", "bash.execute", "deny")
    out = mgr.authorize(CH.CHAT, "bash.execute", _ran)
    assert out == "The bash.execute action is not allowed. Do NOT retry."   # block STRING, not a dict
    assert db.execute("SELECT reason FROM policy_blocked_log").fetchone()["reason"] == "deny"


# 3. unknown key on a no-human channel: lazily creates ONE 'ask' row AND escalates to deny (D2)
def test_unknown_key_provisions_ask_then_escalates(mgr: PolicyManager, db: sqlite3.Connection) -> None:
    out = mgr.authorize(CH.SUBCONSCIOUS, "newtool.action", _ran)
    out2 = mgr.authorize(CH.SUBCONSCIOUS, "newtool.action", _ran)  # no duplicate row
    assert out == out2 == "The newtool.action action is not allowed. Do NOT retry."   # both blocked (strings)
    rows = db.execute("SELECT setting FROM policy WHERE permission='newtool.action'").fetchall()
    assert [r["setting"] for r in rows] == ["ask"]                       # provisioned exactly once
    assert db.execute("SELECT reason FROM policy_blocked_log ORDER BY id").fetchone()["reason"] == "user_unavailable"


# 4. interactive CHAT 'ask': approved runs, denied blocks (gate monkeypatched, never broadcasts off-channel)
@pytest.mark.parametrize("approved,should_run", [(True, True), (False, False)])
def test_chat_ask_follows_user_verdict(mgr: PolicyManager, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch, approved: bool, should_run: bool) -> None:
    monkeypatch.setattr(PolicyManager, "_ask_user", lambda self, permission, channel, should_stop=None: approved)
    _seed(db, "chat", "email.send", "ask")
    out = mgr.authorize(CH.CHAT, "email.send", _ran)
    if should_run:
        assert out == _ran()
    else:
        assert out == "The email.send action is not allowed. Do NOT retry."   # block STRING
        assert db.execute("SELECT reason FROM policy_blocked_log").fetchone()["reason"] == "user_denied"


# 5. data layer: seed → get_all hides internal; upsert flips + rejects bad input; reset restores; blocked-log roundtrip
def test_data_layer_roundtrip(mgr: PolicyManager) -> None:
    assert mgr.apply_seed() > 0
    assert all(r["setting"] != "internal" for r in mgr.get_all())               # internal hidden in Brain
    assert {"channel": "chat", "permission": "email.search", "setting": "allow"} in mgr.get_all()
    assert mgr.upsert("chat", "email.manage", "deny") == 1
    assert any(r["permission"] == "email.manage" and r["setting"] == "deny" for r in mgr.get_all())
    assert mgr.upsert("nope", "x", "allow") == 0 and mgr.upsert("chat", "x", "bogus") == 0   # invalid rejected
    mgr.reset_to_defaults()
    assert any(r["permission"] == "email.manage" and r["setting"] == "ask" for r in mgr.get_all())
    mgr._log_blocked("subconscious", "email.manage", "user_unavailable")
    assert mgr.get_blocked_log()[0]["action_id"] == "email.manage"
    assert mgr.clear_blocked_log() == 1 and mgr.get_blocked_log() == []
