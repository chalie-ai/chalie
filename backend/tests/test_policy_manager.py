"""Unit tests for PolicyManager — flat (channel, permission, setting) gate.

Five tests cover the whole surface: allow/internal run, deny+log, lazy-provision
+ D2 escalation, interactive CHAT verdict (approve/deny), and the data layer
(seed → get_all hides internal → upsert → reset → blocked-log roundtrip)."""
import contextlib
import sqlite3

import pytest

from services.policy_manager import PolicyManager
from services.processor_config import ProcessorConfig

pytestmark = pytest.mark.unit

CH = ProcessorConfig.POLICY_CHANNEL


@pytest.fixture()
def db():
    conn = sqlite3.connect(":memory:")
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
    yield conn
    conn.close()


class _FakeDB:
    def __init__(self, conn): self._conn = conn
    def connection(self):
        @contextlib.contextmanager
        def _ctx(): yield self._conn
        return _ctx()


@pytest.fixture()
def mgr(db):
    return PolicyManager(_FakeDB(db))


def _ran():
    # Mirrors the real callback contract: Ability.execute returns a STRING, so
    # authorize() returns that string verbatim on allow.
    return "RAN"


def _seed(db, channel, permission, setting):
    db.execute("INSERT INTO policy (channel, permission, setting) VALUES (?,?,?)",
               (channel, permission, setting))
    db.commit()


# 1. allow + internal both run the callback (internal == allow at the gate)
@pytest.mark.parametrize("setting", ["allow", "internal"])
def test_allow_and_internal_run_callback(mgr, db, setting):
    _seed(db, "chat", "email.search", setting)
    assert mgr.authorize(CH.CHAT, "email.search", _ran) == _ran()


# 2. deny blocks with the shared message and logs reason 'deny'
def test_deny_blocks_and_logs(mgr, db):
    _seed(db, "chat", "bash.execute", "deny")
    out = mgr.authorize(CH.CHAT, "bash.execute", _ran)
    assert out == "The bash.execute action is not allowed. Do NOT retry."   # block STRING, not a dict
    assert db.execute("SELECT reason FROM policy_blocked_log").fetchone()["reason"] == "deny"


# 3. unknown key on a no-human channel: lazily creates ONE 'ask' row AND escalates to deny (D2)
def test_unknown_key_provisions_ask_then_escalates(mgr, db):
    out = mgr.authorize(CH.SUBCONSCIOUS, "newtool.action", _ran)
    out2 = mgr.authorize(CH.SUBCONSCIOUS, "newtool.action", _ran)  # no duplicate row
    assert out == out2 == "The newtool.action action is not allowed. Do NOT retry."   # both blocked (strings)
    rows = db.execute("SELECT setting FROM policy WHERE permission='newtool.action'").fetchall()
    assert [r["setting"] for r in rows] == ["ask"]                       # provisioned exactly once
    assert db.execute("SELECT reason FROM policy_blocked_log ORDER BY id").fetchone()["reason"] == "user_unavailable"


# 4. interactive CHAT 'ask': approved runs, denied blocks (gate monkeypatched, never broadcasts off-channel)
@pytest.mark.parametrize("approved,should_run", [(True, True), (False, False)])
def test_chat_ask_follows_user_verdict(mgr, db, monkeypatch, approved, should_run):
    monkeypatch.setattr(PolicyManager, "_ask_user", lambda self, permission, channel: approved)
    _seed(db, "chat", "email.send", "ask")
    out = mgr.authorize(CH.CHAT, "email.send", _ran)
    if should_run:
        assert out == _ran()
    else:
        assert out == "The email.send action is not allowed. Do NOT retry."   # block STRING
        assert db.execute("SELECT reason FROM policy_blocked_log").fetchone()["reason"] == "user_denied"


# 5. data layer: seed → get_all hides internal; upsert flips + rejects bad input; reset restores; blocked-log roundtrip
def test_data_layer_roundtrip(mgr):
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
