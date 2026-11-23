import json
import queue
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

import services.database as _db_gateway
from configs.enums.policy_channel import PolicyChannel
from services.ask_user import AskUser
from services.file_mapper_service import FileMapperService
from services.policy_manager import PolicyManager
from services.websocket import Websocket

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
        # Bind the Model connection getter — same boot step run.py runs once at
        # startup (``Database().bind()``). Repeated per test because each Database()
        # captures this test's patched path, so the getter must point at the current
        # test's in-memory handle.
        _db_gateway.Database().bind()
        try:
            yield conn
        finally:
            _db_gateway._local.conns = {}
            _db_gateway._local.depths = {}
    conn.close()


@pytest.fixture()
def mgr(db: sqlite3.Connection) -> PolicyManager:
    return PolicyManager()


class _RecordingClient:
    """A real socket-registry subscriber: implements the registry's send(str)
    protocol and captures every broadcast frame, so a test can assert on exactly
    what the interface would have received."""

    def __init__(self) -> None:
        self.frames: list[dict[str, object]] = []
        self.opened: queue.Queue[str] = queue.Queue()

    def close(self) -> None:
        """Nothing to release: the captured frames stay readable after the registry lets go."""

    def send(self, data: str) -> None:
        frame = json.loads(data)
        self.frames.append(frame)
        if frame["type"] == "ask_opened":
            self.opened.put(frame["ask_id"])


@pytest.fixture()
def recorder() -> Iterator[_RecordingClient]:
    client = _RecordingClient()
    Websocket._connect(client)
    try:
        yield client
    finally:
        Websocket._disconnect(client)


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
    _seed(db, "chat", "email.read", setting)
    assert mgr.authorize(CH.CHAT, "email.read", _ran) == _ran()


# 1b. INTERNAL tools ALWAYS bypass — every channel, no row, even over a deny row.
# The tools that spawn a subagent never gate: what the subagent then calls is
# gated on its own permission, under the channel of the agent that spawned it.
@pytest.mark.parametrize("channel", [CH.CHAT, CH.SUBCONSCIOUS, CH.EXTERNAL_AGENT])
@pytest.mark.parametrize("permission", ["search", "browser.open", "recall", "save_graph",
                                        "subtask.start", "code_agent", "web_search", "web_browse", "pim"])
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


# 3. unknown key: lazily creates ONE 'ask' row and asks the user even on a turn with no origin
#    (a subconscious turn) — the ask simply carries no origin. A denial blocks with the shared
#    message and audits reason user_denied, once per call.
def test_unknown_key_provisions_one_ask_row_and_asks_even_with_no_origin(
    mgr: PolicyManager, db: sqlite3.Connection, recorder: _RecordingClient,
) -> None:
    gave_up = threading.Event()

    def deny_both_asks() -> None:
        for _ in range(2):
            try:
                AskUser.reply(recorder.opened.get(timeout=5.0), False)
            except queue.Empty:
                gave_up.set()                         # no ask opened: the gate's should_stop ends the wait
                return

    answerer = threading.Thread(target=deny_both_asks)
    answerer.start()
    out = mgr.authorize(CH.SUBCONSCIOUS, "newtool.action", _ran, should_stop=gave_up.is_set)
    out2 = mgr.authorize(CH.SUBCONSCIOUS, "newtool.action", _ran, should_stop=gave_up.is_set)  # no duplicate row
    answerer.join()

    assert not gave_up.is_set(), "the gate never asked the user"
    assert out == out2 == "The newtool.action action is not allowed. Do NOT retry."
    rows = db.execute("SELECT setting FROM policy WHERE permission='newtool.action'").fetchall()
    assert [r["setting"] for r in rows] == ["ask"]                       # provisioned exactly once
    reasons = [r["reason"] for r in db.execute("SELECT reason FROM policy_blocked_log ORDER BY id")]
    assert reasons == ["user_denied", "user_denied"]
    asked = [(f["kind"], f["data"]) for f in recorder.frames if f["type"] == "ask_opened"]
    no_origin = {"action_id": "newtool.action", "summary": "", "origin": None}
    assert asked == [("permission", no_origin), ("permission", no_origin)]
    assert AskUser.open_asks() == []


# 5. data layer: seed → get_all hides internal; upsert flips + rejects bad input; reset restores; blocked-log roundtrip
def test_data_layer_roundtrip(mgr: PolicyManager) -> None:
    assert mgr.apply_seed() > 0
    assert all(r["setting"] != "internal" for r in mgr.get_all())               # internal hidden in Brain
    assert mgr.upsert("chat", "bash.modify_file", "deny") == 1
    assert any(r["permission"] == "bash.modify_file" and r["setting"] == "deny" for r in mgr.get_all())
    assert mgr.upsert("nope", "x", "allow") == 0 and mgr.upsert("chat", "x", "bogus") == 0   # invalid rejected
    assert mgr.upsert("chat", "subtask.start", "deny") == 0                     # INTERNAL tool: never user-gated
    assert not any(r["permission"] == "subtask.start" for r in mgr.get_all())
    mgr.reset_to_defaults()
    assert any(r["permission"] == "bash.modify_file" and r["setting"] == "ask" for r in mgr.get_all())
    mgr._log_blocked("subconscious", "bash.modify_file", "user_denied")
    assert mgr.get_blocked_log()[0]["action_id"] == "bash.modify_file"
    assert mgr.clear_blocked_log() == 1 and mgr.get_blocked_log() == []


# 6. apply_seed reaps every row the seed file no longer lists (older seeds carried
#    a row for each tool that spawns a subagent; every boot must converge them out)
#    while a user's setting on a still-seeded permission survives untouched.
def test_apply_seed_reaps_unseeded_rows(mgr: PolicyManager, db: sqlite3.Connection) -> None:
    _seed(db, "chat", "pim", "allow")                      # stale rows from an older seed
    _seed(db, "chat", "subtask.start", "deny")
    _seed(db, "subconscious", "search", "deny")            # bare tool name, not tool.action
    _seed(db, "chat", "bash.read", "deny")                 # user's own setting — must survive
    _seed(db, "external_agent", "run_script", "allow")     # existing install — keeps allow over the deny seed
    mgr.apply_seed()
    perms = {r["permission"] for r in mgr.get_all()}
    assert not perms & {"pim", "subtask.start", "search"}
    # bash.read and run_script ARE in the seed, so the reap keys on membership and
    # leaves each existing setting in place (INSERT OR IGNORE never overwrites a row).
    settings = {(r["channel"], r["permission"]): r["setting"] for r in mgr.get_all()}
    assert settings[("chat", "bash.read")] == "deny"
    assert settings[("external_agent", "run_script")] == "allow"
