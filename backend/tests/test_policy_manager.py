import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

import services.database as _db_gateway
from configs.enums.policy_channel import PolicyChannel
from services.file_mapper_service import FileMapperService
from services.policy_manager import _ASK_NO_SURFACE, _GATE_POLL_SECONDS, PolicyManager, _permission_gates
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

    def send(self, data: str) -> None:
        self.frames.append(json.loads(data))


@pytest.fixture()
def recorder() -> Iterator[_RecordingClient]:
    client = _RecordingClient()
    Websocket._connect(client)
    try:
        yield client
    finally:
        Websocket._disconnect(client)


def _origin() -> dict[str, object]:
    # The shape MessageProcessor.origin builds for a main-spine user turn.
    return {"type": "user", "turn_id": 7, "forked": False}


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
    _seed(db, "chat", "web_search", setting)
    assert mgr.authorize(CH.CHAT, "web_search", _ran) == _ran()


# 1b. INTERNAL tools ALWAYS bypass — every channel, no row, even over a deny row.
# email/calendar/contacts are the pim delegate's inner surface: the user-facing
# permission is the outer `pim` tool, so the inner calls must never gate.
@pytest.mark.parametrize("channel", [CH.CHAT, CH.SUBCONSCIOUS, CH.EXTERNAL_AGENT])
@pytest.mark.parametrize("permission", ["run_script", "search", "browser.open", "recall", "save_graph",
                                        "email.send", "calendar.create_event", "contacts.get"])
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


# 3. unknown key on a surfaceless channel: lazily creates ONE 'ask' row AND denies with the
#    steer sentence — nobody can answer a prompt on a turn with no origin, so the model is told
#    to ask the user for access instead of retrying
def test_unknown_key_provisions_ask_then_escalates(mgr: PolicyManager, db: sqlite3.Connection) -> None:
    out = mgr.authorize(CH.SUBCONSCIOUS, "newtool.action", _ran)
    out2 = mgr.authorize(CH.SUBCONSCIOUS, "newtool.action", _ran)  # no duplicate row
    assert out == out2 == (
        'The newtool.action action is blocked by policy on the subconscious channel: it is set to "ask" '
        "and nobody can answer a prompt here. Do NOT retry. Tell the user it was blocked by policy "
        'and ask them to set newtool.action to "allow" for the subconscious channel in Brain → Policies.'
    )
    rows = db.execute("SELECT setting FROM policy WHERE permission='newtool.action'").fetchall()
    assert [r["setting"] for r in rows] == ["ask"]                       # provisioned exactly once
    assert db.execute("SELECT reason FROM policy_blocked_log ORDER BY id").fetchone()["reason"] == "user_unavailable"


# 5. data layer: seed → get_all hides internal; upsert flips + rejects bad input; reset restores; blocked-log roundtrip
def test_data_layer_roundtrip(mgr: PolicyManager) -> None:
    assert mgr.apply_seed() > 0
    assert all(r["setting"] != "internal" for r in mgr.get_all())               # internal hidden in Brain
    assert mgr.upsert("chat", "bash.modify_file", "deny") == 1
    assert any(r["permission"] == "bash.modify_file" and r["setting"] == "deny" for r in mgr.get_all())
    assert mgr.upsert("nope", "x", "allow") == 0 and mgr.upsert("chat", "x", "bogus") == 0   # invalid rejected
    assert mgr.upsert("chat", "email.send", "deny") == 0                        # INTERNAL tool: never user-gated
    assert not any(r["permission"] == "email.send" for r in mgr.get_all())
    mgr.reset_to_defaults()
    assert any(r["permission"] == "bash.modify_file" and r["setting"] == "ask" for r in mgr.get_all())
    mgr._log_blocked("subconscious", "bash.modify_file", "user_unavailable")
    assert mgr.get_blocked_log()[0]["action_id"] == "bash.modify_file"
    assert mgr.clear_blocked_log() == 1 and mgr.get_blocked_log() == []


# 6. apply_seed reaps every row the seed file no longer lists (older seeds carried
#    email.*/calendar.*/contacts.* rows; every boot must converge them out) while a
#    user's setting on a still-seeded permission survives untouched.
def test_apply_seed_reaps_unseeded_rows(mgr: PolicyManager, db: sqlite3.Connection) -> None:
    _seed(db, "chat", "email.send", "ask")                 # stale rows from an older seed
    _seed(db, "chat", "calendar.create_event", "ask")
    _seed(db, "subconscious", "search", "deny")            # bare tool name, not tool.action
    _seed(db, "chat", "bash.read", "deny")                 # user's own setting — must survive
    mgr.apply_seed()
    perms = {r["permission"] for r in mgr.get_all()}
    assert not perms & {"email.send", "calendar.create_event", "search"}
    # bash.read IS in the seed, so the reap keys on membership and leaves the
    # user's 'deny' in place (INSERT OR IGNORE never overwrites an existing row).
    assert any(r["permission"] == "bash.read" and r["setting"] == "deny" for r in mgr.get_all())


# 7. 'ask' on a turn with no surface (origin None — discovery, subconscious, external agent, a
#    delegate whose caller had none): deny at once with the steer sentence, audit reason
#    user_unavailable, and NOTHING goes over the socket — there is no card to park on.
def test_ask_with_no_surface_denies_with_steer_and_never_broadcasts(
    mgr: PolicyManager, db: sqlite3.Connection, recorder: _RecordingClient,
) -> None:
    _seed(db, "chat", "pim", "ask")
    out = mgr.authorize(CH.CHAT, "pim", _ran, summary="Check tomorrow's calendar")
    assert out == _ASK_NO_SURFACE.format(permission="pim", channel="chat")
    row = db.execute("SELECT context, reason FROM policy_blocked_log").fetchone()
    assert (row["context"], row["reason"]) == ("chat", "user_unavailable")
    assert recorder.frames == []
    assert mgr.pending() == []


# 8. 'ask' on a turn WITH a surface: the broadcast frame is the wire contract (origin + summary),
#    pending() lists that same frame while parked, and an answer given the way
#    POST /api/policies/respond gives it (result + event) runs or blocks the callback, after
#    which a permission_resolved frame goes out and pending() is empty again.
@pytest.mark.parametrize("verdict,should_run", [("approved", True), ("denied", False)])
def test_ask_with_surface_broadcasts_origin_frame_and_resolves(
    mgr: PolicyManager, db: sqlite3.Connection, recorder: _RecordingClient, verdict: str, should_run: bool,
) -> None:
    _seed(db, "chat", "pim", "ask")
    deadline = time.monotonic() + 5.0
    seen: dict[str, object] = {}

    def _answer_like_the_respond_endpoint() -> None:
        while not recorder.frames and time.monotonic() < deadline:
            time.sleep(0.01)
        if not recorder.frames:
            return                                   # the should_stop deadline below fails the test loudly
        seen["pending_while_parked"] = list(mgr.pending())
        gate = _permission_gates[cast(str, recorder.frames[0]["request_id"])]
        gate["result"] = verdict
        cast(threading.Event, gate["event"]).set()

    answerer = threading.Thread(target=_answer_like_the_respond_endpoint)
    answerer.start()
    out = mgr.authorize(
        CH.CHAT, "pim", _ran,
        should_stop=lambda: time.monotonic() > deadline,
        origin=_origin(), summary="Check tomorrow's calendar",
    )
    answerer.join(timeout=5.0)

    request, resolved = recorder.frames
    rid = request["request_id"]
    assert isinstance(rid, str) and len(rid) == 36
    assert request == {
        "type": "permission_request",
        "request_id": rid,
        "action_id": "pim",
        "summary": "Check tomorrow's calendar",
        "origin": {"type": "user", "turn_id": 7, "forked": False},
    }
    assert seen["pending_while_parked"] == [request]
    assert resolved == {"type": "permission_resolved", "request_id": rid}
    assert mgr.pending() == []
    if should_run:
        assert out == _ran()
        assert db.execute("SELECT count(*) FROM policy_blocked_log").fetchone()[0] == 0
    else:
        assert out == "The pim action is not allowed. Do NOT retry."
        assert db.execute("SELECT reason FROM policy_blocked_log").fetchone()["reason"] == "user_denied"


# 9. a cancelled turn unparks the gate on the next poll: blocked, no audit row (not a verdict),
#    and the card is withdrawn with a permission_resolved frame.
def test_cancelled_turn_unparks_gate_resolves_card_and_logs_nothing(
    mgr: PolicyManager, db: sqlite3.Connection, recorder: _RecordingClient,
) -> None:
    _seed(db, "chat", "pim", "ask")
    started = time.monotonic()
    out = mgr.authorize(CH.CHAT, "pim", _ran, should_stop=lambda: True, origin=_origin(), summary="x")
    assert time.monotonic() - started < 2 * _GATE_POLL_SECONDS
    assert out == "The pim action is not allowed. Do NOT retry."
    assert [f["type"] for f in recorder.frames] == ["permission_request", "permission_resolved"]
    assert recorder.frames[0]["request_id"] == recorder.frames[1]["request_id"]
    assert mgr.pending() == []
    assert db.execute("SELECT count(*) FROM policy_blocked_log").fetchone()[0] == 0
