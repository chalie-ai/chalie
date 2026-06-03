"""Act-trail as a query (spec §4c / §4) — observable behaviour only.

The trail is an audit log: ``Ability.record()`` INSERTs one raw row per tool
call; ``Ability.fetch_by_transcript_id()`` SELECTs them oldest→newest. On top of
that the loop adds three behaviours — an ephemeral-purge lifecycle, a
``trail_compaction`` slice boundary, and narration rows (DB row + WS event).

These tests cover those behaviours. They deliberately do NOT test field shapes,
flag-storage contracts, constants, boolean-predicate enumeration, or
"function X does not do Y" guards.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ── helpers ───────────────────────────────────────────────────────────────────

def _insert_transcript(db_conn, channel="user", role="user", content="hello"):
    """Insert a transcript row and return its id."""
    db_conn.execute(
        "INSERT INTO transcript (channel, role, content, created_at) "
        "VALUES (?, ?, ?, datetime('now'))",
        (channel, role, content),
    )
    db_conn.commit()
    return db_conn.execute("SELECT last_insert_rowid() AS id").fetchone()[0]


def _count_tool_rows(db_conn, transcript_id):
    return db_conn.execute(
        "SELECT COUNT(*) FROM tool_calls WHERE transcript_id = ?",
        (transcript_id,),
    ).fetchone()[0]


def _fetch_tool_rows(db_conn, transcript_id):
    rows = db_conn.execute(
        "SELECT id, tool_name, params, result, ephemeral FROM tool_calls "
        "WHERE transcript_id = ? ORDER BY id",
        (transcript_id,),
    ).fetchall()
    return [dict(zip(("id", "tool_name", "params", "result", "ephemeral"), r)) for r in rows]


def _make_flat_mp(db_conn, channel="test", broadcast=False, current_iteration=0):
    """Build a flat-path MessageProcessor bound to a real transcript anchor.

    Single shared factory — replaces the per-class ``_make_flat_mp`` copies.
    """
    from services.processor_config import ProcessorConfig
    from services.message_processor import MessageProcessor

    uid = _insert_transcript(db_conn, channel=channel)
    config = ProcessorConfig(
        channel=channel,
        role="test",
        policy_channel=ProcessorConfig.POLICY_CHANNEL.CHAT,
        build_user_prompt=lambda mp: "",
        build_user_definition=lambda mp: "",
        build_system_prompt=lambda mp: "",
        always_available=[],
        discoverable=[],
        blocked=frozenset(),
        max_iterations=1,
        skip_transcript=True,
        skip_input_row=True,
        suppress_history=True,
        broadcast_to="user" if broadcast else None,
        memory_seed=False,
        post_turn=None,
    )
    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "hi", None)
    mp.config = config
    mp.uid = uid
    mp.current_iteration = current_iteration
    mp.deadline = None
    mp.cancel_event = threading.Event()
    mp.thinking_level = "low"
    mp.thinking_exploration = None
    mp.active_tools = []
    return mp


def _make_response(text):
    """Minimal LLM-response stand-in for narration tests."""
    r = MagicMock()
    r.text = text
    r.tool_calls = None
    return r


# ── F2/F3: the write and the read ─────────────────────────────────────────────

class TestRecordAndFetch:
    """record() INSERTs one row; fetch_by_transcript_id() SELECTs them in order."""

    def test_record_inserts_one_row(self, db):
        """One call to record() → exactly one tool_calls row.  Spec §4c / F2."""
        from abilities._base import Ability

        uid = _insert_transcript(db, channel="test_f2")
        before = _count_tool_rows(db, uid)
        Ability.record(
            tool_name="weather", params={"location": "Paris"},
            result="Sunny 25°C", transcript_id=uid, ephemeral=True,
        )
        assert _count_tool_rows(db, uid) == before + 1, "record() must write exactly one row"

    def test_fetch_returns_rows_in_id_order(self, db):
        """Rows returned oldest→newest by autoincrement id, not created_at.  Spec §4c / F3."""
        from abilities._base import Ability

        uid = _insert_transcript(db, channel="test_f3")
        Ability.record(tool_name="first", params={}, result="1", transcript_id=uid)
        Ability.record(tool_name="second", params={}, result="2", transcript_id=uid)
        Ability.record(tool_name="third", params={}, result="3", transcript_id=uid)

        names = [r["tool_name"] for r in Ability.fetch_by_transcript_id(uid)]
        assert names == ["first", "second", "third"], (
            "fetch_by_transcript_id must return rows in ascending id order"
        )

    def test_fetch_scoped_to_transcript_id(self, db):
        """fetch_by_transcript_id must not return rows from other transcripts.  Spec §4c / F3."""
        from abilities._base import Ability

        uid_a = _insert_transcript(db, channel="test_f3a")
        uid_b = _insert_transcript(db, channel="test_f3b")
        Ability.record(tool_name="ta", params={}, result="ra", transcript_id=uid_a)
        Ability.record(tool_name="tb", params={}, result="rb", transcript_id=uid_b)

        assert all(r["tool_name"] == "ta" for r in Ability.fetch_by_transcript_id(uid_a)), \
            "uid_a must only return its rows"
        assert all(r["tool_name"] == "tb" for r in Ability.fetch_by_transcript_id(uid_b)), \
            "uid_b must only return its rows"


# ── F5/F6/F7: compaction slice boundary ───────────────────────────────────────

class TestCompactionBoundary:
    """_render_act_trail slices from the latest trail_compaction; history compaction is not a boundary."""

    def test_slices_from_latest_trail_compaction(self, db):
        """Rows before the last trail_compaction are excluded from the trail.  Spec §4c / F5."""
        from abilities._base import Ability

        mp = _make_flat_mp(db, channel="test_f5")
        Ability.record(tool_name="old_tool", params={}, result="old",
                       transcript_id=mp.uid, ephemeral=True)
        Ability.record(tool_name="trail_compaction", params={}, result="summary text",
                       transcript_id=mp.uid, ephemeral=True)
        Ability.record(tool_name="new_tool", params={}, result="new",
                       transcript_id=mp.uid, ephemeral=True)

        trail = mp._render_act_trail()
        assert "old_tool" not in trail, "rows before trail_compaction must be excluded"
        assert "new_tool" in trail or "summary text" in trail, (
            "rows from trail_compaction onward must be included"
        )

    def test_no_compaction_all_rows_included(self, db):
        """With no trail_compaction row, all rows appear in trail.  Spec §4c / F6."""
        from abilities._base import Ability

        mp = _make_flat_mp(db, channel="test_f6")
        Ability.record(tool_name="tool_x", params={}, result="rx",
                       transcript_id=mp.uid, ephemeral=True)
        Ability.record(tool_name="tool_y", params={}, result="ry",
                       transcript_id=mp.uid, ephemeral=True)

        trail = mp._render_act_trail()
        assert "tool_x" in trail and "tool_y" in trail, (
            "all rows must appear when no compaction exists"
        )

    def test_history_compaction_not_a_trail_boundary(self, db):
        """tool_name='compaction' (history) does NOT reset the trail slice.  Spec §4c / F7."""
        from abilities._base import Ability

        mp = _make_flat_mp(db, channel="test_f7")
        Ability.record(tool_name="tool_before", params={}, result="before",
                       transcript_id=mp.uid, ephemeral=True)
        Ability.record(tool_name="compaction", params={"status": "success"},
                       result="history summary", transcript_id=mp.uid, ephemeral=False)
        Ability.record(tool_name="tool_after", params={}, result="after",
                       transcript_id=mp.uid, ephemeral=True)

        trail = mp._render_act_trail()
        assert "tool_before" in trail, "'compaction' (history) row must not slice the trail"
        assert "tool_after" in trail, "tool_after must appear in trail"


# ── F9: _has_trail gates rendering ────────────────────────────────────────────

class TestHasTrail:
    """_has_trail() decides whether the trail is rendered into the prompt at all."""

    def test_has_trail_true_after_real_tool(self, db):
        """_has_trail() → True after a non-compaction row is recorded.  Spec §4c / F9."""
        from abilities._base import Ability

        mp = _make_flat_mp(db, channel="test_f9")
        Ability.record(tool_name="weather", params={}, result="Sunny",
                       transcript_id=mp.uid, ephemeral=True)
        assert mp._has_trail() is True, "trail with a real tool row must return True"


# ── F11: ephemeral purge lifecycle ────────────────────────────────────────────

class TestEphemeralPurge:
    """Ephemeral rows are purged at turn end; durable rows survive."""

    def test_purge_ephemeral_deletes_ephemeral_rows(self, db):
        """_purge_ephemeral_tool_calls removes ephemeral=1 rows, keeps ephemeral=0.  Spec §4c / F11."""
        from abilities._base import Ability

        mp = _make_flat_mp(db, channel="test_f11")
        Ability.record(tool_name="weather", params={}, result="Sunny",
                       transcript_id=mp.uid, ephemeral=True)
        Ability.record(tool_name="thinking", params={}, result="analysis",
                       transcript_id=mp.uid, ephemeral=False)  # durable — must survive

        def _eph(flag):
            return db.execute(
                "SELECT COUNT(*) FROM tool_calls WHERE transcript_id=? AND ephemeral=?",
                (mp.uid, flag),
            ).fetchone()[0]

        before_eph, before_dur = _eph(1), _eph(0)
        mp._purge_ephemeral_tool_calls()

        assert before_eph > 0, "test setup: should have had ephemeral rows"
        assert _eph(1) == 0, "_purge_ephemeral_tool_calls must delete all ephemeral rows"
        assert _eph(0) == before_dur, "durable rows must survive the purge"


# ── F14/N3: narration writes a trail row and emits a WS event ─────────────────

class TestNarration:
    """_record_narration writes a 'narration' trail row and emits an act_narration WS event."""

    def test_narration_creates_trail_row(self, db):
        """_record_narration writes a tool_calls row with tool_name='narration'.  Spec §4 / F14."""
        mp = _make_flat_mp(db, channel="test_f14")
        mp._record_narration(_make_response("Let me check the weather for you."))

        rows = _fetch_tool_rows(db, mp.uid)
        assert len(rows) == 1, "_record_narration must write one trail row"
        assert rows[0]["tool_name"] == "narration", "trail row must have tool_name='narration'"

    def test_narration_emits_correct_payload(self, db):
        """_record_narration emits {type, text, step} to WS when broadcast_to is set.  Spec §4 / N3."""
        import abilities._base as _base_mod

        mp = _make_flat_mp(db, channel="test_n3", broadcast=True, current_iteration=3)
        emitted = []
        with patch.object(_base_mod, "_emit",
                          side_effect=lambda config, event: emitted.append(event)):
            mp._record_narration(_make_response("I'll look that up for you."))

        events = [e for e in emitted if e.get("type") == "act_narration"]
        assert len(events) == 1, "_record_narration must emit exactly one act_narration event"
        assert "text" in events[0], "narration event must have 'text' field"
        assert events[0]["step"] == 3, "step must equal current_iteration at time of emission"

    def test_narration_no_emit_when_broadcast_to_none(self, db):
        """No WS event emitted when config.broadcast_to is None.  Spec §4 / N3 / N1."""
        import abilities._base as _base_mod

        mp = _make_flat_mp(db, channel="test_n3c", broadcast=False)
        emitted = []
        with patch.object(_base_mod, "_emit",
                          side_effect=lambda config, event: emitted.append(event)):
            mp._record_narration(_make_response("thinking out loud"))

        events = [e for e in emitted if e.get("type") == "act_narration"]
        assert events == [], "no act_narration event when broadcast_to is None"
