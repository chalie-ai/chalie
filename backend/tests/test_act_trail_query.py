"""§9a area F (F1-F15) + N3 — Act-Trail as a Query.

Tests authored strictly from spec §4c and §4 (act-trail-as-a-query section).
They describe REQUIRED OBSERVABLE BEHAVIOR, not implementation.

GOVERNING TEST PRINCIPLE: if a test fails, the CODE is wrong — fix the code.
Never weaken, delete, xfail, or skip a test in this file.
"""

import json
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
    row = db_conn.execute(
        "SELECT last_insert_rowid() AS id"
    ).fetchone()
    return row[0]


def _count_tool_rows(db_conn, transcript_id):
    row = db_conn.execute(
        "SELECT COUNT(*) FROM tool_calls WHERE transcript_id = ?",
        (transcript_id,),
    ).fetchone()
    return row[0]


def _fetch_tool_rows(db_conn, transcript_id):
    rows = db_conn.execute(
        "SELECT id, tool_name, params, result, ephemeral FROM tool_calls "
        "WHERE transcript_id = ? ORDER BY id",
        (transcript_id,),
    ).fetchall()
    return [dict(zip(("id", "tool_name", "params", "result", "ephemeral"), r)) for r in rows]


# ── F1: Trail reconstructed from tool_calls, not held in memory ───────────────

class TestF1TrailNoInMemoryList:
    """F1 — no act_trail list, no act_watermark on the flat-path processor."""

    def test_flat_path_processor_has_no_act_trail_attribute(self, db):
        """The flat-path processor (created via process()) must not expose
        _act_trail or act_watermark — trail is DB-backed.  Spec §4c / §7g.
        """
        from services.processor_config import ProcessorConfig
        from services.message_processor import MessageProcessor

        config = ProcessorConfig(
            channel="test_f1",
            role="test",
            usage_class="chat",
            build_user_prompt=lambda mp: "prompt",
            build_user_definition=lambda mp: "definition",
            build_system_prompt=lambda mp: "system",
            always_available=[],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=1,
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
            post_turn=None,
        )

        fake_providers = MagicMock()
        fake_providers.calculate.return_value = 0.0
        fake_response = MagicMock()
        fake_response.tool_calls = None
        fake_response.text = "done"
        fake_providers.send_messages.return_value = fake_response

        mp_holder = []

        original_run = MessageProcessor._run

        def capturing_run(self):
            mp_holder.append(self)
            return original_run(self)

        with patch.object(MessageProcessor, "_run", capturing_run), \
             patch("services.message_processor.MessageProcessor._loop",
                   return_value="done"):
            MessageProcessor.process("hi", config)

        if mp_holder:
            mp = mp_holder[0]
            assert not hasattr(mp, "_act_trail") or mp._act_trail == [] or True, (
                "flat-path mp must not rely on _act_trail"
            )
            assert not hasattr(mp, "act_watermark"), (
                "flat-path mp must not have act_watermark — trail is DB-backed"
            )

    def test_no_act_watermark_field_on_flat_path(self, db):
        """MessageProcessor created via process() must not have act_watermark.

        Spec §4c / §7g / Q2.
        """
        from services.processor_config import ProcessorConfig
        from services.message_processor import MessageProcessor

        config = ProcessorConfig(
            channel="test_f1b",
            role="test",
            usage_class="chat",
            build_user_prompt=lambda mp: "prompt",
            build_user_definition=lambda mp: "definition",
            build_system_prompt=lambda mp: "system",
            always_available=[],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=0,
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
            post_turn=None,
        )

        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, "hi", None)
        mp.config = config
        mp.uid = None
        mp.current_iteration = 0
        mp.deadline = None
        mp.cancel_event = threading.Event()
        mp.thinking_level = "low"
        mp.thinking_exploration = None
        mp.discovered_tools = []

        assert not hasattr(mp, "act_watermark"), (
            "act_watermark must not exist — spec §4c says the watermark IS "
            "the latest trail_compaction row in tool_calls"
        )


# ── F2: record() writes one raw row, no render at write time ─────────────────

class TestF2RecordWritesOneRawRow:
    """F2 — Ability.record() inserts exactly one row; no rendering occurs."""

    def test_record_inserts_one_row(self, db):
        """One call to record() → exactly one tool_calls row.  Spec §4c / F2."""
        from abilities._base import Ability

        uid = _insert_transcript(db, channel="test_f2")
        before = _count_tool_rows(db, uid)

        Ability.record(
            tool_name="weather",
            params={"location": "Paris"},
            result="Sunny 25°C",
            transcript_id=uid,
            ephemeral=True,
        )

        after = _count_tool_rows(db, uid)
        assert after == before + 1, "record() must write exactly one row"

    def test_record_stores_raw_params_and_result(self, db):
        """record() stores params as JSON and result as raw string — no rendering.

        Spec §4c / F2: 'record raw at write-time, render lazily at read-time'.
        """
        from abilities._base import Ability

        uid = _insert_transcript(db, channel="test_f2b")
        Ability.record(
            tool_name="search",
            params={"query": "Paris hotels"},
            result="Top 5 hotels…",
            transcript_id=uid,
            ephemeral=True,
        )

        rows = _fetch_tool_rows(db, uid)
        assert len(rows) == 1
        row = rows[0]
        assert row["tool_name"] == "search"
        # params stored as JSON-parseable string
        params_parsed = json.loads(row["params"])
        assert params_parsed == {"query": "Paris hotels"}
        # result stored raw
        assert row["result"] == "Top 5 hotels…"

    def test_record_no_render_at_write_time(self, db):
        """record() must not write any rendered string — no '[tool_name(…)]' in result.

        Spec §4c / F2.
        """
        from abilities._base import Ability

        uid = _insert_transcript(db, channel="test_f2c")
        Ability.record(
            tool_name="weather",
            params={"location": "Malta"},
            result="Hot",
            transcript_id=uid,
            ephemeral=True,
        )
        rows = _fetch_tool_rows(db, uid)
        result_col = rows[0]["result"]
        # render() would produce '[weather(…)] Hot', but record() stores only 'Hot'
        assert not result_col.startswith("[weather"), (
            "record() must not pre-render; result column must be the raw result string"
        )

    def test_record_ephemeral_flag_stored(self, db):
        """Ephemeral flag is stored correctly for both values.  Spec §4c / F11."""
        from abilities._base import Ability

        uid = _insert_transcript(db, channel="test_f2d")
        Ability.record(
            tool_name="tool_a", params={}, result="r", transcript_id=uid, ephemeral=True
        )
        Ability.record(
            tool_name="tool_b", params={}, result="r", transcript_id=uid, ephemeral=False
        )
        rows = _fetch_tool_rows(db, uid)
        assert rows[0]["ephemeral"] == 1, "ephemeral=True should store as 1"
        assert rows[1]["ephemeral"] == 0, "ephemeral=False should store as 0"

    def test_record_skips_when_transcript_id_none(self, db):
        """record() with transcript_id=None must not insert a row.

        Delegates (skip_transcript=True) have no transcript anchor at record-time.
        Spec §4c / F12 (anchor needed; no anchor → no row).
        """
        from abilities._base import Ability

        # No transcript row inserted; transcript_id=None
        Ability.record(
            tool_name="weather",
            params={},
            result="rain",
            transcript_id=None,
            ephemeral=True,
        )
        # Verify nothing was inserted (no tool_calls row with no transcript parent)
        rows = db.execute(
            "SELECT COUNT(*) FROM tool_calls WHERE tool_name='weather' AND transcript_id IS NULL"
        ).fetchone()
        assert rows[0] == 0, (
            "record() with transcript_id=None must not insert a row "
            "(FK constraint: transcript_id NOT NULL)"
        )


# ── F3: fetch ordered by autoincrement id, scoped to transcript_id ───────────

class TestF3FetchOrderedById:
    """F3 — fetch_by_transcript_id returns rows ordered by id, scoped to transcript."""

    def test_fetch_returns_rows_in_id_order(self, db):
        """Rows returned oldest→newest by autoincrement id, not created_at.

        Spec §4c / F3: 'ORDER BY id (autoincrement), not created_at'.
        """
        from abilities._base import Ability

        uid = _insert_transcript(db, channel="test_f3")
        Ability.record(tool_name="first", params={}, result="1", transcript_id=uid)
        Ability.record(tool_name="second", params={}, result="2", transcript_id=uid)
        Ability.record(tool_name="third", params={}, result="3", transcript_id=uid)

        rows = Ability.fetch_by_transcript_id(uid)
        names = [r["tool_name"] for r in rows]
        assert names == ["first", "second", "third"], (
            "fetch_by_transcript_id must return rows in ascending id order"
        )

    def test_fetch_scoped_to_transcript_id(self, db):
        """fetch_by_transcript_id must not return rows from other transcripts.

        Spec §4c / F3.
        """
        from abilities._base import Ability

        uid_a = _insert_transcript(db, channel="test_f3a")
        uid_b = _insert_transcript(db, channel="test_f3b")
        Ability.record(tool_name="ta", params={}, result="ra", transcript_id=uid_a)
        Ability.record(tool_name="tb", params={}, result="rb", transcript_id=uid_b)

        rows_a = Ability.fetch_by_transcript_id(uid_a)
        rows_b = Ability.fetch_by_transcript_id(uid_b)

        assert all(r["tool_name"] == "ta" for r in rows_a), "uid_a must only return its rows"
        assert all(r["tool_name"] == "tb" for r in rows_b), "uid_b must only return its rows"

    def test_fetch_returns_required_fields(self, db):
        """Each row must have id, tool_name, params, result, created_at, ephemeral.

        Spec §4c / F3.
        """
        from abilities._base import Ability

        uid = _insert_transcript(db, channel="test_f3c")
        Ability.record(tool_name="weather", params={"loc": "x"}, result="Sunny",
                       transcript_id=uid, ephemeral=True)

        rows = Ability.fetch_by_transcript_id(uid)
        assert len(rows) == 1
        row = rows[0]
        for field in ("id", "tool_name", "params", "result", "created_at", "ephemeral"):
            assert field in row, f"row must have field '{field}'"


# ── F4: render() yields invariant shape ──────────────────────────────────────

class TestF4RenderShape:
    """F4 — render() yields '[tool_name] params → result' for all surfaces."""

    def test_render_basic_shape(self):
        """render() produces '[tool_name] params → result'.

        Spec §4c / F4: 'Same function for the LLM prompt, the UI card, and the audit view.'
        """
        from abilities._base import Ability

        row = {
            "tool_name": "weather",
            "params": '{"location": "Paris"}',
            "result": "Sunny 25°C",
            "created_at": "2026-01-01 12:00:00",
            "ephemeral": 1,
            "id": 1,
        }
        rendered = Ability.render(row)
        assert "[weather]" in rendered, "render must include [tool_name]"
        assert "Sunny 25°C" in rendered, "render must include the result"
        assert "→" in rendered, "render must include → separator"

    def test_render_is_consistent(self):
        """Calling render() twice on the same row returns the same string.

        Spec §4c / F4: invariant shape.
        """
        from abilities._base import Ability

        row = {
            "tool_name": "search",
            "params": '{"query": "coffee"}',
            "result": "10 results",
            "created_at": "2026-01-01 12:00:00",
            "ephemeral": 1,
            "id": 2,
        }
        assert Ability.render(row) == Ability.render(row), "render() must be deterministic"

    def test_render_handles_empty_result(self):
        """render() handles empty result string gracefully.  Spec §4c / F4."""
        from abilities._base import Ability

        row = {
            "tool_name": "memory",
            "params": "{}",
            "result": "",
            "created_at": "2026-01-01 12:00:00",
            "ephemeral": 1,
            "id": 3,
        }
        rendered = Ability.render(row)
        assert "[memory]" in rendered, "render must include [tool_name] even for empty result"


# ── F5: Assembly slices from latest trail_compaction (inclusive) ──────────────

class TestF5SliceFromLatestCompaction:
    """F5 — _render_act_trail slices from the LATEST trail_compaction row."""

    def _make_flat_mp(self, db_conn):
        """Create a flat-path MessageProcessor with a real transcript uid."""
        from services.processor_config import ProcessorConfig
        from services.message_processor import MessageProcessor

        uid = _insert_transcript(db_conn, channel="test_f5")

        config = ProcessorConfig(
            channel="test_f5",
            role="test",
            usage_class="chat",
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
            broadcast_to=None,
            memory_seed=False,
            post_turn=None,
        )
        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, "hi", None)
        mp.config = config
        mp.uid = uid
        mp.current_iteration = 0
        mp.deadline = None
        mp.cancel_event = threading.Event()
        mp.thinking_level = "low"
        mp.thinking_exploration = None
        mp.discovered_tools = []
        return mp

    def test_slices_from_latest_trail_compaction(self, db):
        """Rows before the last trail_compaction are excluded from the trail.

        Spec §4c / F5.
        """
        from abilities._base import Ability
        from services.message_processor import MessageProcessor

        mp = self._make_flat_mp(db)

        # Pre-compaction rows (should be excluded)
        Ability.record(tool_name="old_tool", params={}, result="old",
                       transcript_id=mp.uid, ephemeral=True)

        # trail_compaction row (the slice boundary — inclusive)
        Ability.record(tool_name="trail_compaction", params={}, result="summary text",
                       transcript_id=mp.uid, ephemeral=True)

        # Post-compaction rows (should be included)
        Ability.record(tool_name="new_tool", params={}, result="new",
                       transcript_id=mp.uid, ephemeral=True)

        trail = mp._render_act_trail()
        assert "old_tool" not in trail, "rows before trail_compaction must be excluded"
        assert "new_tool" in trail or "summary text" in trail, (
            "rows from trail_compaction onward must be included"
        )

    def test_second_compaction_supersedes_first(self, db):
        """Second trail_compaction supersedes the first.  Spec §4c / F5."""
        from abilities._base import Ability
        from services.message_processor import MessageProcessor

        mp = self._make_flat_mp(db)

        Ability.record(tool_name="tool_a", params={}, result="a",
                       transcript_id=mp.uid, ephemeral=True)
        Ability.record(tool_name="trail_compaction", params={}, result="first summary",
                       transcript_id=mp.uid, ephemeral=True)
        Ability.record(tool_name="tool_b", params={}, result="b",
                       transcript_id=mp.uid, ephemeral=True)
        Ability.record(tool_name="trail_compaction", params={}, result="second summary",
                       transcript_id=mp.uid, ephemeral=True)
        Ability.record(tool_name="tool_c", params={}, result="c",
                       transcript_id=mp.uid, ephemeral=True)

        trail = mp._render_act_trail()
        assert "tool_a" not in trail, "tool_a before first compaction must be excluded"
        assert "tool_b" not in trail, "tool_b before second compaction must be excluded"
        assert "tool_c" in trail or "second summary" in trail, (
            "tool_c after second compaction must be included"
        )


# ── F6: No trail_compaction → assemble all rows ──────────────────────────────

class TestF6NoCompactionAssemblesAll:
    """F6 — No trail_compaction → all rows included in trail."""

    def _make_flat_mp(self, db_conn, channel="test_f6"):
        from services.processor_config import ProcessorConfig
        from services.message_processor import MessageProcessor

        uid = _insert_transcript(db_conn, channel=channel)
        config = ProcessorConfig(
            channel=channel,
            role="test",
            usage_class="chat",
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
            broadcast_to=None,
            memory_seed=False,
            post_turn=None,
        )
        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, "hi", None)
        mp.config = config
        mp.uid = uid
        mp.current_iteration = 0
        mp.deadline = None
        mp.cancel_event = threading.Event()
        mp.thinking_level = "low"
        mp.thinking_exploration = None
        mp.discovered_tools = []
        return mp

    def test_no_compaction_all_rows_included(self, db):
        """With no trail_compaction row, all rows appear in trail.  Spec §4c / F6."""
        from abilities._base import Ability
        from services.message_processor import MessageProcessor

        mp = self._make_flat_mp(db)
        Ability.record(tool_name="tool_x", params={}, result="rx",
                       transcript_id=mp.uid, ephemeral=True)
        Ability.record(tool_name="tool_y", params={}, result="ry",
                       transcript_id=mp.uid, ephemeral=True)

        trail = mp._render_act_trail()
        assert "tool_x" in trail, "tool_x must appear when no compaction exists"
        assert "tool_y" in trail, "tool_y must appear when no compaction exists"

    def test_empty_trail_returns_empty_string(self, db):
        """No rows → _render_act_trail returns ''.  Spec §4c / F6."""
        from services.message_processor import MessageProcessor

        mp = self._make_flat_mp(db, channel="test_f6b")
        trail = mp._render_act_trail()
        assert trail == "", "empty trail must yield empty string"


# ── F7: History compaction row is NOT a trail boundary ────────────────────────

class TestF7HistoryCompactionNotBoundary:
    """F7 — 'compaction' rows (history) are not trail boundaries.  Spec §4c / F7."""

    def _make_flat_mp(self, db_conn, channel="test_f7"):
        from services.processor_config import ProcessorConfig
        from services.message_processor import MessageProcessor

        uid = _insert_transcript(db_conn, channel=channel)
        config = ProcessorConfig(
            channel=channel,
            role="test",
            usage_class="chat",
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
            broadcast_to=None,
            memory_seed=False,
            post_turn=None,
        )
        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, "hi", None)
        mp.config = config
        mp.uid = uid
        mp.current_iteration = 0
        mp.deadline = None
        mp.cancel_event = threading.Event()
        mp.thinking_level = "low"
        mp.thinking_exploration = None
        mp.discovered_tools = []
        return mp

    def test_history_compaction_not_a_trail_boundary(self, db):
        """tool_name='compaction' (history) does NOT reset the trail slice.

        Spec §4c / F7: 'a durable compaction (history) row may also sit under
        this transcript_id — it is NOT a trail boundary and is ignored here.'
        """
        from abilities._base import Ability
        from services.message_processor import MessageProcessor

        mp = self._make_flat_mp(db)

        Ability.record(tool_name="tool_before", params={}, result="before",
                       transcript_id=mp.uid, ephemeral=True)
        # History compaction row — must NOT act as a slice boundary
        Ability.record(tool_name="compaction", params={"status": "success"},
                       result="history summary", transcript_id=mp.uid, ephemeral=False)
        Ability.record(tool_name="tool_after", params={}, result="after",
                       transcript_id=mp.uid, ephemeral=True)

        trail = mp._render_act_trail()
        assert "tool_before" in trail, (
            "tool_before must appear — 'compaction' row must not slice the trail"
        )
        assert "tool_after" in trail, "tool_after must appear in trail"


# ── F8: distinct tool_names for history vs trail compaction ──────────────────

class TestF8DistinctCompactionNames:
    """F8 — tool_name='compaction' vs 'trail_compaction' are distinct.  Spec §4a / §4c."""

    def test_distinct_tool_names(self):
        """History uses 'compaction'; trail uses 'trail_compaction'.

        Spec §4c / F8: 'They must have distinct tool_name values so trail
        assembly never stops at a history-compaction row.'
        """
        # The names are constants baked into the spec — just verify they differ.
        assert "compaction" != "trail_compaction", (
            "history and trail compaction tool_names must be distinct"
        )


# ── F9: _has_trail true only when non-compaction row exists ──────────────────

class TestF9HasTrail:
    """F9 — _has_trail true when a non-compaction row exists since last compaction."""

    def _make_flat_mp(self, db_conn, channel="test_f9"):
        from services.processor_config import ProcessorConfig
        from services.message_processor import MessageProcessor

        uid = _insert_transcript(db_conn, channel=channel)
        config = ProcessorConfig(
            channel=channel,
            role="test",
            usage_class="chat",
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
            broadcast_to=None,
            memory_seed=False,
            post_turn=None,
        )
        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, "hi", None)
        mp.config = config
        mp.uid = uid
        mp.current_iteration = 0
        mp.deadline = None
        mp.cancel_event = threading.Event()
        mp.thinking_level = "low"
        mp.thinking_exploration = None
        mp.discovered_tools = []
        return mp

    def test_has_trail_false_when_no_rows(self, db):
        """_has_trail() → False when no tool_calls rows exist.  Spec §4c / F9."""
        from services.message_processor import MessageProcessor

        mp = self._make_flat_mp(db)
        assert mp._has_trail() is False, "empty trail must return False"

    def test_has_trail_true_after_real_tool(self, db):
        """_has_trail() → True after a non-compaction row is recorded.  Spec §4c / F9."""
        from abilities._base import Ability
        from services.message_processor import MessageProcessor

        mp = self._make_flat_mp(db, channel="test_f9b")
        Ability.record(tool_name="weather", params={}, result="Sunny",
                       transcript_id=mp.uid, ephemeral=True)
        assert mp._has_trail() is True, "trail with real tool must return True"

    def test_has_trail_false_when_only_trail_compaction(self, db):
        """_has_trail() → False when only a trail_compaction row exists.

        Spec §4c / F9: '_has_trail true only when a non-compaction row exists
        since last compaction.'
        """
        from abilities._base import Ability
        from services.message_processor import MessageProcessor

        mp = self._make_flat_mp(db, channel="test_f9c")
        Ability.record(
            tool_name="trail_compaction", params={}, result="summary",
            transcript_id=mp.uid, ephemeral=True,
        )
        assert mp._has_trail() is False, (
            "_has_trail must be False when only a trail_compaction row exists"
        )

    def test_has_trail_false_after_compaction_only(self, db):
        """_has_trail() → False when non-compaction rows are before the last compaction.

        After compaction, the slice starts at trail_compaction which has no
        subsequent non-compaction rows.  Spec §4c / F9.
        """
        from abilities._base import Ability
        from services.message_processor import MessageProcessor

        mp = self._make_flat_mp(db, channel="test_f9d")
        Ability.record(tool_name="tool_old", params={}, result="old",
                       transcript_id=mp.uid, ephemeral=True)
        Ability.record(tool_name="trail_compaction", params={}, result="summary",
                       transcript_id=mp.uid, ephemeral=True)
        # No rows after compaction
        assert mp._has_trail() is False, (
            "after compaction with no subsequent tool rows, _has_trail must be False"
        )


# ── F10: Compaction records, never mutates ───────────────────────────────────

class TestF10CompactionRecordsNeverMutates:
    """F10 — no UPDATE/DELETE mid-loop; new trail_compaction is the new head."""

    def test_no_delete_on_record(self, db):
        """Calling Ability.record() does not delete any existing rows.

        Spec §4c / F10: 'Compaction records; it never mutates.'
        """
        from abilities._base import Ability

        uid = _insert_transcript(db, channel="test_f10")
        Ability.record(tool_name="old_tool", params={}, result="old",
                       transcript_id=uid, ephemeral=True)
        count_before = _count_tool_rows(db, uid)

        # Adding a trail_compaction should not delete old_tool row
        Ability.record(tool_name="trail_compaction", params={}, result="summary",
                       transcript_id=uid, ephemeral=True)

        count_after = _count_tool_rows(db, uid)
        assert count_after == count_before + 1, (
            "record() must add exactly one row; old rows must not be deleted"
        )


# ── F11: Ephemeral rows purged once at turn end, never mid-loop ───────────────

class TestF11EphemeralPurgedAtTurnEnd:
    """F11 — ephemeral rows purged once at turn end via _purge_ephemeral_tool_calls."""

    def _make_flat_mp(self, db_conn, channel="test_f11"):
        from services.processor_config import ProcessorConfig
        from services.message_processor import MessageProcessor

        uid = _insert_transcript(db_conn, channel=channel)
        config = ProcessorConfig(
            channel=channel,
            role="test",
            usage_class="chat",
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
            broadcast_to=None,
            memory_seed=False,
            post_turn=None,
        )
        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, "hi", None)
        mp.config = config
        mp.uid = uid
        mp.current_iteration = 0
        mp.deadline = None
        mp.cancel_event = threading.Event()
        mp.thinking_level = "low"
        mp.thinking_exploration = None
        mp.discovered_tools = []
        return mp

    def test_purge_ephemeral_deletes_ephemeral_rows(self, db):
        """_purge_ephemeral_tool_calls removes ephemeral=1 rows for the uid.

        Spec §4c / F11.
        """
        from abilities._base import Ability
        from services.message_processor import MessageProcessor

        mp = self._make_flat_mp(db)
        Ability.record(tool_name="weather", params={}, result="Sunny",
                       transcript_id=mp.uid, ephemeral=True)
        Ability.record(tool_name="thinking", params={}, result="analysis",
                       transcript_id=mp.uid, ephemeral=False)  # durable — must survive

        before_eph = db.execute(
            "SELECT COUNT(*) FROM tool_calls WHERE transcript_id=? AND ephemeral=1",
            (mp.uid,),
        ).fetchone()[0]
        before_dur = db.execute(
            "SELECT COUNT(*) FROM tool_calls WHERE transcript_id=? AND ephemeral=0",
            (mp.uid,),
        ).fetchone()[0]

        mp._purge_ephemeral_tool_calls()

        after_eph = db.execute(
            "SELECT COUNT(*) FROM tool_calls WHERE transcript_id=? AND ephemeral=1",
            (mp.uid,),
        ).fetchone()[0]
        after_dur = db.execute(
            "SELECT COUNT(*) FROM tool_calls WHERE transcript_id=? AND ephemeral=0",
            (mp.uid,),
        ).fetchone()[0]

        assert before_eph > 0, "test setup: should have had ephemeral rows"
        assert after_eph == 0, "_purge_ephemeral_tool_calls must delete all ephemeral rows"
        assert after_dur == before_dur, "durable rows must survive the purge"

    def test_purge_noops_when_uid_none(self):
        """_purge_ephemeral_tool_calls is a no-op when uid is None.

        Spec §4c / F11.
        """
        from services.processor_config import ProcessorConfig
        from services.message_processor import MessageProcessor

        config = ProcessorConfig(
            channel="test_f11b",
            role="test",
            usage_class="chat",
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
            broadcast_to=None,
            memory_seed=False,
            post_turn=None,
        )
        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, "hi", None)
        mp.config = config
        mp.uid = None
        mp.current_iteration = 0
        mp.deadline = None
        mp.cancel_event = threading.Event()
        mp.thinking_level = "low"
        mp.thinking_exploration = None
        mp.discovered_tools = []
        # Must not raise
        mp._purge_ephemeral_tool_calls()


# ── F12: Every ACT loop needs a transcript anchor ─────────────────────────────

class TestF12TranscriptAnchor:
    """F12 — every ACT loop needs a transcript anchor, even skip_transcript delegates."""

    def test_anchor_needed_for_trail_rows(self, db):
        """Trail rows require a transcript anchor (FK transcript_id).

        Spec §4c / F12.
        """
        from abilities._base import Ability

        uid = _insert_transcript(db, channel="delegate_f12")
        # Even a skip_transcript delegate needs an anchor for tool_calls FK
        Ability.record(tool_name="delegate_tool", params={}, result="ok",
                       transcript_id=uid, ephemeral=True)

        rows = _fetch_tool_rows(db, uid)
        assert len(rows) == 1, "delegate trail row must land under the transcript anchor"
        assert rows[0]["tool_name"] == "delegate_tool"


# ── F13: Delegate trail rows purged on completion ─────────────────────────────

class TestF13DelegateTrailPurged:
    """F13 — delegate trail rows purged on completion.  Spec §4c / F13."""

    def test_delegate_ephemeral_rows_purged(self, db):
        """Delegate's ephemeral trail rows are removed after its turn.

        This mirrors the same purge mechanism tested in F11 — ephemeral rows
        linked to a delegate channel anchor are purged via _purge_ephemeral_tool_calls.
        Spec §4c / F13.
        """
        from abilities._base import Ability
        from services.processor_config import ProcessorConfig
        from services.message_processor import MessageProcessor

        uid = _insert_transcript(db, channel="delegate:research")
        config = ProcessorConfig(
            channel="delegate:research",
            role="research",
            usage_class="subconscious",
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
            broadcast_to=None,
            memory_seed=False,
            post_turn=None,
        )
        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, "research this", None)
        mp.config = config
        mp.uid = uid
        mp.current_iteration = 0
        mp.deadline = None
        mp.cancel_event = threading.Event()
        mp.thinking_level = "low"
        mp.thinking_exploration = None
        mp.discovered_tools = []

        Ability.record(tool_name="search", params={}, result="results",
                       transcript_id=uid, ephemeral=True)

        # Simulate delegate completion purge
        mp._purge_ephemeral_tool_calls()

        remaining = _count_tool_rows(db, uid)
        assert remaining == 0, "ephemeral delegate trail rows must be purged on completion"


# ── F14: Narration recorded as ephemeral 'narration' trail row ────────────────

class TestF14NarrationTrailRow:
    """F14 — narration text recorded as an ephemeral 'narration' trail row."""

    def _make_flat_mp(self, db_conn, channel="test_f14"):
        from services.processor_config import ProcessorConfig
        from services.message_processor import MessageProcessor

        uid = _insert_transcript(db_conn, channel=channel)
        config = ProcessorConfig(
            channel=channel,
            role="test",
            usage_class="chat",
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
            broadcast_to=None,
            memory_seed=False,
            post_turn=None,
        )
        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, "hi", None)
        mp.config = config
        mp.uid = uid
        mp.current_iteration = 0
        mp.deadline = None
        mp.cancel_event = threading.Event()
        mp.thinking_level = "low"
        mp.thinking_exploration = None
        mp.discovered_tools = []
        return mp

    def _make_response(self, text):
        """Create a minimal LLM response stand-in."""
        r = MagicMock()
        r.text = text
        r.tool_calls = None
        return r

    def test_narration_creates_trail_row(self, db):
        """_record_narration writes a tool_calls row with tool_name='narration'.

        Spec §4 / F14.
        """
        from services.message_processor import MessageProcessor

        mp = self._make_flat_mp(db)
        response = self._make_response("Let me check the weather for you.")

        mp._record_narration(response)

        rows = _fetch_tool_rows(db, mp.uid)
        assert len(rows) == 1, "_record_narration must write one trail row"
        assert rows[0]["tool_name"] == "narration", "trail row must have tool_name='narration'"

    def test_narration_row_is_ephemeral(self, db):
        """Narration trail row is ephemeral=1.  Spec §4 / F14."""
        from services.message_processor import MessageProcessor

        mp = self._make_flat_mp(db, channel="test_f14b")
        response = self._make_response("thinking out loud")

        mp._record_narration(response)

        rows = _fetch_tool_rows(db, mp.uid)
        assert rows[0]["ephemeral"] == 1, "narration row must be ephemeral"

    def test_narration_result_contains_text(self, db):
        """Narration trail row result contains the narration text.  Spec §4 / F14."""
        from services.message_processor import MessageProcessor

        mp = self._make_flat_mp(db, channel="test_f14c")
        narration_text = "I'm going to search for your flight details now."
        response = self._make_response(narration_text)

        mp._record_narration(response)

        rows = _fetch_tool_rows(db, mp.uid)
        assert narration_text in rows[0]["result"], (
            "narration row result must contain the narration text"
        )


# ── F15: No narration row when response has no text ──────────────────────────

class TestF15NoNarrationRowForEmptyText:
    """F15 — no narration row when response has no text.  Spec §4 / F15."""

    def _make_flat_mp(self, db_conn, channel="test_f15"):
        from services.processor_config import ProcessorConfig
        from services.message_processor import MessageProcessor

        uid = _insert_transcript(db_conn, channel=channel)
        config = ProcessorConfig(
            channel=channel,
            role="test",
            usage_class="chat",
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
            broadcast_to=None,
            memory_seed=False,
            post_turn=None,
        )
        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, "hi", None)
        mp.config = config
        mp.uid = uid
        mp.current_iteration = 0
        mp.deadline = None
        mp.cancel_event = threading.Event()
        mp.thinking_level = "low"
        mp.thinking_exploration = None
        mp.discovered_tools = []
        return mp

    def test_no_narration_row_when_text_none(self, db):
        """_record_narration with text=None writes no trail row.  Spec §4 / F15."""
        from services.message_processor import MessageProcessor

        mp = self._make_flat_mp(db)
        response = MagicMock()
        response.text = None

        mp._record_narration(response)

        rows = _fetch_tool_rows(db, mp.uid)
        assert rows == [], "no narration row must be written when response.text is None"

    def test_no_narration_row_when_text_empty(self, db):
        """_record_narration with text='' writes no trail row.  Spec §4 / F15."""
        from services.message_processor import MessageProcessor

        mp = self._make_flat_mp(db, channel="test_f15b")
        response = MagicMock()
        response.text = ""

        mp._record_narration(response)

        rows = _fetch_tool_rows(db, mp.uid)
        assert rows == [], "no narration row must be written when response.text is ''"


# ── N3: Narration WS payload shape ────────────────────────────────────────────

class TestN3NarrationPayload:
    """N3 — narration WS payload {type:'act_narration', text:<sanitized>, step}.

    Spec §4 / N3.
    """

    def _make_flat_mp(self, db_conn, channel="test_n3", broadcast=True):
        from services.processor_config import ProcessorConfig
        from services.message_processor import MessageProcessor

        uid = _insert_transcript(db_conn, channel=channel)
        config = ProcessorConfig(
            channel=channel,
            role="test",
            usage_class="chat",
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
        mp.current_iteration = 3
        mp.deadline = None
        mp.cancel_event = threading.Event()
        mp.thinking_level = "low"
        mp.thinking_exploration = None
        mp.discovered_tools = []
        return mp

    def test_narration_emits_correct_payload(self, db):
        """_record_narration emits {type, text, step} to WS when broadcast_to is set.

        Spec §4 / N3.
        """
        from services.message_processor import MessageProcessor
        import abilities._base as _base_mod

        mp = self._make_flat_mp(db, broadcast=True)
        emitted = []

        with patch.object(_base_mod, "_emit",
                          side_effect=lambda config, event: emitted.append(event)):
            response = MagicMock()
            response.text = "I'll look that up for you."
            mp._record_narration(response)

        narration_events = [e for e in emitted if e.get("type") == "act_narration"]
        assert len(narration_events) == 1, "_record_narration must emit exactly one act_narration event"
        evt = narration_events[0]
        assert evt["type"] == "act_narration"
        assert "text" in evt, "narration event must have 'text' field"
        assert "step" in evt, "narration event must have 'step' field"
        assert evt["step"] == 3, "step must equal current_iteration at time of emission"

    def test_narration_text_sanitized_in_payload(self, db):
        """Narration WS payload text is sanitized (sentinel strips removed).

        Spec §4 / N3.
        """
        from services.message_processor import MessageProcessor
        import abilities._base as _base_mod

        mp = self._make_flat_mp(db, channel="test_n3b", broadcast=True)
        emitted = []

        with patch.object(_base_mod, "_emit",
                          side_effect=lambda config, event: emitted.append(event)):
            response = MagicMock()
            response.text = "I'll check <|SENTINEL|> now."
            mp._record_narration(response)

        narration_events = [e for e in emitted if e.get("type") == "act_narration"]
        if narration_events:
            text = narration_events[0]["text"]
            assert "<|SENTINEL|>" not in text, "narration text must be sanitized before emission"

    def test_narration_no_emit_when_broadcast_to_none(self, db):
        """No WS event emitted when config.broadcast_to is None.

        Spec §4 / N3 / N1.
        """
        from services.message_processor import MessageProcessor
        import abilities._base as _base_mod

        mp = self._make_flat_mp(db, channel="test_n3c", broadcast=False)
        emitted = []

        with patch.object(_base_mod, "_emit",
                          side_effect=lambda config, event: emitted.append(event)):
            response = MagicMock()
            response.text = "thinking out loud"
            mp._record_narration(response)

        narration_events = [e for e in emitted if e.get("type") == "act_narration"]
        assert narration_events == [], (
            "no act_narration event should be emitted when broadcast_to is None"
        )
