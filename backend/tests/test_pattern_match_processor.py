"""
Feature tests for PatternMatchProcessor and its processor-scoped tools.

Each test answers exactly one question: does this service do what it claims?
Real in-memory SQLite via the `db` fixture (schema.sql through
SchemaConvergenceService). MemoryStore is the real production implementation.

LLM boundary: Providers.instance().send_messages is the ONLY boundary that
is stubbed — everything else (DB, DataGraphService, save_pattern, save_graph)
runs against real production code.

Per feedback_test_philosophy.md: hot path only, no mock theater.
"""

import json
from unittest.mock import patch

import pytest

from services.llm_service import LLMResponse

pytestmark = pytest.mark.unit


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_llm_response(text="", tool_calls=None):
    """Return a minimal LLMResponse with the given tool_calls list."""
    return LLMResponse(
        text=text,
        model="test-model",
        provider="mock",
        tool_calls=tool_calls,
    )


def _tool_call(tool_name, **kwargs):
    """Return a minimal tool_call dict matching the format MessageProcessor expects."""
    return {"name": tool_name, "input": kwargs}


def _seed_transcripts(db, count):
    """Seed `count` transcript rows and return the list of inserted IDs.

    SQLite auto-increments, so we rely on SELECT after INSERT to get IDs.
    """
    for i in range(count):
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('test', 'user', ?, datetime('now', ? || ' seconds'))",
            (f"msg {i}", str(i)),
        )
    db.commit()
    rows = db.execute("SELECT id FROM transcript ORDER BY id").fetchall()
    return [r[0] for r in rows]


def _seed_cursor(db, cursor_value):
    """Insert a data_graph row for pattern_match_cursor."""
    db.execute(
        "INSERT INTO data_graph (kind, key, value, active, first_seen_at, last_confirmed_at, source) "
        "VALUES ('system', 'pattern_match_cursor', ?, 1, datetime('now'), datetime('now'), 'test')",
        (str(cursor_value),),
    )
    db.commit()


def _seed_pattern(db, name, confidence, active=1):
    """Insert an active behavioral_pattern row with the given confidence."""
    value = json.dumps({
        "name": name,
        "frequency": "daily",
        "time_anchor": "",
        "summary": f"test pattern {name}",
        "confidence": confidence,
        "last_seen_at": "2026-04-26T00:00:00+00:00",
        "evidence_transcript_ids": [1, 2],
    })
    db.execute(
        "INSERT INTO data_graph (kind, key, value, active, first_seen_at, last_confirmed_at, source) "
        "VALUES ('behavioral_pattern', ?, ?, ?, datetime('now'), datetime('now'), 'pattern_match')",
        (name, value, active),
    )
    db.commit()


def _fetch_pattern(db, name):
    """Fetch the active behavioral_pattern row for a given name."""
    row = db.execute(
        "SELECT id, value, active FROM data_graph "
        "WHERE kind='behavioral_pattern' AND key=? ORDER BY id DESC LIMIT 1",
        (name,),
    ).fetchone()
    if row is None:
        return None
    return {"id": row[0], "value": json.loads(row[1]), "active": row[2]}


def _fetch_pattern_sql_cols(db, name):
    """Fetch the SQL reinforcement columns for the most-recent behavioral_pattern row."""
    row = db.execute(
        "SELECT evidence_count, storage_strength, retrieval_weight, last_accessed_at "
        "FROM data_graph "
        "WHERE kind='behavioral_pattern' AND key=? ORDER BY id DESC LIMIT 1",
        (name,),
    ).fetchone()
    if row is None:
        return None
    return {
        "evidence_count": row[0],
        "storage_strength": row[1],
        "retrieval_weight": row[2],
        "last_accessed_at": row[3],
    }


def _fetch_cursor(db):
    """Read the pattern_match_cursor value from data_graph. Returns int or None."""
    row = db.execute(
        "SELECT value FROM data_graph "
        "WHERE kind='system' AND key='pattern_match_cursor' AND active=1 "
        "ORDER BY id DESC LIMIT 1",
    ).fetchone()
    if row is None:
        return None
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return None


# ── Shared patch targets ─────────────────────────────────────────────────────

_PROVIDERS_INSTANCE = "services.providers.Providers.instance"


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestSkipsWhenDeltaBelowThreshold:
    """Tests 1+2 — step returns 'skip' when delta is below the 50-transcript threshold.

    Case A: no transcripts at all (cold boot) → no cursor row written.
    Case B: cursor=10, only 30 transcripts (delta=20) → cursor unchanged.
    """

    @pytest.mark.parametrize("scenario", ["cold_boot", "under_delta"])
    def test_skips_when_below_threshold(self, scenario, db, store):
        from services.subconscious_worker import SubconsciousWorker

        if scenario == "under_delta":
            _seed_cursor(db, 10)
            for i in range(30):
                db.execute(
                    "INSERT INTO transcript (channel, role, content) VALUES ('t', 'user', ?)",
                    (f"msg {i}",),
                )
            db.commit()

        worker = SubconsciousWorker(tick_sec=10, idle_window_sec=60)
        result = worker._step_pattern_match()

        assert result.startswith("skip"), f"Expected skip for {scenario!r}, got: {result!r}"

        if scenario == "cold_boot":
            cursor = _fetch_cursor(db)
            assert cursor is None, (
                f"Expected no cursor row on cold boot, but cursor={cursor}"
            )
        else:
            row = db.execute(
                "SELECT value FROM data_graph "
                "WHERE kind='system' AND key='pattern_match_cursor' "
                "ORDER BY last_confirmed_at DESC LIMIT 1"
            ).fetchone()
            assert row is not None
            assert row[0] == "10", f"Expected cursor='10', got {row[0]!r}"


class TestFiftyPlusDeltaFiresAndWritesPattern:
    """Test 3 — 60 transcripts, no cursor → processor fires, confidence=7.0."""

    def test_50_plus_delta_fires_and_writes_new_pattern_confidence_7(self, db, store):
        from services.subconscious_worker import SubconsciousWorker

        _seed_transcripts(db, 60)

        tc = _tool_call(
            "save_pattern",
            name="morning_run",
            frequency="weekday",
            time_anchor="07:00",
            summary="goes for a run in the morning",
            evidence_transcript_ids=[1, 2, 3],
        )
        llm_response_with_tool = _make_llm_response(tool_calls=[tc])
        llm_response_clean = _make_llm_response(tool_calls=None)

        call_count = {"n": 0}

        def _fake_send(system_prompt, messages, job=None, tools=None, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return llm_response_with_tool
            return llm_response_clean

        with patch(_PROVIDERS_INSTANCE) as mock_inst:
            mock_inst.return_value.send_messages.side_effect = _fake_send
            mock_inst.return_value.get_context_limit.return_value = 32_000
            mock_inst.return_value.calculate.return_value = 0.0

            worker = SubconsciousWorker(tick_sec=10, idle_window_sec=60)
            result = worker._step_pattern_match()

        assert "fired" in result, f"Expected 'fired' in result, got: {result!r}"

        pattern = _fetch_pattern(db, "morning_run")
        assert pattern is not None, "Expected behavioral_pattern row for 'morning_run'"
        assert pattern["active"] == 1
        assert pattern["value"]["confidence"] == pytest.approx(7.0)


class TestSavePatternConfidenceCap:
    """Tests 4+5 — reinforce always caps at 10.0, whether starting below or at the cap.

    seed_confidence=4.0 → 4+7=11 → capped to 10.0.
    seed_confidence=10.0 → already at cap → stays 10.0.
    """

    @pytest.mark.parametrize("seed_confidence", [4.0, 10.0])
    def test_reinforce_caps_at_10(self, seed_confidence, db, store):
        from services.subconscious_worker import SubconsciousWorker

        _seed_pattern(db, "morning_run", confidence=seed_confidence)
        _seed_transcripts(db, 60)

        tc = _tool_call(
            "save_pattern",
            name="morning_run",
            frequency="weekday",
            time_anchor="07:00",
            summary="goes for a run in the morning",
            evidence_transcript_ids=[1, 2],
        )
        call_count = {"n": 0}

        def _fake_send(system_prompt, messages, job=None, tools=None, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_llm_response(tool_calls=[tc])
            return _make_llm_response(tool_calls=None)

        with patch(_PROVIDERS_INSTANCE) as mock_inst:
            mock_inst.return_value.send_messages.side_effect = _fake_send
            mock_inst.return_value.get_context_limit.return_value = 32_000
            mock_inst.return_value.calculate.return_value = 0.0

            worker = SubconsciousWorker(tick_sec=10, idle_window_sec=60)
            worker._step_pattern_match()

        pattern = _fetch_pattern(db, "morning_run")
        assert pattern is not None
        assert pattern["value"]["confidence"] == pytest.approx(10.0), (
            f"seed={seed_confidence}: expected confidence=10.0 (cap), "
            f"got {pattern['value']['confidence']}"
        )


class TestUntouchedPatternDecaysAndSoftDeletes:
    """Test 6 — untouched patterns decay −0.005; pinned at 0 → soft-deleted."""

    def test_untouched_pattern_decays_pinned_zero_soft_deletes(self, db, store):
        from services.subconscious_worker import SubconsciousWorker

        # Pattern A: confidence=0.5 → should decay to 0.495.
        # Pattern B: confidence=0.005 → decays to 0.0 → active=0.
        _seed_pattern(db, "pattern_a", confidence=0.5)
        _seed_pattern(db, "pattern_b", confidence=0.005)
        _seed_transcripts(db, 60)

        # LLM returns NO tool calls → postTurn decay fires on all untouched rows.
        def _fake_send(system_prompt, messages, job=None, tools=None, **kw):
            return _make_llm_response(tool_calls=None)

        with patch(_PROVIDERS_INSTANCE) as mock_inst:
            mock_inst.return_value.send_messages.side_effect = _fake_send
            mock_inst.return_value.get_context_limit.return_value = 32_000
            mock_inst.return_value.calculate.return_value = 0.0

            worker = SubconsciousWorker(tick_sec=10, idle_window_sec=60)
            worker._step_pattern_match()

        # Re-fetch through the raw connection (db fixture is the connection itself).
        row_a = db.execute(
            "SELECT value, active FROM data_graph "
            "WHERE kind='behavioral_pattern' AND key='pattern_a'"
        ).fetchone()
        row_b = db.execute(
            "SELECT value, active FROM data_graph "
            "WHERE kind='behavioral_pattern' AND key='pattern_b'"
        ).fetchone()

        assert row_a is not None
        val_a = json.loads(row_a[0])
        assert val_a["confidence"] == pytest.approx(0.495, abs=1e-6), (
            f"Expected pattern_a confidence≈0.495, got {val_a['confidence']}"
        )
        assert row_a[1] == 1, "pattern_a should still be active"

        assert row_b is not None
        val_b = json.loads(row_b[0])
        assert val_b["confidence"] == pytest.approx(0.0, abs=1e-6), (
            f"Expected pattern_b confidence=0.0, got {val_b['confidence']}"
        )
        assert row_b[1] == 0, "pattern_b should be soft-deleted (active=0)"


class TestSaveGraphRoutesThroughDataGraphService:
    """Test 7 — save_graph routes through DataGraphService with source='pattern_match'."""

    def test_save_graph_routes_with_source_tag(self, db, store):
        from services.subconscious_worker import SubconsciousWorker

        _seed_transcripts(db, 60)

        tc = _tool_call(
            "save_graph",
            kind="user_specific",
            key="residence",
            value="Lisbon",
        )
        call_count = {"n": 0}

        def _fake_send(system_prompt, messages, job=None, tools=None, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_llm_response(tool_calls=[tc])
            return _make_llm_response(tool_calls=None)

        with patch(_PROVIDERS_INSTANCE) as mock_inst:
            mock_inst.return_value.send_messages.side_effect = _fake_send
            mock_inst.return_value.get_context_limit.return_value = 32_000
            mock_inst.return_value.calculate.return_value = 0.0

            worker = SubconsciousWorker(tick_sec=10, idle_window_sec=60)
            worker._step_pattern_match()

        # The key may be LUT-canonicalised by DataGraphService.store().
        # Assert on source='pattern_match' and kind='user_specific' only.
        row = db.execute(
            "SELECT source FROM data_graph "
            "WHERE source='pattern_match' AND kind='user_specific' LIMIT 1"
        ).fetchone()
        assert row is not None, (
            "Expected at least one data_graph row with source='pattern_match' "
            "and kind='user_specific' after save_graph tool call"
        )


class TestSaveGraphBehavioralPatternKindReturnsError:
    """Test 8 — save_graph with kind='behavioral_pattern' returns invalid_kind error."""

    def test_save_graph_kind_behavioral_pattern_returns_invalid_kind(self, db, store):
        from abilities.save_graph import SaveGraph

        # SaveGraph reads its budget counter via current_processor() + getattr.
        # No processor is bound here — getattr falls back to 0, validation
        # short-circuits before any DB write.
        instance = SaveGraph()
        result = instance.execute(
            "test",
            {"kind": "behavioral_pattern", "key": "x", "value": "y"},
            None,
        )

        assert isinstance(result, dict), f"SaveGraph.execute returned non-dict: {result!r}"
        assert result.get("error") == "invalid_kind", (
            f"Expected error='invalid_kind', got: {result}"
        )
        assert result.get("kind") == "behavioral_pattern"

        # No row should have been written to data_graph.
        row = db.execute(
            "SELECT id FROM data_graph WHERE kind='behavioral_pattern' AND key='x'"
        ).fetchone()
        assert row is None, "No data_graph row should exist after rejected save_graph"


class TestSavePatternBudgetCapAt20:
    """Test 9 — 21 save_pattern calls → 20 land, 21st returns budget_exceeded."""

    def test_save_pattern_budget_cap_at_20(self, db, store):
        from services.subconscious_worker import SubconsciousWorker

        _seed_transcripts(db, 60)

        # Build 21 tool calls with unique names.
        tcs_batch = [
            _tool_call(
                "save_pattern",
                name=f"pattern_{i:02d}",
                frequency="daily",
                time_anchor="",
                summary=f"test pattern {i}",
                evidence_transcript_ids=[1, 2],
            )
            for i in range(21)
        ]

        call_count = {"n": 0}

        def _fake_send(system_prompt, messages, job=None, tools=None, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Return all 21 tool calls in the first pass.
                return _make_llm_response(tool_calls=tcs_batch)
            # Subsequent iterations: no more tool calls → loop exits.
            return _make_llm_response(tool_calls=None)

        with patch(_PROVIDERS_INSTANCE) as mock_inst:
            mock_inst.return_value.send_messages.side_effect = _fake_send
            mock_inst.return_value.get_context_limit.return_value = 32_000
            mock_inst.return_value.calculate.return_value = 0.0

            worker = SubconsciousWorker(tick_sec=10, idle_window_sec=60)
            worker._step_pattern_match()

        # Exactly 20 behavioral_pattern rows should exist (21st was rejected).
        count = db.execute(
            "SELECT COUNT(*) FROM data_graph "
            "WHERE kind='behavioral_pattern' AND source='pattern_match' AND active=1"
        ).fetchone()[0]
        assert count == 20, f"Expected 20 rows (budget cap), got {count}"


class TestSaveGraphBudgetCapAt50:
    """Test 10 — 51 save_graph calls → 50 land, 51st returns budget_exceeded."""

    def test_save_graph_budget_cap_at_50(self, db, store):
        from abilities.save_graph import SaveGraph
        from services.message_processor import bind_current_processor

        # SaveGraph reads/writes its budget counter via current_processor()
        # + getattr/setattr. Bind a stub as the current processor for the
        # duration of the test so the counter persists across 51 calls.
        class _StubProcessor:
            _save_graph_calls = 0

        stub_processor = _StubProcessor()
        instance = SaveGraph()

        with bind_current_processor(stub_processor):
            results = [
                instance.execute(
                    "test",
                    {"kind": "misc", "key": f"key_{i}", "value": f"value_{i}"},
                    None,
                )
                for i in range(51)
            ]

        # First 50 must succeed.
        for i, r in enumerate(results[:50]):
            assert r.get("ok") is True, f"Call {i} expected ok=True, got: {r}"

        # 51st must return budget_exceeded.
        assert results[50].get("budget_exceeded") is True, (
            f"Expected budget_exceeded on 51st call, got: {results[50]}"
        )
        assert results[50].get("tool") == "save_graph"

        # Counter must be exactly 50.
        assert stub_processor._save_graph_calls == 50, (
            f"Expected _save_graph_calls=50, got {stub_processor._save_graph_calls}"
        )


class TestReinforcementDiminishingBoost:
    """Test 12 — each successive reinforce adds a smaller storage_strength boost and
    the value is capped at 1.0 (TKT-581 fix, diminishing returns)."""

    def test_multiple_reinforcements_give_diminishing_boost(self, db, store):
        from abilities.save_pattern import _upsert_pattern

        # Seed with SQL defaults (evidence_count=1, storage_strength=0.5).
        _seed_pattern(db, "evening_read", confidence=3.0)

        tc_params = {
            "name": "evening_read",
            "frequency": "daily",
            "time_anchor": "evening",
            "summary": "reads before bed",
            "evidence_transcript_ids": [1, 2],
        }

        # Reinforce 3 times directly via the production upsert function.
        strengths = []
        for _ in range(3):
            _upsert_pattern({**tc_params, "evidence": tc_params["evidence_transcript_ids"]})
            cols = _fetch_pattern_sql_cols(db, "evening_read")
            strengths.append(cols["storage_strength"])

        # evidence_count must be 1 (initial) + 3 (reinforcements) = 4.
        cols = _fetch_pattern_sql_cols(db, "evening_read")
        assert cols["evidence_count"] == 4, (
            f"Expected evidence_count=4 after 3 reinforcements, got {cols['evidence_count']}"
        )

        # strength must grow after each reinforce.
        assert strengths[1] > strengths[0], (
            f"Expected strength to grow after 2nd reinforce: {strengths}"
        )
        assert strengths[2] > strengths[1], (
            f"Expected strength to grow after 3rd reinforce: {strengths}"
        )

        # Each boost must be smaller than the previous (diminishing returns).
        boost_1 = strengths[0] - 0.5
        boost_2 = strengths[1] - strengths[0]
        boost_3 = strengths[2] - strengths[1]
        assert boost_2 < boost_1, (
            f"Expected diminishing boost: boost_2={boost_2:.5f} should be < boost_1={boost_1:.5f}"
        )
        assert boost_3 < boost_2, (
            f"Expected diminishing boost: boost_3={boost_3:.5f} should be < boost_2={boost_2:.5f}"
        )

        # strength must not exceed 1.0.
        assert cols["storage_strength"] <= 1.0, (
            f"storage_strength must be capped at 1.0, got {cols['storage_strength']}"
        )
