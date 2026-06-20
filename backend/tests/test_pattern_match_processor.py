"""Feature tests for PatternMatchProcessor."""

import contextlib
import json
import sqlite3
from collections.abc import Callable, Generator
from typing import cast

import pytest

from services.llm_clients.base import ProviderClient
from services.provider_api import ProviderApiRequest, ProviderApiResponse
from services.providers import Providers

pytestmark = pytest.mark.unit


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_llm_response(
    text: str = "",
    tool_calls: list[dict[str, object]] | None = None,
) -> ProviderApiResponse:
    return ProviderApiResponse(
        text=text,
        model="test-model",
        provider="mock",
        tool_calls=tool_calls,
    )


class _FakeLLMService(ProviderClient):

    CONTENT_FIELD_LABEL = "message.content"

    def __init__(self, send_fn: Callable[[], ProviderApiResponse]) -> None:
        self._send_fn = send_fn

    def get_context_limit(self) -> int:
        return 200_000

    def estimate_request_tokens(self, dto: ProviderApiRequest) -> int:
        return 1

    def send(self, dto: ProviderApiRequest) -> ProviderApiResponse:
        return self._send_fn()


@contextlib.contextmanager
def _inject_fake_client(
    send_fn: Callable[[], ProviderApiResponse],
) -> Generator[None, None, None]:
    original = getattr(Providers, "_resolve")
    setattr(Providers, "_resolve", lambda self, *_a, **_kw: _FakeLLMService(send_fn))
    try:
        yield
    finally:
        setattr(Providers, "_resolve", original)


def _tool_call(tool_name: str, **kwargs: object) -> dict[str, object]:
    """Return a minimal tool_call dict matching the format MessageProcessor expects."""
    return {"name": tool_name, "input": kwargs}


def _seed_transcripts(db: sqlite3.Connection, count: int) -> list[object]:
    for i in range(count):
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at) "
            "VALUES ('user', 'user', ?, datetime('now', ? || ' seconds'))",
            (f"msg {i}", str(i)),
        )
    db.commit()
    rows = db.execute("SELECT id FROM transcript ORDER BY id").fetchall()
    return [r[0] for r in rows]


def _seed_cursor(db: sqlite3.Connection, cursor_value: int) -> None:
    db.execute(
        "INSERT INTO data_graph (kind, key, value, active, first_seen_at, last_confirmed_at, source) "
        "VALUES ('system', 'pattern_match_cursor', ?, 1, datetime('now'), datetime('now'), 'test')",
        (str(cursor_value),),
    )
    db.commit()


def _seed_pattern(
    db: sqlite3.Connection,
    name: str,
    confidence: float,
    active: int = 1,
) -> None:
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


def _fetch_pattern(
    db: sqlite3.Connection,
    name: str,
) -> dict[str, object] | None:
    """Fetch the active behavioral_pattern row for a given name."""
    row = db.execute(
        "SELECT id, value, active FROM data_graph "
        "WHERE kind='behavioral_pattern' AND key=? ORDER BY id DESC LIMIT 1",
        (name,),
    ).fetchone()
    if row is None:
        return None
    return {"id": row[0], "value": json.loads(row[1]), "active": row[2]}


def _fetch_pattern_sql_cols(
    db: sqlite3.Connection,
    name: str,
) -> dict[str, object] | None:
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


def _fetch_cursor(db: sqlite3.Connection) -> int | None:
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


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestSkipsWhenDeltaBelowThreshold:
    """Tests 1+2 — step returns 'skip' when delta is below the 50-transcript threshold."""

    @pytest.mark.parametrize("scenario", ["cold_boot", "under_delta"])
    def test_skips_when_below_threshold(
        self, scenario: str, db: sqlite3.Connection, store: object
    ) -> None:
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

    def test_50_plus_delta_fires_and_writes_new_pattern_confidence_7(
        self, db: sqlite3.Connection, store: object
    ) -> None:
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

        call_count: dict[str, int] = {"n": 0}

        def _fake_send() -> ProviderApiResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return llm_response_with_tool
            return llm_response_clean

        with _inject_fake_client(_fake_send):
            worker = SubconsciousWorker(tick_sec=10, idle_window_sec=60)
            result = worker._step_pattern_match()

        assert "fired" in result, f"Expected 'fired' in result, got: {result!r}"

        pattern = _fetch_pattern(db, "morning_run")
        assert pattern is not None, "Expected behavioral_pattern row for 'morning_run'"
        assert pattern["active"] == 1
        assert cast(dict[str, object], pattern["value"])["confidence"] == pytest.approx(7.0)


class TestSavePatternConfidenceCap:
    """Tests 4+5 — reinforce always caps at 10.0, whether starting below or at the cap."""

    @pytest.mark.parametrize("seed_confidence", [4.0, 10.0])
    def test_reinforce_caps_at_10(
        self, seed_confidence: float, db: sqlite3.Connection, store: object
    ) -> None:
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
        call_count: dict[str, int] = {"n": 0}

        def _fake_send() -> ProviderApiResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_llm_response(tool_calls=[tc])
            return _make_llm_response(tool_calls=None)

        with _inject_fake_client(_fake_send):

            worker = SubconsciousWorker(tick_sec=10, idle_window_sec=60)
            worker._step_pattern_match()

        pattern = _fetch_pattern(db, "morning_run")
        assert pattern is not None
        assert cast(dict[str, object], pattern["value"])["confidence"] == pytest.approx(10.0), (
            f"seed={seed_confidence}: expected confidence=10.0 (cap), "
            f"got {cast(dict[str, object], pattern['value'])['confidence']}"
        )


class TestUntouchedPatternDecaysAndSoftDeletes:
    """Test 6 — untouched patterns decay −0.005; pinned at 0 → soft-deleted."""

    def test_untouched_pattern_decays_pinned_zero_soft_deletes(
        self, db: sqlite3.Connection, store: object
    ) -> None:
        from services.subconscious_worker import SubconsciousWorker

        # Pattern A: confidence=0.5 → should decay to 0.495.
        # Pattern B: confidence=0.005 → decays to 0.0 → active=0.
        _seed_pattern(db, "pattern_a", confidence=0.5)
        _seed_pattern(db, "pattern_b", confidence=0.005)
        _seed_transcripts(db, 60)

        # LLM returns NO tool calls → postTurn decay fires on all untouched rows.
        def _fake_send() -> ProviderApiResponse:
            return _make_llm_response(tool_calls=None)

        with _inject_fake_client(_fake_send):

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

    def test_save_graph_routes_with_source_tag(
        self, db: sqlite3.Connection, store: object
    ) -> None:
        from services.subconscious_worker import SubconsciousWorker

        _seed_transcripts(db, 60)

        tc = _tool_call(
            "save_graph",
            kind="user_specific",
            key="residence",
            value="Lisbon",
        )
        call_count: dict[str, int] = {"n": 0}

        def _fake_send() -> ProviderApiResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_llm_response(tool_calls=[tc])
            return _make_llm_response(tool_calls=None)

        with _inject_fake_client(_fake_send):

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
    """Test 8 — save_graph with kind='behavioral_pattern' returns a loud"""

    def test_save_graph_kind_behavioral_pattern_returns_invalid_kind(
        self, db: sqlite3.Connection, store: object
    ) -> None:
        from abilities.save_graph import ALLOWED_KINDS, SaveGraph

        # SaveGraph reads its budget counter via self.mp + getattr.
        # No processor is bound here — getattr falls back to 0, validation
        # short-circuits before any DB write.
        instance = SaveGraph()
        result = instance.run(
            {"kind": "behavioral_pattern", "key": "x", "value": "y"},
        )

        # run() now returns an error ToolResult — the invalid kind is loud.
        assert result.status == "error", f"Expected error result, got: {result!r}"
        assert result.code == "invalid-param", (
            f"Expected code='invalid-param', got: {result.code}"
        )
        # The valid ladder lists the real storable kinds and excludes the rejected one.
        assert tuple(ALLOWED_KINDS) == result.valid
        assert "behavioral_pattern" not in result.valid

        # No row should have been written to data_graph.
        row = db.execute(
            "SELECT id FROM data_graph WHERE kind='behavioral_pattern' AND key='x'"
        ).fetchone()
        assert row is None, "No data_graph row should exist after rejected save_graph"


class TestSavePatternBudgetCapAt20:
    """Test 9 — 21 save_pattern calls → 20 land, 21st returns budget_exceeded."""

    def test_save_pattern_budget_cap_at_20(
        self, db: sqlite3.Connection, store: object
    ) -> None:
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

        call_count: dict[str, int] = {"n": 0}

        def _fake_send() -> ProviderApiResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Return all 21 tool calls in the first pass.
                return _make_llm_response(tool_calls=tcs_batch)
            # Subsequent iterations: no more tool calls → loop exits.
            return _make_llm_response(tool_calls=None)

        with _inject_fake_client(_fake_send):

            worker = SubconsciousWorker(tick_sec=10, idle_window_sec=60)
            worker._step_pattern_match()

        # Exactly 20 behavioral_pattern rows should exist (21st was rejected).
        count = db.execute(
            "SELECT COUNT(*) FROM data_graph "
            "WHERE kind='behavioral_pattern' AND source='pattern_match' AND active=1"
        ).fetchone()[0]
        assert count == 20, f"Expected 20 rows (budget cap), got {count}"


class TestSaveGraphBudgetCapAt50:
    """Test 10 — 51 save_graph calls in one tick → 50 land, 51st hits the budget cap."""

    def test_save_graph_budget_cap_at_50(
        self, db: sqlite3.Connection, store: object
    ) -> None:
        from services.subconscious_worker import SubconsciousWorker

        _seed_transcripts(db, 60)

        # 51 unique save_graph calls.
        tcs_batch = [
            _tool_call(
                "save_graph",
                kind="misc",
                key=f"fact_{i:02d}",
                value=f"value_{i:02d}",
            )
            for i in range(51)
        ]

        call_count: dict[str, int] = {"n": 0}

        def _fake_send() -> ProviderApiResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Return all 51 tool calls in the first pass.
                return _make_llm_response(tool_calls=tcs_batch)
            # Subsequent iterations: no more tool calls → loop exits.
            return _make_llm_response(tool_calls=None)

        with _inject_fake_client(_fake_send):
            worker = SubconsciousWorker(tick_sec=10, idle_window_sec=60)
            worker._step_pattern_match()

        # Exactly 50 misc rows should have landed (51st rejected by the budget cap).
        count = db.execute(
            "SELECT COUNT(*) FROM data_graph "
            "WHERE kind='misc' AND source='pattern_match'"
        ).fetchone()[0]
        assert count == 50, f"Expected 50 rows (budget cap), got {count}"


class TestReinforcementDiminishingBoost:
    """Test 12 — each successive reinforce adds a smaller storage_strength boost and"""

    def test_multiple_reinforcements_give_diminishing_boost(
        self, db: sqlite3.Connection, store: object
    ) -> None:
        from services.subconscious_worker import SubconsciousWorker

        # Seed with SQL defaults (evidence_count=1, storage_strength=0.5).
        _seed_pattern(db, "evening_read", confidence=3.0)

        tc = _tool_call(
            "save_pattern",
            name="evening_read",
            frequency="daily",
            time_anchor="evening",
            summary="reads before bed",
            evidence_transcript_ids=[1, 2],
        )

        def _reinforce_once(
            tool_call: dict[str, object],
        ) -> Callable[[], ProviderApiResponse]:
            """A fresh LLM stub: one save_pattern call on the first turn, then stop."""
            state: dict[str, int] = {"n": 0}

            def _send() -> ProviderApiResponse:
                state["n"] += 1
                if state["n"] == 1:
                    return _make_llm_response(tool_calls=[tool_call])
                return _make_llm_response(tool_calls=None)

            return _send

        # Three real background ticks. Each needs a fresh >=50-transcript window
        # (the worker advances pattern_match_cursor to the latest id on a fire),
        # so seed 60 more transcripts before each tick. "evening_read" is touched
        # every turn → the post-turn decay hook excludes it, so storage_strength
        # evolves purely by reinforcement.
        strengths: list[float] = []
        for tick in range(3):
            _seed_transcripts(db, 60)
            with _inject_fake_client(_reinforce_once(tc)):
                worker = SubconsciousWorker(tick_sec=10, idle_window_sec=60)
                result = worker._step_pattern_match()
            assert result.startswith("fired"), (
                f"tick {tick}: expected the matcher to fire, got {result!r}"
            )
            cols = _fetch_pattern_sql_cols(db, "evening_read")
            assert cols is not None
            strengths.append(cast(float, cols["storage_strength"]))

        # evidence_count must be 1 (initial) + 3 (reinforcements) = 4.
        cols = _fetch_pattern_sql_cols(db, "evening_read")
        assert cols is not None
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
        assert cast(float, cols["storage_strength"]) <= 1.0, (
            f"storage_strength must be capped at 1.0, got {cols['storage_strength']}"
        )


def _seed_located_transcripts(
    db: sqlite3.Connection,
    count: int,
    *,
    channel: str,
) -> None:
    """Seed `count` location-tagged transcripts on `channel`."""
    for i in range(count):
        db.execute(
            "INSERT INTO transcript (channel, role, content, created_at, "
            "location_lat, location_lon, location_name) "
            "VALUES (?, 'user', ?, datetime('now', ? || ' seconds'), "
            "35.9, 14.5, 'Valletta')",
            (channel, f"at the gym {i}", str(i)),
        )
    db.commit()


class TestGeoPassProvenance:
    """Test 13 — the GEO pass writes behavioural patterns with source='geo_pattern'."""

    def test_geo_pass_writes_pattern_with_geo_provenance(
        self, db: sqlite3.Connection, store: object
    ) -> None:
        from services.subconscious_worker import SubconsciousWorker

        # 35 located rows on the user channel — above the geo _MIN_DELTA (30).
        _seed_located_transcripts(db, 35, channel="user")

        tc = _tool_call(
            "save_pattern",
            name="harbour_gym",
            frequency="daily",
            time_anchor="18:00",
            summary="user trains at the harbour gym daily",
            evidence_transcript_ids=[1, 2, 3],
        )
        call_count: dict[str, int] = {"n": 0}

        def _fake_send() -> ProviderApiResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_llm_response(tool_calls=[tc])
            return _make_llm_response(tool_calls=None)

        with _inject_fake_client(_fake_send):
            worker = SubconsciousWorker(tick_sec=10, idle_window_sec=60)
            result = worker._step_geo_patterns()

        assert "fired" in result, f"Expected the geo step to fire, got: {result!r}"

        row = db.execute(
            "SELECT source FROM data_graph "
            "WHERE kind='behavioral_pattern' AND key='harbour_gym' AND active=1 "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None, "geo pass must write the behavioural_pattern row"
        assert row[0] == "geo_pattern", (
            f"geo-pass pattern must carry source='geo_pattern', got {row[0]!r} "
            "(a 'pattern_match' tag here means the provenance fix regressed)"
        )

    def test_geo_pass_skips_when_located_rows_are_on_muted_channel(
        self, db: sqlite3.Connection, store: object
    ) -> None:
        """Channel filter (anchor J): location-tagged rows on a NON-user-activity"""
        from services.source_profiles import profile_for
        from services.subconscious_worker import SubconsciousWorker

        channel = "delegate:research"
        assert not profile_for(channel).geo_is_user, (
            "test premise: delegate:* must not count as user geo-activity"
        )

        # 50 located rows, all on a muted channel — well above _MIN_DELTA.
        _seed_located_transcripts(db, 50, channel=channel)

        def _fake_send() -> ProviderApiResponse:
            raise AssertionError(
                "the geo step must NOT fire (and so never call the LLM) when the "
                "only located rows are on a non-user-activity channel"
            )

        with _inject_fake_client(_fake_send):
            worker = SubconsciousWorker(tick_sec=10, idle_window_sec=60)
            result = worker._step_geo_patterns()

        assert result.startswith("skip"), (
            f"expected the geo step to skip on muted-channel geo rows, got: {result!r}"
        )
        # No behavioural pattern was written.
        assert db.execute(
            "SELECT COUNT(*) FROM data_graph WHERE kind='behavioral_pattern'"
        ).fetchone()[0] == 0
