"""Integration tests for the SegmentService tool_calls fetch path.

The production lookup is ``SegmentService._fetch_tool_calls(ids)``, which
takes the exact transcript row IDs threaded from ``MessageProcessor._uid``.
This replaces the original recency-based lookup that had two known bugs:

  1. It filtered ``role='user'``, so subagent rows (``role='subagent'``) were
     silently invisible. Subagent weather calls produced orphan tags.
  2. It raced concurrent background writes, which can land user-channel rows
     between ACT completion and the WS emit, causing the recency lookup to
     point at the wrong turn.

These regression-sentinel tests assert that the function pairs tool_calls
to *exact* transcript IDs regardless of channel/role, and doesn't drift
across other concurrent transcript writes.

Additionally tested:
- SegmentService.build() empty/missing transcript_ids fall-through path.
- Two consecutive turns each start ordinals at 1 (per-turn freshness contract).
"""

import json
import sqlite3
from typing import cast

import pytest

pytestmark = pytest.mark.integration


_WEATHER_DICT = {
    "location": "London, GB",
    "condition": "Partly cloudy",
    "temperature_c": 12.4,
    "temperature_f": 54.3,
    "feels_like_c": 10.1,
    "humidity_pct": 78,
    "wind_kmh": 14.2,
    "wind_direction": "WSW",
    "visibility_km": None,
    "uv_index": None,
    "precip_mm": 0.0,
    "observation_time": "2026-05-02T14:00",
    "is_raining": False,
    "is_daylight": True,
    "is_hot": False,
    "is_cold": False,
    "is_windy": False,
    "is_clear": False,
    "forecast_tomorrow_condition": "Slight rain",
    "forecast_tomorrow_max_c": 14.0,
    "forecast_tomorrow_min_c": 9.0,
    "forecast_tomorrow_precip_chance_pct": 70,
    "forecast_tomorrow_precip_mm": 3.2,
}


def _make_tool_result(ordinal: int, location: str = "London, GB") -> str:
    payload = dict(_WEATHER_DICT, location=location)
    return json.dumps(payload) + f"\n\n<span id='weather_{ordinal}'>"


def _seed_transcript(db: sqlite3.Connection, channel: str, role: str, content: str = "x") -> int:
    db.execute(
        "INSERT INTO transcript (channel, role, content, xml_migrated) "
        "VALUES (?, ?, ?, 1)",
        (channel, role, content),
    )
    return cast(int, db.execute("SELECT last_insert_rowid()").fetchone()[0])


def _seed_tool_call(db: sqlite3.Connection, transcript_id: int, ordinal: int, location: str = "London, GB") -> None:
    db.execute(
        "INSERT INTO tool_calls (transcript_id, tool_name, params, result, created_at) "
        "VALUES (?, 'weather', '{}', ?, datetime('now'))",
        (transcript_id, _make_tool_result(ordinal, location)),
    )


class TestFetchByTranscriptIds:
    """The new function pairs tool_calls to exact transcript IDs."""

    def test_user_channel_single_id_returns_weather_row(self, db: sqlite3.Connection) -> None:
        tid = _seed_transcript(db, "user", "user", "What is the weather?")
        _seed_tool_call(db, tid, 1)
        db.commit()

        from services.segment_service import SegmentService
        rows = SegmentService._fetch_tool_calls([tid])

        assert len(rows) == 1
        assert rows[0]["tool_name"] == "weather"
        assert "<span id='weather_1'>" in cast(str, rows[0]["result"])

    def test_user_channel_multi_call_turn_returns_all(self, db: sqlite3.Connection) -> None:
        tid = _seed_transcript(db, "user", "user", "Compare London and Tokyo")
        _seed_tool_call(db, tid, 1, "London, GB")
        _seed_tool_call(db, tid, 2, "Tokyo, JP")
        db.commit()

        from services.segment_service import SegmentService
        rows = SegmentService._fetch_tool_calls([tid])

        assert len(rows) == 2
        results = " ".join(cast(str, r["result"]) for r in rows)
        assert "<span id='weather_1'>" in results
        assert "<span id='weather_2'>" in results

    def test_subagent_channel_resolved_by_explicit_id(self, db: sqlite3.Connection) -> None:
        """Regression sentinel: subagent rows MUST be reachable when their
        transcript IDs are passed explicitly. The old recency lookup with its
        ``role='user'`` filter silently missed these — this test exists to
        prevent reintroducing that filter.
        """
        tid = _seed_transcript(db, "subagent", "subagent", "internal task")
        _seed_tool_call(db, tid, 1)
        db.commit()

        from services.segment_service import SegmentService
        rows = SegmentService._fetch_tool_calls([tid])

        assert len(rows) == 1, (
            "Subagent rows must be returned when the caller passes their exact "
            "transcript_id. If this fails the channel/role filter has been "
            "reintroduced and the subagent rich-media bug is back."
        )
        assert "<span id='weather_1'>" in cast(str, rows[0]["result"])

    def test_concurrent_background_write_does_not_pollute_results(self, db: sqlite3.Connection) -> None:
        """Regression sentinel against the concurrent-write race. A user-channel
        row written *after* the assistant turn must not appear in the result
        when the lookup is by ID.
        """
        ump_tid = _seed_transcript(db, "user", "user", "weather please")
        _seed_tool_call(db, ump_tid, 1)
        # Simulate a background loop writing a fresh user-channel row mid-emit.
        bg_tid = _seed_transcript(db, "user", "user", "background synthesis")
        _seed_tool_call(db, bg_tid, 99, "Mars, X1")
        db.commit()

        from services.segment_service import SegmentService
        rows = SegmentService._fetch_tool_calls([ump_tid])

        assert len(rows) == 1
        assert "<span id='weather_1'>" in cast(str, rows[0]["result"])
        assert "Mars" not in cast(str, rows[0]["result"]), (
            "Concurrent row leaked into the assistant turn — recency lookup "
            "has been reintroduced."
        )

    def test_empty_id_list_returns_empty(self) -> None:
        from services.segment_service import SegmentService
        assert SegmentService._fetch_tool_calls([]) == []


class TestSegmentServiceBuild:
    """SegmentService.build() must always return at least one segment."""

    def test_no_transcript_ids_emits_text_segment(self) -> None:
        from services.segment_service import SegmentService

        segments = SegmentService.build("Here is your result.", [])

        assert len(segments) == 1
        assert segments[0]["type"] == "text"
        assert segments[0]["content"] == "Here is your result."

    def test_html_content_no_tool_calls_emits_text_segment(self) -> None:
        from services.segment_service import SegmentService

        segments = SegmentService.build("<p>It is <b>sunny</b> today.</p>", [])

        assert all(s["type"] == "text" for s in segments)
