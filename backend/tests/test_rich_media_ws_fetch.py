"""Integration tests for the WS-boundary tool_calls fetch path.

The production lookup is ``_fetch_tool_calls_for_transcript_ids(ids)``, which
takes the exact transcript row IDs threaded from ``MessageProcessor._uid``
via the OutputService metadata. This replaces the original recency-based
lookup that had two known bugs:

  1. It filtered ``role='user'``, so subagent rows (``role='subagent'``) were
     silently invisible. Subagent weather calls produced orphan tags.
  2. It raced the DMN daemon, which can write user-channel rows between ACT
     completion and the WS emit, causing the recency lookup to point at the
     wrong turn.

These regression-sentinel tests assert that the new function pairs tool_calls
to *exact* transcript IDs regardless of channel/role, includes ephemeral=1
rows, and doesn't drift across other concurrent transcript writes. If a
future change ever reintroduces a recency heuristic, these tests fail.

Additionally tested:
- _attach_segments empty/missing transcript_ids fall-through path.
- Two consecutive turns each start ordinals at 1 (per-turn freshness contract).
"""

import json
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


def _seed_transcript(db, channel: str, role: str, content: str = "x") -> int:
    db.execute(
        "INSERT INTO transcript (channel, role, content, xml_migrated) "
        "VALUES (?, ?, ?, 1)",
        (channel, role, content),
    )
    return db.execute("SELECT last_insert_rowid()").fetchone()[0]


def _seed_tool_call(db, transcript_id: int, ordinal: int, location: str = "London, GB") -> None:
    db.execute(
        "INSERT INTO tool_calls (transcript_id, tool_name, params, result, ephemeral, created_at) "
        "VALUES (?, 'weather', '{}', ?, 1, datetime('now'))",
        (transcript_id, _make_tool_result(ordinal, location)),
    )


class TestFetchByTranscriptIds:
    """The new function pairs tool_calls to exact transcript IDs."""

    def test_user_channel_single_id_returns_weather_row(self, db):
        tid = _seed_transcript(db, "user", "user", "What is the weather?")
        _seed_tool_call(db, tid, 1)
        db.commit()

        from api.websocket import _fetch_tool_calls_for_transcript_ids
        rows = _fetch_tool_calls_for_transcript_ids([tid])

        assert len(rows) == 1
        assert rows[0]["tool_name"] == "weather"
        assert "<span id='weather_1'>" in rows[0]["result"]

    def test_user_channel_multi_call_turn_returns_all(self, db):
        tid = _seed_transcript(db, "user", "user", "Compare London and Tokyo")
        _seed_tool_call(db, tid, 1, "London, GB")
        _seed_tool_call(db, tid, 2, "Tokyo, JP")
        db.commit()

        from api.websocket import _fetch_tool_calls_for_transcript_ids
        rows = _fetch_tool_calls_for_transcript_ids([tid])

        assert len(rows) == 2
        results = " ".join(r["result"] for r in rows)
        assert "<span id='weather_1'>" in results
        assert "<span id='weather_2'>" in results

    def test_subagent_channel_resolved_by_explicit_id(self, db):
        """Regression sentinel: subagent rows MUST be reachable when their
        transcript IDs are passed explicitly. The old recency lookup with its
        ``role='user'`` filter silently missed these — this test exists to
        prevent reintroducing that filter.
        """
        tid = _seed_transcript(db, "subagent", "subagent", "internal task")
        _seed_tool_call(db, tid, 1)
        db.commit()

        from api.websocket import _fetch_tool_calls_for_transcript_ids
        rows = _fetch_tool_calls_for_transcript_ids([tid])

        assert len(rows) == 1, (
            "Subagent rows must be returned when the caller passes their exact "
            "transcript_id. If this fails the channel/role filter has been "
            "reintroduced and the subagent rich-media bug is back."
        )
        assert "<span id='weather_1'>" in rows[0]["result"]

    def test_concurrent_dmn_write_does_not_pollute_results(self, db):
        """Regression sentinel against the DMN race. A concurrent user-channel
        row written *after* the assistant turn must not appear in the result
        when the lookup is by ID.
        """
        ump_tid = _seed_transcript(db, "user", "user", "weather please")
        _seed_tool_call(db, ump_tid, 1)
        # Simulate DMN writing a fresh user-channel row mid-emit.
        dmn_tid = _seed_transcript(db, "user", "user", "dmn synthesis")
        _seed_tool_call(db, dmn_tid, 99, "Mars, X1")
        db.commit()

        from api.websocket import _fetch_tool_calls_for_transcript_ids
        rows = _fetch_tool_calls_for_transcript_ids([ump_tid])

        assert len(rows) == 1
        assert "<span id='weather_1'>" in rows[0]["result"]
        assert "Mars" not in rows[0]["result"], (
            "DMN row leaked into the assistant turn — recency lookup has "
            "been reintroduced."
        )

    def test_empty_id_list_returns_empty(self):
        from api.websocket import _fetch_tool_calls_for_transcript_ids
        assert _fetch_tool_calls_for_transcript_ids([]) == []


class TestAttachSegments:
    """_attach_segments is the WS-boundary chokepoint that calls into
    rich_media_parser. It must never raise and always set ``segments``.
    """

    def test_no_transcript_ids_emits_text_segment(self):
        from api.websocket import _attach_segments

        message_evt = {"type": "message", "content": ""}
        content = "Here is your result."
        _attach_segments(message_evt, content, [])

        assert "segments" in message_evt
        segments = message_evt["segments"]
        assert len(segments) == 1
        assert segments[0]["type"] == "text"
        assert segments[0]["content"] == "Here is your result."

    def test_html_content_no_tool_calls_emits_text_segment(self):
        from api.websocket import _attach_segments

        message_evt = {"type": "message"}
        _attach_segments(message_evt, "<p>It is <b>sunny</b> today.</p>", [])

        assert "segments" in message_evt
        assert all(s["type"] == "text" for s in message_evt["segments"])


class TestOrdinalCounterAcrossConsecutiveTurns:
    """Per-turn ordinal contract — fresh dispatcher means fresh ordinals.

    The contract is now made explicit by ``MessageProcessor.send()`` calling
    ``dispatcher.reset_turn_ordinals()`` after instantiation. Even if the
    dispatcher were ever reused across turns, the reset would still hold.
    """

    def test_two_dispatcher_instances_each_start_at_ordinal_1(self):
        from services.act_dispatcher_service import ActDispatcherService

        ordinals = []

        def _capture(channel, action):
            ordinals.append(action.get("_rich_media_ordinal"))
            return "ok"

        d1 = ActDispatcherService()
        d1.handlers["weather"] = _capture
        d1.dispatch_action("user", {"type": "weather"})

        d2 = ActDispatcherService()
        d2.handlers["weather"] = _capture
        d2.dispatch_action("user", {"type": "weather"})

        assert ordinals == [1, 1], (
            f"Expected [1, 1] (ordinal resets on new dispatcher instance), got {ordinals}"
        )

    def test_explicit_reset_turn_ordinals_restarts_counter(self):
        from services.act_dispatcher_service import ActDispatcherService

        ordinals = []

        def _capture(channel, action):
            ordinals.append(action.get("_rich_media_ordinal"))
            return "ok"

        d = ActDispatcherService()
        d.handlers["weather"] = _capture
        d.dispatch_action("user", {"type": "weather"})
        d.dispatch_action("user", {"type": "weather"})
        # End of turn — dispatcher reused for hypothetical next turn.
        d.reset_turn_ordinals()
        d.dispatch_action("user", {"type": "weather"})

        assert ordinals == [1, 2, 1], (
            f"Expected [1, 2, 1] after explicit reset, got {ordinals}"
        )
