# All tool_calls rows are now durable (no ephemeral column); the retention janitor
# in DecayEngineService removes rows older than 7 days.
#
# Feature tests for the /conversation/recent refresh path (get_recent_history)
# under the recursive MP-chain turn model. A tool turn is now a CHAIN of rows:
#
#     input row (user)
#       → assistant STEP row  (emits the tool calls; tools anchor HERE)
#       → assistant FINAL row (carries the synthesis spans)
#
# all sharing the turn_id the input row opened. So the refresh path must:
#   * render each assistant row's OWN tool calls as chips ({tool_name, summary}),
#     anchored to the row that emitted them (the step row), AND
#   * resolve rich-media spans TURN-WIDE — the span lives on the final row but the
#     tool that produced the card ran on the step row, so pairing needs every
#     transcript id in the turn, not just the row's own.
#
# These drive the REAL production entry point (get_recent_history) against the
# REAL db, seeding turns through the production writers (transcript_service +
# ActTrail) — no hand-rolled INSERTs, no mocks.

import json
import sqlite3
from typing import cast

import pytest

pytestmark = pytest.mark.unit


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

_TOOL_RESULT = (
    json.dumps(_WEATHER_DICT)
    + "\n\n"
    + "This tool supports rich-media rendering. "
    + "wrap your synthesis in <span id='weather_1'>your synthesis here</span>."
)


def _weather_result(loc: str, ordinal: int) -> str:
    payload = dict(_WEATHER_DICT, location=loc)
    return json.dumps(payload) + f"\n\n<span id='weather_{ordinal}'>"


class TestConversationRecentSegments:

    def _seed_plain_turn(self, user_text: str, final_text: str) -> tuple[int, int]:
        """A turn with no tools: input row + a single final assistant row."""
        from services.transcript_service import Transcript
        uid = Transcript.write_input_row("user", "user", user_text)
        turn = Transcript.turn_id_of_row(uid)
        final_id = Transcript.write_assistant_row("user", final_text, turn_id=turn)
        return uid, final_id

    def _seed_tool_turn(
        self,
        user_text: str,
        step_text: str,
        final_text: str,
        tool_calls: list[dict[str, object]],
    ) -> tuple[int, int, int]:
        """A tool turn in the chain model: input row → assistant STEP row that
        emitted the tools (tools anchor to the step row, carrying their
        act_summary) → assistant FINAL row carrying the synthesis spans. All
        three share the turn_id the input row opened.

        Seeded through the production writers + ActTrail.record — the same path
        the live chain takes — so the refresh path reads exactly what production
        persisTranscript."""
        from services.transcript_service import Transcript
        from services.act_trail import ActTrail
        uid = Transcript.write_input_row("user", "user", user_text)
        turn = Transcript.turn_id_of_row(uid)
        step_id = Transcript.write_assistant_row("user", step_text, turn_id=turn)
        trail = ActTrail()
        for tc in tool_calls:
            trail.record(
                tool_name=cast(str, tc["tool_name"]),
                params=cast(dict[str, object], tc.get("params", {})),
                result=cast(str, tc["result"]),
                transcript_id=step_id,
                summary=cast(str, tc.get("summary", "")),
            )
        final_id = Transcript.write_assistant_row("user", final_text, turn_id=turn)
        return uid, step_id, final_id

    def test_plain_assistant_row_gets_one_text_segment(self, db: sqlite3.Connection) -> None:
        self._seed_plain_turn("Hello", "Hi there!")

        from api.conversation import get_recent_history
        messages, _, _ = get_recent_history(limit=10)

        asst = next(m for m in messages if m["role"] == "assistant")
        assert asst["segments"] == [{"type": "text", "content": "Hi there!"}]
        assert not asst.get("tool_calls")          # no tools on a plain turn

    def test_user_row_has_no_segments_field(self, db: sqlite3.Connection) -> None:
        self._seed_plain_turn("Hello", "Hi!")

        from api.conversation import get_recent_history
        messages, _, _ = get_recent_history(limit=10)

        user_msg = next(m for m in messages if m["role"] == "user")
        assert "segments" not in user_msg

    def test_final_row_pairs_a_rich_segment_from_the_step_rows_tool(self, db: sqlite3.Connection) -> None:
        """Turn-wide resolve: the weather tool ran on the STEP row, the synthesis
        span lives on the FINAL row. The refresh path must reach across the turn
        to pair them — the rich card lands on the final row's segmenTranscript."""
        _, _, final_id = self._seed_tool_turn(
            "What's the weather in London?",
            "Let me check the weather.",
            "Here is the weather. <span id='weather_1'>Partly cloudy, 12°C.</span>",
            [{"tool_name": "weather", "params": {"location": "London"},
              "result": _TOOL_RESULT, "summary": "Checking London weather"}],
        )

        from api.conversation import get_recent_history
        messages, _, _ = get_recent_history(limit=10)

        final = next(m for m in messages if m["id"] == str(final_id))
        rich = [s for s in cast(list[dict[str, object]], final["segments"]) if s["type"] == "rich"]
        assert len(rich) == 1
        assert rich[0]["tag"] == "weather_1"
        assert rich[0]["synthesis"] == "Partly cloudy, 12°C."
        assert cast(dict[str, object], rich[0]["payload"])["location"] == "London, GB"

    def test_step_row_carries_its_own_tool_chip_with_summary(self, db: sqlite3.Connection) -> None:
        """Each assistant row surfaces ONLY the tools it emitted, with the
        ability's persisted act_summary — the blue box the frontend prinTranscript."""
        _, step_id, final_id = self._seed_tool_turn(
            "What's the weather in London?",
            "Let me check the weather.",
            "Here is the weather. <span id='weather_1'>Partly cloudy, 12°C.</span>",
            [{"tool_name": "weather", "params": {"location": "London"},
              "result": _TOOL_RESULT, "summary": "Checking London weather"}],
        )

        from api.conversation import get_recent_history
        messages, _, _ = get_recent_history(limit=10)
        by_id = {m["id"]: m for m in messages}

        step = by_id[str(step_id)]
        assert step["tool_calls"] == [
            {"tool_name": "weather", "summary": "Checking London weather"}
        ]
        # The tool ran on the step row, so the final synthesis row shows no chip.
        assert not by_id[str(final_id)].get("tool_calls")

    def test_content_field_preserved_on_assistant_rows(self, db: sqlite3.Connection) -> None:
        _, final_id = self._seed_plain_turn("Q", "Plain response with no spans.")

        from api.conversation import get_recent_history
        messages, _, _ = get_recent_history(limit=10)

        asst = next(m for m in messages if m["id"] == str(final_id))
        assert asst["content"] == "Plain response with no spans."

    def test_two_city_turn_final_row_has_two_rich_segments(self, db: sqlite3.Connection) -> None:
        _, step_id, final_id = self._seed_tool_turn(
            "Compare London and Tokyo weather",
            "Checking both cities.",
            "London: <span id='weather_1'>Cloudy.</span> "
            "Tokyo: <span id='weather_2'>Clear.</span>",
            [
                {"tool_name": "weather", "params": {"location": "London"},
                 "result": _weather_result("London, GB", 1), "summary": "London"},
                {"tool_name": "weather", "params": {"location": "Tokyo"},
                 "result": _weather_result("Tokyo, JP", 2), "summary": "Tokyo"},
            ],
        )

        from api.conversation import get_recent_history
        messages, _, _ = get_recent_history(limit=10)
        by_id = {m["id"]: m for m in messages}

        final = by_id[str(final_id)]
        rich = [s for s in cast(list[dict[str, object]], final["segments"]) if s["type"] == "rich"]
        assert {s["tag"] for s in rich} == {"weather_1", "weather_2"}
        # Both chips live on the step row that emitted them.
        assert [t["tool_name"] for t in cast(list[dict[str, object]], by_id[str(step_id)]["tool_calls"])] == ["weather", "weather"]

    def test_each_turn_gets_its_own_segments(self, db: sqlite3.Connection) -> None:
        # Turn 1: a weather tool turn (step + final assistant rows).
        self._seed_tool_turn(
            "London weather?",
            "Checking.",
            "<span id='weather_1'>Cloudy.</span>",
            [{"tool_name": "weather", "params": {"location": "London"},
              "result": _weather_result("London, GB", 1), "summary": "x"}],
        )
        # Turn 2: a plain turn (final assistant row only).
        self._seed_plain_turn("Thanks", "You are welcome!")

        from api.conversation import get_recent_history
        messages, _, _ = get_recent_history(limit=10)

        asst = [m for m in messages if m["role"] == "assistant"]
        # turn 1 yields a step + a final row; turn 2 yields one final row.
        assert len(asst) == 3
        assert any(s["type"] == "rich" for m in asst for s in cast(list[dict[str, object]], m["segments"]))
        plain = next(m for m in asst if m["content"] == "You are welcome!")
        assert all(s["type"] == "text" for s in cast(list[dict[str, object]], plain["segments"]))
