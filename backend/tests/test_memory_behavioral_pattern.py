"""Unit tests for behavioral_pattern support in the memory ability.

Three behaviours under test:
1. BehavioralPattern.render() produces a human-readable line from a valid JSON
   string and degrades gracefully on invalid input.
2. _recall_payload() projects every hit into a structured row that carries its
   own kind field (behavioral_pattern, user_specific, …).
3. behavioral_pattern stays in the cross-kind recall span — both the memory
   tool's _DG_RECALL_KINDS and the recall service's _VERTICALS — so its rows are
   never silently dropped from recall.

All are pure-function or constant-wiring assertions: deterministic and
embedder-free, matching the unit tier.
"""

import json
from typing import cast

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. _render_behavioral_pattern — pure-function, no IO
# ---------------------------------------------------------------------------


class TestRenderBehavioralPattern:

    def test_valid_json_produces_readable_line(self) -> None:
        """Full JSON with all fields renders as 'name (freq @ anchor): summary [confidence=N]'."""
        from models.behavioral_pattern import BehavioralPattern

        raw = json.dumps({
            "name": "dawn_meditation_practice",
            "frequency": "daily",
            "time_anchor": "06:00",
            "summary": "Practises 20 minutes of silent meditation before breakfast",
            "confidence": 8.0,
        })
        result = BehavioralPattern.render(raw)

        assert result.startswith("dawn_meditation_practice"), (
            f"Expected name at start, got: {result!r}"
        )
        assert "daily" in result
        assert "06:00" in result
        assert "Practises 20 minutes" in result
        assert "[confidence=8.0]" in result

    def test_missing_time_anchor_omits_at_segment(self) -> None:
        """When time_anchor is absent the '@ HH:MM' segment is omitted entirely."""
        from models.behavioral_pattern import BehavioralPattern

        raw = json.dumps({
            "name": "evening_walk",
            "frequency": "weekdays",
            "summary": "30 min walk after dinner",
            "confidence": 5.5,
        })
        result = BehavioralPattern.render(raw)

        assert "@" not in result, (
            f"No time_anchor means no '@', got: {result!r}"
        )
        assert "evening_walk (weekdays)" in result

    def test_invalid_json_returns_raw_string(self) -> None:
        """Unparseable JSON is returned verbatim — no crash, no data loss."""
        from models.behavioral_pattern import BehavioralPattern

        raw = "not valid { json"
        result = BehavioralPattern.render(raw)

        assert result == raw, (
            f"Expected raw passthrough for invalid JSON, got: {result!r}"
        )

    def test_non_string_input_returns_raw(self) -> None:
        """Non-string input (e.g. already-decoded dict) is handled without crash."""
        from models.behavioral_pattern import BehavioralPattern

        result = BehavioralPattern.render(cast(str, None))
        assert result == "", f"None input should produce empty string, got: {result!r}"


# ---------------------------------------------------------------------------
# 2. _recall_payload — pure-function projection, no IO
# ---------------------------------------------------------------------------


class TestRecallPayload:

    def _make_hit(self, kind: str, key: str = "some_key", text: str = "some text", relevance: str = "high") -> dict[str, object]:
        return {"id": key, "kind": kind, "text": text, "relevance": relevance}

    def test_behavioral_pattern_row_carries_kind_field(self) -> None:
        """A behavioral_pattern hit projects to a row whose kind field is set."""
        from services.memory_retrieval import _recall_payload

        hit = self._make_hit(kind="behavioral_pattern", key="dawn_meditation_practice",
                              text="dawn_meditation_practice (daily @ 06:00): ...")
        rows = _recall_payload([hit])

        assert len(rows) == 1
        row = rows[0]
        assert row["kind"] == "behavioral_pattern"
        assert row["id"] == "dawn_meditation_practice"
        assert row["content"] == "dawn_meditation_practice (daily @ 06:00): ..."
        assert row["score"] == "high"

    def test_every_row_carries_its_kind_field(self) -> None:
        """Unlike the old prose format, kind is a first-class field on EVERY row."""
        from services.memory_retrieval import _recall_payload

        hit = self._make_hit(kind="user_specific", key="residence", text="Valletta")
        rows = _recall_payload([hit])

        assert len(rows) == 1
        row = rows[0]
        assert row["kind"] == "user_specific"
        assert row["id"] == "residence"
        assert row["content"] == "Valletta"

    def test_mixed_results_each_keep_their_own_kind(self) -> None:
        """In a mixed list every row reports its own kind verbatim."""
        from services.memory_retrieval import _recall_payload

        hits = [
            self._make_hit(kind="behavioral_pattern", key="morning_run",
                           text="morning_run (daily): runs 5km [confidence=7.0]"),
            self._make_hit(kind="user_specific", key="food_preference", text="pasta"),
        ]
        rows = _recall_payload(hits)

        assert len(rows) == 2
        by_id = {r["id"]: r for r in rows}
        assert by_id["morning_run"]["kind"] == "behavioral_pattern"
        assert by_id["food_preference"]["kind"] == "user_specific"


# ---------------------------------------------------------------------------
# 3. behavioral_pattern stays in the cross-kind recall span (the wiring guard)
# ---------------------------------------------------------------------------


class TestBehavioralPatternInRecallSpan:
    """behavioral_pattern must remain a recall lane. The live data proves it is
    a valuable, fully-indexed kind (rich cosine-searchable summary prose), so
    dropping it from the span is a silent regression. These assert the wiring
    directly — deterministic and embedder-free — unlike an end-to-end surfacing
    check, which can only skip in an env with no embedding service and so proves
    nothing in the unit tier."""

    def test_tool_recall_span_lists_behavioral_pattern(self) -> None:
        """The memory tool's cross-kind data-graph recall span includes it."""
        from models.behavioral_pattern import BehavioralPattern
        from services.memory_retrieval import _DG_RECALL_KINDS

        assert BehavioralPattern.KIND in _DG_RECALL_KINDS, (
            f"behavioral_pattern dropped from the tool recall span: {_DG_RECALL_KINDS}"
        )

    def test_recall_service_fuses_behavioral_pattern_vertical(self) -> None:
        """The modelless recall service fuses it as one of its verticals."""
        from models.behavioral_pattern import BehavioralPattern
        from services.memory_recall_service import _VERTICALS

        assert BehavioralPattern in _VERTICALS, (
            f"behavioral_pattern vertical missing from the recall span: {_VERTICALS}"
        )
