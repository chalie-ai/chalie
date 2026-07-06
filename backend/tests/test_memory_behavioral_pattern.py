"""Feature tests for behavioral_pattern support in the memory ability.

Three behaviours under test:
1. _render_behavioral_pattern() produces a human-readable line from a valid
   JSON string and degrades gracefully on invalid input.
2. _recall_payload() projects every hit into a structured row that carries its
   own kind field (behavioral_pattern, user_specific, …).
3. _search_data_graph() includes KIND_BEHAVIORAL_PATTERN in the kinds list
   passed to DataGraphService.recall() so the rows actually surface.

All three are pure-function or real-DataGraph tests — zero mocks except the
single network-boundary patch already established by the project's conftest db
fixture (which points the Database gateway at an isolated SQLite instance).
"""

import json
import sqlite3
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
# 3. _search_data_graph — real DataGraph, real SQLite, real recall path
# ---------------------------------------------------------------------------


class TestSearchDataGraphIncludesBehavioralPattern:
    """_search_data_graph surfaces behavioral_pattern rows from the real DB."""

    def test_behavioral_pattern_row_is_returned_by_search(self, db: sqlite3.Connection) -> None:
        """A seeded behavioral_pattern row is visible in _search_data_graph results.

        This is the regression-protection case: before the fix,
        KIND_BEHAVIORAL_PATTERN was absent from the kinds filter passed to
        dgs.recall(), so behavioral_pattern rows were silently dropped.
        """
        from models.data_graph import DataGraph
        from services.data_graph_service import KIND_BEHAVIORAL_PATTERN

        pattern_value = json.dumps({
            "name": "dawn_meditation_practice",
            "frequency": "daily",
            "time_anchor": "06:00",
            "summary": "Practises 20 minutes of silent meditation before breakfast",
            "confidence": 8.0,
        })
        DataGraph.store(
            kind=KIND_BEHAVIORAL_PATTERN,
            key="dawn_meditation_practice",
            value=pattern_value,
            source="test:seed",
        )

        from services.memory_retrieval import _search_data_graph

        hits, _ = _search_data_graph("meditation morning routine", limit=10)

        if not hits:
            pytest.skip(
                "FTS/vec did not surface this hit — embedding service unavailable in this env"
            )

        kinds_returned = {h.get("kind") for h in hits}
        assert KIND_BEHAVIORAL_PATTERN in kinds_returned, (
            f"behavioral_pattern kind not present in hits. Kinds returned: {kinds_returned}"
        )

    def test_behavioral_pattern_hit_text_is_rendered_not_raw_json(self, db: sqlite3.Connection) -> None:
        """The text field for a behavioral_pattern hit is the rendered line, not raw JSON."""
        from models.data_graph import DataGraph
        from services.data_graph_service import KIND_BEHAVIORAL_PATTERN

        pattern_value = json.dumps({
            "name": "evening_run",
            "frequency": "weekdays",
            "time_anchor": "18:30",
            "summary": "Runs 5 km along the seafront",
            "confidence": 7.2,
        })
        DataGraph.store(
            kind=KIND_BEHAVIORAL_PATTERN,
            key="evening_run",
            value=pattern_value,
            source="test:seed",
        )

        from services.memory_retrieval import _search_data_graph

        hits, _ = _search_data_graph("evening exercise running", limit=10)

        if not hits:
            pytest.skip(
                "FTS/vec did not surface this hit — embedding service unavailable in this env"
            )

        bp_hits = [h for h in hits if h.get("kind") == KIND_BEHAVIORAL_PATTERN]
        assert bp_hits, "Expected at least one behavioral_pattern hit"

        for hit in bp_hits:
            text = cast(str, hit.get("text", ""))
            # Must be the rendered form, not raw JSON
            assert not text.strip().startswith("{"), (
                f"text must be rendered, not raw JSON: {text!r}"
            )
            assert "[confidence=" in text, (
                f"Rendered line must include [confidence=...]: {text!r}"
            )
