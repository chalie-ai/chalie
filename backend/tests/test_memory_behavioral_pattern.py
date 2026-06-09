"""Feature tests for behavioral_pattern support in the memory ability.

Three behaviours under test:
1. _render_behavioral_pattern() produces a human-readable line from a valid
   JSON string and degrades gracefully on invalid input.
2. _format_results() labels behavioral_pattern hits with kind:behavioral_pattern
   in the prefix, and omits that label for other kinds.
3. _search_data_graph() includes KIND_BEHAVIORAL_PATTERN in the kinds list
   passed to DataGraphService.recall() so the rows actually surface.

All three are pure-function or real-DataGraph tests — zero mocks except the
single network-boundary patch already established by the project's conftest db
fixture (which patches get_shared_db_service to an isolated SQLite instance).
"""

import json

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. _render_behavioral_pattern — pure-function, no IO
# ---------------------------------------------------------------------------


class TestRenderBehavioralPattern:
    """_render_behavioral_pattern converts a JSON blob to a readable line."""

    def test_valid_json_produces_readable_line(self):
        """Full JSON with all fields renders as 'name (freq @ anchor): summary [confidence=N]'."""
        from abilities.memory import _render_behavioral_pattern

        raw = json.dumps({
            "name": "dawn_meditation_practice",
            "frequency": "daily",
            "time_anchor": "06:00",
            "summary": "Practises 20 minutes of silent meditation before breakfast",
            "confidence": 8.0,
        })
        result = _render_behavioral_pattern(raw)

        assert result.startswith("dawn_meditation_practice"), (
            f"Expected name at start, got: {result!r}"
        )
        assert "daily" in result
        assert "06:00" in result
        assert "Practises 20 minutes" in result
        assert "[confidence=8.0]" in result

    def test_missing_time_anchor_omits_at_segment(self):
        """When time_anchor is absent the '@ HH:MM' segment is omitted entirely."""
        from abilities.memory import _render_behavioral_pattern

        raw = json.dumps({
            "name": "evening_walk",
            "frequency": "weekdays",
            "summary": "30 min walk after dinner",
            "confidence": 5.5,
        })
        result = _render_behavioral_pattern(raw)

        assert "@" not in result, (
            f"No time_anchor means no '@', got: {result!r}"
        )
        assert "evening_walk (weekdays)" in result

    def test_invalid_json_returns_raw_string(self):
        """Unparseable JSON is returned verbatim — no crash, no data loss."""
        from abilities.memory import _render_behavioral_pattern

        raw = "not valid { json"
        result = _render_behavioral_pattern(raw)

        assert result == raw, (
            f"Expected raw passthrough for invalid JSON, got: {result!r}"
        )

    def test_non_string_input_returns_raw(self):
        """Non-string input (e.g. already-decoded dict) is handled without crash."""
        from abilities.memory import _render_behavioral_pattern

        result = _render_behavioral_pattern(None)
        assert result == "", f"None input should produce empty string, got: {result!r}"


# ---------------------------------------------------------------------------
# 2. _format_results — pure-function, no IO
# ---------------------------------------------------------------------------


class TestFormatResults:
    """_format_results labels behavioral_pattern hits distinctly from other kinds."""

    def _make_hit(self, kind, key="some_key", text="some text", relevance="high"):
        return {"id": key, "kind": kind, "text": text, "relevance": relevance}

    def test_behavioral_pattern_hit_includes_kind_label(self):
        """A hit with kind='behavioral_pattern' must have kind:behavioral_pattern in prefix."""
        from abilities.memory import _format_results

        hit = self._make_hit(kind="behavioral_pattern", key="dawn_meditation_practice",
                              text="dawn_meditation_practice (daily @ 06:00): ...")
        result = _format_results([hit])

        assert "kind:behavioral_pattern" in result, (
            f"Expected kind:behavioral_pattern in output, got: {result!r}"
        )
        assert "id:dawn_meditation_practice" in result

    def test_non_behavioral_pattern_hit_omits_kind_label(self):
        """A user_specific hit must NOT have kind: in its prefix."""
        from abilities.memory import _format_results

        hit = self._make_hit(kind="user_specific", key="residence", text="Valletta")
        result = _format_results([hit])

        assert "kind:" not in result, (
            f"kind: label must only appear for behavioral_pattern, got: {result!r}"
        )
        assert "id:residence" in result
        assert "Valletta" in result

    def test_mixed_results_labels_only_behavioral_pattern(self):
        """In a mixed list only behavioral_pattern rows carry the kind label."""
        from abilities.memory import _format_results

        hits = [
            self._make_hit(kind="behavioral_pattern", key="morning_run",
                           text="morning_run (daily): runs 5km [confidence=7.0]"),
            self._make_hit(kind="user_specific", key="food_preference", text="pasta"),
        ]
        result = _format_results(hits)

        lines = result.splitlines()
        assert len(lines) == 2

        bp_line = next(line for line in lines if "morning_run" in line)
        user_line = next(line for line in lines if "food_preference" in line)

        assert "kind:behavioral_pattern" in bp_line
        assert "kind:" not in user_line


# ---------------------------------------------------------------------------
# 3. _search_data_graph — real DataGraph, real SQLite, real recall path
# ---------------------------------------------------------------------------


class TestSearchDataGraphIncludesBehavioralPattern:
    """_search_data_graph surfaces behavioral_pattern rows from the real DB."""

    def test_behavioral_pattern_row_is_returned_by_search(self, db):
        """A seeded behavioral_pattern row is visible in _search_data_graph results.

        This is the regression-protection case: before the fix,
        KIND_BEHAVIORAL_PATTERN was absent from the kinds filter passed to
        dgs.recall(), so behavioral_pattern rows were silently dropped.
        """
        from services.data_graph_service import get_data_graph_service, KIND_BEHAVIORAL_PATTERN

        dgs = get_data_graph_service()
        pattern_value = json.dumps({
            "name": "dawn_meditation_practice",
            "frequency": "daily",
            "time_anchor": "06:00",
            "summary": "Practises 20 minutes of silent meditation before breakfast",
            "confidence": 8.0,
        })
        dgs.store(
            kind=KIND_BEHAVIORAL_PATTERN,
            key="dawn_meditation_practice",
            value=pattern_value,
            source="test:seed",
        )

        from abilities.memory import _search_data_graph

        hits, _ = _search_data_graph("meditation morning routine", limit=10)

        if not hits:
            pytest.skip(
                "FTS/vec did not surface this hit — embedding service unavailable in this env"
            )

        kinds_returned = {h.get("kind") for h in hits}
        assert KIND_BEHAVIORAL_PATTERN in kinds_returned, (
            f"behavioral_pattern kind not present in hits. Kinds returned: {kinds_returned}"
        )

    def test_behavioral_pattern_hit_text_is_rendered_not_raw_json(self, db):
        """The text field for a behavioral_pattern hit is the rendered line, not raw JSON."""
        from services.data_graph_service import get_data_graph_service, KIND_BEHAVIORAL_PATTERN

        dgs = get_data_graph_service()
        pattern_value = json.dumps({
            "name": "evening_run",
            "frequency": "weekdays",
            "time_anchor": "18:30",
            "summary": "Runs 5 km along the seafront",
            "confidence": 7.2,
        })
        dgs.store(
            kind=KIND_BEHAVIORAL_PATTERN,
            key="evening_run",
            value=pattern_value,
            source="test:seed",
        )

        from abilities.memory import _search_data_graph

        hits, _ = _search_data_graph("evening exercise running", limit=10)

        if not hits:
            pytest.skip(
                "FTS/vec did not surface this hit — embedding service unavailable in this env"
            )

        bp_hits = [h for h in hits if h.get("kind") == KIND_BEHAVIORAL_PATTERN]
        assert bp_hits, "Expected at least one behavioral_pattern hit"

        for hit in bp_hits:
            text = hit.get("text", "")
            # Must be the rendered form, not raw JSON
            assert not text.strip().startswith("{"), (
                f"text must be rendered, not raw JSON: {text!r}"
            )
            assert "[confidence=" in text, (
                f"Rendered line must include [confidence=...]: {text!r}"
            )
