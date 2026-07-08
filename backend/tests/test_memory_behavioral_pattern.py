"""Unit tests for behavioral_pattern's place in the memory ability.

Two behaviours under test:
1. _recall_payload() projects every recall hit into a structured row that
   carries its own kind field.
2. behavioral_pattern is deliberately NOT a semantic-recall lane — it is
   surfaced deterministically (PromptService.patterns / top_patterns), so it is
   excluded from both the memory tool's _DG_RECALL_KINDS and the recall
   service's _VERTICALS, and from the search-index write pipeline. This guard
   keeps the recall span and the indexing gate in agreement.

All are pure-function or constant-wiring assertions: deterministic and
embedder-free, matching the unit tier.
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. _recall_payload — pure-function projection, no IO
# ---------------------------------------------------------------------------


class TestRecallPayload:

    def _make_hit(self, kind: str, key: str = "some_key", text: str = "some text", relevance: str = "high") -> dict[str, object]:
        return {"id": key, "kind": kind, "text": text, "relevance": relevance}

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
        assert row["score"] == "high"

    def test_mixed_results_each_keep_their_own_kind(self) -> None:
        """In a mixed list every row reports its own kind verbatim."""
        from services.memory_retrieval import _recall_payload

        hits = [
            self._make_hit(kind="discovery", key="new_cafe",
                           text="A new café opened near the office"),
            self._make_hit(kind="user_specific", key="food_preference", text="pasta"),
        ]
        rows = _recall_payload(hits)

        assert len(rows) == 2
        by_id = {r["id"]: r for r in rows}
        assert by_id["new_cafe"]["kind"] == "discovery"
        assert by_id["food_preference"]["kind"] == "user_specific"


# ---------------------------------------------------------------------------
# 2. behavioral_pattern is surfaced deterministically, never semantically
# ---------------------------------------------------------------------------


class TestBehavioralPatternNotInRecallSpan:
    """behavioral_pattern must NOT be a semantic-recall lane. Patterns reach the
    model deterministically every turn (PromptService.patterns / top_patterns),
    which carries the full active set — a semantic recall over patterns is
    redundant, and vec-indexing a value that is rewritten every turn (confidence
    decays) is pure churn. These assert the recall span and the indexing gate
    agree, so nobody re-introduces a span entry that the write side starves."""

    def test_tool_recall_span_excludes_behavioral_pattern(self) -> None:
        """The memory tool's cross-kind data-graph recall span omits it."""
        from models.behavioral_pattern import BehavioralPattern
        from services.memory_retrieval import _DG_RECALL_KINDS

        assert BehavioralPattern.KIND not in _DG_RECALL_KINDS, (
            f"behavioral_pattern must not be a recall lane: {_DG_RECALL_KINDS}"
        )

    def test_recall_service_does_not_fuse_behavioral_pattern_vertical(self) -> None:
        """The modelless recall service does not fuse it as a vertical."""
        from models.behavioral_pattern import BehavioralPattern
        from services.memory_recall_service import _VERTICALS

        assert BehavioralPattern not in _VERTICALS, (
            f"behavioral_pattern must not be a recall vertical: {_VERTICALS}"
        )

    def test_indexing_gate_excludes_behavioral_pattern(self) -> None:
        """The write-side index gate agrees: the kind earns no FTS/vec posting."""
        from models.behavioral_pattern import BehavioralPattern
        from services.search_expander_service import should_fts_index

        assert should_fts_index(BehavioralPattern.KIND) is False, (
            "behavioral_pattern must stay out of the search-index pipeline so "
            "the indexing gate and the (empty) recall span never disagree."
        )
