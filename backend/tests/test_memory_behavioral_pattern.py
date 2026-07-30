"""Unit tests for the memory ability's recall projection.

``_recall_payload()`` projects every recall hit into a structured row that
carries its own kind field — pure-function, deterministic and embedder-free,
matching the unit tier.

(behavioral_pattern is deliberately not a semantic-recall lane: patterns reach
the model deterministically every turn, so vec-indexing a value rewritten each
turn would be churn. That is a declaration in the model — ``__search__ = None``
— not a behaviour, so it is enforced where it is declared rather than restated
here.)
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
        from services.memory_service import MemoryService

        hit = self._make_hit(kind="user_specific", key="residence", text="Valletta")
        rows = MemoryService._recall_payload([hit])

        assert len(rows) == 1
        row = rows[0]
        assert row["kind"] == "user_specific"
        assert row["id"] == "residence"
        assert row["content"] == "Valletta"
        assert row["score"] == "high"

    def test_mixed_results_each_keep_their_own_kind(self) -> None:
        """In a mixed list every row reports its own kind verbatim."""
        from services.memory_service import MemoryService

        hits = [
            self._make_hit(kind="discovery", key="new_cafe",
                           text="A new café opened near the office"),
            self._make_hit(kind="user_specific", key="food_preference", text="pasta"),
        ]
        rows = MemoryService._recall_payload(hits)

        assert len(rows) == 2
        by_id = {r["id"]: r for r in rows}
        assert by_id["new_cafe"]["kind"] == "discovery"
        assert by_id["food_preference"]["kind"] == "user_specific"
