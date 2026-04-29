"""Feature tests for the memory ability's cos_score → relevance labelling path.

_search_data_graph reads cos_score from data_graph_service.recall() rows and
passes it to _relevance_label().  Before the fix (commit 9a375ad),
retrieval_weight always defaulted to 1.0 so every row was labelled
relevance:high regardless of semantic similarity.

These tests assert the post-fix behaviour: the relevance label in the
formatted output is driven by cos_score, not retrieval_weight.

Mock boundary: services.data_graph_service.get_data_graph_service is patched
to return a stub service whose recall() yields a controlled row.  All memory.py
logic (reading cos_score, calling _relevance_label, formatting the result) runs
against the real production code.
"""

from unittest.mock import MagicMock, patch

import pytest

from abilities.memory import _search_data_graph

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_dgs(recall_rows):
    """Return a MagicMock DataGraphService whose recall() returns recall_rows."""
    fake = MagicMock()
    fake.recall.return_value = recall_rows
    return fake


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMemoryRecallRelevanceLabel:
    """_search_data_graph labels hits by cos_score, not retrieval_weight."""

    def test_low_cos_score_yields_relevance_low_regardless_of_retrieval_weight(self):
        """Row with cos_score=0.3 and retrieval_weight=10.0 must produce relevance:low.

        This is the regression-protection case: the old code used
        retrieval_weight (always ~1.0) so every row was relevance:high.
        After the fix, only cos_score drives the label.
        """
        row = {
            "key": "dentist_appointment",
            "value": "Tuesday at 3pm",
            "cos_score": 0.3,
            "retrieval_weight": 10.0,
        }

        with patch(
            "services.data_graph_service.get_data_graph_service",
            return_value=_mock_dgs([row]),
        ):
            hits, _ = _search_data_graph("dentist", limit=5)

        assert hits, "Expected at least one hit"
        hit = hits[0]
        assert hit["relevance"] == "low", (
            f"cos_score=0.3 must yield relevance:low, got {hit['relevance']!r}"
        )
        assert hit["confidence"] == pytest.approx(0.3), (
            f"confidence must equal cos_score=0.3, got {hit['confidence']}"
        )

    def test_high_cos_score_yields_relevance_high(self):
        """Row with cos_score=0.85 must produce relevance:high."""
        row = {
            "key": "home_city",
            "value": "Valletta",
            "cos_score": 0.85,
            "retrieval_weight": 0.5,
        }

        with patch(
            "services.data_graph_service.get_data_graph_service",
            return_value=_mock_dgs([row]),
        ):
            hits, _ = _search_data_graph("home city location", limit=5)

        assert hits
        assert hits[0]["relevance"] == "high", (
            f"cos_score=0.85 must yield relevance:high, got {hits[0]['relevance']!r}"
        )

    def test_medium_cos_score_yields_relevance_medium(self):
        """Row with cos_score=0.55 must produce relevance:medium."""
        row = {
            "key": "preferred_language",
            "value": "English",
            "cos_score": 0.55,
            "retrieval_weight": 1.0,
        }

        with patch(
            "services.data_graph_service.get_data_graph_service",
            return_value=_mock_dgs([row]),
        ):
            hits, _ = _search_data_graph("language preference", limit=5)

        assert hits
        assert hits[0]["relevance"] == "medium", (
            f"cos_score=0.55 must yield relevance:medium, got {hits[0]['relevance']!r}"
        )
