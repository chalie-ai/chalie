"""Tests for conversational belief correction — digest_worker hook using DataGraphService.

Real-DB flows:
- DataGraphService is instantiated against the test SQLite fixture
- Assertions verify actual DB state changes (row deleted / value updated)
"""

import pytest
from unittest.mock import patch

pytestmark = pytest.mark.unit


# ── Helpers ────────────────────────────────────────────────────────────────────

def _seed_trait(db_conn, key, value, retrieval_weight=0.8):
    """Insert a trait row directly into the data_graph table."""
    now = '2026-01-01T00:00:00+00:00'
    db_conn.execute(
        "INSERT INTO data_graph (kind, key, value, retrieval_weight, "
        "first_seen_at, last_confirmed_at) "
        "VALUES ('user_specific', ?, ?, ?, ?, ?)",
        (key, value, retrieval_weight, now, now),
    )
    db_conn.commit()


def _get_trait(db_conn, key):
    """Return the live (non-deleted, active) data_graph row for the given key, or None."""
    return db_conn.execute(
        "SELECT id, key, value, retrieval_weight, deleted_at "
        "FROM data_graph "
        "WHERE kind='user_specific' AND key=? AND deleted_at IS NULL AND active=1",
        (key,),
    ).fetchone()


def _reset_dgs_singleton():
    """Clear the DataGraphService process singleton so tests see fresh DB."""
    import services.data_graph_service as _dgs_mod
    with _dgs_mod._instance_lock:
        _dgs_mod._instance = None


# ── Pure pattern guards — no DB needed ────────────────────────────────────────

class TestBeliefCorrectionPatternGuards:
    def test_no_correction_pattern_returns_early(self, db):
        """Messages without correction patterns are ignored — no DB access at all."""
        from workers.digest_worker import _run_belief_correction_hook
        _run_belief_correction_hook("The weather is nice today")

    def test_no_self_reference_returns_early(self, db):
        """Messages matching pattern but without I/me/my are ignored (guardrail 1)."""
        from workers.digest_worker import _run_belief_correction_hook
        _run_belief_correction_hook("Don't assume sushi is good")

    def test_hook_error_does_not_raise(self):
        """Hook failures are swallowed (non-fatal)."""
        from workers.digest_worker import _run_belief_correction_hook
        _reset_dgs_singleton()
        with patch('services.data_graph_service.get_data_graph_service', side_effect=Exception('DB down')):
            # Should not raise
            _run_belief_correction_hook("I don't like sushi")
        # Reset after so the poisoned singleton doesn't leak into subsequent tests
        _reset_dgs_singleton()


# ── Real-DB: negation deletes matching trait ──────────────────────────────────

class TestBeliefCorrectionNegation:
    def test_negation_deletes_matching_trait(self, db):
        """'I don't like sushi' when favourite_food=sushi → row soft-deleted in data_graph."""
        from workers.digest_worker import _run_belief_correction_hook

        _reset_dgs_singleton()
        _seed_trait(db, 'favourite_food', 'sushi', retrieval_weight=0.8)
        assert _get_trait(db, 'favourite_food') is not None

        _run_belief_correction_hook("I don't like sushi")

        # Row must now be soft-deleted (deleted_at set)
        deleted_row = db.execute(
            "SELECT deleted_at FROM data_graph "
            "WHERE kind='user_specific' AND key='favourite_food'",
        ).fetchone()
        assert deleted_row is not None
        assert deleted_row['deleted_at'] is not None, (
            "Expected trait to be soft-deleted after negation"
        )

    def test_low_weight_trait_skipped(self, db):
        """Traits below 0.4 retrieval_weight are not deleted (guardrail 2)."""
        from workers.digest_worker import _run_belief_correction_hook

        _reset_dgs_singleton()
        _seed_trait(db, 'favourite_food', 'sushi', retrieval_weight=0.3)

        _run_belief_correction_hook("I don't like sushi")

        # Row must still be alive — guardrail prevents modification
        row = _get_trait(db, 'favourite_food')
        assert row is not None, "Low-weight trait must not be deleted"

    def test_no_matching_trait_value_no_mutation(self, db):
        """Message negates 'weather' but only 'pizza' is stored — no mutation."""
        from workers.digest_worker import _run_belief_correction_hook

        _reset_dgs_singleton()
        _seed_trait(db, 'favourite_food', 'pizza', retrieval_weight=0.8)

        _run_belief_correction_hook("I don't like weather")

        row = _get_trait(db, 'favourite_food')
        assert row is not None, "Unrelated trait must be untouched"
        assert row['value'] == 'pizza'


# ── Real-DB: replacement corrects trait ───────────────────────────────────────

class TestBeliefCorrectionReplacement:
    def test_replacement_corrects_trait_value_in_db(self, db):
        """'Actually my name is Dylan' when name=dan → new row stored via DataGraphService.

        Patches ContradictionClassifierService to return temporal_change so the
        test doesn't depend on the ONNX contradiction model being present.
        """
        from workers.digest_worker import _run_belief_correction_hook

        _reset_dgs_singleton()
        _seed_trait(db, 'name', 'dan', retrieval_weight=0.9)

        mock_cls = patch(
            'services.contradiction_classifier_service.ContradictionClassifierService.check_new_trait',
            return_value={'classification': 'temporal_change', 'reasoning': 'test'},
        )
        with mock_cls:
            _run_belief_correction_hook("Actually my name is Dylan")

        # DataGraphService.store() inserts a new active row on temporal_change
        row = db.execute(
            "SELECT key, value FROM data_graph "
            "WHERE kind='user_specific' AND key='name' AND active=1 AND deleted_at IS NULL "
            "ORDER BY id DESC LIMIT 1",
        ).fetchone()
        assert row is not None, "A name row must exist after replacement"
        assert 'dylan' in row['value'].lower(), (
            f"Expected 'dylan' in updated value, got {row['value']!r}"
        )

    def test_replacement_value_capped_at_3_words(self, db):
        """Replacement captures at most 3 words — trailing clause is not included."""
        from workers.digest_worker import _run_belief_correction_hook

        _reset_dgs_singleton()
        _seed_trait(db, 'name', 'dan', retrieval_weight=0.9)

        mock_cls = patch(
            'services.contradiction_classifier_service.ContradictionClassifierService.check_new_trait',
            return_value={'classification': 'temporal_change', 'reasoning': 'test'},
        )
        with mock_cls:
            _run_belief_correction_hook("Actually my name is Dylan by the way")

        row = db.execute(
            "SELECT value FROM data_graph "
            "WHERE kind='user_specific' AND key='name' AND active=1 AND deleted_at IS NULL "
            "ORDER BY id DESC LIMIT 1",
        ).fetchone()
        assert row is not None
        new_value = row['value']
        assert len(new_value.split()) <= 3, (
            f"Expected ≤ 3 words in replacement value, got {new_value!r}"
        )
