"""Tests for conversational belief correction — digest_worker hook using KnowledgeService.

Hollow mock tests replaced with real-DB flows:
- KnowledgeService is instantiated against the test SQLite fixture
- Assertions verify actual DB state changes (row deleted / value updated)
  rather than checking mock call counts
"""

import pytest

pytestmark = pytest.mark.unit


# ── Helpers ────────────────────────────────────────────────────────────────────

def _seed_trait(db_conn, key, value, confidence=0.8):
    """Insert a trait row directly into the knowledge table."""
    db_conn.execute(
        "INSERT INTO knowledge (kind, entity, key, value, confidence, decay_class) "
        "VALUES ('trait', 'user', ?, ?, ?, 'standard')",
        (key, value, confidence),
    )
    db_conn.commit()


def _get_trait(db_conn, key):
    """Return the live (non-deleted) knowledge row for the given key, or None."""
    return db_conn.execute(
        "SELECT key, value, confidence, deleted_at "
        "FROM knowledge "
        "WHERE entity='user' AND key=? AND deleted_at IS NULL",
        (key,),
    ).fetchone()


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
        from unittest.mock import patch
        with patch('services.database_service.get_shared_db_service', side_effect=Exception('DB down')):
            # Should not raise
            _run_belief_correction_hook("I don't like sushi")


# ── Real-DB: negation deletes matching trait ──────────────────────────────────

class TestBeliefCorrectionNegation:
    def test_negation_deletes_matching_trait(self, db):
        """'I don't like sushi' when favourite_food=sushi → row soft-deleted in DB."""
        from workers.digest_worker import _run_belief_correction_hook

        _seed_trait(db, 'favourite_food', 'sushi', confidence=0.8)
        assert _get_trait(db, 'favourite_food') is not None

        _run_belief_correction_hook("I don't like sushi")

        # Row must now be soft-deleted (deleted_at set)
        deleted_row = db.execute(
            "SELECT deleted_at FROM knowledge WHERE entity='user' AND key='favourite_food'",
        ).fetchone()
        assert deleted_row is not None
        assert deleted_row['deleted_at'] is not None, (
            "Expected trait to be soft-deleted after negation"
        )

    def test_low_confidence_trait_skipped(self, db):
        """Traits below 0.4 confidence are not deleted (guardrail 2)."""
        from workers.digest_worker import _run_belief_correction_hook

        _seed_trait(db, 'favourite_food', 'sushi', confidence=0.3)

        _run_belief_correction_hook("I don't like sushi")

        # Row must still be alive — guardrail prevents modification
        row = _get_trait(db, 'favourite_food')
        assert row is not None, "Low-confidence trait must not be deleted"

    def test_no_matching_trait_value_no_mutation(self, db):
        """Message negates 'weather' but only 'pizza' is stored — no mutation."""
        from workers.digest_worker import _run_belief_correction_hook

        _seed_trait(db, 'favourite_food', 'pizza', confidence=0.8)

        _run_belief_correction_hook("I don't like weather")

        row = _get_trait(db, 'favourite_food')
        assert row is not None, "Unrelated trait must be untouched"
        assert row['value'] == 'pizza'


# ── Real-DB: replacement corrects trait ───────────────────────────────────────

class TestBeliefCorrectionReplacement:
    def test_replacement_corrects_trait_value_in_db(self, db):
        """'Actually my name is Dylan' when name=Dan → value updated to 'dylan' in DB."""
        from workers.digest_worker import _run_belief_correction_hook

        _seed_trait(db, 'name', 'dan', confidence=0.9)

        _run_belief_correction_hook("Actually my name is Dylan")

        row = _get_trait(db, 'name')
        assert row is not None, "Trait row must survive a value replacement"
        assert 'dylan' in row['value'].lower(), (
            f"Expected 'dylan' in updated value, got {row['value']!r}"
        )

    def test_replacement_value_capped_at_3_words(self, db):
        """Replacement captures at most 3 words — trailing clause is not included."""
        from workers.digest_worker import _run_belief_correction_hook

        _seed_trait(db, 'name', 'dan', confidence=0.9)

        _run_belief_correction_hook("Actually my name is Dylan by the way")

        row = _get_trait(db, 'name')
        assert row is not None
        new_value = row['value']
        assert len(new_value.split()) <= 3, (
            f"Expected ≤ 3 words in replacement value, got {new_value!r}"
        )
