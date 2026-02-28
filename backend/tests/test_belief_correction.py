"""Tests for conversational belief correction — UserTraitService and digest_worker hook."""

import pytest
from unittest.mock import MagicMock, patch, call

pytestmark = pytest.mark.unit


# ── UserTraitService.correct_trait ────────────────────────────────────────────

class TestCorrectTrait:
    def _make_service(self, existing=None):
        """Build a UserTraitService with a mocked database."""
        from services.user_trait_service import UserTraitService
        db = MagicMock()
        conn_ctx = MagicMock()
        conn = MagicMock()
        cursor = MagicMock()
        db.connection.return_value.__enter__ = lambda *a: conn
        db.connection.return_value.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = cursor
        # existing row: (id, category) or None
        cursor.fetchone.return_value = existing
        return UserTraitService(db), cursor

    def test_correct_existing_trait_updates_row(self):
        """correct_trait() updates the row when trait already exists."""
        svc, cursor = self._make_service(existing=(1, 'preference'))

        with patch.object(svc, '_generate_embedding_raw', return_value='[0.1,0.2]'):
            result = svc.correct_trait('favourite_food', 'ramen', user_id='primary')

        assert result is True
        # Should have called UPDATE
        update_call = cursor.execute.call_args_list[-1]
        sql = update_call[0][0]
        assert 'UPDATE user_traits' in sql
        assert 'explicit_correction' in sql

    def test_correct_new_trait_inserts_row(self):
        """correct_trait() inserts when trait does not exist."""
        svc, cursor = self._make_service(existing=None)

        with patch.object(svc, '_generate_embedding_raw', return_value='[0.1,0.2]'):
            result = svc.correct_trait('name', 'Dylan', user_id='primary')

        assert result is True
        insert_call = cursor.execute.call_args_list[-1]
        sql = insert_call[0][0]
        assert 'INSERT INTO user_traits' in sql
        assert 'explicit_correction' in sql

    def test_correct_trait_sets_confidence_095(self):
        """correct_trait() always sets confidence to 0.95."""
        svc, cursor = self._make_service(existing=(1, 'core'))

        with patch.object(svc, '_generate_embedding_raw', return_value=None):
            svc.correct_trait('name', 'Dylan', user_id='primary')

        # Confidence 0.95 must appear in the UPDATE args
        update_call = cursor.execute.call_args_list[-1]
        args = update_call[0][1]
        assert 0.95 in args

    def test_correct_trait_preserves_existing_category_when_not_specified(self):
        """correct_trait() preserves existing category if none is supplied."""
        svc, cursor = self._make_service(existing=(1, 'preference'))

        with patch.object(svc, '_generate_embedding_raw', return_value=None):
            svc.correct_trait('favourite_food', 'ramen')

        update_call = cursor.execute.call_args_list[-1]
        args = update_call[0][1]
        # 'preference' (existing category) should be in the args
        assert 'preference' in args

    def test_correct_trait_returns_false_on_db_error(self):
        """correct_trait() returns False when database raises."""
        from services.user_trait_service import UserTraitService
        db = MagicMock()
        db.connection.side_effect = Exception('DB is down')
        svc = UserTraitService(db)

        result = svc.correct_trait('name', 'Dylan')
        assert result is False

    def test_high_confidence_trait_overwritten(self):
        """correct_trait() overwrites even when existing confidence > 0.5 (bypasses >2x gate)."""
        # This tests the key design: >2x threshold is NOT applied here
        svc, cursor = self._make_service(existing=(1, 'core'))

        with patch.object(svc, '_generate_embedding_raw', return_value=None):
            result = svc.correct_trait('name', 'Dylan', user_id='primary')

        assert result is True
        update_call = cursor.execute.call_args_list[-1]
        assert 'UPDATE user_traits' in update_call[0][0]


# ── UserTraitService.delete_trait ─────────────────────────────────────────────

class TestDeleteTrait:
    def _make_service(self, rowcount=1):
        from services.user_trait_service import UserTraitService
        db = MagicMock()
        conn = MagicMock()
        cursor = MagicMock()
        cursor.rowcount = rowcount
        db.connection.return_value.__enter__ = lambda *a: conn
        db.connection.return_value.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = cursor
        return UserTraitService(db), cursor

    def test_delete_trait_returns_true_when_deleted(self):
        svc, cursor = self._make_service(rowcount=1)
        assert svc.delete_trait('favourite_food') is True

    def test_delete_trait_returns_false_when_not_found(self):
        svc, cursor = self._make_service(rowcount=0)
        assert svc.delete_trait('nonexistent') is False

    def test_delete_trait_executes_correct_sql(self):
        svc, cursor = self._make_service(rowcount=1)
        svc.delete_trait('favourite_food', user_id='primary')
        sql = cursor.execute.call_args[0][0]
        assert 'DELETE FROM user_traits' in sql
        assert 'trait_key' in sql

    def test_delete_trait_returns_false_on_db_error(self):
        from services.user_trait_service import UserTraitService
        db = MagicMock()
        db.connection.side_effect = Exception('DB is down')
        svc = UserTraitService(db)
        assert svc.delete_trait('name') is False


# ── UserTraitService.get_all_traits ───────────────────────────────────────────

class TestGetAllTraits:
    def test_returns_list_of_dicts(self):
        from services.user_trait_service import UserTraitService
        db = MagicMock()
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = [
            ('name', 'Dylan', 0.95, 'core'),
            ('favourite_food', 'ramen', 0.8, 'preference'),
        ]
        db.connection.return_value.__enter__ = lambda *a: conn
        db.connection.return_value.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = cursor
        svc = UserTraitService(db)

        result = svc.get_all_traits()
        assert len(result) == 2
        assert result[0]['trait_key'] == 'name'
        assert result[0]['trait_value'] == 'Dylan'
        assert result[0]['confidence'] == 0.95

    def test_returns_empty_list_on_error(self):
        from services.user_trait_service import UserTraitService
        db = MagicMock()
        db.connection.side_effect = Exception('DB is down')
        svc = UserTraitService(db)
        assert svc.get_all_traits() == []


# ── _run_belief_correction_hook ───────────────────────────────────────────────

class TestBeliefCorrectionHook:
    def test_no_correction_pattern_returns_early(self):
        """Messages without correction patterns are ignored — no DB access at all."""
        from workers.digest_worker import _run_belief_correction_hook
        with patch('services.database_service.get_shared_db_service') as mock_db:
            _run_belief_correction_hook("The weather is nice today")
        mock_db.assert_not_called()

    def test_no_self_reference_returns_early(self):
        """Messages matching pattern but without I/me/my are ignored (guardrail 1)."""
        from workers.digest_worker import _run_belief_correction_hook
        with patch('services.database_service.get_shared_db_service') as mock_db:
            _run_belief_correction_hook("Don't assume sushi is good")
        mock_db.assert_not_called()

    def test_negation_deletes_matching_trait(self):
        """'I don't like sushi' when favourite_food=sushi deletes the trait."""
        from workers.digest_worker import _run_belief_correction_hook

        mock_trait_svc = MagicMock()
        mock_trait_svc.get_all_traits.return_value = [
            {'trait_key': 'favourite_food', 'trait_value': 'sushi', 'confidence': 0.8, 'category': 'preference'},
        ]

        with patch('services.database_service.get_shared_db_service', return_value=MagicMock()), \
             patch('services.user_trait_service.UserTraitService', return_value=mock_trait_svc):
            _run_belief_correction_hook("I don't like sushi")

        mock_trait_svc.delete_trait.assert_called_once_with('favourite_food')

    def test_replacement_corrects_trait(self):
        """'Actually my name is Dylan' when name=Dan corrects the trait."""
        from workers.digest_worker import _run_belief_correction_hook

        mock_trait_svc = MagicMock()
        mock_trait_svc.get_all_traits.return_value = [
            {'trait_key': 'name', 'trait_value': 'dan', 'confidence': 0.9, 'category': 'core'},
        ]

        with patch('services.database_service.get_shared_db_service', return_value=MagicMock()), \
             patch('services.user_trait_service.UserTraitService', return_value=mock_trait_svc):
            _run_belief_correction_hook("Actually my name is Dylan")

        mock_trait_svc.correct_trait.assert_called_once()
        call_args = mock_trait_svc.correct_trait.call_args
        assert call_args[0][0] == 'name'
        assert 'dylan' in call_args[0][1].lower()

    def test_low_confidence_trait_skipped(self):
        """Traits below 0.4 confidence are not modified (guardrail 2)."""
        from workers.digest_worker import _run_belief_correction_hook

        mock_trait_svc = MagicMock()
        mock_trait_svc.get_all_traits.return_value = [
            {'trait_key': 'favourite_food', 'trait_value': 'sushi', 'confidence': 0.3, 'category': 'preference'},
        ]

        with patch('services.database_service.get_shared_db_service', return_value=MagicMock()), \
             patch('services.user_trait_service.UserTraitService', return_value=mock_trait_svc):
            _run_belief_correction_hook("I don't like sushi")

        mock_trait_svc.delete_trait.assert_not_called()

    def test_no_matching_trait_value_no_mutation(self):
        """Message matching pattern but trait value not in message — no mutation."""
        from workers.digest_worker import _run_belief_correction_hook

        mock_trait_svc = MagicMock()
        mock_trait_svc.get_all_traits.return_value = [
            {'trait_key': 'favourite_food', 'trait_value': 'pizza', 'confidence': 0.8, 'category': 'preference'},
        ]

        with patch('services.database_service.get_shared_db_service', return_value=MagicMock()), \
             patch('services.user_trait_service.UserTraitService', return_value=mock_trait_svc):
            _run_belief_correction_hook("I don't like weather")

        mock_trait_svc.delete_trait.assert_not_called()
        mock_trait_svc.correct_trait.assert_not_called()

    def test_hook_error_does_not_raise(self):
        """Hook failures are swallowed (non-fatal)."""
        from workers.digest_worker import _run_belief_correction_hook
        with patch('services.database_service.get_shared_db_service', side_effect=Exception('DB down')):
            # Should not raise
            _run_belief_correction_hook("I don't like sushi")

    def test_replacement_value_capped_at_3_words(self):
        """Replacement value is capped at 3 words to avoid trailing clause capture."""
        from workers.digest_worker import _run_belief_correction_hook

        mock_trait_svc = MagicMock()
        mock_trait_svc.get_all_traits.return_value = [
            {'trait_key': 'name', 'trait_value': 'dan', 'confidence': 0.9, 'category': 'core'},
        ]

        with patch('services.database_service.get_shared_db_service', return_value=MagicMock()), \
             patch('services.user_trait_service.UserTraitService', return_value=mock_trait_svc):
            _run_belief_correction_hook("Actually my name is Dylan by the way")

        call_args = mock_trait_svc.correct_trait.call_args
        new_value = call_args[0][1]
        # Should be at most 3 words
        assert len(new_value.split()) <= 3
