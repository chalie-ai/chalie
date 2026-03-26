"""Tests for conversational belief correction — digest_worker hook using KnowledgeService."""

import pytest
from unittest.mock import MagicMock, patch, call

pytestmark = pytest.mark.unit


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

        mock_ks = MagicMock()
        mock_ks.get_by_kind.return_value = [
            {'key': 'favourite_food', 'value': 'sushi', 'confidence': 0.8, 'data': '{"category": "preference"}'},
        ]

        with patch('services.database_service.get_shared_db_service', return_value=MagicMock()), \
             patch('services.knowledge_service.KnowledgeService', return_value=mock_ks):
            _run_belief_correction_hook("I don't like sushi")

        mock_ks.forget.assert_called_once_with('user', 'favourite_food')

    def test_replacement_corrects_trait(self):
        """'Actually my name is Dylan' when name=Dan corrects the trait."""
        from workers.digest_worker import _run_belief_correction_hook

        mock_ks = MagicMock()
        mock_ks.get_by_kind.return_value = [
            {'key': 'name', 'value': 'dan', 'confidence': 0.9, 'data': '{"category": "core"}'},
        ]

        with patch('services.database_service.get_shared_db_service', return_value=MagicMock()), \
             patch('services.knowledge_service.KnowledgeService', return_value=mock_ks):
            _run_belief_correction_hook("Actually my name is Dylan")

        mock_ks.update.assert_called_once()
        call_args = mock_ks.update.call_args
        assert call_args[0][0] == 'user'
        assert call_args[0][1] == 'name'
        assert 'dylan' in call_args[1]['value'].lower()

    def test_low_confidence_trait_skipped(self):
        """Traits below 0.4 confidence are not modified (guardrail 2)."""
        from workers.digest_worker import _run_belief_correction_hook

        mock_ks = MagicMock()
        mock_ks.get_by_kind.return_value = [
            {'key': 'favourite_food', 'value': 'sushi', 'confidence': 0.3, 'data': '{"category": "preference"}'},
        ]

        with patch('services.database_service.get_shared_db_service', return_value=MagicMock()), \
             patch('services.knowledge_service.KnowledgeService', return_value=mock_ks):
            _run_belief_correction_hook("I don't like sushi")

        mock_ks.forget.assert_not_called()

    def test_no_matching_trait_value_no_mutation(self):
        """Message matching pattern but trait value not in message — no mutation."""
        from workers.digest_worker import _run_belief_correction_hook

        mock_ks = MagicMock()
        mock_ks.get_by_kind.return_value = [
            {'key': 'favourite_food', 'value': 'pizza', 'confidence': 0.8, 'data': '{"category": "preference"}'},
        ]

        with patch('services.database_service.get_shared_db_service', return_value=MagicMock()), \
             patch('services.knowledge_service.KnowledgeService', return_value=mock_ks):
            _run_belief_correction_hook("I don't like weather")

        mock_ks.forget.assert_not_called()
        mock_ks.update.assert_not_called()

    def test_hook_error_does_not_raise(self):
        """Hook failures are swallowed (non-fatal)."""
        from workers.digest_worker import _run_belief_correction_hook
        with patch('services.database_service.get_shared_db_service', side_effect=Exception('DB down')):
            # Should not raise
            _run_belief_correction_hook("I don't like sushi")

    def test_replacement_value_capped_at_3_words(self):
        """Replacement value is capped at 3 words to avoid trailing clause capture."""
        from workers.digest_worker import _run_belief_correction_hook

        mock_ks = MagicMock()
        mock_ks.get_by_kind.return_value = [
            {'key': 'name', 'value': 'dan', 'confidence': 0.9, 'data': '{"category": "core"}'},
        ]

        with patch('services.database_service.get_shared_db_service', return_value=MagicMock()), \
             patch('services.knowledge_service.KnowledgeService', return_value=mock_ks):
            _run_belief_correction_hook("Actually my name is Dylan by the way")

        call_args = mock_ks.update.call_args
        new_value = call_args[1]['value']
        # Should be at most 3 words
        assert len(new_value.split()) <= 3
