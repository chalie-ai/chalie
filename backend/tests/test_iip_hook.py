"""Unit tests for the Immediate Identity Promotion (IIP) hook in digest_worker."""

import pytest
from unittest.mock import MagicMock, patch, call


pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_hook(text: str):
    """
    Run _run_iip_hook with mocked IdentityStateService and KnowledgeService.

    Both are imported locally inside _run_iip_hook, so we patch them at their
    source modules.

    Returns (identity_svc_mock, knowledge_svc_mock) for assertion.
    """
    from services.database_service import get_shared_db_service
    mock_identity = MagicMock()
    mock_identity.set_field.return_value = True
    mock_ks = MagicMock()
    mock_ks.store.return_value = True

    with patch('services.identity_state_service.IdentityStateService', return_value=mock_identity), \
         patch('services.knowledge_service.KnowledgeService', return_value=mock_ks):
        from workers.digest_worker import _run_iip_hook
        _run_iip_hook(text, get_shared_db_service())

    return mock_identity, mock_ks


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIIPHook:

    def test_call_me_dylan(self, db):
        """'call me Dylan' → name='Dylan', category='core', confidence=0.95."""
        mock_id, mock_ks = _run_hook("call me Dylan")
        mock_id.set_field.assert_called_once_with(
            'name', 'Dylan', confidence=0.95, provisional=False
        )
        mock_ks.store.assert_called_once()
        kwargs = mock_ks.store.call_args[1]
        assert kwargs['value'] == 'Dylan'
        assert kwargs['kind'] == 'trait'
        assert kwargs['confidence'] == 0.95

    def test_lowercase_name_title_cased(self, db):
        """'call me dylan' (all lowercase) → stored as 'Dylan' via title()."""
        mock_id, mock_ks = _run_hook("call me dylan")
        kwargs = mock_id.set_field.call_args[0]
        assert kwargs[1] == 'Dylan'

    def test_apostrophe_in_name_accepted(self, db):
        """'call me O'Brien' → matched and stored as 'O'Brien'."""
        mock_id, mock_ks = _run_hook("call me O'Brien")
        assert mock_id.set_field.called
        stored_name = mock_id.set_field.call_args[0][1]
        assert "O'Brien" in stored_name or "O'brien" in stored_name

    def test_hyphen_in_name_accepted(self, db):
        """'call me Smith-Jones' → matched and stored."""
        mock_id, mock_ks = _run_hook("call me Smith-Jones")
        assert mock_id.set_field.called
        stored_name = mock_id.set_field.call_args[0][1]
        assert 'Smith' in stored_name

    def test_stopword_not_matched(self, db):
        """'call me maybe' → stopword, no store."""
        mock_id, mock_ks = _run_hook("call me maybe")
        mock_id.set_field.assert_not_called()
        mock_ks.store.assert_not_called()

    def test_stopword_later_not_matched(self, db):
        """'call me later' → stopword, no store."""
        mock_id, mock_ks = _run_hook("call me later")
        mock_id.set_field.assert_not_called()
        mock_ks.store.assert_not_called()

    def test_single_char_not_matched(self, db):
        """'you can call me J' → single char, rejected (len < 2)."""
        mock_id, mock_ks = _run_hook("you can call me J")
        mock_id.set_field.assert_not_called()

    def test_my_name_is_pattern(self, db):
        """'my name is Alex' → matches 'my name is' pattern."""
        mock_id, _ = _run_hook("My name is Alex")
        assert mock_id.set_field.called
        assert mock_id.set_field.call_args[0][1] == 'Alex'

    def test_you_can_call_me_pattern(self, db):
        """'you can call me Sam' → matched."""
        mock_id, _ = _run_hook("you can call me Sam")
        assert mock_id.set_field.called
        assert mock_id.set_field.call_args[0][1] == 'Sam'

    def test_i_go_by_pattern(self, db):
        """'I go by Chris' → matched."""
        mock_id, _ = _run_hook("I go by Chris")
        assert mock_id.set_field.called

    def test_store_failure_does_not_raise(self, db):
        """MemoryStore failure in set_field → store still called; no raise."""
        from services.database_service import get_shared_db_service
        mock_identity = MagicMock()
        mock_identity.set_field.side_effect = ConnectionError("MemoryStore down")
        mock_ks = MagicMock()
        mock_ks.store.return_value = True

        with patch('services.identity_state_service.IdentityStateService', return_value=mock_identity), \
             patch('services.knowledge_service.KnowledgeService', return_value=mock_ks):
            from workers.digest_worker import _run_iip_hook
            # Should not raise
            _run_iip_hook("call me Dylan", get_shared_db_service())

    def test_database_failure_does_not_raise(self, db):
        """Database failure in store → no raise."""
        from services.database_service import get_shared_db_service
        mock_identity = MagicMock()
        mock_identity.set_field.return_value = True
        mock_ks = MagicMock()
        mock_ks.store.side_effect = Exception("DB error")

        with patch('services.identity_state_service.IdentityStateService', return_value=mock_identity), \
             patch('services.knowledge_service.KnowledgeService', return_value=mock_ks):
            from workers.digest_worker import _run_iip_hook
            # Should not raise
            _run_iip_hook("call me Dylan", get_shared_db_service())

    def test_no_match_does_not_call_services(self, db):
        """Unrelated text → no services called."""
        mock_id, mock_ks = _run_hook("What's the weather like today?")
        mock_id.set_field.assert_not_called()
        mock_ks.store.assert_not_called()
