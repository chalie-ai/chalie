"""Tests for memory_skill — unified store/recall/update/forget entry point."""

import pytest
from unittest.mock import patch, MagicMock

from services.innate_skills.memory_skill import (
    handle_memory,
    _auto_classify,
)


pytestmark = pytest.mark.unit


# ── Auto-classify unit tests ────────────────────────────────────────

class TestAutoClassify:

    def test_trait_key(self):
        kind, decay = _auto_classify('user_name')
        assert kind == 'trait'
        assert decay == 'permanent'

    def test_preference_key(self):
        kind, decay = _auto_classify('food_preference')
        assert kind == 'preference'
        assert decay == 'slow'

    def test_procedure_key(self):
        kind, decay = _auto_classify('how_to_deploy')
        assert kind == 'procedure'
        assert decay == 'slow'

    def test_default_classification(self):
        kind, decay = _auto_classify('random_thing')
        assert kind == 'fact'
        assert decay == 'standard'

    def test_birthday_is_trait(self):
        kind, decay = _auto_classify('birthday')
        assert kind == 'trait'
        assert decay == 'permanent'

    def test_favorite_is_preference(self):
        kind, decay = _auto_classify('favorite_food')
        assert kind == 'preference'
        assert decay == 'slow'

    def test_workflow_is_procedure(self):
        kind, decay = _auto_classify('deployment_workflow')
        assert kind == 'procedure'
        assert decay == 'slow'


# ── Store via skill ─────────────────────────────────────────────────

class TestHandleMemoryStore:

    def test_handle_memory_store_basic(self):
        """Stores entries via the skill."""
        mock_ks = MagicMock()
        mock_ks.store.return_value = {'key': 'color', 'value': 'blue'}

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks), \
             patch('services.database_service.get_shared_db_service', return_value=MagicMock()):
            result = handle_memory('topic', {
                'action': 'store',
                'entries': [
                    {'key': 'color', 'value': 'blue', 'kind': 'fact', 'decay_class': 'standard'},
                ],
            })

        assert 'Stored 1' in result
        assert 'color' in result
        mock_ks.store.assert_called_once()

    def test_handle_memory_store_auto_classify(self):
        """Verifies kind/decay_class auto-classification when omitted."""
        mock_ks = MagicMock()
        mock_ks.store.return_value = {'key': 'user_name', 'value': 'Dylan'}
        call_args = {}

        def _capture_store(**kwargs):
            call_args.update(kwargs)
            return {'key': 'user_name', 'value': 'Dylan'}

        mock_ks.store.side_effect = _capture_store

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks), \
             patch('services.database_service.get_shared_db_service', return_value=MagicMock()):
            result = handle_memory('topic', {
                'action': 'store',
                'entries': [
                    {'key': 'user_name', 'value': 'Dylan'},
                ],
            })

        assert 'Stored 1' in result
        # Auto-classify should detect 'name' in key -> trait, permanent
        assert call_args.get('kind') == 'trait'
        assert call_args.get('decay_class') == 'permanent'

    def test_handle_memory_store_no_entries(self):
        """Empty entries list returns error."""
        result = handle_memory('topic', {'action': 'store', 'entries': []})
        assert 'Error' in result

    def test_handle_memory_store_skips_invalid(self):
        """Entries missing key or value are skipped."""
        mock_ks = MagicMock()
        mock_ks.store.return_value = None  # Skipped entries return None

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks), \
             patch('services.database_service.get_shared_db_service', return_value=MagicMock()):
            result = handle_memory('topic', {
                'action': 'store',
                'entries': [
                    {'key': '', 'value': 'no key'},
                    {'key': 'no_value'},
                ],
            })

        assert 'Nothing stored' in result
        mock_ks.store.assert_not_called()


# ── Recall via skill ────────────────────────────────────────────────

class TestHandleMemoryRecall:

    def test_handle_memory_recall(self):
        """Basic recall returns results."""
        mock_ks = MagicMock()
        mock_ks.recall.return_value = [
            {
                'key': 'user_name', 'value': 'Dylan', 'kind': 'trait',
                'confidence': 0.9, 'entity': 'user', 'decay_class': 'permanent',
                'evidence_count': 3,
            },
        ]

        mock_episodic = MagicMock()
        mock_episodic.retrieve_episodes.return_value = []

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks), \
             patch('services.database_service.get_shared_db_service', return_value=MagicMock()), \
             patch('services.episodic_service.EpisodicService', return_value=mock_episodic), \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            result = handle_memory('topic', {
                'action': 'recall',
                'query': 'user name',
            })

        assert 'user_name' in result
        assert 'Dylan' in result

    def test_handle_memory_recall_no_query(self):
        """Missing query returns error."""
        result = handle_memory('topic', {'action': 'recall', 'query': ''})
        assert 'Error' in result

    def test_handle_memory_recall_empty_results(self):
        """Empty results produce structured empty response."""
        mock_ks = MagicMock()
        mock_ks.recall.return_value = []

        mock_episodic = MagicMock()
        mock_episodic.retrieve_episodes.return_value = []

        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0,)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db), \
             patch('services.episodic_service.EpisodicService', return_value=mock_episodic), \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            result = handle_memory('topic', {
                'action': 'recall',
                'query': 'nonexistent thing',
            })

        assert 'No matches' in result


# ── Update via skill ────────────────────────────────────────────────

class TestHandleMemoryUpdate:

    def test_handle_memory_update(self):
        """Updates an entry via the skill."""
        mock_ks = MagicMock()
        mock_ks.update.return_value = {'key': 'color', 'value': 'red'}

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks), \
             patch('services.database_service.get_shared_db_service', return_value=MagicMock()):
            result = handle_memory('topic', {
                'action': 'update',
                'key': 'color',
                'value': 'red',
            })

        assert 'Updated' in result
        assert 'color' in result

    def test_handle_memory_update_no_key(self):
        """Missing key returns error."""
        result = handle_memory('topic', {'action': 'update', 'value': 'red'})
        assert 'Error' in result

    def test_handle_memory_update_no_changes(self):
        """No value or confidence returns error."""
        result = handle_memory('topic', {'action': 'update', 'key': 'color'})
        assert 'Error' in result

    def test_handle_memory_update_not_found(self):
        """Update on nonexistent entry reports not found."""
        mock_ks = MagicMock()
        mock_ks.update.return_value = None

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks), \
             patch('services.database_service.get_shared_db_service', return_value=MagicMock()):
            result = handle_memory('topic', {
                'action': 'update',
                'key': 'ghost',
                'value': 'something',
            })

        assert 'No entry found' in result


# ── Forget via skill ────────────────────────────────────────────────

class TestHandleMemoryForget:

    def test_handle_memory_forget(self):
        """Soft-deletes an entry via the skill."""
        mock_ks = MagicMock()
        mock_ks.forget.return_value = True

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks), \
             patch('services.database_service.get_shared_db_service', return_value=MagicMock()):
            result = handle_memory('topic', {
                'action': 'forget',
                'key': 'old_fact',
            })

        assert 'Forgotten' in result
        assert 'old_fact' in result

    def test_handle_memory_forget_no_key(self):
        """Missing key returns error."""
        result = handle_memory('topic', {'action': 'forget'})
        assert 'Error' in result

    def test_handle_memory_forget_not_found(self):
        """Forget on nonexistent entry reports not found."""
        mock_ks = MagicMock()
        mock_ks.forget.return_value = False

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks), \
             patch('services.database_service.get_shared_db_service', return_value=MagicMock()):
            result = handle_memory('topic', {
                'action': 'forget',
                'key': 'ghost',
            })

        assert 'No entry found' in result


# ── Invalid action ──────────────────────────────────────────────────

class TestHandleMemoryInvalid:

    def test_handle_memory_invalid_action(self):
        """Invalid action returns error string."""
        result = handle_memory('topic', {'action': 'explode'})
        assert 'Unknown action' in result
        assert 'explode' in result

    def test_handle_memory_default_action_is_recall(self):
        """Missing action defaults to recall."""
        # Without a query, recall returns an error about missing query
        result = handle_memory('topic', {})
        assert 'Error' in result  # "no query specified for recall"
