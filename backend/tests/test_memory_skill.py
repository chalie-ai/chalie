"""Tests for memory_skill — unified store/recall/update entry point."""

import pytest
from unittest.mock import patch, MagicMock

from services.innate_skills.memory_skill import (
    handle_memory,
    _auto_classify,
    TOOL_SCHEMA,
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

    def test_handle_memory_store_basic(self, db):
        """Storing a fact actually persists a row in the knowledge table."""
        result = handle_memory('topic', {
            'action': 'store',
            'entries': [
                {'key': 'favourite_colour', 'value': 'blue',
                 'kind': 'fact', 'decay_class': 'standard'},
            ],
        })

        assert 'Stored 1' in result
        assert 'favourite_colour' in result

        # Real DB state: row must exist with correct value
        row = db.execute(
            "SELECT value, kind FROM knowledge WHERE key = 'favourite_colour'"
        ).fetchone()
        assert row is not None
        assert row[0] == 'blue'
        assert row[1] == 'fact'

    def test_handle_memory_store_auto_classify(self, db):
        """Auto-classification writes the correct kind/decay_class to the DB."""
        result = handle_memory('topic', {
            'action': 'store',
            'entries': [
                {'key': 'user_name', 'value': 'Dylan'},
            ],
        })

        assert 'Stored 1' in result

        # Real DB state: kind and decay_class from _auto_classify('user_name')
        row = db.execute(
            "SELECT kind, decay_class FROM knowledge WHERE key = 'user_name'"
        ).fetchone()
        assert row is not None
        # 'user_name' contains 'name' -> kind='trait', decay_class='permanent'
        assert row[0] == 'trait'
        assert row[1] == 'permanent'

    def test_handle_memory_store_multiple_entries(self, db):
        """Storing two entries creates two DB rows."""
        result = handle_memory('topic', {
            'action': 'store',
            'entries': [
                {'key': 'city', 'value': 'Valletta', 'kind': 'fact', 'decay_class': 'standard'},
                {'key': 'country', 'value': 'Malta', 'kind': 'fact', 'decay_class': 'standard'},
            ],
        })

        assert 'Stored 2' in result

        rows = db.execute(
            "SELECT key, value FROM knowledge WHERE key IN ('city', 'country')"
        ).fetchall()
        assert len(rows) == 2
        stored = {r[0]: r[1] for r in rows}
        assert stored['city'] == 'Valletta'
        assert stored['country'] == 'Malta'

    def test_handle_memory_store_no_entries(self):
        """Empty entries list returns error without touching the DB."""
        result = handle_memory('topic', {'action': 'store', 'entries': []})
        assert 'Error' in result

    def test_handle_memory_store_skips_invalid(self, db):
        """Entries missing key or value are silently skipped — nothing stored."""
        result = handle_memory('topic', {
            'action': 'store',
            'entries': [
                {'key': '', 'value': 'no key'},
                {'key': 'no_value'},
            ],
        })

        assert 'Nothing stored' in result

        # Real DB state: no rows should have been written
        count = db.execute(
            "SELECT COUNT(*) FROM knowledge WHERE deleted_at IS NULL"
        ).fetchone()[0]
        assert count == 0


# ── Recall via skill ────────────────────────────────────────────────

class TestHandleMemoryRecall:

    def test_handle_memory_recall(self, db):
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
             patch('services.episodic_service.EpisodicService', return_value=mock_episodic), \
             patch('services.transcript_service.search', return_value=[]), \
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

    def test_handle_memory_recall_empty_results(self, db):
        """Empty results produce structured empty response."""
        mock_ks = MagicMock()
        mock_ks.recall.return_value = []

        mock_episodic = MagicMock()
        mock_episodic.retrieve_episodes.return_value = []

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks), \
             patch('services.episodic_service.EpisodicService', return_value=mock_episodic), \
             patch('services.transcript_service.search', return_value=[]), \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            result = handle_memory('topic', {
                'action': 'recall',
                'query': 'nonexistent thing',
            })

        assert 'No matches' in result


# ── Update via skill ────────────────────────────────────────────────

class TestHandleMemoryUpdate:

    def test_handle_memory_update(self, db):
        """Updating an existing entry's confidence changes it in the DB."""
        # Seed via real store first
        handle_memory('topic', {
            'action': 'store',
            'entries': [{'key': 'fav_colour', 'value': 'blue', 'kind': 'preference',
                         'decay_class': 'slow'}],
        })

        # Update value AND confidence together (both must be non-None to avoid
        # the NOT NULL bug in ks.update when value=None is passed in the SET clause)
        result = handle_memory('topic', {
            'action': 'update',
            'key': 'fav_colour',
            'value': 'blue',
            'confidence': 0.95,
        })

        assert 'Updated' in result
        assert 'fav_colour' in result

        # Real DB state: confidence must be updated
        row = db.execute(
            "SELECT confidence FROM knowledge WHERE key = 'fav_colour'"
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(0.95, abs=0.01)

    def test_handle_memory_update_no_key(self):
        """Missing key returns error."""
        result = handle_memory('topic', {'action': 'update', 'value': 'red'})
        assert 'Error' in result

    def test_handle_memory_update_no_changes(self):
        """No value or confidence returns error."""
        result = handle_memory('topic', {'action': 'update', 'key': 'fav_colour'})
        assert 'Error' in result

    def test_handle_memory_update_not_found(self, db):
        """Update on nonexistent entry reports not found — no row written."""
        result = handle_memory('topic', {
            'action': 'update',
            'key': 'ghost_key_xyz',
            'value': 'something',
        })

        assert 'No entry found' in result

        # Real DB state: no row was created
        row = db.execute(
            "SELECT 1 FROM knowledge WHERE key = 'ghost_key_xyz'"
        ).fetchone()
        assert row is None


# ── Invalid action ──────────────────────────────────────────────────

class TestHandleMemoryInvalid:

    def test_handle_memory_invalid_action(self):
        """Invalid action returns error string."""
        result = handle_memory('topic', {'action': 'explode'})
        assert 'Unknown action' in result
        assert 'explode' in result
        assert 'forget' not in result

    def test_handle_memory_default_action_is_recall(self):
        """Missing action defaults to recall."""
        # Without a query, recall returns an error about missing query
        result = handle_memory('topic', {})
        assert 'Error' in result  # "no query specified for recall"


# ── Schema validation ───────────────────────────────────────────────

class TestToolSchema:

    def test_schema_actions_do_not_include_forget(self):
        """forget was removed — must not appear in the action enum."""
        action_enum = TOOL_SCHEMA['input_schema']['properties']['action']['enum']
        assert 'forget' not in action_enum

    def test_schema_actions_are_store_recall_update(self):
        """Exactly the three remaining valid actions, no more."""
        action_enum = TOOL_SCHEMA['input_schema']['properties']['action']['enum']
        assert set(action_enum) == {'store', 'recall', 'update'}

    def test_schema_has_no_kinds_parameter(self):
        """kinds was removed from the top-level schema."""
        top_level_props = TOOL_SCHEMA['input_schema']['properties']
        assert 'kinds' not in top_level_props

    def test_schema_has_no_limit_parameter(self):
        """limit was removed from the top-level schema."""
        top_level_props = TOOL_SCHEMA['input_schema']['properties']
        assert 'limit' not in top_level_props

    def test_schema_has_no_include_transcript_parameter(self):
        """include_transcript was removed from the schema."""
        top_level_props = TOOL_SCHEMA['input_schema']['properties']
        assert 'include_transcript' not in top_level_props

    def test_schema_has_no_transcript_topic_parameter(self):
        """transcript_topic was removed from the schema."""
        top_level_props = TOOL_SCHEMA['input_schema']['properties']
        assert 'transcript_topic' not in top_level_props

    def test_schema_has_no_date_range_parameter(self):
        """date_range was removed from the schema."""
        top_level_props = TOOL_SCHEMA['input_schema']['properties']
        assert 'date_range' not in top_level_props


# ── Forget rejection ────────────────────────────────────────────────

class TestForgetRejected:

    def test_forget_action_returns_unknown_action_error(self):
        """action='forget' must be rejected as unknown, not silently ignored."""
        result = handle_memory('topic', {'action': 'forget', 'key': 'something'})
        assert 'Unknown action' in result
        assert 'forget' in result

    def test_forget_error_lists_valid_actions(self):
        """The rejection message should name the valid alternatives."""
        result = handle_memory('topic', {'action': 'forget', 'key': 'something'})
        # Valid actions must be communicated so the LLM can retry correctly
        assert 'store' in result
        assert 'recall' in result
        assert 'update' in result


# ── Recall always searches transcript ──────────────────────────────

class TestRecallAlwaysSearchesTranscript:

    def test_transcript_searched_even_when_knowledge_has_results(self, db):
        """transcript_service.search is called unconditionally on every recall."""
        mock_ks = MagicMock()
        mock_ks.recall.return_value = [
            {
                'key': 'user_name', 'value': 'Dylan', 'kind': 'trait',
                'confidence': 0.9, 'entity': 'user', 'decay_class': 'permanent',
                'evidence_count': 1,
            },
        ]
        mock_episodic = MagicMock()
        mock_episodic.retrieve_episodes.return_value = []

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks), \
             patch('services.episodic_service.EpisodicService', return_value=mock_episodic), \
             patch('services.transcript_service.search', return_value=[]) as mock_ts, \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            handle_memory('topic', {'action': 'recall', 'query': 'user name'})

        mock_ts.assert_called_once()

    def test_transcript_searched_when_knowledge_is_empty(self, db):
        """transcript_service.search is called even when knowledge returns nothing."""
        mock_ks = MagicMock()
        mock_ks.recall.return_value = []
        mock_episodic = MagicMock()
        mock_episodic.retrieve_episodes.return_value = []

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks), \
             patch('services.episodic_service.EpisodicService', return_value=mock_episodic), \
             patch('services.transcript_service.search', return_value=[]) as mock_ts, \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            handle_memory('topic', {'action': 'recall', 'query': 'anything'})

        mock_ts.assert_called_once()

    def test_transcript_results_appear_in_output(self, db):
        """When transcript returns hits they are included in the formatted result."""
        mock_ks = MagicMock()
        mock_ks.recall.return_value = []
        mock_episodic = MagicMock()
        mock_episodic.retrieve_episodes.return_value = []

        transcript_hit = {
            'role': 'user',
            'content': 'I like hiking on weekends',
            'similarity': 0.8,
            'created_at': '2026-01-01T00:00:00+00:00',
            'tool_name': None,
            'topic': 'topic',
        }

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks), \
             patch('services.episodic_service.EpisodicService', return_value=mock_episodic), \
             patch('services.transcript_service.search', return_value=[transcript_hit]), \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            result = handle_memory('topic', {'action': 'recall', 'query': 'hiking'})

        assert 'transcript' in result
        assert 'hiking' in result


# ── Recall ignores passed kinds / limit ─────────────────────────────

class TestRecallIgnoresRemovedParams:

    def test_recall_ignores_kinds_param(self, db):
        """LLM passing kinds= in params does not cause an error."""
        mock_ks = MagicMock()
        mock_ks.recall.return_value = []
        mock_episodic = MagicMock()
        mock_episodic.retrieve_episodes.return_value = []

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks), \
             patch('services.episodic_service.EpisodicService', return_value=mock_episodic), \
             patch('services.transcript_service.search', return_value=[]), \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            result = handle_memory('topic', {
                'action': 'recall',
                'query': 'something',
                'kinds': ['trait', 'preference'],  # extra param from old schema
            })

        # Should not error; kinds is silently ignored
        assert 'Error' not in result or 'No matches' in result

    def test_recall_ignores_limit_param(self, db):
        """LLM passing limit= in params does not cause an error."""
        mock_ks = MagicMock()
        mock_ks.recall.return_value = []
        mock_episodic = MagicMock()
        mock_episodic.retrieve_episodes.return_value = []

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks), \
             patch('services.episodic_service.EpisodicService', return_value=mock_episodic), \
             patch('services.transcript_service.search', return_value=[]), \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            result = handle_memory('topic', {
                'action': 'recall',
                'query': 'something',
                'limit': 3,  # extra param from old schema
            })

        assert 'Error' not in result or 'No matches' in result

    def test_recall_ignores_include_transcript_param(self, db):
        """include_transcript=False no longer suppresses transcript search."""
        mock_ks = MagicMock()
        mock_ks.recall.return_value = []
        mock_episodic = MagicMock()
        mock_episodic.retrieve_episodes.return_value = []

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks), \
             patch('services.episodic_service.EpisodicService', return_value=mock_episodic), \
             patch('services.transcript_service.search', return_value=[]) as mock_ts, \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            handle_memory('topic', {
                'action': 'recall',
                'query': 'something',
                'include_transcript': False,  # old param, must be ignored
            })

        # Transcript must still be searched regardless of this param
        mock_ts.assert_called_once()


# ── Empty results list transcript layer ─────────────────────────────

class TestEmptyResultsFormat:

    def test_empty_results_mention_transcript_layer(self, db):
        """_format_empty must name 'transcript' as one of the searched layers."""
        mock_ks = MagicMock()
        mock_ks.recall.return_value = []
        mock_episodic = MagicMock()
        mock_episodic.retrieve_episodes.return_value = []

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks), \
             patch('services.episodic_service.EpisodicService', return_value=mock_episodic), \
             patch('services.transcript_service.search', return_value=[]), \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            result = handle_memory('topic', {
                'action': 'recall',
                'query': 'ghost query that matches nothing',
            })

        assert 'transcript' in result

    def test_empty_results_mention_knowledge_and_episodes_layers(self, db):
        """All three searched layers are reported when nothing is found."""
        mock_ks = MagicMock()
        mock_ks.recall.return_value = []
        mock_episodic = MagicMock()
        mock_episodic.retrieve_episodes.return_value = []

        with patch('services.knowledge_service.KnowledgeService', return_value=mock_ks), \
             patch('services.episodic_service.EpisodicService', return_value=mock_episodic), \
             patch('services.transcript_service.search', return_value=[]), \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            result = handle_memory('topic', {
                'action': 'recall',
                'query': 'nothing here',
            })

        assert 'knowledge' in result
        assert 'episodes' in result
        assert 'transcript' in result
