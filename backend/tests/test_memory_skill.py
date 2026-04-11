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


# ── Contradiction check wiring ────────────────────────────────────────────────
# Tests A-D: _handle_store() <-> _check_trait_contradiction gate
# These tests patch _check_trait_contradiction at the module level (not its
# lazy-import internals) and KnowledgeService/get_shared_db_service so no
# real DB or embedding service is needed.
# ─────────────────────────────────────────────────────────────────────────────

class _FakeKS:
    """Minimal KnowledgeService fake — records store() calls, returns configurable dict."""

    def __init__(self, stored_entry=None):
        self._stored_entry = stored_entry if stored_entry is not None else {'rowid': 99}
        self.store_calls = []

    def store(self, **kwargs):
        self.store_calls.append(kwargs)
        return self._stored_entry


def _params_for(key, value, kind=None):
    """Build a minimal params dict for _handle_store."""
    entry = {'key': key, 'value': value}
    if kind is not None:
        entry['kind'] = kind
    return {'entries': [entry]}


class TestHandleStoreTraitContradictionCheckFires:
    """Test A — _check_trait_contradiction fires for trait entries."""

    def test_a1_explicit_trait_kind_fires_check(self):
        """kind='trait' → _check_trait_contradiction called with (ks, rowid, key, value, 1.0, thread_id, source='chat')."""
        fake_ks = _FakeKS(stored_entry={'rowid': 42})

        with patch('services.knowledge_service.KnowledgeService', return_value=fake_ks), \
             patch('services.database_service.get_shared_db_service', return_value=object()), \
             patch('services.innate_skills.memory_skill._check_trait_contradiction') as mock_ctc:
            from services.innate_skills.memory_skill import _handle_store
            result = _handle_store('ch-1', _params_for('user_name', 'Alice', kind='trait'))

        assert 'Stored' in result
        mock_ctc.assert_called_once_with(
            fake_ks, 42, 'user_name', 'Alice', 1.0,
            thread_id='ch-1',
            source='chat',
        )

    def test_a2_auto_classified_trait_key_fires_check(self):
        """key 'user_name' auto-classifies to trait → contradiction check fires."""
        fake_ks = _FakeKS(stored_entry={'rowid': 7})

        with patch('services.knowledge_service.KnowledgeService', return_value=fake_ks), \
             patch('services.database_service.get_shared_db_service', return_value=object()), \
             patch('services.innate_skills.memory_skill._check_trait_contradiction') as mock_ctc:
            from services.innate_skills.memory_skill import _handle_store
            result = _handle_store('ch-2', _params_for('user_name', 'Bob'))

        assert 'Stored' in result
        assert mock_ctc.call_count == 1
        _, called_new_id, called_key, called_value, called_conf = mock_ctc.call_args.args
        assert called_new_id == 7
        assert called_key == 'user_name'
        assert called_value == 'Bob'
        assert called_conf == 1.0
        assert mock_ctc.call_args.kwargs['thread_id'] == 'ch-2'
        assert mock_ctc.call_args.kwargs['source'] == 'chat'

    def test_a3_rowid_preferred_over_id_when_both_present(self):
        """stored_entry with both 'rowid' and 'id' → rowid is passed to check."""
        fake_ks = _FakeKS(stored_entry={'rowid': 55, 'id': 100})

        with patch('services.knowledge_service.KnowledgeService', return_value=fake_ks), \
             patch('services.database_service.get_shared_db_service', return_value=object()), \
             patch('services.innate_skills.memory_skill._check_trait_contradiction') as mock_ctc:
            from services.innate_skills.memory_skill import _handle_store
            _handle_store('ch-3', _params_for('user_name', 'Carol', kind='trait'))

        assert mock_ctc.call_args.args[1] == 55

    def test_a4_falls_back_to_id_when_rowid_absent(self):
        """stored_entry with 'id' but no 'rowid' → 'id' is used."""
        fake_ks = _FakeKS(stored_entry={'id': 33})

        with patch('services.knowledge_service.KnowledgeService', return_value=fake_ks), \
             patch('services.database_service.get_shared_db_service', return_value=object()), \
             patch('services.innate_skills.memory_skill._check_trait_contradiction') as mock_ctc:
            from services.innate_skills.memory_skill import _handle_store
            _handle_store('ch-4', _params_for('user_name', 'Dave', kind='trait'))

        assert mock_ctc.call_args.args[1] == 33


class TestHandleStoreNonTraitSkipsContradictionCheck:
    """Test B — _check_trait_contradiction must NOT fire for non-trait kinds."""

    @pytest.mark.parametrize('kind', ['fact', 'procedure', 'preference', 'concept', 'rule', 'metric'])
    def test_b1_no_contradiction_check_for_non_trait_kind(self, kind):
        fake_ks = _FakeKS(stored_entry={'rowid': 10})

        with patch('services.knowledge_service.KnowledgeService', return_value=fake_ks), \
             patch('services.database_service.get_shared_db_service', return_value=object()), \
             patch('services.innate_skills.memory_skill._check_trait_contradiction') as mock_ctc:
            from services.innate_skills.memory_skill import _handle_store
            result = _handle_store('ch-5', _params_for(f'{kind}_key', 'some value', kind=kind))

        assert 'Stored' in result
        mock_ctc.assert_not_called()


class TestHandleStoreContradictionCheckExceptionDoesNotAbortStore:
    """Test C — exception in _check_trait_contradiction must not propagate."""

    def test_c1_store_succeeds_when_contradiction_check_raises(self):
        """RuntimeError in _check_trait_contradiction → _handle_store still returns success."""
        fake_ks = _FakeKS(stored_entry={'rowid': 20})

        with patch('services.knowledge_service.KnowledgeService', return_value=fake_ks), \
             patch('services.database_service.get_shared_db_service', return_value=object()), \
             patch('services.innate_skills.memory_skill._check_trait_contradiction',
                   side_effect=RuntimeError('embedding service down')):
            from services.innate_skills.memory_skill import _handle_store
            result = _handle_store('ch-6', _params_for('user_name', 'Eve', kind='trait'))

        assert 'Stored' in result
        assert len(fake_ks.store_calls) == 1

    def test_c2_result_is_success_string_not_error_despite_ctc_failure(self):
        """Return value names the stored key even when contradiction check blew up."""
        fake_ks = _FakeKS(stored_entry={'rowid': 21})

        with patch('services.knowledge_service.KnowledgeService', return_value=fake_ks), \
             patch('services.database_service.get_shared_db_service', return_value=object()), \
             patch('services.innate_skills.memory_skill._check_trait_contradiction',
                   side_effect=ValueError('vector store unavailable')):
            from services.innate_skills.memory_skill import _handle_store
            result = _handle_store('ch-7', _params_for('user_name', 'Frank', kind='trait'))

        assert '[MEMORY] Stored 1 entries' in result
        assert 'Error' not in result


class TestHandleStoreMissingRowidSkipsContradictionCheck:
    """Test D — stored_entry without rowid/id must not call _check_trait_contradiction."""

    def test_d1_empty_dict_stored_entry_skips_ctc(self):
        """stored_entry={} is falsy → inner `if stored_entry and kind == 'trait':` skips."""
        fake_ks = _FakeKS(stored_entry={})

        with patch('services.knowledge_service.KnowledgeService', return_value=fake_ks), \
             patch('services.database_service.get_shared_db_service', return_value=object()), \
             patch('services.innate_skills.memory_skill._check_trait_contradiction') as mock_ctc:
            from services.innate_skills.memory_skill import _handle_store
            _handle_store('ch-8', _params_for('user_name', 'Grace', kind='trait'))

        mock_ctc.assert_not_called()

    def test_d2_zero_rowid_skips_ctc_via_new_id_guard(self):
        """rowid=0 → new_id=0 → `if new_id:` guard is falsy → no CTC call."""
        fake_ks = _FakeKS(stored_entry={'rowid': 0})

        with patch('services.knowledge_service.KnowledgeService', return_value=fake_ks), \
             patch('services.database_service.get_shared_db_service', return_value=object()), \
             patch('services.innate_skills.memory_skill._check_trait_contradiction') as mock_ctc:
            from services.innate_skills.memory_skill import _handle_store
            _handle_store('ch-9', _params_for('user_name', 'Heidi', kind='trait'))

        mock_ctc.assert_not_called()

    def test_d3_none_rowid_and_none_id_skips_ctc(self):
        """rowid=None falls through to id=None → new_id=None → no CTC call."""
        fake_ks = _FakeKS(stored_entry={'rowid': None, 'id': None})

        with patch('services.knowledge_service.KnowledgeService', return_value=fake_ks), \
             patch('services.database_service.get_shared_db_service', return_value=object()), \
             patch('services.innate_skills.memory_skill._check_trait_contradiction') as mock_ctc:
            from services.innate_skills.memory_skill import _handle_store
            _handle_store('ch-10', _params_for('user_name', 'Ivan', kind='trait'))

        mock_ctc.assert_not_called()

