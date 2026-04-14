"""Tests for memory_skill — DataGraphService-backed store/recall entry point."""

import contextlib
import sqlite3

import pytest
from unittest.mock import patch, MagicMock

from services.innate_skills.memory_skill import (
    handle_memory,
    TOOL_SCHEMA,
    _relevance_label,
    _search_data_graph,
    _search_document_artifacts,
)
from services.data_graph_service import DataGraphService, KIND_DOCUMENT
from services.database_service import DatabaseService


# ── Shared fixture for real DataGraphService (document artifact tests) ──────

_DOC_ARTIFACT_DDL = [
    """
    CREATE TABLE IF NOT EXISTS data_graph (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        kind              TEXT NOT NULL,
        key               TEXT NOT NULL,
        value             TEXT,
        storage_strength  REAL NOT NULL DEFAULT 0.5,
        retrieval_weight  REAL NOT NULL DEFAULT 1.0,
        salience_score    REAL NOT NULL DEFAULT 0.0,
        evidence_count    INTEGER NOT NULL DEFAULT 1,
        first_seen_at     TEXT NOT NULL DEFAULT (datetime('now')),
        last_confirmed_at TEXT NOT NULL DEFAULT (datetime('now')),
        last_accessed_at  TEXT,
        source            TEXT,
        deleted_at        TEXT,
        active            INTEGER NOT NULL DEFAULT 1,
        search_queries    TEXT DEFAULT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ms_dg_kind ON data_graph(kind)",
    "CREATE INDEX IF NOT EXISTS idx_ms_dg_key ON data_graph(key)",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS data_graph_fts USING fts5(
        key, value, kind, search_queries,
        tokenize='porter unicode61'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS data_graph_edges (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        from_id          INTEGER NOT NULL REFERENCES data_graph(id) ON DELETE CASCADE,
        to_id            INTEGER NOT NULL REFERENCES data_graph(id) ON DELETE CASCADE,
        edge_type        TEXT NOT NULL DEFAULT 'related',
        strength         REAL NOT NULL DEFAULT 1.0,
        created_at       TEXT NOT NULL DEFAULT (datetime('now')),
        last_accessed_at TEXT,
        UNIQUE (from_id, to_id, edge_type)
    )
    """,
    "CREATE TABLE IF NOT EXISTS data_graph_key_vec (rowid INTEGER PRIMARY KEY, embedding BLOB)",
    "CREATE TABLE IF NOT EXISTS data_graph_value_vec (rowid INTEGER PRIMARY KEY, embedding BLOB)",
]


@pytest.fixture
def _doc_db(tmp_path):
    """Real SQLite DB with data_graph schema for document artifact recall tests."""
    db_path = str(tmp_path / "ms_doc_artifacts.db")
    shared_conn = sqlite3.connect(db_path)
    shared_conn.row_factory = sqlite3.Row
    shared_conn.execute("PRAGMA foreign_keys = ON")
    for ddl in _DOC_ARTIFACT_DDL:
        shared_conn.execute(ddl)
    shared_conn.commit()

    db = DatabaseService.__new__(DatabaseService)
    db._db_path = db_path
    db._init_complete = True
    db.db_path = db_path

    _depth = [0]

    @contextlib.contextmanager
    def _connection():
        _depth[0] += 1
        try:
            yield shared_conn
            if _depth[0] == 1:
                shared_conn.commit()
        except Exception:
            if _depth[0] == 1:
                shared_conn.rollback()
            raise
        finally:
            _depth[0] -= 1

    db.connection = _connection
    yield db
    shared_conn.close()


@pytest.fixture
def _doc_svc(_doc_db):
    """DataGraphService with real SQLite, embeddings disabled."""
    svc = DataGraphService(_doc_db)
    svc._generate_embedding = MagicMock(return_value=None)
    svc._schedule_embeddings = MagicMock()
    svc._schedule_doc2query = MagicMock()
    return svc


pytestmark = pytest.mark.unit


# ── Relevance bucketing ──────────────────────────────────────────────

class TestRelevanceLabel:

    def test_high_at_0_7(self):
        assert _relevance_label(0.7) == "high"

    def test_high_above_0_7(self):
        assert _relevance_label(1.0) == "high"

    def test_medium_at_0_4(self):
        assert _relevance_label(0.4) == "medium"

    def test_medium_below_0_7(self):
        assert _relevance_label(0.69) == "medium"

    def test_low_below_0_4(self):
        assert _relevance_label(0.39) == "low"

    def test_low_at_zero(self):
        assert _relevance_label(0.0) == "low"


# ── TOOL_SCHEMA ─────────────────────────────────────────────────────

class TestToolSchema:

    def test_schema_actions_are_store_recall_reflect(self):
        action_enum = TOOL_SCHEMA['input_schema']['properties']['action']['enum']
        assert set(action_enum) == {'store', 'recall', 'reflect'}

    def test_schema_has_no_update_action(self):
        action_enum = TOOL_SCHEMA['input_schema']['properties']['action']['enum']
        assert 'update' not in action_enum

    def test_schema_has_no_forget_action(self):
        action_enum = TOOL_SCHEMA['input_schema']['properties']['action']['enum']
        assert 'forget' not in action_enum

    def test_schema_has_no_entries_array(self):
        props = TOOL_SCHEMA['input_schema']['properties']
        assert 'entries' not in props

    def test_schema_has_no_entity_param(self):
        props = TOOL_SCHEMA['input_schema']['properties']
        assert 'entity' not in props

    def test_schema_has_no_confidence_param(self):
        props = TOOL_SCHEMA['input_schema']['properties']
        assert 'confidence' not in props

    def test_schema_has_no_decay_class_param(self):
        props = TOOL_SCHEMA['input_schema']['properties']
        assert 'decay_class' not in props

    def test_schema_kind_enum_is_three_llm_visible_kinds(self):
        kind_enum = TOOL_SCHEMA['input_schema']['properties']['kind']['enum']
        assert set(kind_enum) == {'user_specific', 'system', 'misc'}

    def test_schema_kind_does_not_include_moment(self):
        kind_enum = TOOL_SCHEMA['input_schema']['properties']['kind']['enum']
        assert 'moment' not in kind_enum

    def test_schema_required_is_action_only(self):
        required = TOOL_SCHEMA['input_schema']['required']
        assert required == ['action']

    def test_schema_has_key_value_query_properties(self):
        props = TOOL_SCHEMA['input_schema']['properties']
        assert 'key' in props
        assert 'value' in props
        assert 'query' in props


# ── Store ────────────────────────────────────────────────────────────

class TestHandleMemoryStore:

    def _mock_dgs(self, return_value):
        mock = MagicMock()
        mock.store.return_value = return_value
        return mock

    def test_store_calls_dgs_store_with_correct_args(self):
        mock_dgs = self._mock_dgs({'id': 1, 'key': 'user_name', 'value': 'Dylan'})

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {
                'action': 'store',
                'key': 'user_name',
                'value': 'Dylan',
                'kind': 'user_specific',
            })

        mock_dgs.store.assert_called_once_with(
            kind='user_specific',
            key='user_name',
            value='Dylan',
            source='skill:memory:store:topic',
        )
        assert 'Stored' in result
        assert 'user_name' in result

    def test_store_defaults_kind_to_user_specific(self):
        mock_dgs = self._mock_dgs({'id': 2, 'key': 'fav_food', 'value': 'pizza'})

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {
                'action': 'store',
                'key': 'fav_food',
                'value': 'pizza',
            })

        mock_dgs.store.assert_called_once()
        assert mock_dgs.store.call_args.kwargs['kind'] == 'user_specific'
        assert 'Stored' in result

    def test_store_missing_key_returns_error(self):
        result = handle_memory('topic', {'action': 'store', 'value': 'something'})
        assert 'Error' in result
        assert 'key' in result

    def test_store_missing_value_returns_error(self):
        result = handle_memory('topic', {'action': 'store', 'key': 'my_key'})
        assert 'Error' in result
        assert 'value' in result

    def test_store_dgs_returns_none_gives_error(self):
        mock_dgs = self._mock_dgs(None)

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {
                'action': 'store',
                'key': 'some_key',
                'value': 'some_value',
            })

        assert 'failed' in result.lower() or 'error' in result.lower() or 'Store' in result

    def test_store_true_contradiction_returns_conflict_question(self):
        conflict_result = {
            'conflict': True,
            'classification': 'true_contradiction',
            'existing': {'value': 'Malta'},
            'proposed_key': 'hometown',
            'proposed_value': 'London',
        }
        mock_dgs = self._mock_dgs(conflict_result)

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {
                'action': 'store',
                'key': 'hometown',
                'value': 'London',
            })

        assert 'Conflict detected' in result
        assert 'Malta' in result
        assert 'London' in result

    def test_store_ambiguous_contradiction_returns_prompt(self):
        conflict_result = {
            'conflict': True,
            'classification': 'ambiguous',
            'existing': {'value': 'blue'},
            'proposed_key': 'eye_colour',
            'proposed_value': 'green',
        }
        mock_dgs = self._mock_dgs(conflict_result)

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {
                'action': 'store',
                'key': 'eye_colour',
                'value': 'green',
            })

        assert 'Should I store' in result
        assert 'green' in result

    def test_store_system_kind_accepted(self):
        mock_dgs = self._mock_dgs({'id': 3, 'key': 'tone', 'value': 'terse'})

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {
                'action': 'store',
                'key': 'tone',
                'value': 'terse',
                'kind': 'system',
            })

        assert mock_dgs.store.call_args.kwargs['kind'] == 'system'
        assert 'Stored' in result

    def test_store_misc_kind_accepted(self):
        mock_dgs = self._mock_dgs({'id': 4, 'key': 'scratch', 'value': 'temp'})

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {
                'action': 'store',
                'key': 'scratch',
                'value': 'temp',
                'kind': 'misc',
            })

        assert mock_dgs.store.call_args.kwargs['kind'] == 'misc'
        assert 'Stored' in result


# ── Recall ───────────────────────────────────────────────────────────

class TestHandleMemoryRecall:

    def test_recall_missing_query_returns_error(self):
        result = handle_memory('topic', {'action': 'recall', 'query': ''})
        assert 'Error' in result

    def test_recall_calls_dgs_recall(self):
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = [
            {'id': 1, 'key': 'user_name', 'value': 'Dylan',
             'retrieval_weight': 0.9, 'evidence_count': 3, 'composite_score': 2.5},
        ]

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs), \
             patch('services.innate_skills.memory_skill._search_episodes', return_value=([], '0 matches')), \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            result = handle_memory('topic', {'action': 'recall', 'query': 'user name'})

        assert 'user_name' in result
        assert 'Dylan' in result

    def test_recall_formats_relevance_high(self):
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = [
            {'id': 1, 'key': 'fav_food', 'value': 'pizza',
             'retrieval_weight': 0.8, 'evidence_count': 1, 'composite_score': 1.5},
        ]

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs), \
             patch('services.innate_skills.memory_skill._search_episodes', return_value=([], '0 matches')), \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            result = handle_memory('topic', {'action': 'recall', 'query': 'food'})

        assert 'relevance:high' in result

    def test_recall_formats_relevance_medium(self):
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = [
            {'id': 1, 'key': 'city', 'value': 'Valletta',
             'retrieval_weight': 0.5, 'evidence_count': 1, 'composite_score': 1.0},
        ]

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs), \
             patch('services.innate_skills.memory_skill._search_episodes', return_value=([], '0 matches')), \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            result = handle_memory('topic', {'action': 'recall', 'query': 'city'})

        assert 'relevance:medium' in result

    def test_recall_formats_relevance_low(self):
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = [
            {'id': 1, 'key': 'old_fact', 'value': 'stale',
             'retrieval_weight': 0.2, 'evidence_count': 1, 'composite_score': 0.3},
        ]

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs), \
             patch('services.innate_skills.memory_skill._search_episodes', return_value=([], '0 matches')), \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            result = handle_memory('topic', {'action': 'recall', 'query': 'fact'})

        assert 'relevance:low' in result

    def test_recall_empty_returns_no_memories(self):
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = []

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs), \
             patch('services.innate_skills.memory_skill._search_episodes', return_value=([], '0 matches')), \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            result = handle_memory('topic', {'action': 'recall', 'query': 'ghost'})

        assert 'No memories found' in result
        assert 'ghost' in result

    def test_recall_output_format_id_relevance_value(self):
        """Output must match: [id:{key},relevance:{level}] {value}"""
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = [
            {'id': 1, 'key': 'partner_name', 'value': 'Sarah',
             'retrieval_weight': 0.75, 'evidence_count': 2, 'composite_score': 2.1},
        ]

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs), \
             patch('services.innate_skills.memory_skill._search_episodes', return_value=([], '0 matches')), \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            result = handle_memory('topic', {'action': 'recall', 'query': 'partner'})

        assert '[id:partner_name,relevance:high] Sarah' in result


# ── Invalid / unknown actions ────────────────────────────────────────

class TestHandleMemoryInvalid:

    def test_unknown_action_returns_error(self):
        result = handle_memory('topic', {'action': 'explode'})
        assert 'Unknown action' in result
        assert 'explode' in result

    def test_unknown_action_error_lists_valid_actions(self):
        result = handle_memory('topic', {'action': 'explode'})
        assert 'store' in result
        assert 'recall' in result
        assert 'reflect' in result

    def test_update_action_rejected_as_unknown(self):
        result = handle_memory('topic', {'action': 'update', 'key': 'k', 'value': 'v'})
        assert 'Unknown action' in result

    def test_forget_action_rejected_as_unknown(self):
        result = handle_memory('topic', {'action': 'forget', 'key': 'something'})
        assert 'Unknown action' in result

    def test_missing_action_defaults_to_recall_then_errors_no_query(self):
        result = handle_memory('topic', {})
        assert 'Error' in result  # "no query specified for recall"

    def test_no_knowledge_service_imports(self):
        """memory_skill must not import KnowledgeService anywhere."""
        import services.innate_skills.memory_skill as ms
        import inspect
        src = inspect.getsource(ms)
        assert 'KnowledgeService' not in src
        assert 'knowledge_service' not in src


# ── Dead code removed ────────────────────────────────────────────────

class TestDeadCodeRemoved:

    def test_auto_classify_does_not_exist(self):
        import services.innate_skills.memory_skill as ms
        assert not hasattr(ms, '_auto_classify')

    def test_check_trait_contradiction_does_not_exist(self):
        import services.innate_skills.memory_skill as ms
        assert not hasattr(ms, '_check_trait_contradiction')

    def test_handle_update_does_not_exist(self):
        import services.innate_skills.memory_skill as ms
        assert not hasattr(ms, '_handle_update')

    def test_trait_keys_constant_does_not_exist(self):
        import services.innate_skills.memory_skill as ms
        assert not hasattr(ms, '_TRAIT_KEYS')

    def test_preference_keys_constant_does_not_exist(self):
        import services.innate_skills.memory_skill as ms
        assert not hasattr(ms, '_PREFERENCE_KEYS')

    def test_procedure_keys_constant_does_not_exist(self):
        import services.innate_skills.memory_skill as ms
        assert not hasattr(ms, '_PROCEDURE_KEYS')

    def test_search_transcript_does_not_exist(self):
        import services.innate_skills.memory_skill as ms
        assert not hasattr(ms, '_search_transcript')

    def test_format_empty_does_not_exist(self):
        import services.innate_skills.memory_skill as ms
        assert not hasattr(ms, '_format_empty')

    def test_salience_label_does_not_exist(self):
        import services.innate_skills.memory_skill as ms
        assert not hasattr(ms, '_salience_label')


# ── _search_data_graph unit ──────────────────────────────────────────

class TestSearchDataGraph:

    def test_returns_id_relevance_text_shape(self):
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = [
            {'id': 1, 'key': 'user_name', 'value': 'Dylan',
             'retrieval_weight': 0.9, 'evidence_count': 1, 'composite_score': 2.0},
        ]

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            hits, status = _search_data_graph("name", 10)

        assert len(hits) == 1
        assert hits[0]['id'] == 'user_name'
        assert hits[0]['text'] == 'Dylan'
        assert hits[0]['relevance'] == 'high'
        assert '1 matches' in status

    def test_empty_result_returns_zero_matches_status(self):
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = []

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            hits, status = _search_data_graph("ghost", 10)

        assert hits == []
        assert '0 matches' in status

    def test_exception_returns_error_status(self):
        mock_dgs = MagicMock()
        mock_dgs.recall.side_effect = RuntimeError("db exploded")

        with patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            hits, status = _search_data_graph("anything", 10)

        assert hits == []
        assert 'error' in status


# ── Reflect ──────────────────────────────────────────────────────────

class TestHandleMemoryReflect:

    def test_reflect_missing_query_returns_error(self):
        result = handle_memory('topic', {'action': 'reflect', 'query': ''})
        assert 'Error' in result

    def test_reflect_no_episodes_no_dg_returns_no_memories(self):
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = []

        with patch('services.innate_skills.memory_skill.recall_episodes', return_value=([], '0 matches')), \
             patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {'action': 'reflect', 'query': 'cats'})

        assert 'No memories found' in result
        assert 'cats' in result

    def test_reflect_with_episode_shows_main_memory(self):
        fake_ep = {
            'id': 'ep-1', 'gist': 'User has a cat named Blake',
            'salience': 0.9, 'composite_score': 50,
            'transcript_ids': '[]', 'consolidated_from': '[]',
        }
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = []

        with patch('services.innate_skills.memory_skill.recall_episodes',
                   return_value=([fake_ep], '1 matches')), \
             patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {'action': 'reflect', 'query': 'cat'})

        assert '## Main Memory' in result
        assert 'User has a cat named Blake' in result

    def test_reflect_includes_supporting_facts(self):
        fake_ep = {
            'id': 'ep-1', 'gist': 'User has a cat',
            'salience': 0.9, 'composite_score': 50,
            'transcript_ids': '[]', 'consolidated_from': '[]',
        }
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = [
            {'id': 10, 'key': 'cat_name', 'value': 'Blake',
             'retrieval_weight': 0.9, 'composite_score': 2.0},
            {'id': 11, 'key': 'cat_age', 'value': '16 years',
             'retrieval_weight': 0.5, 'composite_score': 1.0},
        ]

        with patch('services.innate_skills.memory_skill.recall_episodes',
                   return_value=([fake_ep], '1 matches')), \
             patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {'action': 'reflect', 'query': 'cat'})

        assert '### Supporting facts:' in result
        assert '[cat_name] Blake' in result
        assert '[cat_age] 16 years' in result

    def test_reflect_dg_limited_to_2(self):
        """Data graph results must be capped at 2 in reflect mode."""
        fake_ep = {
            'id': 'ep-1', 'gist': 'Test episode',
            'salience': 0.5, 'composite_score': 30,
            'transcript_ids': '[]', 'consolidated_from': '[]',
        }
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = [
            {'id': i, 'key': f'k{i}', 'value': f'v{i}',
             'retrieval_weight': 0.9, 'composite_score': 2.0}
            for i in range(5)
        ]

        with patch('services.innate_skills.memory_skill.recall_episodes',
                   return_value=([fake_ep], '1 matches')), \
             patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {'action': 'reflect', 'query': 'test'})

        # _search_data_graph called with limit=2 and non-document kinds
        mock_dgs.recall.assert_called_once_with(
            query='test',
            kinds=['user_specific', 'system', 'misc', 'moment'],
            limit=2,
        )

    def test_reflect_no_episodes_falls_back_to_dg_only(self):
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = [
            {'id': 1, 'key': 'cat_name', 'value': 'Blake',
             'retrieval_weight': 0.9, 'composite_score': 2.0},
        ]

        with patch('services.innate_skills.memory_skill.recall_episodes',
                   return_value=([], '0 matches')), \
             patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):
            result = handle_memory('topic', {'action': 'reflect', 'query': 'cat'})

        assert '### Supporting facts:' in result
        assert '[cat_name] Blake' in result

    def test_reflect_action_accepted_by_schema(self):
        action_enum = TOOL_SCHEMA['input_schema']['properties']['action']['enum']
        assert 'reflect' in action_enum


# ── _search_document_artifacts — real DataGraphService ───────────────────────

class TestSearchDocumentArtifacts:
    """_search_document_artifacts returns hits with correct format using real DB."""

    def test_empty_db_returns_empty_list(self, _doc_svc):
        with patch('services.data_graph_service.get_data_graph_service',
                   return_value=_doc_svc):
            hits = _search_document_artifacts("solar panels")
        assert hits == []

    def test_hit_text_has_document_fragment_prefix(self, _doc_svc):
        """Each hit text is prefixed with [document(id:{doc_id},type:fragment)]."""
        _doc_svc.store(KIND_DOCUMENT, 'doc:solar:000',
                       'Solar panels convert sunlight to electricity.',
                       source='document:solar')

        with patch('services.data_graph_service.get_data_graph_service',
                   return_value=_doc_svc):
            hits = _search_document_artifacts("solar")

        assert len(hits) >= 1
        assert hits[0]['text'].startswith('[document(id:solar,type:fragment)]')

    def test_hit_text_contains_artifact_value(self, _doc_svc):
        """The stored artifact value appears in the hit text."""
        _doc_svc.store(KIND_DOCUMENT, 'doc:energy:000',
                       'Photovoltaic efficiency reached 22 percent.',
                       source='document:energy')

        with patch('services.data_graph_service.get_data_graph_service',
                   return_value=_doc_svc):
            hits = _search_document_artifacts("photovoltaic efficiency")

        assert len(hits) >= 1
        assert 'Photovoltaic efficiency reached 22 percent.' in hits[0]['text']

    def test_doc_id_extracted_from_source_field(self, _doc_svc):
        """doc_id in the formatted prefix comes from the source='document:{doc_id}' field."""
        _doc_svc.store(KIND_DOCUMENT, 'doc:warranty2026:000',
                       'This warranty covers hardware defects.',
                       source='document:warranty2026')

        with patch('services.data_graph_service.get_data_graph_service',
                   return_value=_doc_svc):
            hits = _search_document_artifacts("warranty hardware")

        assert len(hits) >= 1
        assert '[document(id:warranty2026,type:fragment)]' in hits[0]['text']

    def test_doc_id_falls_back_to_key_when_source_missing(self, _doc_svc, _doc_db):
        """When source is NULL/empty, doc_id is extracted from the key 'doc:{id}:{idx}'."""
        # Insert a row with no source via raw SQL
        with _doc_db.connection() as conn:
            conn.execute("""
                INSERT INTO data_graph
                    (kind, key, value, active, storage_strength, retrieval_weight,
                     salience_score, evidence_count, first_seen_at, last_confirmed_at, source)
                VALUES (?, ?, ?, 1, 0.5, 1.0, 0.0, 1, datetime('now'), datetime('now'), NULL)
            """, (KIND_DOCUMENT, 'doc:fallbackdoc:000', 'Fallback content from key field.'))
            rowid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO data_graph_fts(rowid, key, value, kind, search_queries) VALUES (?, ?, ?, ?, ?)",
                (rowid, 'doc:fallbackdoc:000', 'Fallback content from key field.', KIND_DOCUMENT, '')
            )

        with patch('services.data_graph_service.get_data_graph_service',
                   return_value=_doc_svc):
            hits = _search_document_artifacts("fallback content")

        assert len(hits) >= 1
        assert '[document(id:fallbackdoc,type:fragment)]' in hits[0]['text']

    def test_hit_has_id_key_confidence_fields(self, _doc_svc):
        """Each hit dict has 'id', 'text', 'relevance', 'confidence' fields."""
        _doc_svc.store(KIND_DOCUMENT, 'doc:fields:000',
                       'Testing the hit object shape.',
                       source='document:fields')

        with patch('services.data_graph_service.get_data_graph_service',
                   return_value=_doc_svc):
            hits = _search_document_artifacts("hit object shape")

        assert len(hits) >= 1
        hit = hits[0]
        assert 'id' in hit
        assert 'text' in hit
        assert 'relevance' in hit
        assert 'confidence' in hit

    def test_relevance_is_always_high(self, _doc_svc):
        """Document artifact hits always carry relevance='high'."""
        _doc_svc.store(KIND_DOCUMENT, 'doc:highrel:000',
                       'Document artifacts are always high relevance.',
                       source='document:highrel')

        with patch('services.data_graph_service.get_data_graph_service',
                   return_value=_doc_svc):
            hits = _search_document_artifacts("document artifacts relevance")

        assert len(hits) >= 1
        for hit in hits:
            assert hit['relevance'] == 'high'

    def test_limit_caps_results(self, _doc_svc):
        """limit parameter caps the number of results returned."""
        for i in range(5):
            _doc_svc.store(KIND_DOCUMENT, f'doc:limitdoc:{i:03d}',
                           f'Solar panel efficiency measurement {i}.',
                           source='document:limitdoc')

        with patch('services.data_graph_service.get_data_graph_service',
                   return_value=_doc_svc):
            hits = _search_document_artifacts("solar panel efficiency", limit=2)

        assert len(hits) <= 2

    def test_exception_returns_empty_list(self, _doc_svc):
        """If the data graph service raises, returns [] gracefully."""
        broken = MagicMock()
        broken.recall.side_effect = RuntimeError("connection lost")

        with patch('services.data_graph_service.get_data_graph_service',
                   return_value=broken):
            hits = _search_document_artifacts("anything")

        assert hits == []

    def test_non_document_rows_are_excluded(self, _doc_svc):
        """Rows with kind != KIND_DOCUMENT do not appear in document artifact hits."""
        from services.data_graph_service import KIND_USER_SPECIFIC
        _doc_svc.store(KIND_USER_SPECIFIC, 'user_fact',
                       'Solar panels are efficient.', source='skill:memory')
        _doc_svc.store(KIND_DOCUMENT, 'doc:only:000',
                       'Document artifact about solar efficiency.', source='document:only')

        with patch('services.data_graph_service.get_data_graph_service',
                   return_value=_doc_svc):
            hits = _search_document_artifacts("solar efficiency")

        # All returned hits come from document kind — no user_specific rows
        for hit in hits:
            assert 'document(id:' in hit['text']

    def test_recall_integrates_document_artifacts_in_output(self, _doc_svc):
        """End-to-end: handle_memory recall surfaces stored document artifacts."""
        _doc_svc.store(KIND_DOCUMENT, 'doc:integration:000',
                       'Battery storage complements solar installations.',
                       source='document:integration')

        with patch('services.data_graph_service.get_data_graph_service', return_value=_doc_svc), \
             patch('services.innate_skills.memory_skill._search_episodes', return_value=([], '0 matches')), \
             patch('services.innate_skills.memory_skill._store_fok_signal'):
            result = handle_memory('topic', {'action': 'recall', 'query': 'battery storage solar'})

        assert 'Battery storage complements solar installations.' in result
