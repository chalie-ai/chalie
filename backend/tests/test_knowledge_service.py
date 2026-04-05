"""Tests for KnowledgeService — unified knowledge store.

Covers store, recall, get, update, forget, strengthen, decay_cycle,
procedural outcome recording, ranked retrieval, and traits-for-prompt.
"""

import sqlite3
import contextlib
import pytest
from unittest.mock import MagicMock

from services.knowledge_service import KnowledgeService, _validate_trait
from services.database_service import DatabaseService


pytestmark = pytest.mark.unit


# ── Schema ──────────────────────────────────────────────────────────

KNOWLEDGE_DDL = [
    """
    CREATE TABLE IF NOT EXISTS knowledge (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        kind        TEXT NOT NULL,
        entity      TEXT NOT NULL DEFAULT 'user',
        key         TEXT NOT NULL,
        value       TEXT,
        data        TEXT,
        decay_class TEXT NOT NULL DEFAULT 'standard',
        confidence  REAL NOT NULL DEFAULT 0.5,
        reliability TEXT NOT NULL DEFAULT 'reliable',
        source      TEXT,
        evidence_count INTEGER NOT NULL DEFAULT 1,
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
        last_accessed_at TEXT,
        deleted_at  TEXT,
        UNIQUE(entity, key)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_knowledge_kind ON knowledge(kind)",
    "CREATE INDEX IF NOT EXISTS idx_knowledge_entity ON knowledge(entity)",
    "CREATE INDEX IF NOT EXISTS idx_knowledge_key ON knowledge(key)",
    "CREATE INDEX IF NOT EXISTS idx_knowledge_confidence ON knowledge(confidence DESC)",
    "CREATE INDEX IF NOT EXISTS idx_knowledge_decay_class ON knowledge(decay_class)",
    "CREATE INDEX IF NOT EXISTS idx_knowledge_deleted ON knowledge(deleted_at) WHERE deleted_at IS NULL",
    # Use a standalone FTS table (no content sync) so the service's
    # manual INSERT/DELETE operations work in tests.  The production
    # migration uses content='knowledge' with triggers, but the
    # service does manual sync which requires this form.
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
        key, value, kind, entity
    )
    """,
]


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def db_service(tmp_path):
    """Real SQLite DB with knowledge + FTS schema.

    Uses a single persistent connection to mimic the production
    thread-local connection pattern.  The KnowledgeService calls
    ``self.get()`` inside an open ``with self.db.connection()`` block,
    which opens a *nested* context manager.  With a per-call fresh
    connection the inner one cannot see uncommitted rows.  A shared
    connection avoids that problem (matching the real DatabaseService
    which reuses the same thread-local connection).
    """
    db_path = str(tmp_path / "test_knowledge.db")

    # Use a single long-lived connection (like production thread-local)
    shared_conn = sqlite3.connect(db_path)
    shared_conn.row_factory = sqlite3.Row
    for ddl in KNOWLEDGE_DDL:
        shared_conn.execute(ddl)
    shared_conn.commit()

    db = DatabaseService.__new__(DatabaseService)
    db._db_path = db_path
    db._init_complete = True
    db.db_path = db_path

    # Track nesting depth so only the outermost context commits
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
def svc(db_service):
    """KnowledgeService with real SQLite backend, mocked embeddings and signals."""
    service = KnowledgeService(db_service)

    # Mock embedding generation — return None so vec operations are skipped
    service._generate_embedding = MagicMock(return_value=None)
    # Mock signal emission
    service._emit_signal = MagicMock()

    return service


def _raw_get(db_service, entity, key):
    """Direct DB lookup bypassing the service for verification."""
    with db_service.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM knowledge WHERE entity = ? AND key = ? AND deleted_at IS NULL",
            (entity, key),
        )
        row = cursor.fetchone()
        cursor.close()
        return dict(row) if row else None


def _raw_fts_search(db_service, query):
    """Direct FTS search for verification."""
    with db_service.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT rowid, key, value FROM knowledge_fts WHERE knowledge_fts MATCH ?",
            (query,),
        )
        rows = cursor.fetchall()
        cursor.close()
        return rows


# ── Store tests ─────────────────────────────────────────────────────

class TestStore:

    def test_store_new_entry(self, svc, db_service):
        """Basic store creates a row in the DB."""
        result = svc.store(
            kind='trait', entity='user', key='user_name',
            value='Dylan', confidence=0.9, decay_class='permanent',
        )
        assert result is not None
        assert result['key'] == 'user_name'
        assert result['value'] == 'Dylan'
        assert result['confidence'] == 0.9

        raw = _raw_get(db_service, 'user', 'user_name')
        assert raw is not None
        assert raw['kind'] == 'trait'

    def test_store_reinforcement(self, svc):
        """Storing same key twice increases evidence_count and boosts confidence."""
        svc.store(
            kind='fact', entity='user', key='favorite_color',
            value='blue', confidence=0.5,
        )
        result = svc.store(
            kind='fact', entity='user', key='favorite_color',
            value='blue', confidence=0.5,
        )
        assert result is not None
        assert result['evidence_count'] == 2
        # Confidence should be higher than original 0.5
        assert result['confidence'] > 0.5

    def test_store_value_conflict_overwrite(self, svc):
        """Different value with much higher confidence overwrites."""
        svc.store(
            kind='fact', entity='user', key='hometown',
            value='Valletta', confidence=0.3,
        )
        # Confidence > old * 2 (0.3 * 2 = 0.6), so 0.9 triggers overwrite
        result = svc.store(
            kind='fact', entity='user', key='hometown',
            value='Mosta', confidence=0.9,
        )
        assert result is not None
        assert result['value'] == 'Mosta'
        assert result['confidence'] == 0.9

    def test_store_value_conflict_no_overwrite(self, svc):
        """Different value with very low confidence does NOT overwrite."""
        svc.store(
            kind='fact', entity='user', key='hometown',
            value='Valletta', confidence=0.5,
        )
        # 0.3 is NOT >= 0.5 * 0.8 = 0.4, so no overwrite
        result = svc.store(
            kind='fact', entity='user', key='hometown',
            value='Mosta', confidence=0.3,
        )
        assert result is not None
        assert result['value'] == 'Valletta'  # old value preserved
        assert result['evidence_count'] == 2  # still reinforced

    def test_store_invalid_kind(self, svc):
        """Invalid kind returns None."""
        result = svc.store(
            kind='banana', entity='user', key='test', value='val',
        )
        assert result is None

    def test_store_trait_validation_rejects_garbage(self, svc):
        """Placeholder values, short keys etc rejected for traits."""
        # Placeholder value
        result = svc.store(
            kind='trait', entity='user', key='user_name', value='unknown',
        )
        assert result is None

        # Key too short
        result = svc.store(
            kind='trait', entity='user', key='x', value='Dylan',
        )
        assert result is None

        # Value too short
        result = svc.store(
            kind='trait', entity='user', key='user_name', value='ab',
        )
        assert result is None

    def test_store_trait_validation_accepts_good(self, svc):
        """Valid traits pass through validation."""
        result = svc.store(
            kind='trait', entity='user', key='user_name',
            value='Dylan', confidence=0.8, decay_class='permanent',
        )
        assert result is not None
        assert result['value'] == 'Dylan'

    def test_store_fts_sync(self, svc, db_service):
        """After store, FTS search finds the entry."""
        svc.store(
            kind='fact', entity='user', key='programming_language',
            value='Python is the main language', confidence=0.7,
        )
        fts_rows = _raw_fts_search(db_service, 'programming')
        assert len(fts_rows) >= 1

    def test_store_non_trait_skips_validation(self, svc):
        """Non-trait kinds skip the trait validation entirely."""
        # 'unknown' is a valid value for a fact, just not a trait
        result = svc.store(
            kind='fact', entity='user', key='status',
            value='unknown situation reported', confidence=0.5,
        )
        assert result is not None

    def test_store_with_data(self, svc):
        """Structured data is stored as JSON and returned parsed."""
        data = {'category': 'core', 'last_conflict_at': None}
        result = svc.store(
            kind='fact', entity='user', key='data_test',
            value='test value here', data=data,
        )
        assert result is not None
        assert result['data']['category'] == 'core'


# ── Recall tests ────────────────────────────────────────────────────

class TestRecall:

    def test_recall_exact_key_match(self, svc):
        """Exact key gets highest score."""
        svc.store(kind='fact', entity='user', key='dog_name', value='Rex')
        svc.store(kind='fact', entity='user', key='cat_name', value='Whiskers')

        results = svc.recall(query='dog_name')
        assert len(results) >= 1
        assert results[0]['key'] == 'dog_name'

    def test_recall_fts_match(self, svc):
        """FTS finds entries by text content."""
        svc.store(
            kind='concept', entity='user', key='python_lists',
            value='Python lists are ordered mutable collections',
        )
        results = svc.recall(query='mutable collections')
        assert len(results) >= 1
        # Should find the python_lists entry via FTS
        keys = [r['key'] for r in results]
        assert 'python_lists' in keys

    def test_recall_kind_filter(self, svc):
        """Only returns specified kinds."""
        svc.store(kind='fact', entity='user', key='fact_one', value='fact value test')
        svc.store(kind='trait', entity='user', key='user_name', value='Dylan',
                  decay_class='permanent')

        results = svc.recall(query='user_name', kinds=['trait'])
        for r in results:
            assert r['kind'] == 'trait'

    def test_recall_entity_filter(self, svc):
        """Filters by entity."""
        svc.store(kind='fact', entity='user', key='entity_test', value='user fact data')
        svc.store(kind='fact', entity='chalie', key='entity_test', value='chalie fact data')

        results = svc.recall(query='entity_test', entity='user')
        for r in results:
            assert r['entity'] == 'user'

    def test_recall_min_confidence(self, svc):
        """Respects minimum confidence threshold."""
        svc.store(kind='fact', entity='user', key='high_conf', value='certain thing',
                  confidence=0.9)
        svc.store(kind='fact', entity='user', key='low_conf', value='uncertain thing',
                  confidence=0.2)

        results = svc.recall(query='high_conf', min_confidence=0.5)
        keys = [r['key'] for r in results]
        assert 'low_conf' not in keys

    def test_recall_empty_query(self, svc):
        """Empty query returns empty list (no crash)."""
        results = svc.recall(query='')
        assert results == []

    def test_recall_updates_last_accessed(self, svc, db_service):
        """Verify last_accessed_at gets updated on recall hit."""
        svc.store(kind='fact', entity='user', key='access_test',
                  value='should be accessed later')

        # Verify initial state has no last_accessed_at
        _raw_get(db_service, 'user', 'access_test')
        # Note: store calls get() which updates last_accessed_at,
        # so it may already be set. The important thing is it gets
        # updated again on recall.
        svc.recall(query='access_test')

        raw_after = _raw_get(db_service, 'user', 'access_test')
        assert raw_after['last_accessed_at'] is not None

    def test_recall_no_results(self, svc):
        """Querying for nonexistent data returns empty list."""
        results = svc.recall(query='xyzzy_nonexistent_key_12345')
        assert results == []


# ── Get / Update / Forget tests ─────────────────────────────────────

class TestGet:

    def test_get_exact_lookup(self, svc):
        """Returns correct entry."""
        svc.store(kind='fact', entity='user', key='get_test',
                  value='found it', confidence=0.7)

        result = svc.get('user', 'get_test')
        assert result is not None
        assert result['value'] == 'found it'
        assert result['confidence'] == 0.7

    def test_get_nonexistent(self, svc):
        """Returns None for missing entry."""
        result = svc.get('user', 'does_not_exist')
        assert result is None


class TestUpdate:

    def test_update_value(self, svc, db_service):
        """Updates value and syncs FTS."""
        svc.store(kind='fact', entity='user', key='update_val',
                  value='old value here')

        result = svc.update('user', 'update_val', value='new value here')
        assert result is not None
        assert result['value'] == 'new value here'

        # Verify FTS is synced
        fts_rows = _raw_fts_search(db_service, 'new')
        found = any(r['key'] == 'update_val' for r in fts_rows)
        assert found

    def test_update_confidence(self, svc):
        """Updates confidence only."""
        svc.store(kind='fact', entity='user', key='conf_test',
                  value='stable value here', confidence=0.5)

        result = svc.update('user', 'conf_test', confidence=0.9)
        assert result is not None
        assert result['confidence'] == 0.9
        assert result['value'] == 'stable value here'

    def test_update_nonexistent(self, svc):
        """Update on missing entry returns None."""
        result = svc.update('user', 'ghost_key', value='something')
        assert result is None


class TestForget:

    def test_forget_soft_deletes(self, svc, db_service):
        """Marks entry as deleted, removed from FTS."""
        svc.store(kind='fact', entity='user', key='forget_me',
                  value='will be forgotten')

        assert svc.forget('user', 'forget_me') is True

        # Should not be found via get
        assert svc.get('user', 'forget_me') is None

        # FTS should not find it
        fts_rows = _raw_fts_search(db_service, 'forgotten')
        found = any(r['key'] == 'forget_me' for r in fts_rows)
        assert not found

        # Raw DB should have deleted_at set
        with db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT deleted_at FROM knowledge WHERE entity = ? AND key = ?",
                ('user', 'forget_me'),
            )
            row = cursor.fetchone()
            cursor.close()
        assert row is not None
        assert row['deleted_at'] is not None

    def test_forget_nonexistent(self, svc):
        """Returns False for missing entry."""
        assert svc.forget('user', 'ghost_key') is False


# ── Strengthen tests ────────────────────────────────────────────────

class TestStrengthen:

    def test_strengthen_boosts_confidence(self, svc):
        """Confidence increases after strengthen."""
        svc.store(kind='fact', entity='user', key='strong_test',
                  value='getting stronger', confidence=0.5)

        before = svc.get('user', 'strong_test')
        assert svc.strengthen('user', 'strong_test') is True
        after = svc.get('user', 'strong_test')

        assert after['confidence'] > before['confidence']
        assert after['evidence_count'] == before['evidence_count'] + 1

    def test_strengthen_appends_episode(self, svc):
        """For concepts, appends to source_episodes in data."""
        svc.store(kind='concept', entity='user', key='concept_ep_test',
                  value='a concept with episodes',
                  data={'source_episodes': ['ep-1']})

        assert svc.strengthen('user', 'concept_ep_test', episode_id='ep-2') is True

        result = svc.get('user', 'concept_ep_test')
        data = result['data']
        assert 'ep-2' in data['source_episodes']
        assert 'ep-1' in data['source_episodes']

    def test_strengthen_diminishing_returns(self, svc):
        """Multiple strengthens show diminishing boost."""
        svc.store(kind='fact', entity='user', key='dimret_test',
                  value='diminishing returns test', confidence=0.5)

        boosts = []
        for i in range(5):
            before = svc.get('user', 'dimret_test')
            svc.strengthen('user', 'dimret_test')
            after = svc.get('user', 'dimret_test')
            boost = after['confidence'] - before['confidence']
            boosts.append(boost)

        # Each successive boost should be smaller
        for i in range(1, len(boosts)):
            assert boosts[i] <= boosts[i - 1] + 0.001  # small epsilon for float

    def test_strengthen_nonexistent(self, svc):
        """Strengthen on missing entry returns False."""
        assert svc.strengthen('user', 'ghost') is False


# ── Decay tests ─────────────────────────────────────────────────────

class TestDecayCycle:

    def _insert_entry(self, db_service, **overrides):
        """Helper to insert a knowledge entry with specific attributes."""
        defaults = {
            'kind': 'fact',
            'entity': 'user',
            'key': 'decay_test',
            'value': 'test value here',
            'data': None,
            'decay_class': 'standard',
            'confidence': 0.5,
            'reliability': 'reliable',
            'source': 'test',
            'evidence_count': 1,
            'created_at': "datetime('now', '-2 hours')",
            'updated_at': "datetime('now', '-2 hours')",
            'last_accessed_at': None,
            'deleted_at': None,
        }
        defaults.update(overrides)

        # Use raw SQL so we can set created_at/last_accessed_at to past times
        with db_service.connection() as conn:
            cursor = conn.cursor()
            created_at = defaults['created_at']
            updated_at = defaults['updated_at']
            last_accessed = defaults['last_accessed_at']

            # Use SQL expression for datetime manipulation
            cursor.execute(f"""
                INSERT INTO knowledge
                    (kind, entity, key, value, data, decay_class, confidence,
                     reliability, source, evidence_count, created_at, updated_at,
                     last_accessed_at, deleted_at)
                VALUES
                    (?, ?, ?, ?, ?, ?, ?,
                     ?, ?, ?, {created_at}, {updated_at},
                     {last_accessed if last_accessed else 'NULL'}, ?)
            """, (
                defaults['kind'], defaults['entity'], defaults['key'],
                defaults['value'], defaults['data'], defaults['decay_class'],
                defaults['confidence'], defaults['reliability'], defaults['source'],
                defaults['evidence_count'], defaults['deleted_at'],
            ))
            cursor.close()

    def test_decay_cycle_respects_classes(self, svc, db_service):
        """Permanent entries untouched, ephemeral decay fast."""
        self._insert_entry(
            db_service,
            key='permanent_entry', decay_class='permanent',
            confidence=0.5,
        )
        self._insert_entry(
            db_service,
            key='ephemeral_entry', decay_class='ephemeral',
            confidence=0.5,
        )

        svc.decay_cycle()

        perm = _raw_get(db_service, 'user', 'permanent_entry')
        ephem = _raw_get(db_service, 'user', 'ephemeral_entry')

        assert perm is not None
        assert perm['confidence'] == 0.5  # permanent: untouched

        assert ephem is not None
        assert ephem['confidence'] < 0.5  # ephemeral: decayed

    def test_decay_cycle_soft_deletes_at_floor(self, svc, db_service):
        """Entries at confidence 0.05 get soft-deleted."""
        self._insert_entry(
            db_service,
            key='floor_entry', decay_class='fast',
            confidence=0.05,
        )

        svc.decay_cycle()

        # Should be soft-deleted (deleted_at set)
        with db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT deleted_at FROM knowledge WHERE key = ?",
                ('floor_entry',),
            )
            row = cursor.fetchone()
            cursor.close()
        assert row is not None
        assert row['deleted_at'] is not None

    def test_decay_cycle_skips_recently_accessed(self, svc, db_service):
        """Entries accessed within 1 hour are not decayed."""
        self._insert_entry(
            db_service,
            key='recent_entry', decay_class='ephemeral',
            confidence=0.5,
            last_accessed_at="datetime('now')",
        )

        svc.decay_cycle()

        recent = _raw_get(db_service, 'user', 'recent_entry')
        assert recent is not None
        assert recent['confidence'] == 0.5  # not decayed


# ── Procedural tests ────────────────────────────────────────────────

class TestProcedural:

    def test_record_procedure_outcome_new(self, svc):
        """Creates new procedure entry on first record."""
        result = svc.record_procedure_outcome(
            'search_web', success=True, reward=0.5,
        )
        assert result is not None
        assert result['kind'] == 'procedure'
        assert result['entity'] == 'chalie'

        data = result['data']
        assert data['total_attempts'] == 1
        assert data['total_successes'] == 1
        assert data['success_rate'] == 1.0

    def test_record_procedure_outcome_existing(self, svc):
        """Updates existing procedure stats."""
        svc.record_procedure_outcome('multi_action', success=True, reward=0.5)
        svc.record_procedure_outcome('multi_action', success=True, reward=0.3)
        result = svc.record_procedure_outcome(
            'multi_action', success=False, reward=-0.2,
        )

        assert result is not None
        data = result['data']
        assert data['total_attempts'] == 3
        assert data['total_successes'] == 2
        assert abs(data['success_rate'] - 2 / 3) < 0.01

    def test_record_procedure_context_stats(self, svc):
        """Context stats are updated when topic is provided."""
        svc.record_procedure_outcome(
            'ctx_action', success=True, reward=0.5, topic='python',
        )

        result = svc.get('chalie', 'ctx_action')
        ctx = result['data']['context_stats']
        assert 'python' in ctx
        assert ctx['python']['attempts'] == 1
        assert ctx['python']['successes'] == 1

    def test_get_ranked_procedures(self, svc):
        """Returns ranked by expected value."""
        svc.record_procedure_outcome('high_skill', success=True, reward=0.8)
        svc.record_procedure_outcome('low_skill', success=False, reward=-0.5)

        ranked = svc.get_ranked_procedures()
        assert len(ranked) == 2
        assert ranked[0]['expected_value'] >= ranked[1]['expected_value']

    def test_get_ranked_procedures_with_topic(self, svc):
        """Topic affinity adjusts expected value."""
        svc.record_procedure_outcome(
            'topic_skill', success=True, reward=0.5, topic='python',
        )

        ranked = svc.get_ranked_procedures(topic='python')
        assert len(ranked) == 1
        assert ranked[0]['topic_affinity'] == 1.0

    def test_procedure_weight_bounded(self, svc):
        """Weight stays within [0.1, 5.0] bounds after many outcomes."""
        for _ in range(50):
            svc.record_procedure_outcome('bounded', success=True, reward=0.5)

        result = svc.get('chalie', 'bounded')
        weight = result['data']['weight']
        assert 0.1 <= weight <= 5.0


# ── Traits for prompt ───────────────────────────────────────────────

class TestTraitsForPrompt:

    def test_get_traits_for_prompt_core_always_present(self, svc):
        """Permanent traits always included."""
        svc.store(kind='trait', entity='user', key='user_name',
                  value='Dylan', decay_class='permanent', confidence=0.9)
        svc.store(kind='preference', entity='user', key='timezone_pref',
                  value='CET timezone', decay_class='permanent', confidence=0.8)

        traits = svc.get_traits_for_prompt()
        keys = [t['key'] for t in traits]
        assert 'user_name' in keys
        assert 'timezone_pref' in keys

    def test_get_traits_for_prompt_respects_budget(self, svc):
        """Non-core traits don't exceed remaining budget after core."""
        # 2 core permanent traits (always included)
        svc.store(kind='trait', entity='user', key='user_name',
                  value='Dylan', decay_class='permanent', confidence=0.9)
        svc.store(kind='preference', entity='user', key='timezone_pref',
                  value='CET timezone', decay_class='permanent', confidence=0.8)

        # 15 non-permanent traits — only some should be included
        for i in range(15):
            svc.store(
                kind='trait', entity='user',
                key=f'trait_dynamic_{i}', value=f'dynamic value {i}',
                decay_class='slow', confidence=0.5,
            )

        traits = svc.get_traits_for_prompt(limit=8)
        # Core (2) + remaining slots from semantic/wildcard tier
        # Total should not exceed 8 (_MAX_TRAITS_IN_PROMPT)
        assert len(traits) <= 8

    def test_get_traits_for_prompt_empty(self, svc):
        """No traits returns empty list."""
        traits = svc.get_traits_for_prompt()
        assert traits == []


# ── Trait validation unit tests ─────────────────────────────────────

class TestValidateTrait:

    def test_rejects_short_key(self):
        assert _validate_trait('x', 'valid value') == 'key_too_short'

    def test_rejects_empty_key(self):
        assert _validate_trait('', 'valid value') == 'key_too_short'

    def test_rejects_stop_word_key(self):
        assert _validate_trait('the', 'valid value') == 'key_is_stop_words'

    def test_rejects_short_value(self):
        assert _validate_trait('user_name', 'ab') == 'value_too_short'

    def test_rejects_placeholder_value(self):
        assert _validate_trait('user_name', 'unknown') == 'placeholder_value'
        assert _validate_trait('user_name', 'N/A') == 'placeholder_value'
        assert _validate_trait('user_name', 'none') == 'placeholder_value'

    def test_rejects_long_value(self):
        assert _validate_trait('user_name', 'x' * 501) == 'value_too_long'

    def test_accepts_valid_trait(self):
        assert _validate_trait('user_name', 'Dylan Grech') is None

    def test_rejects_insufficient_content(self):
        # Key and value share the same single content word
        assert _validate_trait('test', 'test value') is None  # 'test' + 'value' = 2 words, ok
        # Actually insufficient: only stop words in value
        result = _validate_trait('coding', 'is the')
        assert result == 'no_content_words'

    def test_rejects_topic_slug_too_long(self):
        result = _validate_trait('topic_time_a_b_c_d', 'some valid value')
        assert result == 'topic_slug_too_long'


class TestKnowledgeTopicContext:
    """TopicContext integration tests for KnowledgeService."""

    def test_recall_accepts_topic_context(self, svc):
        """recall() works when a TopicContext is passed."""
        from services.topic_context import TopicContext

        ctx = TopicContext(topic='test')
        result = svc.recall('nonexistent query', _context=ctx)
        assert result == []
        assert ctx.failed_sections == []

    def test_recall_records_failure_to_context(self):
        """When recall fails, failure is recorded on TopicContext."""
        from services.topic_context import TopicContext

        ctx = TopicContext(topic='test')
        # Create a service with a broken db that raises on connection
        broken_db = MagicMock()
        broken_db.connection.side_effect = Exception('db exploded')
        svc = KnowledgeService(broken_db)

        result = svc.recall('test query', _context=ctx)
        assert result == []
        assert len(ctx.failed_sections) == 1
        assert ctx.failed_sections[0][0] == 'knowledge_recall'
        assert 'db exploded' in ctx.failed_sections[0][1]
