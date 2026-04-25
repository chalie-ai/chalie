"""
Unit tests for PatternExtractor — merge functions, promotion logic, and run_once.

Uses real SQLite via a local db_service fixture (no mocks).
Per feedback_test_philosophy.md: hot-path only, no mock theater.
"""

import contextlib
import json
import sqlite3
import uuid

import pytest

from services.pattern_extractor import (
    PatternExtractor,
    _VERTICALS,
    _pattern_key,
    _patterns_match,
    merge_time_routine,
    merge_preference,
    merge_recurring_entity,
    merge_event_cadence,
)
from services.data_graph_service import KIND_BEHAVIORAL_PATTERN
from services.time_utils import utc_now

pytestmark = pytest.mark.unit


# ── Minimal DDL (standalone, no sqlite-vec needed) ────────────────────────────

_DDL = [
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
    "CREATE INDEX IF NOT EXISTS idx_dg_kind ON data_graph(kind)",
    "CREATE INDEX IF NOT EXISTS idx_dg_key  ON data_graph(key)",
    "CREATE INDEX IF NOT EXISTS idx_dg_active ON data_graph(kind, active) WHERE deleted_at IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_dg_confirmed ON data_graph(last_confirmed_at)",
    # Standalone FTS — production uses content-table with triggers; unit tests sync manually.
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS data_graph_fts USING fts5(
        key, value, kind, search_queries,
        tokenize='porter unicode61'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS data_graph_edges (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        from_id          INTEGER NOT NULL,
        to_id            INTEGER NOT NULL,
        edge_type        TEXT NOT NULL DEFAULT 'related',
        strength         REAL NOT NULL DEFAULT 1.0,
        created_at       TEXT NOT NULL DEFAULT (datetime('now')),
        last_accessed_at TEXT,
        UNIQUE (from_id, to_id, edge_type)
    )
    """,
    # Stub vec tables — sqlite-vec not available in unit test environments.
    """
    CREATE TABLE IF NOT EXISTS data_graph_key_vec (
        rowid INTEGER PRIMARY KEY,
        embedding BLOB
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS data_graph_value_vec (
        rowid INTEGER PRIMARY KEY,
        embedding BLOB
    )
    """,
    # LUT miss tracking — required by DataGraphService.store() lut_miss path.
    """
    CREATE TABLE IF NOT EXISTS concept_lut_misses (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        kind       TEXT NOT NULL,
        key        TEXT NOT NULL,
        value_preview TEXT,
        count      INTEGER NOT NULL DEFAULT 1,
        first_seen TEXT NOT NULL,
        last_seen  TEXT NOT NULL,
        UNIQUE(kind, key)
    )
    """,
    # episodes table — needed for super-episode queries.
    """
    CREATE TABLE IF NOT EXISTS episodes (
        id TEXT PRIMARY KEY,
        gist TEXT NOT NULL,
        salience INTEGER NOT NULL DEFAULT 5,
        channel TEXT NOT NULL DEFAULT 'user',
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now')),
        last_accessed_at TEXT,
        access_count INTEGER DEFAULT 0,
        deleted_at TEXT,
        transcript_ids TEXT DEFAULT '[]',
        transcript_id_start INTEGER,
        transcript_id_end INTEGER,
        emotional_valence REAL,
        emotional_arousal REAL,
        consolidated_from TEXT DEFAULT '[]',
        consolidated_into TEXT,
        storage_strength REAL DEFAULT 1.0,
        retrieval_weight REAL DEFAULT 1.0
    )
    """,
]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db_service(tmp_path):
    """Real SQLite DB with data_graph + episodes schema."""
    from services.database_service import DatabaseService

    db_path = str(tmp_path / 'test_pattern_extractor.db')
    shared_conn = sqlite3.connect(db_path)
    shared_conn.row_factory = sqlite3.Row
    shared_conn.execute('PRAGMA foreign_keys = ON')
    for ddl in _DDL:
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
def extractor(db_service):
    """PatternExtractor wired to the test database.

    Embedding generation is disabled (returns None) so vec0 operations
    are skipped — same approach as DataGraphService tests.
    """
    from unittest.mock import patch
    from services.data_graph_service import DataGraphService

    with patch.object(DataGraphService, '_generate_embedding', return_value=None):
        yield PatternExtractor(db_service)


def _seed_super_episode(db_service, gist: str = 'test gist', ep_id: str = None) -> str:
    """Insert a super-episode row and return its id."""
    ep_id = ep_id or str(uuid.uuid4())
    now = utc_now().isoformat()
    with db_service.connection() as conn:
        conn.execute(
            'INSERT INTO episodes (id, gist, channel, created_at, consolidated_from) '
            'VALUES (?, ?, ?, ?, ?)',
            (ep_id, gist, 'user', now, json.dumps([str(uuid.uuid4())])),
        )
    return ep_id


def _seed_pattern(db_service, vertical: str, content: dict) -> str:
    """Insert a behavioral_pattern row directly and return its key."""
    pattern_class = content.get('class', 'preference')
    key = _pattern_key(vertical, pattern_class, content.get('slots', {}))
    now = utc_now().isoformat()
    with db_service.connection() as conn:
        conn.execute(
            'INSERT INTO data_graph (kind, key, value, first_seen_at, last_confirmed_at) '
            'VALUES (?, ?, ?, ?, ?)',
            (KIND_BEHAVIORAL_PATTERN, key, json.dumps(content), now, now),
        )
    return key


def _fetch_pattern(db_service, key: str) -> dict | None:
    """Fetch a pattern row's content dict from data_graph."""
    with db_service.connection() as conn:
        row = conn.execute(
            'SELECT value FROM data_graph WHERE kind=? AND key=? AND active=1 LIMIT 1',
            (KIND_BEHAVIORAL_PATTERN, key),
        ).fetchone()
    if row is None:
        return None
    return json.loads(row['value'])


# ── Pure-function tests ───────────────────────────────────────────────────────

class TestTimeRoutineMerge:

    def test_overlapping_windows_merge_increments_recurrence(self):
        """Two observations with overlapping hour windows merge into one row, recurrence=2."""
        now = utc_now().isoformat()
        ep1, ep2 = str(uuid.uuid4()), str(uuid.uuid4())

        existing = {
            'vertical': 'meal_times',
            'class': 'time_routine',
            'slots': {'event_label': 'dinner', 'day_bucket': 'fri', 'hour_window': '18:30-20:00'},
            'recurrence': 1,
            'sigma_confidence': 0.8,
            'first_seen': now,
            'last_seen': now,
            'evidence_super_ep_ids': [ep1],
            'status': 'candidate',
            'source': 'llm',
            'decay_days': 90,
        }
        observation = {
            'slots': {'event_label': 'dinner', 'day_bucket': 'fri', 'hour_window': '19:00-21:00'},
            'evidence_super_ep_ids': [ep2],
            'confidence': 0.9,
        }

        # Predicate: overlapping windows should match.
        assert _patterns_match(
            'time_routine',
            existing['slots'],
            observation['slots'],
        )

        merged = merge_time_routine(existing, observation, utc_now().isoformat())

        assert merged['recurrence'] == 2
        assert ep1 in merged['evidence_super_ep_ids']
        assert ep2 in merged['evidence_super_ep_ids']
        assert merged['sigma_confidence'] == pytest.approx(existing['sigma_confidence'] + observation['confidence'])

    def test_non_overlapping_windows_do_not_match(self):
        """Hour windows 08:00-09:00 and 19:00-20:00 must not match."""
        slots_a = {'event_label': 'breakfast', 'day_bucket': 'weekday', 'hour_window': '08:00-09:00'}
        slots_b = {'event_label': 'breakfast', 'day_bucket': 'weekday', 'hour_window': '19:00-20:00'}
        assert not _patterns_match('time_routine', slots_a, slots_b)

    def test_different_day_buckets_do_not_match(self):
        """Same event_label + overlapping window but different day_bucket must not match."""
        slots_a = {'event_label': 'dinner', 'day_bucket': 'fri', 'hour_window': '18:00-20:00'}
        slots_b = {'event_label': 'dinner', 'day_bucket': 'mon', 'hour_window': '18:30-20:00'}
        assert not _patterns_match('time_routine', slots_a, slots_b)


class TestPreferenceMerge:

    def test_same_category_choice_case_insensitive(self):
        """Same category + choice (case-insensitive) merge; weight = max not sum."""
        now = utc_now().isoformat()
        ep1, ep2 = str(uuid.uuid4()), str(uuid.uuid4())

        existing = {
            'vertical': 'cuisine_pref',
            'class': 'preference',
            'slots': {'category': 'cuisine', 'choice': 'italian', 'weight': 0.6},
            'recurrence': 1,
            'sigma_confidence': 0.7,
            'first_seen': now,
            'last_seen': now,
            'evidence_super_ep_ids': [ep1],
            'status': 'candidate',
            'source': 'llm',
            'decay_days': 180,
        }
        observation = {
            'slots': {'category': 'Cuisine', 'choice': 'Italian', 'weight': 0.9},
            'evidence_super_ep_ids': [ep2],
            'confidence': 0.85,
        }

        # Case-insensitive predicate.
        assert _patterns_match('preference', existing['slots'], observation['slots'])

        merged = merge_preference(existing, observation, utc_now().isoformat())

        assert merged['recurrence'] == 2
        assert merged['slots']['weight'] == pytest.approx(0.9)  # max, not sum
        assert merged['slots']['weight'] != pytest.approx(existing['slots']['weight'] + observation['slots']['weight'])

    def test_different_choices_do_not_match(self):
        slots_a = {'category': 'cuisine', 'choice': 'italian'}
        slots_b = {'category': 'cuisine', 'choice': 'mexican'}
        assert not _patterns_match('preference', slots_a, slots_b)

    def test_different_categories_do_not_match(self):
        slots_a = {'category': 'cuisine', 'choice': 'italian'}
        slots_b = {'category': 'music', 'choice': 'italian'}
        assert not _patterns_match('preference', slots_a, slots_b)


class TestRecurringEntityMerge:

    def test_normalised_name_match_case(self):
        """'pizza hut' and 'Pizza Hut' must merge via normalised match."""
        now = utc_now().isoformat()
        ep1, ep2 = str(uuid.uuid4()), str(uuid.uuid4())

        existing = {
            'vertical': 'restaurant_pref',
            'class': 'recurring_entity',
            'slots': {'kind': 'restaurant', 'name': 'pizza hut', 'context': 'Friday dinner'},
            'recurrence': 1,
            'sigma_confidence': 0.7,
            'first_seen': now,
            'last_seen': now,
            'evidence_super_ep_ids': [ep1],
            'status': 'candidate',
            'source': 'llm',
            'decay_days': 180,
        }
        observation = {
            'slots': {'kind': 'restaurant', 'name': 'Pizza Hut', 'context': 'Lunch with team'},
            'evidence_super_ep_ids': [ep2],
            'confidence': 0.8,
        }

        assert _patterns_match('recurring_entity', existing['slots'], observation['slots'])

        merged = merge_recurring_entity(existing, observation, utc_now().isoformat())

        assert merged['recurrence'] == 2
        # First-seen name is preserved.
        assert merged['slots']['name'] == 'pizza hut'
        # Context updated to most recent observation.
        assert merged['slots']['context'] == 'Lunch with team'
        assert ep1 in merged['evidence_super_ep_ids']
        assert ep2 in merged['evidence_super_ep_ids']

    def test_edit_distance_match(self):
        """Names with edit distance ≤2 must match."""
        slots_a = {'kind': 'restaurant', 'name': 'Nandos', 'context': ''}
        slots_b = {'kind': 'restaurant', 'name': "Nando's", 'context': ''}
        # apostrophe stripped by normalise → same string → distance 0
        assert _patterns_match('recurring_entity', slots_a, slots_b)

    def test_substring_containment_match(self):
        """Substring containment triggers a match."""
        slots_a = {'kind': 'project', 'name': 'Chalie', 'context': ''}
        slots_b = {'kind': 'project', 'name': 'Chalie App', 'context': ''}
        assert _patterns_match('recurring_entity', slots_a, slots_b)

    def test_different_kinds_do_not_match(self):
        slots_a = {'kind': 'restaurant', 'name': 'Pizza Hut', 'context': ''}
        slots_b = {'kind': 'project', 'name': 'Pizza Hut', 'context': ''}
        assert not _patterns_match('recurring_entity', slots_a, slots_b)


class TestEventCadenceMerge:

    def test_three_observations_7_days_apart_produce_cadence_7(self):
        """Three event_cadence observations with cadence_days=7 produce cadence_days=7."""
        now = utc_now().isoformat()
        ep1, ep2, ep3 = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())

        existing = {
            'vertical': 'meetings',
            'class': 'event_cadence',
            'slots': {'event_type': 'team standup', 'cadence_days': 7, 'next_expected': now},
            'recurrence': 1,
            'sigma_confidence': 0.8,
            'first_seen': now,
            'last_seen': now,
            'evidence_super_ep_ids': [ep1],
            'status': 'candidate',
            'source': 'llm',
            'decay_days': 60,
        }
        obs2 = {
            'slots': {'event_type': 'team standup', 'cadence_days': 7, 'next_expected': now},
            'evidence_super_ep_ids': [ep2],
            'confidence': 0.8,
        }
        obs3 = {
            'slots': {'event_type': 'team standup', 'cadence_days': 7, 'next_expected': now},
            'evidence_super_ep_ids': [ep3],
            'confidence': 0.8,
        }

        assert _patterns_match('event_cadence', existing['slots'], obs2['slots'])
        assert _patterns_match('event_cadence', existing['slots'], obs3['slots'])

        after_2 = merge_event_cadence(existing, obs2, now)
        after_3 = merge_event_cadence(after_2, obs3, now)

        assert after_3['slots']['cadence_days'] == 7
        assert after_3['recurrence'] == 3

    def test_cadence_running_mean(self):
        """Running mean recomputation: gaps of 7 and 14 days → mean 10 after three obs."""
        now = utc_now().isoformat()
        existing = {
            'vertical': 'meetings',
            'class': 'event_cadence',
            'slots': {'event_type': 'sync', 'cadence_days': 7, 'next_expected': now},
            'recurrence': 1,
            'sigma_confidence': 0.5,
            'first_seen': now,
            'last_seen': now,
            'evidence_super_ep_ids': [str(uuid.uuid4())],
            'status': 'candidate',
            'source': 'llm',
            'decay_days': 60,
        }
        obs2 = {
            'slots': {'event_type': 'sync', 'cadence_days': 14, 'next_expected': now},
            'evidence_super_ep_ids': [str(uuid.uuid4())],
            'confidence': 0.5,
        }
        obs3 = {
            'slots': {'event_type': 'sync', 'cadence_days': 9, 'next_expected': now},
            'evidence_super_ep_ids': [str(uuid.uuid4())],
            'confidence': 0.5,
        }
        after_2 = merge_event_cadence(existing, obs2, now)
        assert after_2['slots']['cadence_days'] == 10  # (7+14)/2

        after_3 = merge_event_cadence(after_2, obs3, now)
        # ((10*2)+9)/3 = 29/3 = ~9.67 → rounds to 10
        assert after_3['slots']['cadence_days'] in (9, 10)

    def test_different_event_types_do_not_match(self):
        slots_a = {'event_type': 'standup', 'cadence_days': 7}
        slots_b = {'event_type': 'sprint review', 'cadence_days': 7}
        assert not _patterns_match('event_cadence', slots_a, slots_b)


# ── Promotion test ────────────────────────────────────────────────────────────

class TestPromotion:

    def test_candidate_promotes_when_thresholds_met(self, db_service, extractor):
        """Seed candidate with recurrence=4, sigma_conf=2.4, evidence from 2 supers;
        one more observation pushes recurrence=5, sigma_conf≥3.0; status flips to 'active'.
        """
        ep_ids = [str(uuid.uuid4()) for _ in range(2)]
        now = utc_now().isoformat()

        # Insert the candidate directly into data_graph.
        candidate_content = {
            'vertical': 'meal_times',
            'class': 'time_routine',
            'slots': {'event_label': 'lunch', 'day_bucket': 'weekday', 'hour_window': '12:00-13:00'},
            'recurrence': 4,
            'sigma_confidence': 2.4,
            'first_seen': now,
            'last_seen': now,
            'evidence_super_ep_ids': ep_ids,
            'status': 'candidate',
            'source': 'llm',
            'decay_days': 90,
        }
        key = _seed_pattern(db_service, 'meal_times', candidate_content)

        # Seed a super-episode that will trigger the observation.
        ep_new = _seed_super_episode(db_service, gist='User had lunch at noon on Tuesday')

        # The observation that pushes recurrence to 5 and sigma_conf to ≥3.0.
        observation = {
            'slots': {'event_label': 'lunch', 'day_bucket': 'weekday', 'hour_window': '12:30-13:00'},
            'evidence_super_ep_ids': [ep_new],
            'confidence': 0.7,  # 2.4 + 0.7 = 3.1 ≥ 3.0
        }

        vcfg = _VERTICALS['meal_times']

        # Load existing patterns from DB as the extractor would.
        existing = extractor._fetch_existing_patterns('meal_times')
        assert len(existing) == 1, "Seed row not found"

        _, promoted = extractor._apply_observation('meal_times', vcfg, observation, existing)

        assert promoted, "Expected promotion to fire"

        # Verify persistence.
        persisted = _fetch_pattern(db_service, key)
        assert persisted is not None
        assert persisted['status'] == 'active'
        assert persisted['recurrence'] == 5
        assert persisted['sigma_confidence'] == pytest.approx(3.1)

    def test_candidate_does_not_promote_below_threshold(self, db_service, extractor):
        """Candidate with recurrence=2, sigma_conf=1.0 must stay 'candidate'."""
        now = utc_now().isoformat()
        ep_id = str(uuid.uuid4())
        candidate_content = {
            'vertical': 'meal_times',
            'class': 'time_routine',
            'slots': {'event_label': 'snack', 'day_bucket': 'weekday', 'hour_window': '15:00-16:00'},
            'recurrence': 2,
            'sigma_confidence': 1.0,
            'first_seen': now,
            'last_seen': now,
            'evidence_super_ep_ids': [ep_id, str(uuid.uuid4())],
            'status': 'candidate',
            'source': 'llm',
            'decay_days': 90,
        }
        key = _seed_pattern(db_service, 'meal_times', candidate_content)

        observation = {
            'slots': {'event_label': 'snack', 'day_bucket': 'weekday', 'hour_window': '15:30-16:00'},
            'evidence_super_ep_ids': [str(uuid.uuid4())],
            'confidence': 0.5,  # total = 1.5; min_sigma_conf = 3.0; still below
        }

        vcfg = _VERTICALS['meal_times']
        existing = extractor._fetch_existing_patterns('meal_times')
        _, promoted = extractor._apply_observation('meal_times', vcfg, observation, existing)

        assert not promoted
        persisted = _fetch_pattern(db_service, key)
        assert persisted['status'] == 'candidate'


# ── Integration: run_once with mocked LLM ────────────────────────────────────

class TestRunOnceLLMMocked:
    """run_once is LLM-heavy; test the wiring with a mocked BackgroundLLMProxy."""

    def test_run_once_returns_summary_for_all_verticals(self, db_service, extractor):
        """run_once always returns a key for every vertical."""
        from unittest.mock import patch, MagicMock
        from services.llm_service import LLMResponse

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = LLMResponse(
            text='{"observations": []}', model='m', provider='mock'
        )

        with patch('services.pattern_extractor.create_background_llm_proxy', return_value=mock_llm):
            # Seed one super-episode so the watermark path is exercised.
            _seed_super_episode(db_service, 'User ate sushi for dinner on Friday')
            result = extractor.run_once()

        assert set(result.keys()) == set(_VERTICALS.keys())
        for v in _VERTICALS:
            assert 'observations' in result[v] or 'error' in result[v]

    def test_run_once_vertical_failure_does_not_block_others(self, db_service, extractor):
        """An exception in one vertical's extraction must not prevent others from running.

        LLM failures are caught inside _call_llm and returned as zero observations.
        A catastrophic vertical failure (e.g. DB error) is caught in run_once's outer
        try/except and stored as {'error': '...'}.  We simulate the latter by making
        _extract_vertical raise directly for the first vertical.
        """
        from unittest.mock import patch, MagicMock
        from services.llm_service import LLMResponse

        call_count = [0]
        original_extract = extractor._extract_vertical

        def _patched_extract(vertical_name, vcfg):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError('simulated DB failure')
            return original_extract(vertical_name, vcfg)

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = LLMResponse(
            text='{"observations": []}', model='m', provider='mock'
        )

        _seed_super_episode(db_service, 'User had meetings all day')

        with patch('services.pattern_extractor.create_background_llm_proxy', return_value=mock_llm):
            with patch.object(extractor, '_extract_vertical', side_effect=_patched_extract):
                result = extractor.run_once()

        # At least one vertical has an error, and at least one has observations key.
        statuses = [set(v.keys()) for v in result.values()]
        has_error = any('error' in s for s in statuses)
        has_success = any('observations' in s for s in statuses)
        assert has_error
        assert has_success

    def test_run_once_skips_verticals_with_no_new_super_eps(self, db_service, extractor):
        """When no super-episodes exist, every vertical returns zero observations."""
        from unittest.mock import patch, MagicMock

        mock_llm = MagicMock()

        with patch('services.pattern_extractor.create_background_llm_proxy', return_value=mock_llm):
            result = extractor.run_once()

        # LLM must never be called when there are no super-episodes.
        mock_llm.send_message.assert_not_called()
        for summary in result.values():
            assert summary.get('observations', 0) == 0
