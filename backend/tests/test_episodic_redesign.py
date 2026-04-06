"""
Tests for the Episodic Memory Pipeline Redesign — Phase 0 (Schema) and Phase 1 (Extractor).

Covers the new store_episode() behaviour (10 new columns, deduplication, freshness-optional)
and edge cases in EpisodeExtractorService that were not covered by the original test file.

Database strategy: builds an in-memory SQLite database with the full episodes table
schema (including all Phase 0 columns) directly — bypasses the session-scoped template
fixture so migration idempotency issues in the dev environment do not affect these tests.
"""

import json
import math
import sqlite3
import uuid
import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit

# ── Minimal episodes schema ───────────────────────────────────────────────────
# This is the full post-Phase-0 schema.  It must stay in sync with schema.sql.
# SQLite does not support default values that call functions (datetime('now'))
# in the CREATE TABLE body, so timestamps default to NULL here for simplicity.
_EPISODES_DDL = """
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    intent TEXT NOT NULL,
    context TEXT NOT NULL,
    action TEXT NOT NULL,
    emotion TEXT NOT NULL,
    outcome TEXT NOT NULL,
    gist TEXT NOT NULL,
    salience INTEGER NOT NULL,
    topic TEXT NOT NULL,
    created_at TEXT,
    updated_at TEXT,
    last_accessed_at TEXT,
    access_count INTEGER DEFAULT 0,
    deleted_at TEXT,
    salience_factors TEXT DEFAULT '{}',
    open_loops TEXT DEFAULT '[]',
    transcript_ids TEXT DEFAULT '[]',
    transcript_id_start INTEGER,
    transcript_id_end INTEGER,
    entities TEXT DEFAULT '[]',
    goal_tags TEXT DEFAULT '[]',
    emotional_valence REAL,
    emotional_arousal REAL,
    consolidated_from TEXT DEFAULT '[]',
    storage_strength REAL DEFAULT 1.0,
    retrieval_weight REAL DEFAULT 1.0
)
"""


# ── Fixture helpers ───────────────────────────────────────────────────────────

class _FakeDB:
    """Thin wrapper that satisfies EpisodicService's db_service.connection() API."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    @contextmanager
    def connection(self):
        yield self._conn
        self._conn.commit()


@pytest.fixture
def mem_db():
    """In-memory SQLite with the Phase-0 episodes schema, no migration dependency."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_EPISODES_DDL)
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def episodic_svc(mem_db):
    """EpisodicService backed by a fresh in-memory episodes table."""
    from services.episodic_service import EpisodicService
    fake_db = _FakeDB(mem_db)
    return EpisodicService(fake_db)


def _ep(**overrides) -> dict:
    """Minimal valid episode_data dict — all 8 required fields present."""
    base = {
        'intent': {'type': 'exploration'},
        'context': {'topic': 'test'},
        'action': 'user asked a question',
        'emotion': {'valence': 0.5},
        'outcome': 'answered successfully',
        'gist': 'Test conversation',
        'salience': 5,
        'topic': 'programming',
    }
    base.update(overrides)
    return base


# ── store_episode: all 10 new Phase-0 columns ────────────────────────────────

class TestStoreEpisodeNewColumns:

    def test_transcript_ids_stored_as_json(self, mem_db, episodic_svc):
        """transcript_ids list is persisted as a JSON array."""
        data = _ep(transcript_ids=[10, 20, 30])

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("SELECT transcript_ids FROM episodes WHERE id = ?",
                             (episode_id,)).fetchone()
        assert json.loads(row['transcript_ids']) == [10, 20, 30]

    def test_transcript_id_start_end_stored(self, mem_db, episodic_svc):
        """transcript_id_start and transcript_id_end are stored independently."""
        data = _ep(transcript_id_start=5, transcript_id_end=29)

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute(
            "SELECT transcript_id_start, transcript_id_end FROM episodes WHERE id = ?",
            (episode_id,)
        ).fetchone()
        assert row['transcript_id_start'] == 5
        assert row['transcript_id_end'] == 29

    def test_entities_stored_as_json(self, mem_db, episodic_svc):
        """entities list is persisted as a JSON array."""
        data = _ep(entities=['Python', 'Flask', 'SQLite'])

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("SELECT entities FROM episodes WHERE id = ?",
                             (episode_id,)).fetchone()
        assert json.loads(row['entities']) == ['Python', 'Flask', 'SQLite']

    def test_goal_tags_stored_as_json(self, mem_db, episodic_svc):
        """goal_tags list is persisted as a JSON array."""
        data = _ep(goal_tags=['learn-python', 'build-app'])

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("SELECT goal_tags FROM episodes WHERE id = ?",
                             (episode_id,)).fetchone()
        assert json.loads(row['goal_tags']) == ['learn-python', 'build-app']

    def test_emotional_valence_stored(self, mem_db, episodic_svc):
        """emotional_valence float is stored and retrievable."""
        data = _ep(emotional_valence=0.75)

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("SELECT emotional_valence FROM episodes WHERE id = ?",
                             (episode_id,)).fetchone()
        assert pytest.approx(row['emotional_valence']) == 0.75

    def test_emotional_arousal_stored(self, mem_db, episodic_svc):
        """emotional_arousal float is stored and retrievable."""
        data = _ep(emotional_arousal=0.4)

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("SELECT emotional_arousal FROM episodes WHERE id = ?",
                             (episode_id,)).fetchone()
        assert pytest.approx(row['emotional_arousal']) == 0.4

    def test_consolidated_from_stored_as_json(self, mem_db, episodic_svc):
        """consolidated_from list is persisted as a JSON array."""
        source_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        data = _ep(consolidated_from=source_ids)

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("SELECT consolidated_from FROM episodes WHERE id = ?",
                             (episode_id,)).fetchone()
        assert json.loads(row['consolidated_from']) == source_ids

    def test_storage_strength_stored(self, mem_db, episodic_svc):
        """storage_strength is stored and retrievable."""
        data = _ep(storage_strength=1.5)

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("SELECT storage_strength FROM episodes WHERE id = ?",
                             (episode_id,)).fetchone()
        assert pytest.approx(row['storage_strength']) == 1.5

    def test_retrieval_weight_stored(self, mem_db, episodic_svc):
        """retrieval_weight is stored and retrievable."""
        data = _ep(retrieval_weight=0.8)

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("SELECT retrieval_weight FROM episodes WHERE id = ?",
                             (episode_id,)).fetchone()
        assert pytest.approx(row['retrieval_weight']) == 0.8

    def test_new_columns_default_when_absent(self, mem_db, episodic_svc):
        """When new optional columns are omitted, defaults are applied."""
        data = _ep()  # no new columns

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("""
            SELECT transcript_ids, entities, goal_tags, emotional_valence,
                   emotional_arousal, consolidated_from, storage_strength, retrieval_weight,
                   transcript_id_start, transcript_id_end
            FROM episodes WHERE id = ?
        """, (episode_id,)).fetchone()

        assert json.loads(row['transcript_ids']) == []
        assert json.loads(row['entities']) == []
        assert json.loads(row['goal_tags']) == []
        assert row['emotional_valence'] is None
        assert row['emotional_arousal'] is None
        assert json.loads(row['consolidated_from']) == []
        assert pytest.approx(row['storage_strength']) == 1.0
        assert pytest.approx(row['retrieval_weight']) == 1.0
        assert row['transcript_id_start'] is None
        assert row['transcript_id_end'] is None

    def test_all_ten_new_columns_round_trip(self, mem_db, episodic_svc):
        """All 10 Phase-0 columns survive a full store-and-read round trip."""
        src_ids = [str(uuid.uuid4())]
        data = _ep(
            transcript_ids=[1, 2, 3],
            transcript_id_start=1,
            transcript_id_end=3,
            entities=['Alice', 'Bob'],
            goal_tags=['goal-a'],
            emotional_valence=-0.3,
            emotional_arousal=0.6,
            consolidated_from=src_ids,
            storage_strength=1.2,
            retrieval_weight=0.9,
        )

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("""
            SELECT transcript_ids, transcript_id_start, transcript_id_end,
                   entities, goal_tags, emotional_valence, emotional_arousal,
                   consolidated_from, storage_strength, retrieval_weight
            FROM episodes WHERE id = ?
        """, (episode_id,)).fetchone()

        assert json.loads(row['transcript_ids']) == [1, 2, 3]
        assert row['transcript_id_start'] == 1
        assert row['transcript_id_end'] == 3
        assert json.loads(row['entities']) == ['Alice', 'Bob']
        assert json.loads(row['goal_tags']) == ['goal-a']
        assert pytest.approx(row['emotional_valence']) == -0.3
        assert pytest.approx(row['emotional_arousal']) == 0.6
        assert json.loads(row['consolidated_from']) == src_ids
        assert pytest.approx(row['storage_strength']) == 1.2
        assert pytest.approx(row['retrieval_weight']) == 0.9


# ── store_episode: deduplication on overlapping transcript ranges ─────────────

class TestStoreEpisodeDeduplication:

    def test_exact_range_overlap_is_skipped(self, mem_db, episodic_svc):
        """Storing an episode with the exact same transcript range returns the
        existing episode ID and does not insert a second row."""
        data = _ep(transcript_id_start=1, transcript_id_end=25)

        first_id = episodic_svc.store_episode(data)
        second_id = episodic_svc.store_episode(data)

        assert second_id == first_id

        count = mem_db.execute(
            "SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL"
        ).fetchone()[0]
        assert count == 1

    def test_overlapping_range_is_skipped(self, mem_db, episodic_svc):
        """An episode whose range partially overlaps an existing one is skipped.

        Overlap condition: existing.start <= new.end AND existing.end >= new.start
        """
        first_id = episodic_svc.store_episode(
            _ep(transcript_id_start=1, transcript_id_end=25)
        )
        # New range [10, 35] overlaps [1, 25]
        second_id = episodic_svc.store_episode(
            _ep(transcript_id_start=10, transcript_id_end=35)
        )

        assert second_id == first_id

        count = mem_db.execute(
            "SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL"
        ).fetchone()[0]
        assert count == 1

    def test_contained_range_is_skipped(self, mem_db, episodic_svc):
        """A range entirely inside an existing episode's range is a duplicate."""
        first_id = episodic_svc.store_episode(
            _ep(transcript_id_start=1, transcript_id_end=50)
        )
        # [10, 20] is fully inside [1, 50]
        second_id = episodic_svc.store_episode(
            _ep(transcript_id_start=10, transcript_id_end=20)
        )

        assert second_id == first_id

    def test_adjacent_non_overlapping_range_is_stored(self, mem_db, episodic_svc):
        """Episodes with non-overlapping ranges are both stored."""
        episodic_svc.store_episode(_ep(transcript_id_start=1, transcript_id_end=25))
        episodic_svc.store_episode(_ep(transcript_id_start=26, transcript_id_end=50))

        count = mem_db.execute(
            "SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL"
        ).fetchone()[0]
        assert count == 2

    def test_soft_deleted_episode_does_not_block_new_store(self, mem_db, episodic_svc):
        """A soft-deleted episode's range should not block a fresh store."""
        first_id = episodic_svc.store_episode(
            _ep(transcript_id_start=1, transcript_id_end=25)
        )
        # Soft-delete the first episode
        mem_db.execute(
            "UPDATE episodes SET deleted_at = datetime('now') WHERE id = ?",
            (first_id,)
        )
        mem_db.commit()

        # Same range should now be stored fresh
        second_id = episodic_svc.store_episode(
            _ep(transcript_id_start=1, transcript_id_end=25)
        )

        assert second_id != first_id
        active = mem_db.execute(
            "SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL"
        ).fetchone()[0]
        assert active == 1

    def test_no_transcript_range_always_stores(self, mem_db, episodic_svc):
        """Episodes without transcript_id_start/end skip deduplication check."""
        id1 = episodic_svc.store_episode(_ep())
        id2 = episodic_svc.store_episode(_ep())

        assert id1 != id2

        count = mem_db.execute(
            "SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL"
        ).fetchone()[0]
        assert count == 2


# ── EpisodeExtractorService: additional edge cases ───────────────────────────
# These supplement the cases already in test_episode_extractor_service.py.

def _make_valid_ep(transcript_ids: list) -> dict:
    return {
        'intent': {'type': 'exploration', 'direction': 'understand topic'},
        'context': 'User learning Python',
        'action': 'explained decorators',
        'emotion': {'valence': 'positive', 'intensity': 'low'},
        'outcome': 'understood',
        'gist': 'User asked about Python decorators.',
        'salience_factors': {
            'novelty': 2, 'emotional_weight': 1, 'goal_relevance': 2,
            'decision_made': False, 'open_loop_created': False,
        },
        'open_loops': [],
        'transcript_ids': transcript_ids,
        'entities': [],
        'goal_tags': [],
        'emotional_valence': 0.3,
        'emotional_arousal': 0.2,
        'traits': [],
    }


def _make_extractor(llm_response_text: str):
    """Build an EpisodeExtractorService with a mocked LLM."""
    from services.llm_service import LLMResponse

    mock_llm = MagicMock()
    mock_llm.send_message.return_value = LLMResponse(
        text=llm_response_text,
        model='test-model',
        provider='mock',
    )

    with patch('services.episode_extractor_service.ConfigService.resolve_agent_config',
               return_value={}), \
         patch('services.episode_extractor_service.ConfigService.get_agent_prompt',
               return_value='Prompt: {{transcript_window}} Topic: {{topic}}'), \
         patch('services.episode_extractor_service.create_llm_service',
               return_value=mock_llm):
        from services.episode_extractor_service import EpisodeExtractorService
        svc = EpisodeExtractorService()

    svc._llm = mock_llm
    return svc


def _make_entry(entry_id: int, role: str = 'user', content: str = 'hello') -> dict:
    return {
        'id': entry_id,
        'role': role,
        'content': content,
        'created_at': '2026-04-06T10:00:00+00:00',
        'tool_name': None,
    }


class TestExtractorSingleObjectResponse:

    def test_llm_returns_single_object_not_array_returns_empty(self):
        """LLM that returns a single JSON object (not array) must be rejected."""
        episode = _make_valid_ep([1])
        svc = _make_extractor(json.dumps(episode))  # object, not [object]
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result == []

    def test_llm_returns_object_wrapped_in_markdown_returns_empty(self):
        """Single object in a markdown code block is also rejected."""
        episode = _make_valid_ep([1])
        response = f"```json\n{json.dumps(episode)}\n```"
        svc = _make_extractor(response)
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result == []


class TestExtractorEntriesWithoutId:

    def test_entries_missing_id_field_episode_skipped(self):
        """Entries without an 'id' key are excluded from valid_entry_ids set,
        so episodes referencing only invalid IDs are skipped entirely."""
        entry_without_id = {'role': 'user', 'content': 'hello', 'created_at': 'now'}
        episode = _make_valid_ep([999])  # 999 won't appear in valid IDs
        svc = _make_extractor(json.dumps([episode]))

        result = svc.extract([entry_without_id], 'test')

        assert result == []

    def test_mixed_entries_with_and_without_id(self):
        """Only entries that have an 'id' key contribute valid IDs."""
        entry_no_id = {'role': 'user', 'content': 'hello', 'created_at': 'now'}
        entry_with_id = _make_entry(42)
        episode = _make_valid_ep([42, 999])  # 42 valid, 999 invalid
        svc = _make_extractor(json.dumps([episode]))

        result = svc.extract([entry_no_id, entry_with_id], 'test')

        assert result[0]['transcript_ids'] == [42]


class TestTraitValidation:

    def test_trait_missing_key_field_still_returned(self):
        """A trait dict missing required keys is still passed through — the extractor
        does not validate trait field completeness, that is the caller's concern."""
        episode = _make_valid_ep([1])
        episode['traits'] = [{'value': 'Dylan', 'kind': 'trait'}]  # no 'key'
        svc = _make_extractor(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        # Episode is still returned with the incomplete trait intact
        assert len(result) == 1
        assert result[0]['traits'] == [{'value': 'Dylan', 'kind': 'trait'}]

    def test_empty_traits_list_preserved(self):
        """An explicit empty traits list is preserved as-is."""
        episode = _make_valid_ep([1])
        episode['traits'] = []
        svc = _make_extractor(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['traits'] == []

    def test_traits_as_dict_instead_of_list_replaced_with_empty(self):
        """A traits value that is a dict (not a list) is normalised to []."""
        episode = _make_valid_ep([1])
        episode['traits'] = {'key': 'name', 'value': 'Alice'}  # dict not list
        svc = _make_extractor(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['traits'] == []


class TestEmotionalClampingEdgeCases:

    def test_nan_string_valence_clamped_to_none(self):
        """A non-numeric string for valence becomes None (can't float() a word)."""
        episode = _make_valid_ep([1])
        episode['emotional_valence'] = 'positive'  # string, not numeric
        svc = _make_extractor(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        # The _clamp helper returns None for non-numeric values
        assert result[0]['emotional_valence'] is None

    def test_nan_string_arousal_clamped_to_none(self):
        """A non-numeric string for arousal becomes None."""
        episode = _make_valid_ep([1])
        episode['emotional_arousal'] = 'high'
        svc = _make_extractor(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['emotional_arousal'] is None

    def test_extremely_large_positive_valence_clamped_to_one(self):
        """A very large positive valence is clamped to 1.0."""
        episode = _make_valid_ep([1])
        episode['emotional_valence'] = 1e9
        svc = _make_extractor(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['emotional_valence'] == 1.0

    def test_extremely_large_negative_valence_clamped_to_minus_one(self):
        """A very large negative valence is clamped to -1.0."""
        episode = _make_valid_ep([1])
        episode['emotional_valence'] = -1e9
        svc = _make_extractor(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['emotional_valence'] == -1.0

    def test_extremely_large_arousal_clamped_to_one(self):
        """A very large arousal value is clamped to 1.0."""
        episode = _make_valid_ep([1])
        episode['emotional_arousal'] = 1e9
        svc = _make_extractor(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['emotional_arousal'] == 1.0

    def test_numeric_string_valence_is_cast_and_clamped(self):
        """A numeric string like '0.5' is cast to float and preserved in range."""
        episode = _make_valid_ep([1])
        episode['emotional_valence'] = '0.5'  # valid numeric string
        svc = _make_extractor(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['emotional_valence'] == pytest.approx(0.5)

    def test_zero_valence_preserved(self):
        """Zero is a valid valence value and must not become None."""
        episode = _make_valid_ep([1])
        episode['emotional_valence'] = 0.0
        svc = _make_extractor(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['emotional_valence'] == pytest.approx(0.0)

    def test_zero_arousal_preserved(self):
        """Zero is the minimum valid arousal value and must not be dropped."""
        episode = _make_valid_ep([1])
        episode['emotional_arousal'] = 0.0
        svc = _make_extractor(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['emotional_arousal'] == pytest.approx(0.0)


class TestExtractorEmptyEntriesList:

    def test_empty_entries_skips_llm_call(self):
        """When entries list is empty the LLM must not be called at all."""
        svc = _make_extractor('[]')

        result = svc.extract([], 'any-topic')

        assert result == []
        svc._llm.send_message.assert_not_called()


class TestExtractorLlmFailure:

    def test_llm_exception_returns_empty_list(self):
        """If the LLM call itself raises, extract() must return [] not propagate."""
        from services.llm_service import LLMResponse

        mock_llm = MagicMock()
        mock_llm.send_message.side_effect = RuntimeError("LLM unavailable")

        with patch('services.episode_extractor_service.ConfigService.resolve_agent_config',
                   return_value={}), \
             patch('services.episode_extractor_service.ConfigService.get_agent_prompt',
                   return_value='Prompt: {{transcript_window}} Topic: {{topic}}'), \
             patch('services.episode_extractor_service.create_llm_service',
                   return_value=mock_llm):
            from services.episode_extractor_service import EpisodeExtractorService
            svc = EpisodeExtractorService()

        svc._llm = mock_llm
        entries = [_make_entry(1, content='hello')]

        result = svc.extract(entries, 'test')

        assert result == []
