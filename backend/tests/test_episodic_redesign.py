"""
Tests for the Episodic Memory Pipeline Redesign — Phase 0 (Schema) and Phase 1 (Extractor).

Covers the new store_episode() behaviour (10 new columns, deduplication, freshness-optional)
and edge cases in EpisodeExtractorService that were not covered by the original test file.

Database strategy: builds an in-memory SQLite database using the real schema.sql so that
column definitions, constraints, and indexes are never out of sync with production.
"""

import json
import re
import sqlite3
import uuid
from pathlib import Path
import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit

_SCHEMA_PATH = Path(__file__).parent.parent / "schema.sql"


def _build_schema(conn: sqlite3.Connection) -> None:
    """Apply the full production schema.sql to an in-memory connection.

    Tries to load sqlite-vec first. If unavailable, skips vec0 statements so
    that the rest of the schema (episodes, FTS5, indexes) still applies.
    """
    sql = _SCHEMA_PATH.read_text()

    vec_available = False
    try:
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        vec_available = True
    except Exception:
        pass

    if vec_available:
        conn.executescript(sql)
    else:
        for stmt in re.split(r';', sql):
            s = stmt.strip()
            if not s or 'vec0' in s.lower():
                continue
            try:
                conn.execute(s)
            except Exception:
                pass
        conn.commit()


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
    """In-memory SQLite built from the real schema.sql — no DDL duplication."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _build_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def episodic_svc(mem_db):
    """EpisodicService backed by a fresh in-memory episodes table."""
    from services.episodic_service import EpisodicService
    fake_db = _FakeDB(mem_db)
    return EpisodicService(fake_db)


def _ep(**overrides) -> dict:
    """Minimal valid episode_data dict — 3 required fields."""
    base = {
        'gist': 'Test conversation',
        'salience': 5,
        'channel': 'programming',
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

    def test_emotional_valence_stored(self, mem_db, episodic_svc):
        """emotional_valence float is stored and retrievable."""
        data = _ep(emotional_valence=0.75)

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("SELECT emotional_valence FROM episodes WHERE id = ?",
                             (episode_id,)).fetchone()
        assert row['emotional_valence'] == pytest.approx(0.75)

    def test_emotional_arousal_stored(self, mem_db, episodic_svc):
        """emotional_arousal float is stored and retrievable."""
        data = _ep(emotional_arousal=0.4)

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("SELECT emotional_arousal FROM episodes WHERE id = ?",
                             (episode_id,)).fetchone()
        assert row['emotional_arousal'] == pytest.approx(0.4)

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
        assert row['storage_strength'] == pytest.approx(1.5)

    def test_retrieval_weight_stored(self, mem_db, episodic_svc):
        """retrieval_weight is stored and retrievable."""
        data = _ep(retrieval_weight=0.8)

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("SELECT retrieval_weight FROM episodes WHERE id = ?",
                             (episode_id,)).fetchone()
        assert row['retrieval_weight'] == pytest.approx(0.8)

    def test_new_columns_default_when_absent(self, mem_db, episodic_svc):
        """When optional columns are omitted, defaults are applied."""
        data = _ep()

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("""
            SELECT transcript_ids, emotional_valence,
                   emotional_arousal, consolidated_from, storage_strength, retrieval_weight,
                   transcript_id_start, transcript_id_end
            FROM episodes WHERE id = ?
        """, (episode_id,)).fetchone()

        assert json.loads(row['transcript_ids']) == []
        assert row['emotional_valence'] is None
        assert row['emotional_arousal'] is None
        assert json.loads(row['consolidated_from']) == []
        assert row['storage_strength'] == pytest.approx(1.0)
        assert row['retrieval_weight'] == pytest.approx(1.0)
        assert row['transcript_id_start'] is None
        assert row['transcript_id_end'] is None

    def test_new_columns_round_trip(self, mem_db, episodic_svc):
        """All optional columns survive a full store-and-read round trip."""
        src_ids = [str(uuid.uuid4())]
        data = _ep(
            transcript_ids=[1, 2, 3],
            transcript_id_start=1,
            transcript_id_end=3,
            emotional_valence=-0.3,
            emotional_arousal=0.6,
            consolidated_from=src_ids,
            storage_strength=1.2,
            retrieval_weight=0.9,
        )

        episode_id = episodic_svc.store_episode(data)

        row = mem_db.execute("""
            SELECT transcript_ids, transcript_id_start, transcript_id_end,
                   emotional_valence, emotional_arousal,
                   consolidated_from, storage_strength, retrieval_weight
            FROM episodes WHERE id = ?
        """, (episode_id,)).fetchone()

        assert json.loads(row['transcript_ids']) == [1, 2, 3]
        assert row['transcript_id_start'] == 1
        assert row['transcript_id_end'] == 3
        assert row['emotional_valence'] == pytest.approx(-0.3)
        assert row['emotional_arousal'] == pytest.approx(0.6)
        assert json.loads(row['consolidated_from']) == src_ids
        assert row['storage_strength'] == pytest.approx(1.2)
        assert row['retrieval_weight'] == pytest.approx(0.9)


# ── store_episode: overlap → super episode ────────────────────────────────────

class TestStoreEpisodeOverlap:

    def test_exact_range_overlap_stores_both_and_creates_super(self, mem_db, episodic_svc):
        """Storing an episode with the exact same transcript range stores the new episode
        and creates a super episode linking both — does not skip or deduplicate."""
        data = _ep(transcript_id_start=1, transcript_id_end=25,
                   transcript_ids=[1, 2, 3], gist='First episode')

        first_id = episodic_svc.store_episode(data)
        second_id = episodic_svc.store_episode(
            _ep(transcript_id_start=1, transcript_id_end=25,
                transcript_ids=[1, 2, 3], gist='Second episode')
        )

        assert second_id != first_id

        count = mem_db.execute(
            "SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL"
        ).fetchone()[0]
        # first + second + super = 3
        assert count == 3

    def test_overlapping_range_stores_new_and_creates_super(self, mem_db, episodic_svc):
        """An episode whose range partially overlaps an existing one is stored,
        and a super episode is created linking both."""
        first_id = episodic_svc.store_episode(
            _ep(transcript_id_start=1, transcript_id_end=25,
                transcript_ids=list(range(1, 26)), gist='First')
        )
        second_id = episodic_svc.store_episode(
            _ep(transcript_id_start=10, transcript_id_end=35,
                transcript_ids=list(range(10, 36)), gist='Second')
        )

        assert second_id != first_id

        supers = mem_db.execute(
            "SELECT consolidated_from FROM episodes WHERE consolidated_from != '[]' AND consolidated_from IS NOT NULL"
        ).fetchall()
        assert len(supers) == 1
        cf = json.loads(supers[0][0])
        assert first_id in cf
        assert second_id in cf

    def test_super_episode_has_merged_transcript_range(self, mem_db, episodic_svc):
        """Super episode's transcript range spans both source episodes."""
        episodic_svc.store_episode(
            _ep(transcript_id_start=1, transcript_id_end=20,
                transcript_ids=list(range(1, 21)), gist='A')
        )
        episodic_svc.store_episode(
            _ep(transcript_id_start=10, transcript_id_end=30,
                transcript_ids=list(range(10, 31)), gist='B')
        )

        row = mem_db.execute(
            "SELECT transcript_id_start, transcript_id_end FROM episodes "
            "WHERE consolidated_from != '[]' AND consolidated_from IS NOT NULL"
        ).fetchone()
        assert row[0] == 1
        assert row[1] == 30

    def test_adjacent_non_overlapping_range_is_stored_without_super(self, mem_db, episodic_svc):
        """Episodes with non-overlapping ranges are both stored with no super episode."""
        episodic_svc.store_episode(_ep(transcript_id_start=1, transcript_id_end=25))
        episodic_svc.store_episode(_ep(transcript_id_start=26, transcript_id_end=50))

        count = mem_db.execute(
            "SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL"
        ).fetchone()[0]
        assert count == 2

        supers = mem_db.execute(
            "SELECT COUNT(*) FROM episodes WHERE consolidated_from != '[]'"
        ).fetchone()[0]
        assert supers == 0

    def test_soft_deleted_episode_does_not_trigger_super(self, mem_db, episodic_svc):
        """A soft-deleted episode's range is excluded from overlap detection."""
        first_id = episodic_svc.store_episode(
            _ep(transcript_id_start=1, transcript_id_end=25)
        )
        mem_db.execute(
            "UPDATE episodes SET deleted_at = datetime('now') WHERE id = ?",
            (first_id,)
        )
        mem_db.commit()

        second_id = episodic_svc.store_episode(
            _ep(transcript_id_start=1, transcript_id_end=25)
        )

        assert second_id != first_id
        active = mem_db.execute(
            "SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL"
        ).fetchone()[0]
        # only the second episode is active (no super created for deleted overlap)
        assert active == 1

    def test_no_transcript_range_always_stores_without_super(self, mem_db, episodic_svc):
        """Episodes without transcript_id_start/end skip overlap check entirely."""
        id1 = episodic_svc.store_episode(_ep())
        id2 = episodic_svc.store_episode(_ep())

        assert id1 != id2

        count = mem_db.execute(
            "SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL"
        ).fetchone()[0]
        assert count == 2

        supers = mem_db.execute(
            "SELECT COUNT(*) FROM episodes WHERE consolidated_from != '[]'"
        ).fetchone()[0]
        assert supers == 0


# ── EpisodeExtractorService: additional edge cases ───────────────────────────
# These supplement the cases already in test_episode_extractor_service.py.

def _make_valid_ep(entry_range: list) -> dict:
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
        'entry_range': entry_range,
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
         patch('services.episode_extractor_service._PROMPT_PATH') as mock_path, \
         patch('services.episode_extractor_service.create_llm_service',
               return_value=mock_llm):
        mock_path.read_text.return_value = 'Prompt: {{transcript_window}} Topic: {{topic}}'
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
        episode = _make_valid_ep([1, 1])
        svc = _make_extractor(json.dumps(episode))  # object, not [object]
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result == []

    def test_llm_returns_object_wrapped_in_markdown_returns_empty(self):
        """Single object in a markdown code block is also rejected."""
        episode = _make_valid_ep([1, 1])
        response = f"```json\n{json.dumps(episode)}\n```"
        svc = _make_extractor(response)
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result == []


class TestExtractorEntriesWithoutId:

    def test_entries_missing_id_field_transcript_ids_empty_skips(self):
        """When no entries in the range have an 'id' key, the episode is skipped."""
        entry_without_id = {'role': 'user', 'content': 'hello', 'created_at': 'now'}
        episode = _make_valid_ep([1, 1])
        svc = _make_extractor(json.dumps([episode]))

        result = svc.extract([entry_without_id], 'test')

        assert result == []

    def test_mixed_entries_only_those_with_id_in_range(self):
        """entry_range is a transcript-ID range; only entries with 'id' in that range contribute."""
        entry_no_id = {'role': 'user', 'content': 'hello', 'created_at': 'now'}
        entry_with_id = _make_entry(42)
        # range [1, 42] — the id-less entry is ignored, the entry with id=42 matches
        episode = _make_valid_ep([1, 42])
        svc = _make_extractor(json.dumps([episode]))

        result = svc.extract([entry_no_id, entry_with_id], 'test')

        assert result[0]['transcript_ids'] == [42]


class TestTraitValidation:

    def test_trait_missing_key_field_still_returned(self):
        """A trait dict missing required keys is still passed through — the extractor
        does not validate trait field completeness, that is the caller's concern."""
        episode = _make_valid_ep([1, 1])
        episode['traits'] = [{'value': 'Dylan', 'kind': 'trait'}]  # no 'key'
        svc = _make_extractor(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        # Episode is still returned with the incomplete trait intact
        assert len(result) == 1
        assert result[0]['traits'] == [{'value': 'Dylan', 'kind': 'trait'}]

    def test_empty_traits_list_preserved(self):
        """An explicit empty traits list is preserved as-is."""
        episode = _make_valid_ep([1, 1])
        episode['traits'] = []
        svc = _make_extractor(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['traits'] == []

    def test_traits_as_dict_instead_of_list_replaced_with_empty(self):
        """A traits value that is a dict (not a list) is normalised to []."""
        episode = _make_valid_ep([1, 1])
        episode['traits'] = {'key': 'name', 'value': 'Alice'}  # dict not list
        svc = _make_extractor(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['traits'] == []


class TestEmotionalClampingEdgeCases:

    def test_nan_string_valence_clamped_to_none(self):
        """A non-numeric string for valence becomes None (can't float() a word)."""
        episode = _make_valid_ep([1, 1])
        episode['emotional_valence'] = 'positive'  # string, not numeric
        svc = _make_extractor(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        # The _clamp helper returns None for non-numeric values
        assert result[0]['emotional_valence'] is None

    def test_nan_string_arousal_clamped_to_none(self):
        """A non-numeric string for arousal becomes None."""
        episode = _make_valid_ep([1, 1])
        episode['emotional_arousal'] = 'high'
        svc = _make_extractor(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['emotional_arousal'] is None

    def test_extremely_large_positive_valence_clamped_to_one(self):
        """A very large positive valence is clamped to 1.0."""
        episode = _make_valid_ep([1, 1])
        episode['emotional_valence'] = 1e9
        svc = _make_extractor(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['emotional_valence'] == pytest.approx(1.0)

    def test_extremely_large_negative_valence_clamped_to_minus_one(self):
        """A very large negative valence is clamped to -1.0."""
        episode = _make_valid_ep([1, 1])
        episode['emotional_valence'] = -1e9
        svc = _make_extractor(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['emotional_valence'] == pytest.approx(-1.0)

    def test_extremely_large_arousal_clamped_to_one(self):
        """A very large arousal value is clamped to 1.0."""
        episode = _make_valid_ep([1, 1])
        episode['emotional_arousal'] = 1e9
        svc = _make_extractor(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['emotional_arousal'] == pytest.approx(1.0)

    def test_numeric_string_valence_is_cast_and_clamped(self):
        """A numeric string like '0.5' is cast to float and preserved in range."""
        episode = _make_valid_ep([1, 1])
        episode['emotional_valence'] = '0.5'  # valid numeric string
        svc = _make_extractor(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['emotional_valence'] == pytest.approx(0.5)

    def test_zero_valence_preserved(self):
        """Zero is a valid valence value and must not become None."""
        episode = _make_valid_ep([1, 1])
        episode['emotional_valence'] = 0.0
        svc = _make_extractor(json.dumps([episode]))
        entries = [_make_entry(1)]

        result = svc.extract(entries, 'test')

        assert result[0]['emotional_valence'] == pytest.approx(0.0)

    def test_zero_arousal_preserved(self):
        """Zero is the minimum valid arousal value and must not be dropped."""
        episode = _make_valid_ep([1, 1])
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

        mock_llm = MagicMock()
        mock_llm.send_message.side_effect = RuntimeError("LLM unavailable")

        with patch('services.episode_extractor_service.ConfigService.resolve_agent_config',
                   return_value={}), \
             patch('services.episode_extractor_service._PROMPT_PATH') as mock_path, \
             patch('services.episode_extractor_service.create_llm_service',
                   return_value=mock_llm):
            mock_path.read_text.return_value = 'Prompt: {{transcript_window}} Topic: {{topic}}'
            from services.episode_extractor_service import EpisodeExtractorService
            svc = EpisodeExtractorService()

        svc._llm = mock_llm
        entries = [_make_entry(1, content='hello')]

        result = svc.extract(entries, 'test')

        assert result == []
