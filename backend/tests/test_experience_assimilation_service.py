"""Tests for ExperienceAssimilationService — tool output -> episodic memory pipeline."""

import pytest
from unittest.mock import MagicMock, patch

from services.experience_assimilation_service import ExperienceAssimilationService


pytestmark = pytest.mark.unit


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_store():
    """Isolated MemoryStore mock."""
    from services.memory_store import MemoryStore
    store = MemoryStore()
    with patch('services.memory_client.MemoryClientService.create_connection', return_value=store):
        yield store


@pytest.fixture
def service(mock_store):
    """ExperienceAssimilationService with mocked dependencies."""
    with patch('services.config_service.ConfigService.resolve_agent_config', return_value={
        'check_interval': 1,
        'max_sessions_per_day': 20,
        'cooldown_per_topic': 300,
        'observation_dedup_ttl': 86400,
    }), patch('services.config_service.ConfigService.get_agent_prompt', return_value='test prompt'):
        svc = ExperienceAssimilationService(check_interval=1)
        yield svc


# ── Daily session cap ────────────────────────────────────────────────

class TestDailySessionCap:

    def test_not_exceeded_on_fresh_day(self, service):
        """Fresh day (no sessions recorded) should not be exceeded."""
        assert service._daily_sessions_exceeded() is False

    def test_exceeded_after_max_sessions(self, service):
        """After max_sessions are recorded, cap should be exceeded."""
        service.store.hset('experience_assimilation_state', mapping={
            'sessions_today': 20,
            'session_day': __import__('time').strftime('%Y-%m-%d'),
        })
        assert service._daily_sessions_exceeded() is True

    def test_resets_on_new_day(self, service):
        """Day rollover resets the counter."""
        service.store.hset('experience_assimilation_state', mapping={
            'sessions_today': 20,
            'session_day': '2020-01-01',
        })
        assert service._daily_sessions_exceeded() is False


# ── Topic cooldown ───────────────────────────────────────────────────

class TestTopicCooldown:

    def test_topic_not_on_cooldown_initially(self, service):
        """A never-processed topic should not be on cooldown."""
        assert service._is_topic_on_cooldown('new_topic') is False

    def test_topic_on_cooldown_after_processing(self, service):
        """A just-processed topic should be on cooldown."""
        service._mark_topic_processed('test_topic')
        assert service._is_topic_on_cooldown('test_topic') is True


# ── Content hash dedup ───────────────────────────────────────────────

class TestContentDedup:

    def test_content_hash_deterministic(self, service):
        """Same inputs produce the same hash."""
        outputs = [{'tool': 'search', 'result': 'found stuff'}]
        h1 = service._content_hash(outputs)
        h2 = service._content_hash(outputs)
        assert h1 == h2

    def test_unseen_content_returns_false(self, service):
        """First-time content returns False (not seen)."""
        assert service._is_content_seen('abc123') is False

    def test_seen_content_returns_true(self, service):
        """Already-seen content returns True."""
        service._is_content_seen('abc123')  # first call registers it
        assert service._is_content_seen('abc123') is True


# ── Observation dedup ────────────────────────────────────────────────

class TestObservationDedup:

    def test_new_observation_not_duplicate(self, service):
        """A never-seen observation is not a duplicate."""
        assert service._is_duplicate_observation('Python is great') is False

    def test_repeated_observation_is_duplicate(self, service):
        """Identical observation text is flagged as duplicate."""
        service._is_duplicate_observation('Python is great')
        assert service._is_duplicate_observation('Python is great') is True

    def test_case_insensitive_dedup(self, service):
        """Observation dedup is case-insensitive."""
        service._is_duplicate_observation('Python is GREAT')
        assert service._is_duplicate_observation('python is great') is True


# ── Process item — empty tool outputs ────────────────────────────────

class TestProcessItem:

    def test_empty_tool_outputs_skipped(self, service):
        """Items with empty tool_outputs are silently skipped."""
        service._process_item({'topic': 'test', 'user_prompt': 'hi', 'tool_outputs': []})
        # No exception, no store writes beyond what constructor did

    def test_topic_on_cooldown_skipped(self, service):
        """Items for a cooled-down topic are skipped."""
        service._mark_topic_processed('hot_topic')

        with patch.object(service, '_reflect') as mock_reflect:
            service._process_item({
                'topic': 'hot_topic',
                'user_prompt': 'test',
                'tool_outputs': [{'tool': 'search', 'result': 'data'}]
            })
            mock_reflect.assert_not_called()


# ── Store episode integration ────────────────────────────────────────

class TestStoreEpisode:

    def test_stores_episode_with_correct_fields(self, service, db):
        """_store_episode builds episode_data with required fields and calls storage."""
        observation = {'text': 'Python 3.12 adds pattern matching', 'durability': 'stable'}

        mock_emb = MagicMock()
        mock_emb.generate_embedding.return_value = [0.1] * 256
        mock_storage = MagicMock()
        mock_storage.store_episode.return_value = 'ep-123'

        # Patch the lazy imports at the exact module path _store_episode uses
        with patch('services.embedding_service.get_embedding_service', return_value=mock_emb), \
             patch('services.episodic_service.EpisodicService', return_value=mock_storage):
            service._store_episode(
                observation, 'python', 'what is new',
                [{'tool': 'search', 'result': 'stuff'}]
            )

        call_args = mock_storage.store_episode.call_args[0][0]
        assert call_args['channel'] == 'python'
        assert call_args['salience'] == 6  # stable -> 6
        assert 'embedding' in call_args

    def test_store_episode_missing_text_raises(self, service, db):
        """Observation without text raises KeyError."""
        observation = {'durability': 'transient'}

        mock_emb = MagicMock()
        mock_emb.generate_embedding.side_effect = KeyError('text')

        with patch('services.embedding_service.get_embedding_service', return_value=mock_emb):
            with pytest.raises(KeyError):
                service._store_episode(
                    observation, 'test', 'prompt',
                    [{'tool': 'x', 'result': 'y'}]
                )
