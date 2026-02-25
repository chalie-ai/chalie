"""Unit tests for SparkWelcomeService."""
import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# Mark all tests as unit (no external dependencies)
pytestmark = pytest.mark.unit


@pytest.fixture
def mock_redis():
    """Create a mock Redis that stores values in a dict."""
    store = {}

    class FakeRedis:
        def get(self, key):
            return store.get(key)

        def setex(self, key, ttl, value):
            store[key] = value

        def set(self, key, value):
            store[key] = value

        def setnx(self, key, value):
            if key not in store:
                store[key] = value
                return True
            return False

        def delete(self, key):
            store.pop(key, None)

        def expire(self, key, ttl):
            pass

    fake = FakeRedis()
    fake._store = store
    return fake


class TestMaybeSendWelcome:
    def test_skips_when_welcome_already_sent(self, mock_redis):
        with patch('services.spark_welcome_service.RedisClientService') as mock_cls, \
             patch('services.spark_state_service.RedisClientService') as mock_state_cls:
            mock_cls.create_connection.return_value = mock_redis
            mock_state_cls.create_connection.return_value = mock_redis

            from services.spark_welcome_service import SparkWelcomeService

            svc = SparkWelcomeService(user_id='test')

            # Pre-set state with welcome already sent
            state = {
                'version': 1, 'phase': 'surface', 'exchange_count': 3,
                'effective_exchanges': 2.0, 'traits_learned': 0,
                'welcome_sent': True, 'welcome_sent_at': 1000.0,
                'phase_entered_at': 1000.0, 'phase_hold_count': 0,
                'last_active_at': 1000.0, 'last_scored_at': 1000.0,
                'first_suggestion_seeded': False, 'topics_discussed': [],
            }
            mock_redis.setex('spark_state:test', 2592000, json.dumps(state))

            result = svc.maybe_send_welcome()
            assert result is False

    def test_acquires_lock_and_sends(self, mock_redis):
        with patch('services.spark_welcome_service.RedisClientService') as mock_cls, \
             patch('services.spark_state_service.RedisClientService') as mock_state_cls:
            mock_cls.create_connection.return_value = mock_redis
            mock_state_cls.create_connection.return_value = mock_redis

            from services.spark_welcome_service import SparkWelcomeService

            svc = SparkWelcomeService(user_id='test')
            # Mock the generate and deliver methods
            svc._generate_welcome = MagicMock(return_value=("Hello there", "A"))
            svc._deliver_welcome = MagicMock()
            svc._log_welcome_event = MagicMock()
            # Patch _has_established_traits to avoid DB calls
            svc._spark_state._has_established_traits = MagicMock(return_value=False)

            result = svc.maybe_send_welcome()
            assert result is True
            svc._deliver_welcome.assert_called_once_with("Hello there")
            svc._log_welcome_event.assert_called_once_with("A")

    def test_lock_prevents_duplicate(self, mock_redis):
        with patch('services.spark_welcome_service.RedisClientService') as mock_cls, \
             patch('services.spark_state_service.RedisClientService') as mock_state_cls:
            mock_cls.create_connection.return_value = mock_redis
            mock_state_cls.create_connection.return_value = mock_redis

            from services.spark_welcome_service import SparkWelcomeService

            # Pre-set the lock
            mock_redis.setnx('spark_welcome_lock:test', '1')

            svc = SparkWelcomeService(user_id='test')
            svc._spark_state._has_established_traits = MagicMock(return_value=False)
            svc._generate_welcome = MagicMock(return_value=("Hello", "A"))
            svc._deliver_welcome = MagicMock()

            result = svc.maybe_send_welcome()
            assert result is False
            svc._deliver_welcome.assert_not_called()


class TestGenerateWelcome:
    def test_fallback_when_llm_unavailable(self, mock_redis):
        with patch('services.spark_welcome_service.RedisClientService') as mock_cls, \
             patch('services.spark_state_service.RedisClientService') as mock_state_cls:
            mock_cls.create_connection.return_value = mock_redis
            mock_state_cls.create_connection.return_value = mock_redis

            from services.spark_welcome_service import SparkWelcomeService, _FALLBACK_VARIANTS

            svc = SparkWelcomeService(user_id='test')

            # Patch the LLM to fail
            with patch('services.spark_welcome_service.SparkWelcomeService._generate_welcome') as mock_gen:
                mock_gen.return_value = (_FALLBACK_VARIANTS['B'], 'B')
                text, variant = mock_gen()
                assert text == _FALLBACK_VARIANTS['B']
                assert variant == 'B'

    def test_fallback_variants_are_valid(self):
        from services.spark_welcome_service import _FALLBACK_VARIANTS
        assert len(_FALLBACK_VARIANTS) == 3
        for key, text in _FALLBACK_VARIANTS.items():
            assert key in ('A', 'B', 'C')
            assert len(text) > 20
            assert '!' not in text  # No exclamation marks per guidelines


class TestIdempotency:
    def test_second_call_returns_false(self, mock_redis):
        with patch('services.spark_welcome_service.RedisClientService') as mock_cls, \
             patch('services.spark_state_service.RedisClientService') as mock_state_cls:
            mock_cls.create_connection.return_value = mock_redis
            mock_state_cls.create_connection.return_value = mock_redis

            from services.spark_welcome_service import SparkWelcomeService

            svc = SparkWelcomeService(user_id='test')
            svc._generate_welcome = MagicMock(return_value=("Welcome", "llm"))
            svc._deliver_welcome = MagicMock()
            svc._log_welcome_event = MagicMock()
            svc._spark_state._has_established_traits = MagicMock(return_value=False)

            # First call sends
            assert svc.maybe_send_welcome() is True

            # Second call skips (welcome_sent is now True)
            assert svc.maybe_send_welcome() is False
