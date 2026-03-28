"""
Tests for backend/api/push.py
"""

import pytest
import json
from unittest.mock import patch, MagicMock, Mock
from api.push import (
    _get_vapid_keys, push_bp,
    send_push_to_all, SUBSCRIPTIONS_KEY, VAPID_KEYS_KEY
)
from services.memory_store import MemoryStore


@pytest.mark.unit
class TestPushNotifications:
    """Test push notification endpoints."""

    @pytest.fixture
    def client(self):
        """Create Flask test client with push blueprint."""
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(push_bp)
        app.config['TESTING'] = True
        return app.test_client()

    @pytest.fixture(autouse=True)
    def bypass_auth(self):
        """Bypass session auth for all tests."""
        with patch('services.auth_session_service.validate_session', return_value=True):
            yield

    def test_vapid_key_generation(self):
        """VAPID keys should be generated and stored."""
        store = MemoryStore()  # Empty store — get() returns None, triggering key generation

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store), \
             patch.dict('os.environ', {}, clear=True):
            keys = _get_vapid_keys()

            assert 'public' in keys
            assert 'private' in keys
            assert len(keys['public']) > 0
            assert len(keys['private']) > 0

    def test_vapid_key_from_env(self):
        """Env vars should take precedence."""
        with patch.dict('os.environ', {
            'VAPID_PUBLIC_KEY': 'env_public',
            'VAPID_PRIVATE_KEY': 'env_private'
        }):
            keys = _get_vapid_keys()

            assert keys['public'] == 'env_public'
            assert keys['private'] == 'env_private'

    def test_vapid_key_from_cache(self):
        """MemoryStore cache should be used if available."""
        cached_keys = {'public': 'cached_public', 'private': 'cached_private'}
        store = MemoryStore()
        store.set(VAPID_KEYS_KEY, json.dumps(cached_keys))

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store), \
             patch.dict('os.environ', {}, clear=True):
            keys = _get_vapid_keys()

            assert keys == cached_keys

    def test_subscribe_stores_subscription(self, client):
        """Subscribe should store subscription in MemoryStore."""
        store = MemoryStore()
        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store):
            subscription = {
                'endpoint': 'https://example.com/push',
                'keys': {'p256dh': 'key1', 'auth': 'key2'}
            }

            response = client.post('/push/subscribe', json=subscription)

            assert response.status_code == 201

    def test_subscribe_invalid_payload(self, client):
        """Invalid subscription should return 400."""
        store = MemoryStore()
        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store):
            response = client.post('/push/subscribe', json={})

        assert response.status_code == 400

    def test_unsubscribe_removes_subscription(self, client):
        """Unsubscribe should remove subscription from MemoryStore."""
        store = MemoryStore()
        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store):
            subscription = {'endpoint': 'https://example.com/push'}

            response = client.post('/push/unsubscribe', json=subscription)

            assert response.status_code == 200

    def test_send_push_to_all(self):
        """Send push should call webpush for all subscriptions."""
        store = MemoryStore()
        store.sadd(SUBSCRIPTIONS_KEY, json.dumps({'endpoint': 'https://example.com/push', 'keys': {}}))

        mock_webpush_fn = MagicMock()
        MockWebPushException = type('WebPushException', (Exception,), {})
        mock_pywebpush = MagicMock()
        mock_pywebpush.webpush = mock_webpush_fn
        mock_pywebpush.WebPushException = MockWebPushException

        with patch.dict('sys.modules', {'pywebpush': mock_pywebpush}), \
             patch('services.memory_client.MemoryClientService.create_connection', return_value=store), \
             patch('api.push._get_vapid_keys', return_value={'public': 'pub', 'private': 'priv'}):
            send_push_to_all("Test", "Body")

            assert mock_webpush_fn.called

    def test_send_push_stale_cleanup(self):
        """410 responses should remove stale subscriptions from the real store.

        Pre-populates two subscriptions; the mock webpush function succeeds for the
        first and raises a 410 WebPushException for the second.  After the call, the
        stale subscription must have been removed via ``store.srem``, which is
        verified by asserting the set shrank from 2 to 1 member.
        """
        store = MemoryStore()
        sub1 = json.dumps({'endpoint': 'https://example.com/push1'})
        sub2 = json.dumps({'endpoint': 'https://example.com/push2'})
        store.sadd(SUBSCRIPTIONS_KEY, sub1, sub2)

        MockWebPushException = type('WebPushException', (Exception,), {})
        mock_webpush_fn = MagicMock()
        mock_pywebpush = MagicMock()
        mock_pywebpush.webpush = mock_webpush_fn
        mock_pywebpush.WebPushException = MockWebPushException

        response_410 = Mock()
        response_410.status_code = 410
        exc = MockWebPushException()
        exc.response = response_410

        mock_webpush_fn.side_effect = [None, exc]

        with patch.dict('sys.modules', {'pywebpush': mock_pywebpush}), \
             patch('services.memory_client.MemoryClientService.create_connection', return_value=store), \
             patch('api.push._get_vapid_keys', return_value={'public': 'pub', 'private': 'priv'}):
            send_push_to_all("Test", "Body")

            # One subscription returned 410 — it must have been removed from the store.
            assert len(store.smembers(SUBSCRIPTIONS_KEY)) == 1

    def test_send_push_no_subscriptions_skips_webpush(self):
        """No subscriptions should skip webpush entirely."""
        store = MemoryStore()  # Empty store — smembers returns empty set, skipping webpush

        MockWebPushException = type('WebPushException', (Exception,), {})
        mock_webpush_fn = MagicMock()
        mock_pywebpush = MagicMock()
        mock_pywebpush.webpush = mock_webpush_fn
        mock_pywebpush.WebPushException = MockWebPushException

        with patch.dict('sys.modules', {'pywebpush': mock_pywebpush}), \
             patch('services.memory_client.MemoryClientService.create_connection', return_value=store), \
             patch('api.push._get_vapid_keys', return_value={'public': 'pub', 'private': 'priv'}):
            send_push_to_all("Test", "Body")

            assert not mock_webpush_fn.called
