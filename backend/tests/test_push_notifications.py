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
        mock_r = MagicMock()
        mock_r.get.return_value = None

        with patch('services.redis_client.RedisClientService.create_connection', return_value=mock_r), \
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
        """Redis cache should be used if available."""
        cached_keys = {'public': 'cached_public', 'private': 'cached_private'}
        mock_r = MagicMock()
        mock_r.get.return_value = json.dumps(cached_keys)

        with patch('services.redis_client.RedisClientService.create_connection', return_value=mock_r), \
             patch.dict('os.environ', {}, clear=True):
            keys = _get_vapid_keys()

            assert keys == cached_keys

    def test_subscribe_stores_subscription(self, client):
        """Subscribe should store subscription in Redis."""
        mock_r = MagicMock()
        with patch('services.redis_client.RedisClientService.create_connection', return_value=mock_r):
            subscription = {
                'endpoint': 'https://example.com/push',
                'keys': {'p256dh': 'key1', 'auth': 'key2'}
            }

            response = client.post('/push/subscribe', json=subscription)

            assert response.status_code == 201

    def test_subscribe_invalid_payload(self, client):
        """Invalid subscription should return 400."""
        mock_r = MagicMock()
        with patch('services.redis_client.RedisClientService.create_connection', return_value=mock_r):
            response = client.post('/push/subscribe', json={})

        assert response.status_code == 400

    def test_unsubscribe_removes_subscription(self, client):
        """Unsubscribe should remove subscription from Redis."""
        mock_r = MagicMock()
        with patch('services.redis_client.RedisClientService.create_connection', return_value=mock_r):
            subscription = {'endpoint': 'https://example.com/push'}

            response = client.post('/push/unsubscribe', json=subscription)

            assert response.status_code == 200

    def test_send_push_to_all(self):
        """Send push should call webpush for all subscriptions."""
        mock_r = MagicMock()
        mock_r.smembers.return_value = {
            json.dumps({'endpoint': 'https://example.com/push', 'keys': {}})
        }

        mock_webpush_fn = MagicMock()
        MockWebPushException = type('WebPushException', (Exception,), {})
        mock_pywebpush = MagicMock()
        mock_pywebpush.webpush = mock_webpush_fn
        mock_pywebpush.WebPushException = MockWebPushException

        with patch.dict('sys.modules', {'pywebpush': mock_pywebpush}), \
             patch('services.redis_client.RedisClientService.create_connection', return_value=mock_r), \
             patch('api.push._get_vapid_keys', return_value={'public': 'pub', 'private': 'priv'}):
            send_push_to_all("Test", "Body")

            assert mock_webpush_fn.called

    def test_send_push_stale_cleanup(self):
        """410 responses should remove stale subscriptions."""
        mock_r = MagicMock()
        mock_r.smembers.return_value = {
            json.dumps({'endpoint': 'https://example.com/push1'}),
            json.dumps({'endpoint': 'https://example.com/push2'})
        }

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
             patch('services.redis_client.RedisClientService.create_connection', return_value=mock_r), \
             patch('api.push._get_vapid_keys', return_value={'public': 'pub', 'private': 'priv'}):
            send_push_to_all("Test", "Body")

            assert mock_r.srem.called

    def test_send_push_no_subscriptions_skips_webpush(self):
        """No subscriptions should skip webpush entirely."""
        mock_r = MagicMock()
        mock_r.smembers.return_value = set()

        MockWebPushException = type('WebPushException', (Exception,), {})
        mock_webpush_fn = MagicMock()
        mock_pywebpush = MagicMock()
        mock_pywebpush.webpush = mock_webpush_fn
        mock_pywebpush.WebPushException = MockWebPushException

        with patch.dict('sys.modules', {'pywebpush': mock_pywebpush}), \
             patch('services.redis_client.RedisClientService.create_connection', return_value=mock_r), \
             patch('api.push._get_vapid_keys', return_value={'public': 'pub', 'private': 'priv'}):
            send_push_to_all("Test", "Body")

            assert not mock_webpush_fn.called
