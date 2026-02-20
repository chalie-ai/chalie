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
    def client(self, mock_redis):
        """Create Flask test client."""
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(push_bp)
        return app.test_client()

    def test_vapid_key_generation(self, mock_redis):
        """VAPID keys should be generated and stored."""
        with patch('api.push.RedisClientService.create_connection', return_value=mock_redis), \
             patch.dict('os.environ', {}, clear=True):
            keys = _get_vapid_keys()

            assert 'public' in keys
            assert 'private' in keys
            assert len(keys['public']) > 0
            assert len(keys['private']) > 0

    def test_vapid_key_from_env(self, mock_redis):
        """Env vars should take precedence."""
        with patch.dict('os.environ', {
            'VAPID_PUBLIC_KEY': 'env_public',
            'VAPID_PRIVATE_KEY': 'env_private'
        }):
            keys = _get_vapid_keys()

            assert keys['public'] == 'env_public'
            assert keys['private'] == 'env_private'

    def test_vapid_key_from_cache(self, mock_redis):
        """Redis cache should be used if available."""
        cached_keys = {'public': 'cached_public', 'private': 'cached_private'}
        mock_redis.get.return_value = json.dumps(cached_keys)

        with patch('api.push.RedisClientService.create_connection', return_value=mock_redis), \
             patch.dict('os.environ', {}, clear=True):
            keys = _get_vapid_keys()

            assert keys == cached_keys

    def test_subscribe_stores_subscription(self, client, mock_redis):
        """Subscribe should store subscription in Redis."""
        with patch('api.push.RedisClientService.create_connection', return_value=mock_redis):
            subscription = {
                'endpoint': 'https://example.com/push',
                'keys': {'p256dh': 'key1', 'auth': 'key2'}
            }

            with patch('api.push.require_session'):
                response = client.post('/push/subscribe', json=subscription)

            assert response.status_code == 201

    def test_subscribe_invalid_payload(self, client, mock_redis):
        """Invalid subscription should return 400."""
        with patch('api.push.require_session'):
            response = client.post('/push/subscribe', json={})

        assert response.status_code == 400

    def test_unsubscribe_removes_subscription(self, client, mock_redis):
        """Unsubscribe should remove subscription from Redis."""
        with patch('api.push.RedisClientService.create_connection', return_value=mock_redis):
            subscription = {'endpoint': 'https://example.com/push'}

            with patch('api.push.require_session'):
                response = client.post('/push/unsubscribe', json=subscription)

            assert response.status_code == 200

    def test_send_push_to_all(self, mock_redis):
        """Send push should call webpush for all subscriptions."""
        mock_redis.smembers.return_value = {
            json.dumps({'endpoint': 'https://example.com/push', 'keys': {}})
        }

        with patch('api.push.RedisClientService.create_connection', return_value=mock_redis), \
             patch('api.push.webpush') as mock_webpush, \
             patch('api.push._get_vapid_keys', return_value={'public': 'pub', 'private': 'priv'}):
            send_push_to_all("Test", "Body")

            # webpush should have been called
            assert mock_webpush.called

    def test_send_push_stale_cleanup(self, mock_redis):
        """410 responses should remove stale subscriptions."""
        mock_redis.smembers.return_value = {
            json.dumps({'endpoint': 'https://example.com/push1'}),
            json.dumps({'endpoint': 'https://example.com/push2'})
        }

        with patch('api.push.RedisClientService.create_connection', return_value=mock_redis), \
             patch('api.push.webpush') as mock_webpush, \
             patch('api.push._get_vapid_keys', return_value={'public': 'pub', 'private': 'priv'}):
            # First subscription succeeds, second returns 410
            response_410 = Mock()
            response_410.status_code = 410
            exception_410 = Exception()
            exception_410.response = response_410

            mock_webpush.side_effect = [None, exception_410]

            send_push_to_all("Test", "Body")

            # Should have removed stale subscription
            assert mock_redis.srem.called

    def test_send_push_no_subscriptions(self, mock_redis):
        """No subscriptions should return early."""
        mock_redis.smembers.return_value = set()

        with patch('api.push.RedisClientService.create_connection', return_value=mock_redis):
            # Should not error
            send_push_to_all("Test", "Body")
