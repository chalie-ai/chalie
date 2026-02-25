"""
Web Push blueprint — subscription management + VAPID key endpoint.
"""

import json
import logging
import os

from flask import Blueprint, jsonify, request

from .auth import require_session

logger = logging.getLogger(__name__)

push_bp = Blueprint('push', __name__)

SUBSCRIPTIONS_KEY = 'push:subscriptions'
VAPID_KEYS_KEY = 'push:vapid_keys'


def _get_vapid_keys():
    """Load or generate VAPID keys. Checks env vars first, then Redis.

    Returns dict with 'public' (unpadded URL-safe base64 of raw EC public key)
    and 'private' (PEM-encoded private key string).
    """
    import base64

    pub = os.environ.get('VAPID_PUBLIC_KEY')
    priv = os.environ.get('VAPID_PRIVATE_KEY')
    if pub and priv:
        return {'public': pub, 'private': priv}

    from services.redis_client import RedisClientService
    redis = RedisClientService.create_connection()

    cached = redis.get(VAPID_KEYS_KEY)
    if cached:
        keys = json.loads(cached)
        # Old keys stored in PEM format are incompatible with py_vapid's from_string()
        if keys.get('private', '').startswith('-----'):
            logger.warning("[Push] Cached VAPID key in PEM format, regenerating")
            redis.delete(VAPID_KEYS_KEY)
        else:
            return keys

    # Generate new VAPID key pair (P-256 / prime256v1)
    from cryptography.hazmat.primitives.asymmetric import ec

    private_key = ec.generate_private_key(ec.SECP256R1())

    # Raw 32-byte private scalar as URL-safe base64 (py_vapid's from_string() expects this)
    priv_raw = private_key.private_numbers().private_value.to_bytes(32, 'big')
    priv_b64 = base64.urlsafe_b64encode(priv_raw).decode('ascii').rstrip('=')

    # Uncompressed EC point → unpadded URL-safe base64 (for applicationServerKey)
    from cryptography.hazmat.primitives import serialization
    pub_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )
    pub_b64 = base64.urlsafe_b64encode(pub_bytes).decode('ascii').rstrip('=')

    keys = {'public': pub_b64, 'private': priv_b64}
    redis.set(VAPID_KEYS_KEY, json.dumps(keys))
    logger.info("[Push] Generated and stored new VAPID keys (raw base64 format)")
    return keys


@push_bp.route('/push/vapid-key', methods=['GET'])
def vapid_public_key():
    """Return the VAPID public key (needed by the browser to subscribe)."""
    try:
        keys = _get_vapid_keys()
        return jsonify({'publicKey': keys['public']}), 200
    except Exception as e:
        logger.error(f"[Push] Failed to get VAPID key: {e}", exc_info=True)
        return jsonify({'error': 'Failed to get VAPID key'}), 500


@push_bp.route('/push/subscribe', methods=['POST'])
@require_session
def push_subscribe():
    """Store a push subscription."""
    subscription = request.get_json()
    if not subscription or 'endpoint' not in subscription:
        return jsonify({'error': 'Invalid subscription'}), 400

    try:
        from services.redis_client import RedisClientService
        redis = RedisClientService.create_connection()
        redis.sadd(SUBSCRIPTIONS_KEY, json.dumps(subscription))
        logger.info(f"[Push] Stored subscription: {subscription['endpoint'][:60]}...")
        return jsonify({'ok': True}), 201
    except Exception as e:
        logger.error(f"[Push] Subscribe error: {e}", exc_info=True)
        return jsonify({'error': 'Failed to store subscription'}), 500


@push_bp.route('/push/unsubscribe', methods=['POST'])
@require_session
def push_unsubscribe():
    """Remove a push subscription."""
    subscription = request.get_json()
    if not subscription or 'endpoint' not in subscription:
        return jsonify({'error': 'Invalid subscription'}), 400

    try:
        from services.redis_client import RedisClientService
        redis = RedisClientService.create_connection()
        redis.srem(SUBSCRIPTIONS_KEY, json.dumps(subscription))
        return jsonify({'ok': True}), 200
    except Exception as e:
        logger.error(f"[Push] Unsubscribe error: {e}", exc_info=True)
        return jsonify({'error': 'Failed to remove subscription'}), 500


def send_push_to_all(title, body, tag='chalie-message'):
    """Send a web push notification to all stored subscriptions."""
    try:
        from pywebpush import webpush, WebPushException
        from services.redis_client import RedisClientService

        keys = _get_vapid_keys()
        redis = RedisClientService.create_connection()
        subscriptions = redis.smembers(SUBSCRIPTIONS_KEY)

        if not subscriptions:
            return

        payload_dict = {
            'title': title,
            'body': body,
            'tag': tag,
            'url': '/',
        }

        payload = json.dumps(payload_dict)

        vapid_key = keys['private']

        stale = []
        for raw_sub in subscriptions:
            sub = json.loads(raw_sub)
            try:
                webpush(
                    subscription_info=sub,
                    data=payload,
                    vapid_private_key=vapid_key,
                    vapid_claims={'sub': 'mailto:chalie@localhost'},
                )
            except WebPushException as e:
                if e.response and e.response.status_code in (404, 410):
                    stale.append(raw_sub)
                else:
                    logger.warning(f"[Push] Failed to send: {e}")
            except Exception as e:
                logger.warning(f"[Push] Send error: {e}")

        # Clean up expired subscriptions
        for raw_sub in stale:
            redis.srem(SUBSCRIPTIONS_KEY, raw_sub)
            logger.info("[Push] Removed stale subscription")

    except Exception as e:
        logger.warning(f"[Push] send_push_to_all error: {e}")
