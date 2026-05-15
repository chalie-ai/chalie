"""
Policies blueprint — per-action permission control (allow / ask / deny).
"""

import json
import logging

from flask import Blueprint, jsonify, request

from .auth import require_session

logger = logging.getLogger(__name__)

policies_bp = Blueprint('policies', __name__, url_prefix='/api/policies')


@policies_bp.route('', methods=['GET'])
@require_session
def get_policies():
    """Return all policy rules grouped by action_id with defaults filled in.

    Response 200::

        {
          "policies": {
            "email.search": {"chat": "allow", "subagent": "allow", "subconscious": "allow"},
            "email.manage": {"chat": "ask", "subagent": "ask", "subconscious": "deny"},
            ...
          }
        }
    """
    try:
        from services.database_service import get_shared_db_service
        from services.policy_service import PolicyService

        svc = PolicyService(get_shared_db_service())
        policies = svc.get_all()
        return jsonify({"policies": policies}), 200
    except Exception as exc:
        logger.error("[POLICIES API] GET failed: %s", exc)
        return jsonify({"error": "Failed to load policies"}), 500


@policies_bp.route('', methods=['PUT'])
@require_session
def update_policies():
    """Upsert a batch of policy rules.

    Request body::

        {"rules": [{"action_id": "email.manage", "context": "chat", "state": "allow"}, ...]}

    Response 200::

        {"updated": 3}
    """
    try:
        data = request.get_json(silent=True) or {}
        rules = data.get('rules', [])

        if not isinstance(rules, list):
            return jsonify({"error": "rules must be a list"}), 400

        for rule in rules:
            if not all(k in rule for k in ('action_id', 'context', 'state')):
                return jsonify({"error": "Each rule must have action_id, context, state"}), 400

        from services.database_service import get_shared_db_service
        from services.policy_service import PolicyService

        svc = PolicyService(get_shared_db_service())
        affected = svc.upsert(rules)
        return jsonify({"updated": affected}), 200
    except Exception as exc:
        logger.error("[POLICIES API] PUT failed: %s", exc)
        return jsonify({"error": "Failed to update policies"}), 500


@policies_bp.route('/reset', methods=['POST'])
@require_session
def reset_policies():
    """Reset all policies to defaults, optionally filtered by context.

    Request body (optional)::

        {"context": "chat"}

    Response 200::

        {"reset": 147}
    """
    try:
        data = request.get_json(silent=True) or {}
        context = data.get('context')

        from services.database_service import get_shared_db_service
        from services.policy_service import PolicyService

        svc = PolicyService(get_shared_db_service())
        affected = svc.reset_to_defaults(context)
        return jsonify({"reset": affected}), 200
    except Exception as exc:
        logger.error("[POLICIES API] Reset failed: %s", exc)
        return jsonify({"error": "Failed to reset policies"}), 500


@policies_bp.route('/respond', methods=['POST'])
@require_session
def respond_permission():
    """Write the user's allow/deny decision to Redis for the blocked dispatch thread.

    The chat WebSocket receive loop is blocked during chat processing, so the
    frontend sends permission responses via this REST endpoint instead.
    """
    body = request.get_json(silent=True) or {}
    request_id = body.get('request_id', '')
    approved = bool(body.get('approved', False))
    if not request_id:
        return jsonify(error='request_id required'), 400
    try:
        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()
        store.setex(f'policy:response:{request_id}', 60, json.dumps({'approved': approved}))
        return jsonify(ok=True), 200
    except Exception as exc:
        logger.error("[POLICIES API] Respond failed: %s", exc)
        return jsonify(error='Failed to write response'), 500


@policies_bp.route('/blocked', methods=['GET'])
@require_session
def get_blocked_log():
    """Return recent blocked-action entries.

    Query params: limit (default 50), offset (default 0).

    Response 200::

        {"entries": [...], "count": 42}
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)

        from services.database_service import get_shared_db_service
        from services.policy_service import PolicyService

        svc = PolicyService(get_shared_db_service())
        entries = svc.get_blocked_log(limit=limit, offset=offset)
        return jsonify({"entries": entries, "count": len(entries)}), 200
    except Exception as exc:
        logger.error("[POLICIES API] Blocked log failed: %s", exc)
        return jsonify({"error": "Failed to load blocked log"}), 500


@policies_bp.route('/blocked', methods=['DELETE'])
@require_session
def clear_blocked_log():
    """Clear all entries from the blocked log.

    Response 200::

        {"cleared": 12}
    """
    try:
        from services.database_service import get_shared_db_service
        from services.policy_service import PolicyService

        svc = PolicyService(get_shared_db_service())
        cleared = svc.clear_blocked_log()
        return jsonify({"cleared": cleared}), 200
    except Exception as exc:
        logger.error("[POLICIES API] Clear blocked log failed: %s", exc)
        return jsonify({"error": "Failed to clear blocked log"}), 500
