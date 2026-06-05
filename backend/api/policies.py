"""
Policies blueprint — per-action permission control (allow / ask / deny).
"""

import logging

from flask import Blueprint, jsonify, request

from .auth import require_session

logger = logging.getLogger(__name__)

policies_bp = Blueprint('policies', __name__, url_prefix='/api/policies')


def _tag_display_rows(rows: list[dict]) -> None:
    """Annotate every row in place with a display ``label`` (and ``group`` for
    MCP rows) so the Brain UI renders friendly names instead of raw permissions.

    - ``_mcp_<server>_<tool>`` -> group=server title, label=humanized tool name.
    - native ``tool.action``    -> label=humanized action only (``search_files.glob``
      -> ``Glob``); a bare ``tool`` with no action -> humanized tool name.

    MCP resolution is guarded: a misconfigured/absent MCP store must never break
    the page — MCP rows then fall through to the native branch.
    """
    from services.mcp_client_service import McpClientService, humanize_segment
    mcp_labels: dict = {}
    try:
        mcp_labels = McpClientService().label_mcp_permissions([r['permission'] for r in rows])
    except Exception as exc:  # noqa: BLE001 — display enrichment must not 500 the page
        logger.warning("[POLICIES API] MCP row tagging skipped: %s", exc)
    for r in rows:
        info = mcp_labels.get(r['permission'])
        if info:
            r['group'] = info['group']
            r['label'] = info['label']
        else:
            _base, _, action = r['permission'].partition('.')
            r['label'] = humanize_segment(action or _base)


@policies_bp.route('', methods=['GET'])
@require_session
def get_policies():
    """Return all policy rows (flat triples), excluding internal rows.

    Response 200:: {"policies": [{"channel","permission","setting"}, ...]}

    Every row carries a display ``label`` (humanized); ``_mcp_*`` rows also carry
    a ``group`` (server title) so the Brain UI groups them by MCP server instead
    of showing one raw ``_mcp_<server>_<tool>`` group per tool.
    """
    try:
        from services.database_service import get_shared_db_service
        from services.policy_manager import PolicyManager
        rows = PolicyManager(get_shared_db_service()).get_all()
        _tag_display_rows(rows)
        return jsonify({"policies": rows}), 200
    except Exception as exc:
        logger.error("[POLICIES API] GET failed: %s", exc)
        return jsonify({"error": "Failed to load policies"}), 500


@policies_bp.route('', methods=['PUT'])
@require_session
def update_policies():
    """Single-cell upsert.  Body:: {"channel","permission","setting"}  ->  {"updated": 1}"""
    try:
        data = request.get_json(silent=True) or {}
        if not all(k in data for k in ('channel', 'permission', 'setting')):
            return jsonify({"error": "channel, permission, setting required"}), 400
        from services.database_service import get_shared_db_service
        from services.policy_manager import PolicyManager
        affected = PolicyManager(get_shared_db_service()).upsert(
            data['channel'], data['permission'], data['setting'])
        return jsonify({"updated": affected}), 200
    except Exception as exc:
        logger.error("[POLICIES API] PUT failed: %s", exc)
        return jsonify({"error": "Failed to update policies"}), 500


@policies_bp.route('/reset', methods=['POST'])
@require_session
def reset_policies():
    """Re-apply the static seed (wipe + reseed).  -> {"reset": N}"""
    try:
        from services.database_service import get_shared_db_service
        from services.policy_manager import PolicyManager
        affected = PolicyManager(get_shared_db_service()).reset_to_defaults()
        return jsonify({"reset": affected}), 200
    except Exception as exc:
        logger.error("[POLICIES API] Reset failed: %s", exc)
        return jsonify({"error": "Failed to reset policies"}), 500


@policies_bp.route('/respond', methods=['POST'])
@require_session
def respond_permission():
    """Wake the blocked ACT dispatch thread with the user's allow/deny decision.

    The ACT loop thread is parked on threading.Event.wait() inside
    PolicyManager._ask_user().  This handler resolves the gate so
    the thread wakes instantly with zero CPU overhead.
    """
    body = request.get_json(silent=True) or {}
    request_id = body.get('request_id', '')
    approved = bool(body.get('approved', False))
    if not request_id:
        return jsonify(error='request_id required'), 400
    try:
        from services.policy_manager import _permission_gates
        gate = _permission_gates.get(request_id)
        if gate is None:
            # Gate already resolved or request_id unknown — respond gracefully
            logger.warning("[POLICIES API] No gate found for request_id=%s", request_id)
            return jsonify(ok=True), 200
        gate['result'] = 'approved' if approved else 'denied'
        gate['event'].set()
        return jsonify(ok=True), 200
    except Exception as exc:
        logger.error("[POLICIES API] Respond failed: %s", exc)
        return jsonify(error='Failed to resolve permission gate'), 500


@policies_bp.route('/blocked', methods=['GET'])
@require_session
def get_blocked_log():
    """Return recent blocked-action entries.

    Query params: limit (default 50).

    Response 200:: {"entries": [...], "count": 42}
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        from services.database_service import get_shared_db_service
        from services.policy_manager import PolicyManager
        svc = PolicyManager(get_shared_db_service())
        entries = svc.get_blocked_log(limit=limit)
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
        from services.policy_manager import PolicyManager

        svc = PolicyManager(get_shared_db_service())
        cleared = svc.clear_blocked_log()
        return jsonify({"cleared": cleared}), 200
    except Exception as exc:
        logger.error("[POLICIES API] Clear blocked log failed: %s", exc)
        return jsonify({"error": "Failed to clear blocked log"}), 500
