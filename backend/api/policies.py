"""
Policies blueprint — per-action permission control (allow / ask / deny).
"""

import logging
import threading
from typing import TYPE_CHECKING, cast

from flask import request
from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from .auth import require_session

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

policies_bp = Namespace('policies', description='Per-action permission control', path='/api/policies')


def _tag_display_rows(rows: "list[dict[str, str]]") -> None:
    """Annotate every row in place with a display ``label`` (and ``group`` for
    MCP rows) so the Brain UI renders friendly names instead of raw permissions.

    - ``_mcp_<server>_<tool>`` -> group=server title, label=humanized tool name.
    - native ``tool.action``    -> label=humanized action only (``search_files.glob``
      -> ``Glob``); a bare ``tool`` with no action -> humanized tool name.

    MCP resolution is guarded: a misconfigured/absent MCP store must never break
    the page — MCP rows then fall through to the native branch.
    """
    from services.mcp_client_service import McpClientService, humanize_segment
    mcp_labels: dict[str, dict[str, str]] = {}
    try:
        mcp_labels = McpClientService().label_mcp_permissions(
            [r['permission'] for r in rows]
        )
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


@policies_bp.route('')
class PoliciesResource(Resource):
    @require_session
    @policies_bp.response(200, "Success")
    @policies_bp.response(500, "Internal server error")
    def get(self) -> ResponseReturnValue:
        try:
            from services.database_service import get_shared_db_service
            from services.policy_manager import PolicyManager
            rows = PolicyManager(get_shared_db_service()).get_all()
            _tag_display_rows(rows)
            return {"policies": rows}, 200
        except Exception as exc:
            logger.error("[POLICIES API] GET failed: %s", exc)
            return {"error": "Failed to load policies"}, 500

    @require_session
    @policies_bp.response(200, "Success")
    @policies_bp.response(400, "Bad request")
    @policies_bp.response(500, "Internal server error")
    def put(self) -> ResponseReturnValue:
        try:
            data = request.get_json(silent=True) or {}
            if not all(k in data for k in ('channel', 'permission', 'setting')):
                return {"error": "channel, permission, setting required"}, 400
            from services.database_service import get_shared_db_service
            from services.policy_manager import PolicyManager
            affected = PolicyManager(get_shared_db_service()).upsert(
                data['channel'], data['permission'], data['setting'])
            return {"updated": affected}, 200
        except Exception as exc:
            logger.error("[POLICIES API] PUT failed: %s", exc)
            return {"error": "Failed to update policies"}, 500


@policies_bp.route('/reset')
class PoliciesResetResource(Resource):
    @require_session
    @policies_bp.response(200, "Success")
    @policies_bp.response(500, "Internal server error")
    def post(self) -> ResponseReturnValue:
        """Re-apply the static seed (wipe + reseed)."""
        try:
            from services.database_service import get_shared_db_service
            from services.policy_manager import PolicyManager
            affected = PolicyManager(get_shared_db_service()).reset_to_defaults()
            return {"reset": affected}, 200
        except Exception as exc:
            logger.error("[POLICIES API] Reset failed: %s", exc)
            return {"error": "Failed to reset policies"}, 500


@policies_bp.route('/respond')
class PoliciesRespondResource(Resource):
    @require_session
    @policies_bp.response(200, "Success")
    @policies_bp.response(400, "Bad request")
    @policies_bp.response(500, "Internal server error")
    def post(self) -> ResponseReturnValue:
        """Wake the blocked ACT dispatch thread with the user's allow/deny decision.

        The ACT loop thread is parked on threading.Event.wait() inside
        PolicyManager._ask_user().  This handler resolves the gate so
        the thread wakes instantly with zero CPU overhead.
        """
        body = request.get_json(silent=True) or {}
        request_id = body.get('request_id', '')
        approved = bool(body.get('approved', False))
        if not request_id:
            return {"error": "request_id required"}, 400
        try:
            from services.policy_manager import _permission_gates
            gate = _permission_gates.get(request_id)
            if gate is None:
                # Gate already resolved or request_id unknown — respond gracefully
                logger.warning("[POLICIES API] No gate found for request_id=%s", request_id)
                return {"ok": True}, 200
            gate['result'] = 'approved' if approved else 'denied'
            cast(threading.Event, gate['event']).set()
            return {"ok": True}, 200
        except Exception as exc:
            logger.error("[POLICIES API] Respond failed: %s", exc)
            return {"error": "Failed to resolve permission gate"}, 500


@policies_bp.route('/blocked')
class PoliciesBlockedResource(Resource):
    @require_session
    @policies_bp.response(200, "Success")
    @policies_bp.response(500, "Internal server error")
    def get(self) -> ResponseReturnValue:
        try:
            limit = request.args.get('limit', 50, type=int)
            from services.database_service import get_shared_db_service
            from services.policy_manager import PolicyManager
            svc = PolicyManager(get_shared_db_service())
            entries = svc.get_blocked_log(limit=limit)
            return {"entries": entries, "count": len(entries)}, 200
        except Exception as exc:
            logger.error("[POLICIES API] Blocked log failed: %s", exc)
            return {"error": "Failed to load blocked log"}, 500

    @require_session
    @policies_bp.response(200, "Success")
    @policies_bp.response(500, "Internal server error")
    def delete(self) -> ResponseReturnValue:
        """Clear all entries from the blocked log."""
        try:
            from services.database_service import get_shared_db_service
            from services.policy_manager import PolicyManager

            svc = PolicyManager(get_shared_db_service())
            cleared = svc.clear_blocked_log()
            return {"cleared": cleared}, 200
        except Exception as exc:
            logger.error("[POLICIES API] Clear blocked log failed: %s", exc)
            return {"error": "Failed to clear blocked log"}, 500