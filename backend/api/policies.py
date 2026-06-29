"""Policies namespace — per-action permission control (allow / ask / deny).

Not pure CRUD: a single-cell policy upsert plus four non-CRUD action/utility
endpoints (reset, respond, blocked-log read/clear). Each is DTO-typed through the
foundation boundary (``@expects``/``@responds``); mutation/action endpoints
return 204 no-body on success; read endpoints return bare lists. The
``_tag_display_rows`` display-enrichment helper stays here (it annotates rows
for the Brain UI, it is not serialization).
"""

import logging
import threading
from typing import TYPE_CHECKING, cast
from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from services.time_utils import parse_utc
from .auth import require_session
from .dto import Error, expects, register_dto, responds
from .dto.boundary import error
from .dto.policy import Policy, PolicyUpsert
from .dto.policy_blocked import BlockedEntry, BlockedQuery
from .dto.policy_respond import PolicyRespond

if TYPE_CHECKING:
    from services.policy_manager import PolicyManager

logger = logging.getLogger(__name__)

_ERR_LOAD = "Failed to load policies"
_ERR_UPDATE = "Failed to update policies"
_ERR_RESET = "Failed to reset policies"
_ERR_RESPOND = "Failed to resolve permission gate"
_ERR_BLOCKED = "Failed to load blocked log"
_ERR_CLEAR = "Failed to clear blocked log"

policies_ns = Namespace("policies", description="Per-action permission control", path="/api/policies")

register_dto(
    policies_ns,
    Policy,
    PolicyUpsert,
    BlockedEntry,
    BlockedQuery,
    PolicyRespond,
    Error,
)

_P = policies_ns.models


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


def _get_policy_manager() -> "PolicyManager":
    from services.database_service import get_shared_db_service
    from services.policy_manager import PolicyManager
    return PolicyManager(get_shared_db_service())


def _policy_dto(row: "dict[str, str]") -> Policy:
    """Build a :class:`Policy` DTO from a display-enriched service row."""
    return Policy(
        channel=row["channel"],
        permission=row["permission"],
        setting=row["setting"],
        label=row.get("label", ""),
        group=row.get("group"),
    )


def _blocked_dto(row: "dict[str, str]") -> BlockedEntry:
    """Build a :class:`BlockedEntry` DTO from a service row.

    ``created_at`` is persisted as an ISO-8601 string; ``parse_utc`` lifts it to a
    timezone-aware ``datetime`` so the foundation serializer re-emits canonical UTC.
    """
    return BlockedEntry(
        action_id=row["action_id"],
        context=row["context"],
        reason=row["reason"],
        created_at=parse_utc(row["created_at"]),
    )


@policies_ns.route("")
class PoliciesResource(Resource):
    @require_session
    @policies_ns.response(200, "All policies", model=_P["Policy"])
    @policies_ns.response(500, _ERR_LOAD, model=_P["Error"])
    @responds(Policy, code=200)
    def get(self) -> list[Policy] | ResponseReturnValue:
        """List all non-internal policies with display labels."""
        try:
            rows = _get_policy_manager().get_all()
            _tag_display_rows(rows)
            return [_policy_dto(row) for row in rows]
        except Exception as exc:
            logger.error("[POLICIES API] GET failed: %s", exc)
            return error(_ERR_LOAD, 500)

    @require_session
    @policies_ns.expect(_P["PolicyUpsert"])
    @policies_ns.response(204, "Policy upserted")
    @policies_ns.response(422, "Validation failed", model=_P["Error"])
    @policies_ns.response(500, _ERR_UPDATE, model=_P["Error"])
    @responds(code=204)
    @expects(PolicyUpsert)
    def put(self, dto: PolicyUpsert) -> None | ResponseReturnValue:
        """Upsert a single policy cell. Invalid channel/setting is a no-op 204."""
        try:
            _get_policy_manager().upsert(dto.channel, dto.permission, dto.setting)
            return None
        except Exception as exc:
            logger.error("[POLICIES API] PUT failed: %s", exc)
            return error(_ERR_UPDATE, 500)


@policies_ns.route("/reset")
class PoliciesResetResource(Resource):
    @require_session
    @policies_ns.response(204, "Policies reset to defaults")
    @policies_ns.response(500, _ERR_RESET, model=_P["Error"])
    @responds(code=204)
    def post(self) -> None | ResponseReturnValue:
        """Re-apply the static seed (wipe + reseed)."""
        try:
            _get_policy_manager().reset_to_defaults()
            return None
        except Exception as exc:
            logger.error("[POLICIES API] Reset failed: %s", exc)
            return error(_ERR_RESET, 500)


@policies_ns.route("/respond")
class PoliciesRespondResource(Resource):
    @require_session
    @policies_ns.expect(_P["PolicyRespond"])
    @policies_ns.response(204, "Permission gate resolved")
    @policies_ns.response(422, "Validation failed", model=_P["Error"])
    @policies_ns.response(500, _ERR_RESPOND, model=_P["Error"])
    @responds(code=204)
    @expects(PolicyRespond)
    def post(self, dto: PolicyRespond) -> None | ResponseReturnValue:
        """Wake the blocked ACT dispatch thread with the user's allow/deny decision.

        The ACT loop thread is parked on ``threading.Event.wait()`` inside
        ``PolicyManager._ask_user()``. This handler resolves the gate so the
        thread wakes instantly. An unknown or already-resolved ``request_id`` is a
        graceful success no-op (204), NOT a not-found error — the gate-absent path
        is by design.
        """
        try:
            from services.policy_manager import _permission_gates
            gate = _permission_gates.get(dto.request_id)
            if gate is not None:
                gate['result'] = 'approved' if dto.approved else 'denied'
                cast(threading.Event, gate['event']).set()
            else:
                logger.warning("[POLICIES API] No gate found for request_id=%s", dto.request_id)
            return None
        except Exception as exc:
            logger.error("[POLICIES API] Respond failed: %s", exc)
            return error(_ERR_RESPOND, 500)


@policies_ns.route("/blocked")
class PoliciesBlockedResource(Resource):
    @require_session
    @policies_ns.param("limit", "Max entries to return (1-500, default 50)", type="integer", _in="query")
    @policies_ns.response(200, "Blocked-action log", model=_P["BlockedEntry"])
    @policies_ns.response(500, _ERR_BLOCKED, model=_P["Error"])
    @responds(BlockedEntry, code=200)
    @expects(BlockedQuery, source="args")
    def get(self, dto: BlockedQuery) -> list[BlockedEntry] | ResponseReturnValue:
        """List recent blocked-action log entries."""
        try:
            entries = _get_policy_manager().get_blocked_log(limit=dto.limit)
            return [_blocked_dto(entry) for entry in entries]
        except Exception as exc:
            logger.error("[POLICIES API] Blocked log failed: %s", exc)
            return error(_ERR_BLOCKED, 500)

    @require_session
    @policies_ns.response(204, "Blocked log cleared")
    @policies_ns.response(500, _ERR_CLEAR, model=_P["Error"])
    @responds(code=204)
    def delete(self) -> None | ResponseReturnValue:
        """Clear all entries from the blocked log."""
        try:
            _get_policy_manager().clear_blocked_log()
            return None
        except Exception as exc:
            logger.error("[POLICIES API] Clear blocked log failed: %s", exc)
            return error(_ERR_CLEAR, 500)