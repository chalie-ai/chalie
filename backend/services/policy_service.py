"""
Policy Service — per-action permission control (allow / ask / deny).

Enforcement point: Ability.dispatch() calls PolicyService.enforce() between
handler lookup and execution.

Four independent contexts: chat, subagent, subconscious, external_agent.
Three states: allow (proceed), ask (block for user confirmation), deny (reject).
"""

import json
import logging
import threading
from typing import Literal

from services.time_utils import utc_now
from utils.data_utils import parse_json_column

logger = logging.getLogger(__name__)

State = Literal["allow", "ask", "deny"]
Context = Literal["chat", "subagent", "subconscious", "external_agent"]

# ── Permission gate registry ──────────────────────────────────────────────────
# Maps request_id → {'event': threading.Event, 'result': str}
# The REST endpoint POST /api/policies/respond resolves a gate by calling
# gate['event'].set() after writing gate['result'] = 'approved' | 'denied'.
# Formerly lived in act_dispatcher_service; moved here in T3.
_permission_gates: dict[str, dict] = {}


def _build_action_description(action_id: str, action: dict) -> str:
    """Build a human-readable one-liner describing the action for the permission card.

    Formerly in act_dispatcher_service; moved here in T3 alongside the
    permission-gate registry.
    """
    _VERBS = {
        'email.read': 'Reading an email',
        'email.search': 'Searching emails',
        'email.draft': 'Drafting an email',
        'email.send': 'Sending an email',
        'email.reply': 'Replying to an email',
        'email.forward': 'Forwarding an email',
        'email.manage': 'Managing an email',
        'calendar.update_event': 'Updating a calendar event',
        'calendar.create_event': 'Creating a calendar event',
        'schedule.create': 'Creating a scheduled task',
        'schedule.cancel': 'Cancelling a scheduled task',
        'browser.interact': 'Interacting with a webpage',
        'browser.render': 'Reading a webpage',
        'document.delete': 'Deleting a document',
        'document.create': 'Creating a document',
        'list.delete': 'Deleting a list',
        'code_eval': 'Running code',
    }
    _CONTEXT_KEYS: dict[str, list[str]] = {
        'email.manage': ['operation'],
        'email.draft': ['to', 'subject'],
        'email.send': ['to', 'subject'],
        'email.reply': ['uid'],
        'email.forward': ['uid', 'to'],
        'email.search': ['sender', 'subject', 'keyword'],
        'schedule.create': ['description'],
        'schedule.cancel': ['description'],
        'browser.interact': ['url'],
        'browser.render': ['url'],
        'document.delete': ['name'],
        'document.create': ['name'],
        'list.delete': ['list_name'],
        'calendar.update_event': ['summary'],
        'calendar.create_event': ['summary'],
    }

    base = _VERBS.get(action_id)
    if not base:
        base = action_id.replace('.', ' ').replace('_', ' ').title()

    if action_id == 'email.manage':
        op = action.get('operation', '')
        if op:
            base = op.replace('_', ' ').title() + ' an email'

    context_keys = _CONTEXT_KEYS.get(action_id, [])
    for key in context_keys:
        val = action.get(key)
        if val and isinstance(val, str) and key != 'operation':
            return f"{base} — {val}"
    return base

VALID_STATES: set[str] = {"allow", "ask", "deny"}
VALID_CONTEXTS: set[str] = {"chat", "subagent", "subconscious", "external_agent"}

USAGE_CLASS_TO_CONTEXT: dict[str, Context] = {
    "chat": "chat",
    "subagent": "subagent",
    "subconscious": "subconscious",
    "external_agent": "external_agent",
}

# ── System tools — always allowed, invisible to PolicyManager UI ─────────────
# Derived from abilities with SYSTEM = True at first access.
_system_tools_cache: frozenset[str] | None = None


def _get_system_tools() -> frozenset[str]:
    global _system_tools_cache
    if _system_tools_cache is not None:
        return _system_tools_cache
    from abilities._registry import AbilityRegistry
    visible = {a.NAME for a in AbilityRegistry.policy_visible()}
    _system_tools_cache = frozenset(
        a.NAME for a in AbilityRegistry.all() if a.NAME not in visible
    )
    return _system_tools_cache


def _is_system_action(action_id: str) -> bool:
    """True when *action_id* belongs to a SYSTEM ability.

    ``_get_system_tools()`` records bare ability names (e.g. ``memory``), but
    enforcement resolves ids that may carry a sub-action (``memory.recall``).
    A sub-action of a system ability is itself a system action, so match the
    name prefix as well as the full id.
    """
    system_tools = _get_system_tools()
    if action_id in system_tools:
        return True
    return action_id.split(".", 1)[0] in system_tools

# ── Default policy matrix ────────────────────────────────────────────────────
#
# Keys: action_id → {context: state}.  Missing context = "deny" for
# subconscious, "ask" for chat/subagent (safe fallback).

_CHAT_ALLOW: dict[str, State] = {
    # Reads
    "bash.read": "allow",
    "browser.interact": "allow", "browser.render": "allow",
    "browser.screenshot": "allow", "browser.monitor": "allow",
    "calendar.list_events": "allow", "calendar.get_event": "allow",
    "chalie_docs": "allow",
    "contacts.list": "allow", "contacts.get": "allow",
    "document.search": "allow", "document.list": "allow", "document.view": "allow",
    "email.search": "allow", "email.read": "allow",
    "find_skills": "allow",
    "find_tools": "allow",
    "home.get_state": "allow",
    "home.list_automations": "allow",
    "home.list_devices": "allow",
    "home.subscribe_events": "allow",
    "ubiquiti.list_devices": "allow", "ubiquiti.list_clients": "allow",
    "ubiquiti.get_info": "allow",
    "list.list_all": "allow", "list.view": "allow",
    "news": "allow",
    "programming_docs_search": "allow",
    "read": "allow",
    "code_eval": "allow",
    "document.upload": "allow",
    "schedule.cancel": "allow", "schedule.create": "allow", "schedule.list": "allow", "schedule.search": "allow",
    "search": "allow",
    "search_files.glob": "allow", "search_files.grep": "allow",
    "skill_builder.create": "allow",
    "skill_builder.delete": "allow",
    "skill_builder.edit": "allow",
    "skill_builder.list": "allow",
    "weather": "allow",
    # Reversible writes
    "document.create": "allow", "document.restore": "allow",
    "email.draft": "allow",
    "list.create": "allow", "list.add": "allow", "list.check": "allow",
    "list.remove": "allow", "list.clear": "allow", "list.rename": "allow",
    "place.save": "allow", "place.list": "allow", "place.get": "allow", "place.delete": "allow",
    "subagent.spawn": "allow", "subagent.list": "allow",
    "subagent.stop": "allow", "subagent.direct": "allow",
    # Delegate tools (subagent-as-tools, spec §5b) — focused read/research
    # agents that replace the former subagent types; allow by default in chat,
    # mirroring the prior subagent allow defaults.
    "web_search": "allow", "research": "allow",
    "summariser": "allow", "web_browse": "allow",
}

_CHAT_ASK: dict[str, State] = {
    "calendar.update_event": "ask",
    "document.delete": "ask",
    "email.forward": "ask",
    "email.manage": "ask",
    "email.reply": "ask",
    "email.send": "ask",
    "bash.compound": "ask",
    "bash.execute": "ask",
    "bash.installation": "ask",
    "bash.modify_file": "ask",
    "bash.remote_execution": "ask",
    "bash.web_fetch": "ask",
    "file_permissions": "ask",
    "file_write": "ask",
    "home.control": "ask",
    "home.trigger_automation": "ask",
    "ubiquiti.control_client": "ask",
    "ubiquiti.control_device": "ask",
    "ubiquiti.manage_wlan": "ask",
    "ubiquiti.manage_port_forward": "ask",
    "ubiquiti.manage_traffic_rule": "ask",
    "ubiquiti.authorize_guest": "ask",
    "list.delete": "ask",
    "web_download": "ask",
}

_SUBCONSCIOUS_ALLOW: dict[str, State] = {
    # Reads
    "calendar.list_events": "allow", "calendar.get_event": "allow",
    "chalie_docs": "allow",
    "contacts.list": "allow", "contacts.get": "allow",
    "document.search": "allow", "document.list": "allow", "document.view": "allow",
    "email.search": "allow", "email.read": "allow",
    "find_skills": "allow",
    "find_tools": "allow",
    "home.get_state": "allow",
    "home.list_automations": "allow",
    "home.list_devices": "allow",
    "ubiquiti.list_devices": "allow", "ubiquiti.list_clients": "allow",
    "ubiquiti.get_info": "allow",
    "list.list_all": "allow", "list.view": "allow",
    "news": "allow",
    "programming_docs_search": "allow",
    "read": "allow",
    "code_eval": "allow",
    "schedule.list": "allow", "schedule.search": "allow",
    "search": "allow",
    "search_files.glob": "allow", "search_files.grep": "allow",
    "subagent.list": "allow", "subagent.stop": "allow",
    "weather": "allow",
    # Internal writes
    "document.create": "allow", "document.restore": "allow",
    "list.create": "allow", "list.add": "allow", "list.check": "allow",
    "list.remove": "allow", "list.clear": "allow", "list.rename": "allow",
}

_EXTERNAL_AGENT_ALLOW: dict[str, State] = {
    # Reads
    "browser.render": "allow", "browser.screenshot": "allow", "browser.monitor": "allow",
    "calendar.list_events": "allow", "calendar.get_event": "allow",
    "chalie_docs": "allow",
    "contacts.list": "allow", "contacts.get": "allow",
    "document.search": "allow", "document.list": "allow", "document.view": "allow",
    "email.search": "allow", "email.read": "allow",
    "find_skills": "allow",
    "find_tools": "allow",
    "home.get_state": "allow",
    "home.list_automations": "allow",
    "home.list_devices": "allow",
    "home.subscribe_events": "allow",
    "ubiquiti.list_devices": "allow", "ubiquiti.list_clients": "allow",
    "ubiquiti.get_info": "allow",
    "list.list_all": "allow", "list.view": "allow",
    "news": "allow",
    "programming_docs_search": "allow",
    "read": "allow",
    "code_eval": "allow",
    "schedule.list": "allow", "schedule.search": "allow",
    "search": "allow",
    "search_files.glob": "allow", "search_files.grep": "allow",
    "subagent.spawn": "allow", "subagent.list": "allow",
    "subagent.stop": "allow", "subagent.direct": "allow",
    "weather": "allow",
    # Reversible writes
    "document.create": "allow", "document.restore": "allow",
    "list.create": "allow", "list.add": "allow", "list.check": "allow",
    "list.remove": "allow", "list.clear": "allow", "list.rename": "allow",
}

_EXTERNAL_AGENT_DENY: dict[str, State] = {
    # Sensitive actions — no user to confirm, so deny by default
    "bash.read": "deny",
    "bash.execute": "deny",
    "bash.modify_file": "deny",
    "bash.web_fetch": "deny",
    "bash.installation": "deny",
    "bash.remote_execution": "deny",
    "bash.compound": "deny",
    "browser.interact": "deny",
    "calendar.update_event": "deny",
    "document.delete": "deny",
    "email.forward": "deny",
    "email.manage": "deny",
    "email.reply": "deny",
    "email.send": "deny",
    "email.draft": "deny",
    "file_permissions": "deny",
    "file_write": "deny",
    "home.control": "deny",
    "home.trigger_automation": "deny",
    "ubiquiti.control_client": "deny",
    "ubiquiti.control_device": "deny",
    "ubiquiti.manage_wlan": "deny",
    "ubiquiti.manage_port_forward": "deny",
    "ubiquiti.manage_traffic_rule": "deny",
    "ubiquiti.authorize_guest": "deny",
    "list.delete": "deny",
    "schedule.create": "deny",
    "schedule.cancel": "deny",
}


_SUBAGENT_OVERRIDE: dict[str, State] = {
    "subagent.spawn": "ask",
    "subagent.direct": "ask",
    "skill_builder.create": "ask",
    "skill_builder.edit": "ask",
    "skill_builder.delete": "ask",
    "skill_builder.list": "ask",
}


def _build_defaults() -> dict[str, dict[Context, State]]:
    """Build the full default matrix from the ability registry."""
    from abilities._registry import AbilityRegistry

    all_action_ids: list[str] = []
    for ability in AbilityRegistry.policy_visible():
        schema = ability.INPUT_SCHEMA
        actions = schema.get("properties", {}).get("action", {}).get("enum", [])
        if actions:
            for act in actions:
                all_action_ids.append(f"{ability.NAME}.{act}")
        else:
            all_action_ids.append(ability.NAME)

    defaults: dict[str, dict[Context, State]] = {}
    for action_id in sorted(all_action_ids):
        chat_default = _CHAT_ALLOW.get(action_id, _CHAT_ASK.get(action_id, "ask"))
        defaults[action_id] = {
            "chat": chat_default,
            "subagent": _SUBAGENT_OVERRIDE.get(action_id, chat_default),
            "subconscious": _SUBCONSCIOUS_ALLOW.get(action_id, "deny"),
            "external_agent": _EXTERNAL_AGENT_ALLOW.get(action_id, _EXTERNAL_AGENT_DENY.get(action_id, "deny")),
        }
    return defaults


_defaults_cache: dict[str, dict[Context, State]] | None = None
_defaults_lock = threading.Lock()


def get_defaults() -> dict[str, dict[Context, State]]:
    global _defaults_cache
    if _defaults_cache is not None:
        return _defaults_cache
    with _defaults_lock:
        if _defaults_cache is None:
            _defaults_cache = _build_defaults()
    return _defaults_cache


def get_policy_meta() -> list[dict]:
    """Return label and category metadata for every policy-visible action_id.

    Derived from the AbilityRegistry — no hardcoded list.
    Returns a sorted list of {action_id, label, category} dicts.
    """
    from abilities._registry import AbilityRegistry

    entries: list[dict] = []
    for ability in AbilityRegistry.policy_visible():
        category = getattr(ability, "POLICY_CATEGORY", "") or ability.NAME.replace("_", " ").title()
        labels = getattr(ability, "POLICY_LABELS", {})
        schema = ability.INPUT_SCHEMA
        actions = schema.get("properties", {}).get("action", {}).get("enum", [])
        if actions:
            for act in actions:
                action_id = f"{ability.NAME}.{act}"
                label = labels.get(act, act.replace("_", " ").title())
                entries.append({"action_id": action_id, "label": label, "category": category})
        else:
            action_id = ability.NAME
            label = labels.get("", ability.SUMMARY if hasattr(ability, "SUMMARY") else action_id)
            entries.append({"action_id": action_id, "label": label, "category": category})
    entries.sort(key=lambda e: (e["category"], e["action_id"]))
    return entries


def reset_defaults_cache() -> None:
    global _defaults_cache
    _defaults_cache = None


class PolicyService:

    def __init__(self, db_service):
        self.db = db_service

    def seed_defaults(self) -> int:
        """Insert default policy rules for any action_ids not yet in the DB.

        Returns the number of rows inserted.
        """
        defaults = get_defaults()
        inserted = 0
        now = utc_now().isoformat()
        with self.db.connection() as conn:
            cursor = conn.cursor()
            for action_id, contexts in defaults.items():
                for context, state in contexts.items():
                    cursor.execute(
                        "INSERT OR IGNORE INTO policy_rules "
                        "(action_id, context, state, updated_at) VALUES (?, ?, ?, ?)",
                        (action_id, context, state, now),
                    )
                    inserted += cursor.rowcount
            conn.commit()
        if inserted:
            logger.info("[POLICY] Seeded %d default policy rules", inserted)
        return inserted

    def check(self, action_id: str, context: str) -> State:
        """Look up the policy state for an action in a given context.

        Falls back to defaults if no DB row exists.
        """
        if _is_system_action(action_id):
            return "allow"
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT state FROM policy_rules WHERE action_id = ? AND context = ?",
                (action_id, context),
            )
            row = cursor.fetchone()
        if row:
            return row[0]
        defaults = get_defaults()
        ctx_defaults = defaults.get(action_id, {})
        return ctx_defaults.get(context, "ask")

    def get_all(self) -> dict[str, dict[str, str]]:
        """Return all rules grouped by action_id, with defaults filled in.

        Returns: {action_id: {context: state, ...}, ...}
        """
        defaults = get_defaults()
        result: dict[str, dict[str, str]] = {}
        for action_id, contexts in defaults.items():
            result[action_id] = dict(contexts)

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT action_id, context, state FROM policy_rules")
            for row in cursor.fetchall():
                aid, ctx, state = row[0], row[1], row[2]
                if _is_system_action(aid):
                    continue
                if aid not in result:
                    result[aid] = {}
                result[aid][ctx] = state

        return result

    def upsert(self, rules: list[dict]) -> int:
        """Upsert a batch of policy rules.

        Args:
            rules: List of {action_id, context, state} dicts.

        Returns: Number of rows affected.
        """
        now = utc_now().isoformat()
        affected = 0
        with self.db.connection() as conn:
            cursor = conn.cursor()
            for rule in rules:
                action_id = rule["action_id"]
                context = rule["context"]
                state = rule["state"]
                if context not in VALID_CONTEXTS:
                    logger.warning("[POLICY] Invalid context %r, skipping", context)
                    continue
                if state not in VALID_STATES:
                    logger.warning("[POLICY] Invalid state %r, skipping", state)
                    continue
                cursor.execute(
                    "INSERT INTO policy_rules (action_id, context, state, updated_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(action_id, context) DO UPDATE SET state = ?, updated_at = ?",
                    (action_id, context, state, now, state, now),
                )
                affected += cursor.rowcount
            conn.commit()
        return affected

    def reset_to_defaults(self, context: str | None = None) -> int:
        """Reset all rules (or rules for a specific context) to defaults.

        Returns: Number of rows affected.
        """
        defaults = get_defaults()
        now = utc_now().isoformat()
        affected = 0
        with self.db.connection() as conn:
            cursor = conn.cursor()
            for action_id, contexts in defaults.items():
                for ctx, state in contexts.items():
                    if context and ctx != context:
                        continue
                    cursor.execute(
                        "INSERT INTO policy_rules (action_id, context, state, updated_at) "
                        "VALUES (?, ?, ?, ?) "
                        "ON CONFLICT(action_id, context) DO UPDATE SET state = ?, updated_at = ?",
                        (action_id, ctx, state, now, state, now),
                    )
                    affected += cursor.rowcount
            conn.commit()
        return affected

    # ── Blocked log ──────────────────────────────────────────────────────────

    def log_blocked(
        self,
        action_id: str,
        context: str,
        reason: str,
        params: dict | None = None,
    ) -> None:
        """Append an entry to the blocked-actions audit log."""
        now = utc_now().isoformat()
        params_json = json.dumps(params, default=str) if params else None
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO policy_blocked_log "
                "(action_id, context, reason, params_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (action_id, context, reason, params_json, now),
            )
            conn.commit()

    def get_blocked_log(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Return recent blocked-action entries, most recent first."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, action_id, context, reason, params_json, created_at "
                "FROM policy_blocked_log ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = cursor.fetchall()
        return [
            {
                "id": r[0],
                "action_id": r[1],
                "context": r[2],
                "reason": r[3],
                "params": parse_json_column(r[4], default=None),
                "created_at": r[5],
            }
            for r in rows
        ]

    def clear_blocked_log(self) -> int:
        """Delete all entries from the blocked log. Returns rows deleted."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM policy_blocked_log")
            affected = cursor.rowcount
            conn.commit()
        return affected

    # ── Utility ──────────────────────────────────────────────────────────────

    @staticmethod
    def resolve_action_id(action_type: str, params: dict) -> str:
        """Derive the policy action_id from a dispatch action.

        If the ability's INPUT_SCHEMA declares an 'action' enum, the action_id
        is '<ability_name>.<action_value>'.  Otherwise it's just the ability name.
        """
        sub_action = params.get("action")
        if sub_action:
            return f"{action_type}.{sub_action}"
        return action_type

    @staticmethod
    def resolve_context() -> Context:
        """Read the current processor's USAGE_CLASS and map to policy context."""
        from services.message_processor import current_processor
        proc = current_processor()
        if proc is None:
            return "chat"
        usage_class = getattr(proc, "USAGE_CLASS", "chat")
        return USAGE_CLASS_TO_CONTEXT.get(usage_class, "chat")

    @staticmethod
    def enforce(tool_name: str, params: dict, channel: str) -> "dict | None":
        """Check policy and return a block result dict if denied, or None to proceed.

        This is the gate called by ``Ability.dispatch()`` between handler lookup
        and execution.  It mirrors the logic from the old
        ``ActDispatcherService._enforce_policy()`` but is a static method on
        PolicyService so Ability.dispatch() can call it without importing the
        dead ActDispatcherService.

        Returns:
            None — proceed with execution.
            dict — a result dict with status='policy_denied' / 'error', caller
                   records it and returns its 'result' text.

        Spec §5 / I7 / I8 / AC-6.
        """
        _POLICY_DENY_MSG = (
            "POLICY BLOCK: This action ({action_id}) is permanently blocked by the "
            "user's policy settings (state=deny). Do NOT retry — the user must change "
            "this in Brain → Policies before it can run."
        )
        _POLICY_UNAVAILABLE_MSG = (
            "POLICY BLOCK: This action ({action_id}) requires explicit user approval "
            "but cannot be requested during background processing. "
            "It has been logged for the user to review."
        )
        _POLICY_USER_DENIED_MSG = (
            "POLICY BLOCK: The user was shown a permission prompt for this action "
            "({action_id}) and explicitly denied it. "
            "Do NOT retry this action in this conversation."
        )
        _POLICY_TIMEOUT_MSG = (
            "POLICY BLOCK: This action ({action_id}) requires user approval. "
            "A permission prompt was shown but the user did not respond. "
            "Do NOT retry this action unless the user asks."
        )

        action_id = PolicyService.resolve_action_id(tool_name, params)

        # Unknown actions (not in default matrix) skip enforcement.
        # _mcp_* tools carry explicit seeded policy_rules rows so they must
        # always proceed to check().
        if action_id not in get_defaults() and not tool_name.startswith("_mcp_"):
            return None

        # Derive context from the current processor's usage_class.
        from services.message_processor import current_processor  # noqa: PLC0415
        proc = current_processor()
        usage_class = getattr(proc, "USAGE_CLASS", "chat") if proc else "chat"
        context = USAGE_CLASS_TO_CONTEXT.get(usage_class, "chat")

        from services.database_service import get_shared_db_service  # noqa: PLC0415
        svc = PolicyService(get_shared_db_service())
        state = svc.check(action_id, context)

        if state == "allow":
            return None

        def _preview(p: dict, max_keys: int = 4) -> dict:
            skip = {"type", "action", "_rich_media_ordinal"}
            out: dict = {}
            for k, v in p.items():
                if k in skip:
                    continue
                if len(out) >= max_keys:
                    break
                out[k] = v
            return out

        if state == "deny":
            svc.log_blocked(action_id, context, "policy_deny", _preview(params))
            return {
                "status": "policy_denied",
                "result": _POLICY_DENY_MSG.format(action_id=action_id),
            }

        # state == "ask"
        if context == "subconscious":
            svc.log_blocked(action_id, context, "user_unavailable", _preview(params))
            return {
                "status": "policy_denied",
                "result": _POLICY_UNAVAILABLE_MSG.format(action_id=action_id),
            }

        # Chat / subagent: request user permission via REST → threading.Event.wait().
        # This is the AC-6 blocking permission gate.
        verdict = PolicyService._request_permission(action_id, params, context)
        if verdict != "approved":
            reason = "user_denied" if verdict == "denied" else "timeout"
            msg = (
                _POLICY_USER_DENIED_MSG if verdict == "denied" else _POLICY_TIMEOUT_MSG
            ).format(action_id=action_id)
            svc.log_blocked(action_id, context, reason, _preview(params))
            return {"status": "policy_denied", "result": msg}

        return None  # Approved — proceed.

    @staticmethod
    def _request_permission(action_id: str, params: dict, context: str) -> str:
        """Broadcast a permission_request and wait for user response.

        Uses threading.Event.wait() — zero CPU, OS-parked thread.  The REST
        endpoint POST /api/policies/respond calls gate.set() to wake this thread.

        Returns one of: 'approved', 'denied'.

        Spec §5 / AC-6.
        """
        import uuid as _uuid  # noqa: PLC0415
        try:
            from services.websocket_broker import WebSocketBroker  # noqa: PLC0415

            request_id = str(_uuid.uuid4())
            gate_entry: dict = {"event": threading.Event(), "result": None}
            _permission_gates[request_id] = gate_entry

            WebSocketBroker().broadcast({
                "type": "permission_request",
                "request_id": request_id,
                "action_id": action_id,
                "context": context,
                "description": _build_action_description(action_id, params),
            })

            gate_entry["event"].wait()  # parks here until REST handler fires
            result = gate_entry.get("result", "denied")
            _permission_gates.pop(request_id, None)
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("[PolicyService._request_permission] failed: %s", exc)
            return "approved"  # Fail open on unexpected errors
