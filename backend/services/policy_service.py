"""
Policy Service — per-action permission control (allow / ask / deny).

Enforcement point: ActDispatcherService.dispatch_action() calls
PolicyService.check() between handler lookup and execution.

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

# ── Default policy matrix ────────────────────────────────────────────────────
#
# Keys: action_id → {context: state}.  Missing context = "deny" for
# subconscious, "ask" for chat/subagent (safe fallback).

_CHAT_ALLOW: dict[str, State] = {
    # Reads
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
    "memory.recall": "allow", "memory.reflect": "allow",
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
    "web_download.tmp": "allow",
    "list.create": "allow", "list.add": "allow", "list.check": "allow",
    "list.remove": "allow", "list.clear": "allow", "list.rename": "allow",
    "memory.store": "allow",
    "place.save": "allow", "place.list": "allow", "place.get": "allow", "place.delete": "allow",
    "subagent": "allow",
}

_CHAT_ASK: dict[str, State] = {
    "calendar.update_event": "ask",
    "document.delete": "ask",
    "email.forward": "ask",
    "email.manage": "ask",
    "email.reply": "ask",
    "email.send": "ask",
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
    "memory.forget": "ask",
    "web_download.system": "ask",
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
    "memory.recall": "allow", "memory.reflect": "allow",
    "news": "allow",
    "programming_docs_search": "allow",
    "read": "allow",
    "schedule.list": "allow", "schedule.search": "allow",
    "search": "allow",
    "search_files.glob": "allow", "search_files.grep": "allow",
    "weather": "allow",
    # Internal writes
    "document.create": "allow", "document.restore": "allow",
    "list.create": "allow", "list.add": "allow", "list.check": "allow",
    "list.remove": "allow", "list.clear": "allow", "list.rename": "allow",
    "memory.store": "allow",
    "web_download.tmp": "allow",
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
    "memory.recall": "allow", "memory.reflect": "allow",
    "news": "allow",
    "programming_docs_search": "allow",
    "read": "allow",
    "schedule.list": "allow", "schedule.search": "allow",
    "search": "allow",
    "search_files.glob": "allow", "search_files.grep": "allow",
    "weather": "allow",
    # Reversible writes
    "document.create": "allow", "document.restore": "allow",
    "list.create": "allow", "list.add": "allow", "list.check": "allow",
    "list.remove": "allow", "list.clear": "allow", "list.rename": "allow",
    "memory.store": "allow",
    "web_download.tmp": "allow",
}

_EXTERNAL_AGENT_DENY: dict[str, State] = {
    # Sensitive actions — no user to confirm, so deny by default
    "browser.interact": "deny",
    "calendar.update_event": "deny",
    "code_eval": "deny",
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
    "memory.forget": "deny",
    "schedule.create": "deny",
    "schedule.cancel": "deny",
    "subagent": "deny",
    "web_download.system": "deny",
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
        defaults[action_id] = {
            "chat": _CHAT_ALLOW.get(action_id, _CHAT_ASK.get(action_id, "ask")),
            "subagent": _CHAT_ALLOW.get(action_id, _CHAT_ASK.get(action_id, "ask")),
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
        if action_id in _get_system_tools():
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
                if aid in _get_system_tools():
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
