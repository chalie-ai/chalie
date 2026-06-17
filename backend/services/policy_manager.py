"""PolicyManager — flat (channel, permission, setting) permission gate.

Single entry point: PolicyManager.wrap(channel, permission, callback, error).
Every native AND MCP tool call flows through it (ToolDispatcher.dispatch passes
ToolDispatcher._execute as the callback).

Settings: internal (always allowed, hidden in Brain) · allow · ask · deny.
Channels: ProcessorConfig.PolicyChannel values.

A small set of read-only / scratch / infrastructure tools (``INTERNAL``) ALWAYS
bypass the gate regardless of channel or any seeded row — they are never
user-gated and carry no seed rows. ANY action on these tools runs unconditionally.

The gate is dead simple: short-circuit INTERNAL tools, else SELECT the setting
(lazily creating an 'ask' row on a miss), then run | block | prompt. It has ZERO
knowledge of threads/timeouts — those live in the callback (Ability.execute).
"""

import json
import logging
import threading
import uuid

from services.processor_config import ProcessorConfig
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

CHANNEL = ProcessorConfig.PolicyChannel
VALID_CHANNELS = {c.value for c in CHANNEL}
VALID_SETTINGS = {"internal", "allow", "ask", "deny"}

# Read-only / scratch / infrastructure tools that ALWAYS bypass the gate, on
# every channel, with no seed row. ANY action on these runs unconditionally —
# they are never user-gated and never appear in the Brain policy surface.
INTERNAL = frozenset({
    "browser", "chalie_docs", "chat_history_compactor", "find_skills",
    "find_tools", "memory", "read", "review_tool_calls", "review_transcript",
    "save_graph", "save_pattern", "search", "skill_manager", "thinking",
    "tool_chain_compactor", "web_download",
})

# Channels with no human at a prompt: an `ask` becomes a `deny` (D2).
_NO_HUMAN = frozenset({CHANNEL.SUBCONSCIOUS, CHANNEL.EXTERNAL_AGENT})

_BLOCK = "The {permission} action is not allowed. Do NOT retry."

# request_id -> {"event": threading.Event, "result": str}. Woken by
# POST /api/policies/respond.
_permission_gates: dict[str, dict] = {}


class PolicyManager:
    def __init__(self, db):
        self.db = db

    # ── The single entry point dispatch calls ─────────────────────────────────

    @staticmethod
    def wrap(channel, permission, callback, error=_BLOCK):
        """Gate `callback` for (channel, permission)."""
        from services.database_service import get_shared_db_service  # noqa: PLC0415
        return PolicyManager(get_shared_db_service()).authorize(channel, permission, callback, error)

    # ── The gate: run | block | ask (dead simple) ─────────────────────────────

    def authorize(self, channel, permission, callback, error=_BLOCK):
        if permission.split(".", 1)[0] in INTERNAL:
            return callback()                       # INTERNAL tools always bypass (no channel, no row)
        setting = self._setting(channel.value, permission)
        if setting in ("internal", "allow"):
            return callback()
        if setting == "ask" and channel not in _NO_HUMAN and self._ask_user(permission, channel.value):
            return callback()
        reason = setting if setting == "deny" else ("user_unavailable" if channel in _NO_HUMAN else "user_denied")
        self._log_blocked(channel.value, permission, reason)
        return error.format(permission=permission)   # block STRING (uniform with execute's return)

    # ── Lookup-or-create: the entire provisioning story ───────────────────────

    def _setting(self, channel, permission):
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT setting FROM policy WHERE channel = ? AND permission = ?",
                (channel, permission),
            ).fetchone()
            if row:
                return row[0]
            conn.execute(
                "INSERT OR IGNORE INTO policy (channel, permission, setting) VALUES (?, ?, 'ask')",
                (channel, permission),
            )
            conn.commit()
            return "ask"

    # ── Interactive prompt (CHAT only; fail-open per D6) ──────────────────────

    def _ask_user(self, permission, channel):
        try:
            from services.websocket_broker import WebSocketBroker  # noqa: PLC0415
            rid = str(uuid.uuid4())
            gate = _permission_gates[rid] = {"event": threading.Event(), "result": None}
            WebSocketBroker().broadcast({
                "type": "permission_request",
                "request_id": rid,
                "action_id": permission,
                "context": channel,
            })
            gate["event"].wait()  # parks until POST /api/policies/respond
            return _permission_gates.pop(rid, {}).get("result") == "approved"
        except Exception as exc:  # noqa: BLE001
            logger.warning("[PolicyManager] permission gate failed: %s", exc)
            return True  # fail-open (D6)

    def _log_blocked(self, channel, permission, reason):
        with self.db.connection() as conn:
            conn.execute(
                "INSERT INTO policy_blocked_log (action_id, context, reason, created_at) "
                "VALUES (?, ?, ?, ?)",
                (permission, channel, reason, utc_now().isoformat()),
            )
            conn.commit()

    # ── Brain REST surface (api/policies.py) ──────────────────────────────────

    def get_all(self):
        """All rows EXCLUDING internal (hidden in Brain), as flat triples."""
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT channel, permission, setting FROM policy "
                "WHERE setting != 'internal' ORDER BY channel, permission"
            ).fetchall()
        return [{"channel": r[0], "permission": r[1], "setting": r[2]} for r in rows]

    def upsert(self, channel, permission, setting):
        """Single-cell upsert. Returns rows affected (0 on invalid input)."""
        if channel not in VALID_CHANNELS or setting not in VALID_SETTINGS:
            return 0
        with self.db.connection() as conn:
            cur = conn.execute(
                "INSERT INTO policy (channel, permission, setting) VALUES (?, ?, ?) "
                "ON CONFLICT(channel, permission) DO UPDATE SET setting = ?",
                (channel, permission, setting, setting),
            )
            conn.commit()
            return cur.rowcount

    def get_blocked_log(self, limit=50):
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT action_id, context, reason, created_at FROM policy_blocked_log "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"action_id": r[0], "context": r[1], "reason": r[2], "created_at": r[3]} for r in rows]

    def clear_blocked_log(self):
        with self.db.connection() as conn:
            cur = conn.execute("DELETE FROM policy_blocked_log")
            conn.commit()
            return cur.rowcount

    # ── Seed / reset (static policy_defaults.json) ────────────────────────────

    def apply_seed(self):
        """Load policy_defaults.json via INSERT OR IGNORE. Returns rows inserted."""
        from services.file_mapper_service import FileMapperService  # noqa: PLC0415
        with open(FileMapperService.get_policy_defaults_path()) as f:
            seed = json.load(f)
        inserted = 0
        with self.db.connection() as conn:
            for r in seed:
                inserted += conn.execute(
                    "INSERT OR IGNORE INTO policy (channel, permission, setting) VALUES (?, ?, ?)",
                    (r["channel"], r["permission"], r["setting"]),
                ).rowcount
            conn.commit()
        return inserted

    def reset_to_defaults(self):
        """Wipe and re-apply the static seed."""
        with self.db.connection() as conn:
            conn.execute("DELETE FROM policy")
            conn.commit()
        return self.apply_seed()
