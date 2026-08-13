"""PolicyManager — flat (channel, permission, setting) permission gate.

Single entry point: PolicyManager.wrap(channel, permission, callback, error).
Every native AND MCP tool call flows through it (ToolDispatcher.dispatch passes
ToolDispatcher._execute as the callback).

Settings: internal (always allowed, hidden in Brain) · allow · ask · deny.
Channels: PolicyChannel values.

The seed file (policy_defaults.json) is the single source of truth for the
policy table. ``apply_seed`` converges the table to the seed on every boot: it
INSERT-OR-IGNOREs every seeded row, then DELETES every row whose permission is
neither in the seed nor an MCP tool permission. Retiring an ability needs only
a seed-file edit — never a migration. The seed is exhaustive (80 permissions ×
3 channels = 240 rows), so matching on the permission column alone is correct
and complete.

A small set of tools (``INTERNAL``) ALWAYS bypass the gate regardless of channel
or any seeded row — read-only / scratch / infrastructure tools, plus
delegate-exclusive inner tools whose user-facing permission is the OUTER delegate
tool (``pim`` covers email/calendar/contacts exactly as ``web_search`` covers
search). They are never user-gated and carry no seed rows.

The gate is dead simple: short-circuit INTERNAL tools, else SELECT the setting
(lazily creating an 'ask' row on a miss), then run | block | prompt. The decision
carries no execution timeout — an allowed tool runs to completion in the callback
(Ability.execute). The interactive prompt parks until the user responds OR the
turn's should_stop() returns True, so it never pins the per-channel turn lock.
"""

import json
import logging
import threading
import uuid
from collections.abc import Callable
from typing import cast

from configs.enums.policy_channel import PolicyChannel
from models.policy import Policy
from models.policy_blocked_log import PolicyBlockedLog
from services.time_utils import utc_now
from services.websocket import Websocket

logger = logging.getLogger(__name__)

CHANNEL = PolicyChannel
VALID_CHANNELS = {c.value for c in CHANNEL}
VALID_SETTINGS = {"internal", "allow", "ask", "deny"}

# Tools that ALWAYS bypass the gate, on every channel, with no seed row. Two
# admission grounds: (1) read-only / scratch / infrastructure tools — never
# user-gated; (2) delegate-exclusive inner tools (DISCOVERABLE=False, pinned on
# one delegate config) — the user-facing permission is the OUTER delegate tool
# (``web_search`` covers search/news, ``web_browse`` covers browser,
# ``pim`` covers email/calendar/contacts, ``code_agent`` covers ``run_script``).
# The mutating file primitives (write_file / edit_file / move / make_dir /
# delete / set_permissions) are NOT here: they carry seeded ``allow`` rows, so
# the user can tighten them per channel. ANY action on INTERNAL tools runs unconditionally;
# they never appear in the Brain policy surface, and the reap pass removes any
# stale rows for them so this frozenset stays the single source of truth.
INTERNAL = frozenset({
    "browser", "calendar", "chalie_docs", "chat_history_compactor", "contacts",
    "delete_graph", "email", "find_skills", "find_tools", "image_preview", "mcp_tools", "news",
    "read", "recall", "review_tool_calls", "review_transcript", "run_script", "save_graph",
    "save_map", "save_synthesis", "search", "skill_manager", "web_fetch",
})

# Channels with no human at a prompt: an `ask` becomes a `deny` (D2).
_NO_HUMAN = frozenset({CHANNEL.SUBCONSCIOUS, CHANNEL.EXTERNAL_AGENT})

_BLOCK = "The {permission} action is not allowed. Do NOT retry."

# request_id -> {"event": threading.Event, "result": str}. Woken by
# POST /api/policies/respond.
_permission_gates: dict[str, dict[str, object]] = {}

# How often the interactive ask-gate re-checks the turn's should_stop() while
# parked: small enough that a cancelled turn frees the per-channel
# lock near-instantly, large enough to cost no measurable idle CPU.
_GATE_POLL_SECONDS = 0.25


class PolicyManager:
    # ── The single entry point dispatch calls ─────────────────────────────────

    @staticmethod
    def wrap(
        channel: PolicyChannel,
        permission: str,
        callback: Callable[[], str],
        error: str = _BLOCK,
        should_stop: "Callable[[], bool] | None" = None,
    ) -> str:
        """Gate `callback` for (channel, permission). `should_stop` (the turn's)
        lets a parked `ask` prompt unwind when the turn is cancelled."""
        return PolicyManager().authorize(channel, permission, callback, error, should_stop)

    # ── The gate: run | block | ask (dead simple) ─────────────────────────────

    def authorize(
        self,
        channel: PolicyChannel,
        permission: str,
        callback: Callable[[], str],
        error: str = _BLOCK,
        should_stop: "Callable[[], bool] | None" = None,
    ) -> str:
        if permission.split(".", 1)[0] in INTERNAL:
            return callback()                       # INTERNAL tools always bypass (no channel, no row)
        setting = Policy.get_or_create_default(channel.value, permission)
        if setting in ("internal", "allow"):
            return callback()
        if setting == "ask" and channel not in _NO_HUMAN and self._ask_user(permission, channel.value, should_stop):
            return callback()
        # A turn cancelled while parked on the ask gate denies the tool
        # but is NOT a user verdict — skip the audit row so the blocked-log stays
        # honest. The guard self-no-ops for every non-cancel path (no should_stop,
        # or it never returned True), which keeps the normal deny/unavailable logging.
        if should_stop is None or not should_stop():
            reason = setting if setting == "deny" else ("user_unavailable" if channel in _NO_HUMAN else "user_denied")
            self._log_blocked(channel.value, permission, reason)
        return error.format(permission=permission)   # block STRING (uniform with execute's return)

    # ── Lookup-or-create: the entire provisioning story ───────────────────────

    def _setting(self, channel: str, permission: str) -> str:
        # Kept as a thin wrapper so the gate's call site stays readable; the
        # bespoke classmethod on Policy owns the INSERT OR IGNORE default.
        return Policy.get_or_create_default(channel, permission)

    # ── Interactive prompt (CHAT only; fail-open per D6) ──────────────────────

    def _ask_user(self, permission: str, channel: str, should_stop: "Callable[[], bool] | None" = None) -> bool:
        try:
            from models.ws_message import WsMessage  # noqa: PLC0415
            rid = str(uuid.uuid4())
            gate = _permission_gates[rid] = {"event": threading.Event(), "result": None}
            Websocket.broadcast(WsMessage(
                type="permission_request",
                request_id=rid,
                action_id=permission,
                context=channel,
            ))
            # Park until the user responds (POST /api/policies/respond sets the gate
            # event) OR the turn is cancelled. Polling makes the wait a cooperative
            # cancel checkpoint instead of an unbounded park that would pin the
            # per-channel turn lock forever. A cancel resolves as DENIED;
            # we break and fall through — never raise, since the fail-open except
            # below would swallow it and WRONGLY approve the gated tool.
            event = cast(threading.Event, gate["event"])
            while not event.wait(_GATE_POLL_SECONDS):
                if should_stop is not None and should_stop():
                    break
            return _permission_gates.pop(rid, {}).get("result") == "approved"
        except Exception as exc:  # noqa: BLE001
            logger.warning("[PolicyManager] permission gate failed: %s", exc)
            return True  # fail-open (D6)

    def _log_blocked(self, channel: str, permission: str, reason: str) -> None:
        PolicyBlockedLog.log_blocked(
            action_id=permission,
            context=channel,
            reason=reason,
            created_at=utc_now().isoformat(),
        )

    # ── Brain REST surface (api/policies.py) ──────────────────────────────────

    def get_all(self) -> list[dict[str, str]]:
        """All rows EXCLUDING internal (hidden in Brain), as flat triples."""
        rows = (
            Policy.filter("setting", "internal", "!=")
            .order_by("channel, permission")
            .select("channel", "permission", "setting")
        )
        return [{k: str(v) for k, v in row.items()} for row in rows]

    def upsert(self, channel: str, permission: str, setting: str) -> int:
        """Single-cell upsert. Returns rows affected (0 on invalid input —
        including INTERNAL tools, which are never user-gated so no row may exist)."""
        if channel not in VALID_CHANNELS or setting not in VALID_SETTINGS:
            return 0
        if permission.split(".", 1)[0] in INTERNAL:
            return 0
        return Policy.upsert(channel, permission, setting)

    def get_blocked_log(self, limit: int = 50) -> list[dict[str, str]]:
        rows = (
            PolicyBlockedLog.order_by("created_at DESC")
            .limit(limit)
            .select("action_id", "context", "reason", "created_at")
        )
        return [{k: str(v) for k, v in row.items()} for row in rows]

    def clear_blocked_log(self) -> int:
        return PolicyBlockedLog.clear_all()

    # ── Seed / reset (static policy_defaults.json) ────────────────────────────

    def apply_seed(self) -> int:
        """Converge the policy table to the seed file.

        This runs every boot (run.py). Two passes:
        1. INSERT-OR-IGNORE every seeded row — this is the source of truth.
        2. DELETE every row whose permission is neither in the seed nor an MCP
           tool permission (created lazily and legitimately absent from the seed).

        Retiring an ability needs only a seed-file edit — never a migration.
        Returns the number of rows inserted by pass 1."""
        from services.file_mapper_service import FileMapperService  # noqa: PLC0415
        with open(FileMapperService.get_policy_defaults_path()) as f:
            seed = cast(list[dict[str, str]], json.load(f))
        inserted = 0
        for r in seed:
            inserted += Policy.insert_or_ignore(r["channel"], r["permission"], r["setting"])
        # Reap pass: delete rows whose permission is not in the seed and not MCP
        seed_permissions = {r["permission"] for r in seed}
        reaped = Policy.delete_unseeded(seed_permissions)
        if reaped:
            logger.info("[PolicyManager] reaped %d stale policy rows", reaped)
        return inserted

    def reset_to_defaults(self) -> int:
        """Wipe and re-apply the static seed."""
        Policy.clear_all()
        return self.apply_seed()
