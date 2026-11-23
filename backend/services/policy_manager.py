"""PolicyManager — flat (channel, permission, setting) permission gate.

Single entry point: ``DispatchService._authorize`` → ``PolicyManager.authorize``.
Every native AND MCP tool call flows through it.

Settings: internal (always allowed, hidden in Brain) · allow · ask · deny.
Channels: PolicyChannel values.

The seed file (policy_defaults.json) is the single source of truth for the
policy table. ``apply_seed`` converges the table to the seed on every boot: it
INSERT-OR-IGNOREs every seeded row, then DELETES every row whose permission is
neither in the seed nor an MCP tool permission. Retiring an ability needs only
a seed-file edit — never a migration. The seed is exhaustive (87 permissions ×
3 channels = 261 rows), so matching on the permission column alone is correct
and complete.

A small set of tools (``INTERNAL``) ALWAYS bypass the gate regardless of channel
or any seeded row — read-only / scratch / infrastructure tools, and the tools
that spawn and manage subagents. They are never user-gated and carry no seed
rows. A tool a subagent calls is gated on the channel of the agent at the top of
the chain, since every subagent runs on its spawner's policy channel.

The gate is dead simple: short-circuit INTERNAL tools, else SELECT the setting
(lazily creating an 'ask' row on a miss), then run | block | ask. The decision
carries no execution timeout — an allowed tool runs to completion in the callback
(Ability.execute). An ``ask`` goes to the user through ``AskUser`` whether or
not anyone is watching; the call's ``origin`` (the interactive turn it belongs
to, if any) rides along as plain data. The calling turn's thread waits for the
answer, with no time limit, until the user replies or dismisses the ask or the
turn's should_stop() returns True; it holds no lock while it waits, so other
turns run beside it. An exception while asking propagates: nothing runs
unapproved.
"""

import json
import logging
import threading
from collections.abc import Callable
from typing import cast

from configs.enums.policy_channel import PolicyChannel
from models.policy import Policy
from models.policy_blocked_log import PolicyBlockedLog
from services.ask_user import AskReply, AskUser, Origin, PermissionAsk
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

CHANNEL = PolicyChannel
VALID_CHANNELS = {c.value for c in CHANNEL}
VALID_SETTINGS = {"internal", "allow", "ask", "deny"}

# Tools that ALWAYS bypass the gate, on every channel, with no seed row. Three
# admission grounds: (1) read-only / scratch / infrastructure tools — never
# user-gated; (2) the tools that spawn subagents and those that list, check,
# stop or message them, and the one a subagent messages its spawner with —
# starting, managing and talking to subagents is internal mechanics, never
# asked; (3) delegate-exclusive inner tools of web_search and web_browse
# (search/news, browser), which have no permission of their own. email /
# calendar / contacts are NOT here: they carry seeded per-action rows, gated on
# the spawner's channel; nor is code_agent's run_script, which carries one
# seeded row per channel. Nor are the mutating file primitives (write_file /
# edit_file / move / make_dir / delete / set_permissions): they carry seeded
# ``allow`` rows, so the user can tighten them per channel. ANY action on
# INTERNAL tools runs unconditionally;
# they never appear in the Brain policy surface, and the reap pass removes any
# stale rows for them so this frozenset stays the single source of truth.
INTERNAL = frozenset({
    "agent_status", "browser", "chalie_docs", "chat_history_compactor", "code_agent",
    "delete_graph", "find_skills", "find_tools", "image_preview", "list_agents", "mcp_tools",
    "message_parent", "message_subagent", "news", "pim", "plan", "read", "recall",
    "review_tool_calls", "review_transcript", "save_graph", "save_map",
    "save_synthesis", "search", "skill_manager", "stop_subagent", "subtask", "web_browse",
    "web_fetch", "web_search",
})

_BLOCK = "The {permission} action is not allowed. Do NOT retry."

# How often a waiting ask re-checks the turn's should_stop(): small enough that
# a stopped turn ends its wait near-instantly, large enough to cost no
# measurable idle CPU.
_GATE_POLL_SECONDS = 0.25


class PolicyManager:
    # ── The gate: run | block | ask (dead simple) ─────────────────────────────

    def authorize(
        self,
        channel: PolicyChannel,
        permission: str,
        callback: Callable[[], str],
        error: str = _BLOCK,
        should_stop: "Callable[[], bool] | None" = None,
        origin: dict[str, object] | None = None,
        summary: str = "",
    ) -> str:
        if permission.split(".", 1)[0] in INTERNAL:
            return callback()                       # INTERNAL tools always bypass (no channel, no row)
        setting = Policy.get_or_create_default(channel.value, permission)
        if setting in ("internal", "allow"):
            return callback()
        if setting == "ask" and self._ask_user(permission, channel.value, origin, summary, should_stop):
            return callback()
        # A turn cancelled while waiting on the ask denies the tool
        # but is NOT a user verdict — skip the audit row so the blocked-log stays
        # honest. The guard self-no-ops for every non-cancel path (no should_stop,
        # or it never returned True), which keeps the normal deny/denied logging.
        if should_stop is None or not should_stop():
            self._log_blocked(channel.value, permission, setting if setting == "deny" else "user_denied")
        return error.format(permission=permission)   # block STRING (uniform with execute's return)

    # ── Lookup-or-create: the entire provisioning story ───────────────────────

    def _setting(self, channel: str, permission: str) -> str:
        # Kept as a thin wrapper so the gate's call site stays readable; the
        # bespoke classmethod on Policy owns the INSERT OR IGNORE default.
        return Policy.get_or_create_default(channel, permission)

    # ── Asking the user ───────────────────────────────────────────────────────

    def _ask_user(
        self,
        permission: str,
        channel: str,
        origin: dict[str, object] | None,
        summary: str,
        should_stop: "Callable[[], bool] | None" = None,
    ) -> bool:
        """Ask the user whether ``permission`` may run and wait in this thread
        for the answer; True only when they allow it. A dismissed ask, or one
        cancelled because the turn's should_stop() fired, reads as denied.
        ``channel`` is the policy channel the row was read on — kept for the
        log line only."""
        done = threading.Event()
        outcome: dict[str, bool] = {}

        def on_reply(reply: AskReply) -> None:
            outcome["allowed"] = reply.answered and reply.value is True
            done.set()

        ask = PermissionAsk(
            action_id=permission,
            summary=summary,
            origin=None if origin is None else Origin.model_validate(origin),
        )
        ask_id = AskUser.permission(ask, on_reply)
        logger.info(
            "[PolicyManager] ask parked: ask_id=%s permission=%s channel=%s origin=%s",
            ask_id, permission, channel, origin,
        )
        try:
            while not done.wait(_GATE_POLL_SECONDS):
                if should_stop is not None and should_stop():
                    AskUser.cancel(ask_id)
        finally:
            # An exception out of the wait still closes the ask; once it is
            # answered or cancelled this is a no-op.
            AskUser.cancel(ask_id)
        return outcome["allowed"]

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
