# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
MessageProcessor — the single flat processor for all LLM message turns.

Lifecycle: one instance per turn. Two turns never share the same object.
Do not add `.instance()` / singleton accessors. There are NO subclasses
(spec §7a / P1): every input channel calls the static ``process()`` entry
point with a per-turn ``ProcessorConfig`` that carries all channel-specific
behaviour (channel, role, prompt builders, tool scopes, post-turn hook, …).

The class provides the ACT lifecycle (``process`` → ``_run`` → ``_setup`` →
``_loop`` → ``_record``), tool dispatch through ``ToolDispatcher.dispatch()``, and
the trail/compaction primitives.
"""

import logging
import re
import threading
from typing import TYPE_CHECKING

from services.metrics_accumulator import MetricsAccumulator
from services.time_formatter_service import TimeFormatterService

if TYPE_CHECKING:
    from services.processor_config import ProcessorConfig

logger = logging.getLogger(__name__)

# ── Trail-compaction boundary markers ───────────────────────────────────────────
#
# The two compaction abilities are dispatched programmatically (never
# model-selected) by _dispatch_compaction(). ChatHistoryCompactor writes the
# durable transcript watermark; ToolChainCompactor's recorded tool_calls row is
# the act-trail boundary. Both names are framework markers, not real tool
# activity, so the trail helpers below treat them specially.

#: The act-trail boundary tool. The LAST non-empty row with this name marks the
#: start of the live (un-compacted) trail slice — see _from_last_compaction.
_TRAIL_BOUNDARY_TOOL = "tool_chain_compactor"

#: Both compactors are framework markers: they never count as real trail content
#: (_has_trail) and the chat-history marker is never rendered to the model (its
#: compacted output reaches the model through the checkpoint prepend instead).
_COMPACTOR_TOOLS: "frozenset[str]" = frozenset({"chat_history_compactor", "tool_chain_compactor"})


_LLM_SENTINEL_PATTERNS = (
    re.compile(r'<\|[^|<>]*\|>'),
    re.compile(r'<\|[^|<>]*\|'),
)


def _sanitize_llm_args(value):
    if isinstance(value, str):
        for p in _LLM_SENTINEL_PATTERNS:
            value = p.sub('', value)
        return value.strip()
    if isinstance(value, list):
        return [_sanitize_llm_args(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_llm_args(v) for k, v in value.items()}
    return value


# ── Attachment reader — used by _seed_turn_zero() ─────────────────────────────

def _read_attachment(path: str) -> "tuple[str, str, str]":
    """Read a temp attachment and return (name, base64_content, content_type).

    Applies a path-traversal guard: only paths under the Chalie temp prefix
    (see ``services.tmp_storage``) are accepted. Raises OSError if the path is
    rejected or the file cannot be read.

    Returns:
        (filename, base64-encoded file bytes, MIME type string)
    """
    import base64
    import mimetypes
    import os

    from services.tmp_storage import TMP_PATH_PREFIX

    real = os.path.realpath(path)
    if not real.startswith(TMP_PATH_PREFIX) or not os.path.isfile(real):
        raise OSError(f"Unsafe or missing attachment path: {path!r}")

    with open(real, "rb") as fh:
        file_bytes = fh.read()

    name = os.path.basename(real)
    content_type, _ = mimetypes.guess_type(real)
    content_type = content_type or "application/octet-stream"
    content_b64 = base64.b64encode(file_bytes).decode()
    return name, content_b64, content_type


# ── Traceability spine ────────────────────────────────────────────────────────
#
# The MessageProcessor instance is the "parent" of everything that runs inside a
# turn. Rather than hide it behind a global ContextVar, it is threaded
# explicitly: ``ToolDispatcher(self).dispatch(…)`` binds it onto each per-call
# ability as ``self.MessageProcessor``, and the Providers facade receives it as ``mp=``.
# Wherever we are in a turn we can always reach the parent — and reconstruct the
# full path that got us there — by holding a real reference, not by reaching
# into a thread-local.


class MessageProcessor:
    """The single flat message processor for every channel (spec §1 / §4 / §7a).

    There are no subclasses.  A turn is driven entirely by a per-turn
    ``ProcessorConfig`` (channel, role, prompt builders, tool scope, the
    ``post_turn_hooks`` tuple, …) supplied to the ``process()`` entry point.  The
    lifecycle is ``process → _run → _setup → _loop → _record``; tool calls route
    through ``ToolDispatcher.dispatch`` and the act-trail is reconstructed from
    ``tool_calls`` rows (§4c) rather than held in memory.
    """

    # ── Tool-scope constants ──────────────────────────────────────────────────
    #
    # find_tools reads ``DISCOVERABLE`` (via ``self.MessageProcessor``) to gate
    # which abilities may be surfaced at runtime.  It is a class-level default
    # shared by every flat-path turn.

    # ``find_tools`` is gated to ``WHERE name IN DISCOVERABLE`` so a
    # processor can never discover anything outside this list.
    DISCOVERABLE: list[str] = [
        "bash",
        "browser",
        "calendar",
        "chalie_docs",
        "code_eval",
        "contacts",
        "document",
        "email",
        "file_permissions",
        "file_write",
        "home",
        "list",
        # mcp_manager is DISCOVERABLE (find_tools can surface it) but SYSTEM-for-policy
        # (always-allowed, never shown in Policy Manager).  See McpManagerAbility.
        "mcp_manager",
        "news",
        "place",
        "programming_docs_search",
        "read",
        "review_tool_calls",
        "review_transcript",
        "schedule",
        "search",
        "search_files",
        "skill_builder",
        "timer",
        "ubiquiti",
        "weather",
        "web_browse",
        "web_download",
        "web_search",
    ]
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, raw_input: str, metadata: dict | None = None):
        self._raw_input = raw_input
        self._metadata = metadata or {}
        # Per-turn config — a plain attribute set by `mp.config = X` in
        # process() / the background workers.  None until attached.
        self.config: "ProcessorConfig | None" = None
        self._memory_seed: str | None = None
        # Raw recall query used by pre_act(); kept separate from _memory_seed
        # (which is the formatted tag block) so recall_episodes() can embed
        # the original query for drift computation rather than the block string.
        self._memory_seed_query: str | None = None
        # Tracks the current ACT loop iteration for tool-event emission
        # without thread-local indirection.
        self._current_iteration: int = 0
        # Per-turn log of memory recall queries (seed + llm_recall).
        # Populated by the memory skill recall path; consumed by the next
        # recall call for redundancy-narrow and drift-expand computation.
        # Entries: {'query': str, 'embedding': list[float],
        #           'caller': 'seed'|'llm_recall', 'effective_radius': float}.
        # Never persisted — cleared when the instance is discarded.
        self._memory_query_history: list[dict] = []
        self._act_trail: list[str] = []
        self._loop_exited_cleanly: bool = False
        self._active_tools: list[str] = []
        self._uid: int | None = None
        self._deliberation_scalar: float | None = None   # raw sigmoid for this turn
        self._deliberation_ema: float | None = None      # EMA after this turn's update
        # Accumulator starts immediately so exploration + compaction tokens count.
        self._metrics: MetricsAccumulator = MetricsAccumulator()
        # The mp owns its provider gateway — param-free, scaffolds from self.
        from services.providers import Providers  # noqa: PLC0415
        self.providers = Providers(self)
        # Window-fit state (design §3.3, trim-then-compact). _history_drop is the
        # number of oldest history rows the fit loop has dropped for the current
        # send; _compaction_pending flags that a send had to trim, so _loop
        # compacts the full history into the next turn's checkpoint.
        self._history_drop: int = 0
        self._compaction_pending: bool = False
        # One-shot guard: a real provider 413 (PayloadTooLargeError) despite the
        # estimate-based fit triggers a single collapse-and-retry, then fails loud.
        self._payload_compacted: bool = False
        # Cooperative cancellation flag. Set by stop endpoints to signal the
        # ACT loop to exit at the next iteration boundary. Never raises —
        # the loop checks is_set() at the top of each iteration.
        self._cancel_event: threading.Event = threading.Event()

    def cancel(self) -> None:
        """Signal the ACT loop to exit at the next iteration boundary.

        Public interface for stop endpoints — avoids reaching into the private
        ``_cancel_event`` attribute from outside the class hierarchy.
        """
        self._cancel_event.set()

    # ── Public per-turn attribute aliases ─────────────────────────────────────
    # The flat process() lifecycle reads/writes mp.uid / mp.cancel_event /
    # mp.active_tools; these properties bridge to the private backing fields
    # set up in __init__.

    @property
    def uid(self) -> 'int | None':
        """Public alias for _uid (the flat-path turn's transcript id)."""
        return self._uid

    @uid.setter
    def uid(self, value: 'int | None') -> None:
        self._uid = value

    @property
    def cancel_event(self) -> threading.Event:
        """Public alias for _cancel_event (the flat-path cooperative cancel flag)."""
        return self._cancel_event

    @cancel_event.setter
    def cancel_event(self, value: threading.Event) -> None:
        self._cancel_event = value

    @property
    def active_tools(self) -> list:
        """Public alias for _active_tools — the tool NAMES live this turn (seeded
        with config.always_available by _setup, appended by find_tools).
        build_tools resolves these names to schemas each ACT iteration."""
        return self._active_tools

    @active_tools.setter
    def active_tools(self, value: list) -> None:
        self._active_tools = value

    def _effective_channel(self) -> str:
        """Channel for this turn — read from the flat config (§4)."""
        return self.config.channel

    # ── Thinking-gate (CHANNEL='user' only) ──────────────────────────────────

    def _run_thinking_gate(self) -> None:
        """Regression-head deliberation scoring. Writes self.thinking_level.

        No-op for non-user channels (classifier is OOD for autonomous flows).
        Never raises. On failure → self.thinking_level = 'low', EMA untouched.

        Flat process() path — channel comes from config (§4).
        """
        if self.config.channel != 'user':
            return

        try:
            from services.deliberation_score_service import DeliberationScoreService
            from services.deliberation_ema_service import DeliberationEmaService

            scalar = DeliberationScoreService().classify(self._raw_input)
            ema_svc = DeliberationEmaService()

            if scalar is None:
                self.thinking_level = 'low'
                self._deliberation_scalar = None
                self._deliberation_ema = ema_svc.peek()
                logger.info(
                    "[DELIBERATION] turn=%s scalar=None ema=%s bucket=low fallback=true",
                    self._uid, self._deliberation_ema,
                )
                return

            ema, bucket = ema_svc.update_and_bucket(scalar)
            self.thinking_level = bucket
            self._deliberation_scalar = scalar
            self._deliberation_ema = ema
            logger.info(
                "[DELIBERATION] turn=%s scalar=%.4f ema=%.4f bucket=%s fallback=false",
                self._uid, scalar, ema, bucket,
            )

            if self._uid is not None:
                from services.database_service import get_shared_db_service
                try:
                    db = get_shared_db_service()
                    with db.connection() as conn:
                        conn.execute(
                            "UPDATE transcript SET deliberation_score = ? WHERE id = ?",
                            (scalar, self._uid),
                        )
                except Exception as exc:
                    logger.warning(
                        "[DELIBERATION] persist failed for uid=%s: %s",
                        self._uid, exc,
                    )

        except Exception:
            logger.exception("[DELIBERATION] gate failed; defaulting to 'low'")
            self.thinking_level = 'low'
            self._deliberation_scalar = None

    # ── Flat-MessageProcessor entry point (spec §4 / T2) ─────────────────────
    #
    # process() is the new single entry point for all channels.  It creates a
    # MessageProcessor with per-turn state from the caller-supplied
    # ProcessorConfig, runs the ACT lifecycle, and returns the response text.
    # Old subclasses continue to work via send() until they are migrated in T7-T8.

    @staticmethod
    def process(
        raw_input: str,
        config: "ProcessorConfig",  # noqa: F821 — deferred import avoids circular dep
        metadata: "dict | None" = None,
        cancel_event: "threading.Event | None" = None,
    ) -> str:
        """Single entry point.  Creates an MP, runs the turn, returns text.

        Spec §4 / AC-1 / L1.
        """
        mp = object.__new__(MessageProcessor)
        # Initialise old-path attributes (metrics, cancel, etc.) via old __init__.
        MessageProcessor.__init__(mp, raw_input, metadata)
        # New flat-path attributes (spec §4 field list).
        mp.config = config
        mp.uid: "int | None" = None
        mp.current_iteration: int = 0
        mp.cancel_event: "threading.Event" = (
            cancel_event if cancel_event is not None else threading.Event()
        )
        # Default is 'low' — the deliberation gate must explicitly set medium/high.
        # A 'medium' default would silently apply deliberation pressure to every turn
        # where the gate wasn't run (non-user channels) or crashed — regressing
        # benchmark behaviour on simple recall/chit-chat.
        mp.thinking_level: str = "low"
        return mp._run()

    def _run(self) -> str:
        """Lifecycle wrapper — run setup→loop→record.

        Spec §4.
        """
        self._setup()
        result = self._loop()
        self._record(result)
        return result

    def _setup(self) -> None:
        """Pre-loop.  Executes once per turn.

        1. Seed ACTIVE_TOOLS with this channel's always_available tier.
        2. Write input row to transcript (unless skip_transcript / skip_input_row).
        3. Run thinking gate (user channel only).
        4. Seed turn 0 — framework tool calls before the first LLM turn.

        Spec §4.
        """
        # ACTIVE_TOOLS is live from iteration 0; find_tools appends to it and
        # build_tools resolves it each turn. Empty for compaction / encoder
        # channels whose always_available is empty by design.
        self.active_tools = list(self.config.always_available or [])

        from services.transcript_service import write_input_row

        if not self.config.skip_transcript and not self.config.skip_input_row:
            self.uid = write_input_row(
                self.config.channel, self.config.role, self._raw_input
            )

        if self.config.channel == "user":
            self._run_thinking_gate()

        self._seed_turn_zero()

    def _seed_turn_zero(self) -> None:
        """Framework-issued tool calls fired once before iteration 0.

        Two declarative behaviours, zero hooks.  Each call goes through
        ToolDispatcher.dispatch() so it BLOCKS, records a tool_calls row, and is
        rendered into the trail exactly like an LLM-issued call.  The model's
        first turn already sees memory matches and uploaded documents.

        Spec §4 / §4d / AC-30 / AC-31.
        """
        from abilities._dispatcher import ToolDispatcher  # noqa: PLC0415

        dispatcher = ToolDispatcher(self)

        # a. Memory auto-seed — fire once when the declarative flag is set.
        #    _auto=True marks this as the background seed recall so memory's
        #    _handle_recall does NOT fan out to document.search + schedule.search
        #    (that delegation is reserved for explicit, model-invoked recalls).
        if self.config.memory_seed:
            dispatcher.dispatch("memory", {"action": "recall", "query": self._raw_input, "_auto": True})

        # c. High-deliberation thinking pass — programmatic, never model-visible.
        if getattr(self, "thinking_level", "low") == "high":
            dispatcher.dispatch("thinking", {})

        # b. Attachment uploads — presence-gated, one blocking document.upload
        #    per file.  No second auto document.view: upload IS the ingest.
        for path in (self._metadata.get("attachments") or []):
            try:
                name, content_b64, content_type = _read_attachment(path)
            except OSError as exc:
                logger.warning("[SEED] could not read attachment %s: %s", path, exc)
                continue
            dispatcher.dispatch("document", {
                "action": "upload",
                "name": name,
                "content": content_b64,
                "content_type": content_type,
            })

    def _loop(self) -> str:  # noqa: C901
        """ACT game loop — fit-and-send, compact when history had to be trimmed.

        Fitting is window-only (design §3.3 — trim-then-compact). Providers.send()
        builds the request with the full watermark-bounded history and drops the
        oldest rows until it reserves response headroom (``max(10% window, 8k)``),
        setting ``_compaction_pending`` when any row was dropped. On that flag the
        loop fires _dispatch_compaction() once, summarising the full history into
        the next turn's checkpoint so the dropped messages are not lost — the
        compaction row's own id becomes the new watermark (design: DB is the state
        machine), so the next send re-reads a collapsed history.

        A real provider rejection (PayloadTooLargeError — the token estimate
        under-counted) triggers a single collapse-and-retry; a second rejection
        re-raises so an irreducible request fails loudly instead of looping.
        """
        from abilities._dispatcher import ToolDispatcher  # noqa: PLC0415
        from services.llm_service import PayloadTooLargeError  # noqa: PLC0415
        while True:
            if self._should_stop():
                return ""
            try:
                response = self.providers.send()
            except PayloadTooLargeError:
                if self._payload_compacted:
                    raise                         # already collapsed once — fail loud
                self._payload_compacted = True
                self._dispatch_compaction()
                continue                          # re-read the compacted DB and retry
            if self._compaction_pending:          # a send had to trim history
                self._dispatch_compaction()       # fold the dropped rows into the checkpoint
            if not response.tool_calls:
                return response.text or ""
            dispatcher = ToolDispatcher(self)
            for tc in response.tool_calls:
                if self.cancel_event.is_set():
                    return ""
                dispatcher.dispatch(tc["name"], tc["input"])
            self._record_narration(response)
            self.current_iteration += 1

    def _should_stop(self) -> bool:
        """Single stop check: cancel OR iteration cap.

        Spec §4 / L3-L6.
        """
        if self.cancel_event.is_set():
            return True
        if self.config.max_iterations is not None:
            if self.current_iteration >= self.config.max_iterations:
                return True
        return False

    def _record(self, response_text: str) -> None:
        """Post-loop.  Persist turn + fan-out side-effects.

        Ephemeral trail rows are purged once here at turn end (spec §4c / F11).
        Durable rows (ephemeral=0) survive for audit / previous-messages replay.

        Spec §4 / M4-M5 / C3-C4.
        """
        from services.transcript_service import write_assistant_row

        if self.cancel_event.is_set():
            self._cleanup_cancelled()
            return

        # Purge ephemeral trail rows once at turn end (§4c / F11).
        self._purge_ephemeral_tool_calls()

        if not self.config.skip_transcript:
            write_assistant_row(self.config.channel, response_text)

        # After-turn hooks: mutually independent, failure-isolated (§4.8).  One
        # hook raising is a non-event for the others — the order is undefined and
        # may become concurrent, so each call is isolated (log + continue).
        for hook in self.config.post_turn_hooks:
            try:
                hook.run(self, response_text)
            except Exception as exc:  # noqa: BLE001 — failure isolation contract
                logger.warning(
                    "[post_turn] hook %s failed (isolated): %s",
                    type(hook).__name__,
                    exc,
                )

    def _cleanup_cancelled(self) -> None:
        """Delete DB rows created during a cancelled turn.

        Spec §4 / M5.
        """
        if self.uid is None:
            return
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                conn.execute(
                    "DELETE FROM tool_calls WHERE transcript_id = ?", (self.uid,)
                )
                conn.execute(
                    "DELETE FROM transcript WHERE id = ?", (self.uid,)
                )
            logger.info(
                "[MessageProcessor] %s: cleaned up cancelled turn (uid=%s)",
                self.config.channel,
                self.uid,
            )
        except Exception as exc:
            logger.warning(
                "[MessageProcessor] %s: failed to clean up cancelled turn (uid=%s): %s",
                self.config.channel,
                self.uid,
                exc,
            )

    def _previous_rows(self) -> list:
        """Watermark-bounded transcript rows for this channel (design §3.7).

        SELECT * FROM transcript WHERE channel=? AND id > <watermark> ORDER BY id ASC
        No LIMIT (fixes the 20-row bug). Empty for suppress_history channels and
        post-compaction turns (the compaction row's own id IS the watermark)."""
        if self.config.suppress_history:
            return []
        from services import compaction_persistence, transcript_service  # noqa: PLC0415
        compaction = compaction_persistence.get_compaction(self.config.channel)
        watermark = compaction["compacted_up_to_id"] if compaction else 0
        return transcript_service.get_recent(self.config.channel, since_id=watermark)

    def get_previous_messages(self) -> str:
        """Render the ## Previous Messages block from _previous_rows().

        Renders every watermark-bounded row, minus the oldest ``_history_drop``
        rows the fit loop dropped to reserve response headroom (design §3.3).
        There is NO fixed row cap: history is bounded by context-window size,
        not a turn count. _previous_rows() returns id-ASC, so dropping the first
        ``_history_drop`` entries removes the OLDEST messages."""
        if self.config.suppress_history:
            return ""
        from services.tool_call_service import ToolCallService  # noqa: PLC0415
        entries = self._previous_rows()
        if self._history_drop:
            entries = entries[self._history_drop:]   # drop oldest (fit-to-window)
        if not entries:
            return ""
        all_ids = [e["id"] for e in entries if e.get("id")]
        durable_by_id = ToolCallService().get_by_transcript_ids(all_ids, include_ephemeral=False) if all_ids else {}
        lines: list[str] = []
        for entry in entries:
            ts = _format_ts(entry.get("created_at"), row_kind="transcript", row_id=entry.get("id"))
            raw_role = entry.get("role") or "unknown"
            role_label = "Assistant" if raw_role == "assistant" else raw_role
            content = (entry.get("content") or "").replace("\n", " ").strip()
            lines.append(f"[{ts}] {role_label}: {content}")
            for tc in durable_by_id.get(entry.get("id"), []):
                tc_name = tc.get("tool_name") or tc.get("name") or "tool"
                if tc_name in _NEVER_RENDER_IN_PREVIOUS:
                    continue
                tc_params = _parse_tc_params(tc.get("params"))
                tc_result = tc.get("result") or ""
                lines.append(_render_tool_call_for_previous(tc_name, tc_params, tc_result))
        return "\n".join(lines)

    def drop_oldest_previous_message(self) -> bool:
        """Drop the oldest rendered history row; return False when none remain.

        The window-fit loop (Providers._fit_request) calls this to shrink the
        request until it reserves response headroom. Monotonic and bounded by the
        watermark-bounded row count, so the fit loop always terminates (design
        §3.3)."""
        if self._history_drop < len(self._previous_rows()):
            self._history_drop += 1
            return True
        return False

    # ── Trail API (T4: act-trail-as-a-query) ─────────────────────────────────

    def _purge_ephemeral_tool_calls(self) -> None:
        """Delete all ephemeral=1 tool_calls rows for the current turn's uid.

        Called once at turn end (_record) and on cancel (_cleanup_cancelled).
        Durable rows (ephemeral=0, e.g. 'thinking', 'compaction') survive.
        No-op when uid is None.

        Spec §4c / F11.
        """
        if self.uid is None:
            return
        try:
            from services.database_service import get_shared_db_service  # noqa: PLC0415
            db = get_shared_db_service()
            with db.connection() as conn:
                conn.execute(
                    "DELETE FROM tool_calls WHERE transcript_id = ? AND ephemeral = 1",
                    (self.uid,),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[MessageProcessor] %s: failed to purge ephemeral tool_calls (uid=%s): %s",
                self.config.channel, self.uid, exc,
            )

    def _has_trail(self) -> bool:
        """True when real (non-compactor) tool activity exists since the last
        trail boundary.

        Slices from the last tool_chain_compactor boundary and returns True only
        when at least one row in that slice is NOT a framework compactor marker.
        Used by ToolChainCompactor to silently no-op when there is nothing to
        compact.
        """
        if self.uid is None:
            return False
        from services.act_trail import ActTrail  # noqa: PLC0415
        rows = _from_last_compaction(ActTrail().fetch_by_transcript_id(self.uid))
        return any(r["tool_name"] not in _COMPACTOR_TOOLS for r in rows)

    def _render_act_trail(self) -> str:
        """Assemble the ACT trail string for the current turn.

        Fetches all tool_calls rows for self.uid ordered by id and slices from
        the last tool_chain_compactor boundary (inclusive); the boundary row's
        handover renders in place of the pre-compacted calls. Framework markers
        are filtered: chat_history_compactor rows never reach the model (their
        compacted history arrives via the checkpoint prepend), and an empty
        tool_chain_compactor no-op row is dropped. Returns '' when uid is None.
        """
        if self.uid is None:
            return ""
        from services.act_trail import ActTrail  # noqa: PLC0415
        trail = ActTrail()
        rows = _from_last_compaction(trail.fetch_by_transcript_id(self.uid))
        out: list[str] = []
        for r in rows:
            name = r.get("tool_name")
            if name == "chat_history_compactor":
                continue
            if name == _TRAIL_BOUNDARY_TOOL and not (r.get("result") or "").strip():
                continue
            out.append(trail.render(r))
        return "\n".join(out)

    def _dispatch_compaction(self) -> None:
        """Fire both compactors through the normal tool-dispatch chokepoint.

        No alternative path: this is the SAME machinery as the turn-0
        memory/thinking seeds. Each compactor goes through ToolDispatcher.dispatch
        so it auto-records its tool_calls row AND auto-emits the act-trail WS
        events (act_tool_start/act_tool_end) — that is how compaction shows up in
        the frontend act-trail without any hand-rolled emit.

        - ToolChainCompactor reads the act-trail off this mp and compacts it
          (silent no-op when empty); its recorded row is the new trail boundary.
        - ChatHistoryCompactor reads get_previous_messages() off this mp and
          advances the durable transcript watermark.

        Fired tool_chain first so its handover summarises the real trail before
        the chat-history marker lands. Both are INTERNAL (policy gate bypassed)
        and never-discoverable.

        ``_history_drop`` is reset to 0 first so ChatHistoryCompactor reads the
        FULL watermark-bounded history — the rows the fit loop dropped from the
        live request must still be summarised into the checkpoint (design §3.3).
        """
        from abilities._dispatcher import ToolDispatcher  # noqa: PLC0415

        self._history_drop = 0   # summarise the full (untrimmed) history
        dispatcher = ToolDispatcher(self)
        # compaction tool dispatch
        dispatcher.dispatch("tool_chain_compactor", {"act_summary": "Compacting tool history"})
        dispatcher.dispatch("chat_history_compactor", {"act_summary": "Compacting conversation"})

    def _record_narration(self, response: "object") -> None:  # type: ignore[override]
        """Mid-loop: persist LLM text between iterations as an ephemeral trail row.

        Records a tool_calls row with tool_name='narration', ephemeral=True.
        Emits an act_narration WS event gated on config.broadcast_to.
        No-op when response.text is falsy.

        Spec §4 / F14 / F15 / N3.
        """
        text = getattr(response, "text", None)
        if not text:
            return
        from abilities._event_emitter import ActEventEmitter  # noqa: PLC0415
        from services.act_trail import ActTrail  # noqa: PLC0415
        ActTrail().record(
            tool_name="narration",
            params={},
            result=text,
            transcript_id=self.uid,
            ephemeral=True,
        )
        # The emitter owns the broadcast_to gate — background loops
        # (broadcast_to=None) never emit (N1/N5).
        ActEventEmitter(self.config).emit({
            "type": "act_narration",
            "text": _sanitize_llm_args(text),
            "step": self.current_iteration,
        })


# ── Module-private helpers ────────────────────────────────────────────────────


#: Placeholder rendered when a row has a missing / empty / unparseable
#: ``created_at`` value. Must be exactly 16 characters so the
#: ``[YYYY-MM-DD HH:MM]`` column width in Previous Messages stays stable.
_MISSING_TS_PLACEHOLDER = '????-??-?? ??:??'


def _wrap_with_checkpoint(channel: str, user_body: str) -> str:
    """Wrap the user-message body with a ### Checkpoint envelope.

    When a compaction row exists for ``channel``, prepends the compacted
    summary under a ``### Checkpoint`` header and places the bare body
    under a ``### Current State`` header. Returns the bare body unchanged
    when there is no checkpoint or the stored compacted_text is empty.

    Called by ``send()`` on every ACT iteration, immediately after
    ``getUserPrompt()`` returns.
    """
    from services import compaction_persistence

    row = compaction_persistence.get_compaction(channel)
    if not row:
        return user_body
    compacted = (row.get('compacted_text') or '').strip()
    if not compacted:
        return user_body
    return (
        "### Checkpoint - What you were previously discussing / doing\n"
        f"{compacted}\n"
        "\n"
        "---\n"
        "### Current State - What's happening in the current turn\n"
        f"{user_body}"
    )


def _from_last_compaction(rows: "list[dict]") -> "list[dict]":
    """Return the tail of *rows* starting at the LAST non-empty trail boundary
    (a tool_chain_compactor row whose result holds a handover), inclusive.

    When no such boundary exists, return all rows. An empty-result
    tool_chain_compactor row (the no-trail no-op) is NOT a boundary — it carries
    no handover, so slicing at it would drop real calls behind a blank marker.
    The chat_history_compactor marker is never a boundary either; only the
    act-trail compactor bounds the trail.
    """
    last: "int | None" = None
    for i, r in enumerate(rows):
        if r.get("tool_name") == _TRAIL_BOUNDARY_TOOL and (r.get("result") or "").strip():
            last = i
    return rows if last is None else rows[last:]


#: Durable tool_call names that must never surface in the ## Previous Messages
#: block.  The history ``compaction`` row is stored ``ephemeral=0`` for audit,
#: but its content is already replayed through the checkpoint prepend at the top
#: of the block — rendering it again would double-inject the summary on every
#: subsequent turn (Decision 4B).
_NEVER_RENDER_IN_PREVIOUS: frozenset[str] = frozenset({'compaction'})


def _render_tool_call_for_previous(tool_name: str, params: dict, result: str) -> str:
    """Render one durable tool_call row for the ## Previous Messages block.

    Bare format (no timestamp prefix, no ``TOOL()`` wrapper) — the row inherits
    its owning transcript row's timestamp implicitly by positional placement.

    Format: '[tool_name(k="v",…)] result'
    """
    parts = []
    for k, v in params.items():
        if isinstance(v, str):
            parts.append(f'{k}="{v}"')
        else:
            parts.append(f'{k}={v}')
    param_str = ','.join(parts)
    return f'[{tool_name}({param_str})] {result}'


def _parse_tc_params(raw: object) -> dict:
    """Parse the ``tool_calls.params`` column into a dict for rendering.

    The DB stores params as a JSON-encoded string. Callers may also pass a
    pre-parsed dict (tests mocking the service). This helper normalises both
    paths and returns ``{}`` on any parse failure — the rendered line becomes
    ``[tool_name()] result`` which is still valid per the north star format.
    """
    if raw is None or raw == '':
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        import json
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _format_ts(
    raw: str | None,
    *,
    row_kind: str = 'row',
    row_id: int | None = None,
) -> str:
    """Format a raw SQLite/ISO timestamp into ``YYYY-MM-DD HH:MM`` in the user's
    local timezone.

    Storage is UTC (``utc_now().isoformat()``); the LLM only ever sees local
    wall-clock time. Conversion runs through
    :meth:`TimeFormatterService.local`, which handles tz lookup.

    If ``raw`` is ``None``, empty, or unparseable, return
    ``_MISSING_TS_PLACEHOLDER`` and emit a single warning log so the problem
    is visible in production without spamming. Rationale: ``parse_utc`` falls
    back to ``datetime.min`` (``0001-01-01 00:00``) on bad input, which would
    otherwise silently corrupt Previous Messages with a bogus "year 1" prefix
    the LLM would treat as real context.

    ``row_kind`` + ``row_id`` are logged but not shown to the LLM — the
    placeholder is the only thing the prompt sees.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        logger.warning(
            "[MessageProcessor._format_ts] missing created_at on %s id=%s — "
            "rendering placeholder", row_kind, row_id,
        )
        return _MISSING_TS_PLACEHOLDER

    formatted = TimeFormatterService.local(raw)
    if formatted is None:
        logger.warning(
            "[MessageProcessor._format_ts] unparseable created_at=%r on %s "
            "id=%s — rendering placeholder", raw, row_kind, row_id,
        )
        return _MISSING_TS_PLACEHOLDER

    return formatted
