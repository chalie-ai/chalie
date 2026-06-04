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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import TYPE_CHECKING

from services.metrics_accumulator import MetricsAccumulator
from services.time_formatter_service import TimeFormatterService

if TYPE_CHECKING:
    from services.processor_config import ProcessorConfig

logger = logging.getLogger(__name__)

# ── Compaction rendering helpers ───────────────────────────────────────────────
#
# These parse/format helpers live alongside the other transcript-rendering
# infrastructure in this module (TimeFormatterService, _MISSING_TS_PLACEHOLDER).
# They are called by _run_full_compaction (below), which owns the two-tier
# compaction orchestration triggered when the live token measurement crosses the
# 0.80 / 0.90 thresholds. _SUMMARY_RE parses the <summary>…</summary> block
# produced by the compaction turn (a flat process() run on the compaction
# channel — see configs.channels.CompactionConfig).

_SUMMARY_RE = re.compile(r"<summary>([\s\S]*?)</summary>", re.IGNORECASE)
_COMPACTION_FAILURE_FMT = "[COMPACTION] %s: continuity failure — reason=%s"

# Maximum bytes fed to _SUMMARY_RE; bounds backtracking on malformed LLM output.
_SUMMARY_RE_CAP = 65_536


def _extract_compaction_summary(raw: 'str | None') -> 'str | None':
    """Extract the body of a <summary>…</summary> block from raw LLM output.

    Returns the stripped inner text on success.
    Returns None when:
    - raw is empty or None.
    - no <summary> tags are present in the output.
    """
    if not raw:
        return None
    m = _SUMMARY_RE.search(raw[:_SUMMARY_RE_CAP])
    return m.group(1).strip() if m else None


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
    # shared by every flat-path turn.  THINKING_TIMEOUT bounds the optional
    # high-mode exploration pass.

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
    THINKING_TIMEOUT: int = 600  # seconds — exploration pass budget (independent of ACT)

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
        # Default is 'low' — classifier must explicitly set medium/high.
        # A 'medium' default would silently apply deliberation pressure to every
        # turn where the gate wasn't run (non-user channels) or crashed —
        # regressing benchmark behaviour on simple recall/chit-chat.
        self._thinking_level: str = 'low'
        self._deliberation_scalar: float | None = None   # raw sigmoid for this turn
        self._deliberation_ema: float | None = None      # EMA after this turn's update
        self._thinking_exploration: str | None = None
        # One-shot guard: any overflow recovery (proactive threshold trip
        # OR 413 from the provider) triggers a Stage 2 ACT restart, but only
        # once per turn. The proactive threshold path can mis-fire when the
        # static system_prompt + tools schema alone exceed compact_at — in
        # that case compaction shrinks user_body but the threshold still
        # trips on restart, and without this guard the loop spins forever.
        # After one recovery: send anyway and let the transport 413 path
        # decide whether the compacted body is genuinely too large.
        self._overflow_recovered_this_turn: bool = False
        # Accumulator starts immediately so exploration + compaction tokens count.
        self._metrics: MetricsAccumulator = MetricsAccumulator()
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

    def _run_full_compaction(self, exclude_id: 'int | None' = None) -> 'str | None':
        """Run a full continuity compaction for this channel via the flat path.

        Reads entries since the watermark, builds the continuity-first envelope,
        runs the compaction LLM through ``MessageProcessor.process()`` with
        ``CompactionConfig`` (the single flat loop — no compaction subclass),
        parses ``<summary>``, and writes the append-only ``tool_calls`` audit
        row that the canonical ``compaction_persistence.get_compaction()``
        lookup reads.

        Args:
            exclude_id: When set, filters this transcript ID from the rendered
                entries list — used by the history-compaction path to exclude
                the current turn's input row so the LLM does not see a partial /
                unanswered user message.

        Returns:
            The extracted ``<summary>`` body on success, None on failure
            (unchanged contract — callers and tests rely on it).
        """
        from configs.channels import CompactionConfig
        from services import compaction_persistence

        channel = self._effective_channel()
        transcript_id = self._uid

        prior = compaction_persistence.get_compaction(channel)
        watermark = prior['compacted_up_to_id'] if prior else 0
        prev_text = (prior.get('compacted_text') or '').strip() if prior else ''

        all_entries = list(compaction_persistence.get_entries_since(channel, watermark))
        if exclude_id is not None:
            entries = [e for e in all_entries if e.get('id') != exclude_id]
        else:
            entries = all_entries

        # Nothing to compact — bail before hitting the LLM.
        if not entries and not prev_text:
            logger.warning(
                "[COMPACTION] %s: compaction invoked with no entries and no prior "
                "checkpoint — skipping LLM call",
                channel,
            )
            return None

        rendered = [_format_compaction_entry(e) for e in entries]
        compaction_input = _build_compaction_input(prev_text, rendered)
        in_chars = len(compaction_input)

        try:
            raw_output = (
                MessageProcessor.process(compaction_input, CompactionConfig()) or ''
            ).strip()
        except Exception as exc:
            reason = f"LLM error: {exc}"
            logger.error(_COMPACTION_FAILURE_FMT, channel, reason)
            _write_compaction_audit_row(
                transcript_id, watermark=watermark, status='failure',
                summary='', reason=reason,
            )
            return None

        if not raw_output:
            reason = "LLM returned empty output"
            logger.warning(_COMPACTION_FAILURE_FMT, channel, reason)
            _write_compaction_audit_row(
                transcript_id, watermark=watermark, status='failure',
                summary='', reason=reason,
            )
            return None

        summary = _extract_compaction_summary(raw_output)
        if not summary:
            reason = "no <summary> tags in LLM output"
            logger.warning(_COMPACTION_FAILURE_FMT, channel, reason)
            _write_compaction_audit_row(
                transcript_id, watermark=watermark, status='failure',
                summary='', reason=reason,
            )
            return None

        new_watermark = max((e.get('id', 0) for e in entries), default=watermark)
        _write_compaction_audit_row(
            transcript_id, watermark=new_watermark, status='success', summary=summary,
        )
        logger.info(
            "[COMPACTION] %s: continuity success — in=%d chars, out=%d chars, "
            "watermark %d→%d",
            channel, in_chars, len(summary), watermark, new_watermark,
        )
        return summary

    def _effective_channel(self) -> str:
        """Channel for this turn — read from the flat config (§4)."""
        return self.config.channel

    # ── Thinking-gate (CHANNEL='user' only) ──────────────────────────────────

    def _run_thinking_gate(self) -> None:
        """Regression-head deliberation scoring. Writes self._thinking_level.

        No-op for non-user channels (classifier is OOD for autonomous flows).
        Never raises. On failure → self._thinking_level = 'low', EMA untouched.

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
                self._thinking_level = 'low'
                self._deliberation_scalar = None
                self._deliberation_ema = ema_svc.peek()
                logger.info(
                    "[DELIBERATION] turn=%s scalar=None ema=%s bucket=low fallback=true",
                    self._uid, self._deliberation_ema,
                )
                self._thinking_exploration = None
                return

            ema, bucket = ema_svc.update_and_bucket(scalar)
            self._thinking_level = bucket
            self._deliberation_scalar = scalar
            self._deliberation_ema = ema
            logger.info(
                "[DELIBERATION] turn=%s scalar=%.4f ema=%.4f bucket=%s fallback=false",
                self._uid, scalar, ema, bucket,
            )

            if self._thinking_level == 'high':
                try:
                    with ThreadPoolExecutor(max_workers=1) as _pool:
                        _future = _pool.submit(self._run_thinking_exploration)
                        try:
                            self._thinking_exploration = _future.result(
                                timeout=self.THINKING_TIMEOUT
                            )
                        except FuturesTimeoutError:
                            logger.warning(
                                "[THINKING] exploration exceeded THINKING_TIMEOUT=%ds"
                                " — proceeding without exploration",
                                self.THINKING_TIMEOUT,
                            )
                            self._thinking_exploration = None
                except Exception as exc:
                    logger.info(
                        "[THINKING] exploration failed (%s) — high turn proceeds "
                        "without exploration", exc,
                    )
                    self._thinking_exploration = None
                if self._thinking_exploration is not None:
                    self._persist_exploration_to_tool_calls(self._uid)
            else:
                self._thinking_exploration = None

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
            self._thinking_level = 'low'
            self._deliberation_scalar = None
            self._thinking_exploration = None

    def _run_thinking_exploration(self) -> 'str | None':
        """One same-job exploration pass for high-mode turns.

        Asks the model to think out loud about the user's request: assess
        gaps in its knowledge, evaluate which tools would help, and flag
        non-obvious aspects. Output is Chain-of-Thought that gets
        re-injected into the ACT loop via the config's get_user_prompt
        (which reads ``thinking_exploration``) so the model can act on its
        own reasoning.

        Tools schema is sent so the model can reason about available
        capabilities, but the prompt instructs it not to invoke them.
        Any tool_calls in the response are discarded (single-pass only).

        The model may output 'NOTHING' if the request is straightforward,
        in which case None is returned and no exploration is injected.

        Returns None on any failure (network, provider rejection, etc).
        Logged at INFO. NEVER raises.
        """
        from services.providers import Providers

        _EXPLORATION_PREFIX = (
            "Think out loud about the user's request before responding.\n\n"
            "Consider:\n"
            "- What does the ideal response look like? What would make it genuinely useful?\n"
            "- Do you already know enough to answer well, or are there gaps?\n"
            "- Would any of your available tools fill those gaps? Which ones, in what order?\n"
            "- Is there anything non-obvious about this request you might miss on a first read?\n\n"
            "Whatever you output here will be shown to you as Chain of Thought on the next "
            "pass — write to your future self. Be specific: name the tools you plan to use, "
            "flag uncertainties, note key facts you want to remember to include.\n\n"
            "If the request is straightforward and you have nothing useful to say to yourself, "
            "output exactly: NOTHING\n\n"
            "DO NOT INVOKE TOOLS — they are disabled in this phase. Think only."
            "\n\n---\n\n"
        )

        try:
            from abilities._registry import AbilityRegistry  # noqa: PLC0415

            # Flat process() path — config carries all per-turn surfaces (§4).
            config = self.config
            user_body = config.get_user_prompt(self)
            user_body = _wrap_with_checkpoint(config.channel, user_body)
            system_prompt = config.get_system_prompt(self)
            tools = AbilityRegistry.build_tools(self)

            response = Providers.instance().send_messages(
                system_prompt,
                [{'role': 'user', 'content': _EXPLORATION_PREFIX + user_body}],
                job=config.job,
                tools=tools,
                thinking_mode='high',
                mp=self,
            )
            # Token accumulation happens at the send gateway (§4e), not here.

            if response.tool_calls:
                logger.debug(
                    "[THINKING] exploration model attempted %d tool call(s) — discarded",
                    len(response.tool_calls),
                )

            text = (response.text or '').strip()
            if text.upper() == 'NOTHING':
                return None
            return text if text else None

        except Exception as exc:
            logger.info("[THINKING] exploration failed (%s)", exc)
            return None

    def _persist_exploration_to_tool_calls(self, transcript_id: 'int | None') -> None:
        """Insert the exploration text as a durable tool_calls row.

        Stored with tool_name='thinking', ephemeral=0 so it survives
        compaction and surfaces as part of the durable audit trail.
        Persistence failure logs INFO and does NOT abort the turn.
        """
        if transcript_id is None or self._thinking_exploration is None:
            return
        from services.act_trail import ActTrail  # noqa: PLC0415
        try:
            ActTrail().record(
                tool_name='thinking',
                params={},
                result=self._thinking_exploration,
                transcript_id=transcript_id,
                ephemeral=False,
            )
        except Exception as exc:
            logger.info(
                "[THINKING] failed to persist exploration to tool_calls (%s)", exc
            )

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
        mp.thinking_level: str = "low"
        mp.thinking_exploration: "str | None" = None
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
            # Sync the exploration result onto the flat-path attribute so that
            # config.get_user_prompt() (e.g. UserConfig.get_user_prompt) can read
            # it via getattr(mp, 'thinking_exploration', None).
            self.thinking_exploration = self._thinking_exploration

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

    #: Channels that are themselves compaction loops.  Inside these channels the
    #: compaction thresholds must NOT fire (D14 — recursion guard).
    _COMPACTION_CHANNELS: frozenset[str] = frozenset({"compaction"})

    def _loop(self) -> str:  # noqa: C901
        """ACT game loop — spec §4 / AC-1.  ≤30 lines."""
        from abilities._dispatcher import ToolDispatcher  # noqa: PLC0415
        from abilities._registry import AbilityRegistry  # noqa: PLC0415
        from services.providers import Providers  # noqa: PLC0415
        p = Providers.instance()
        in_compaction = self.config.channel in self._COMPACTION_CHANNELS
        while True:
            if self._should_stop(): return ""  # noqa: E701
            prompt = self.config.get_user_prompt(self)
            prompt = _wrap_with_checkpoint(self.config.channel, prompt)
            system = self.config.get_system_prompt(self)
            tools = AbilityRegistry.build_tools(self)
            pct = p.calculate(system, prompt, tools, job=self.config.job, mp=self)
            if pct > 0.80:
                if in_compaction:
                    # D14: recursion guard — never compact-of-compaction.
                    logger.warning(
                        "[COMPACTION] recursion guard: %s payload at %.0f%% — "
                        "proceeding without compaction",
                        self.config.channel, pct * 100,
                    )
                elif pct > 0.90 and self._has_trail():
                    if not self._compact_trail(): return ""  # noqa: E701
                    continue
                else:
                    if not self._compact_history(): return ""  # noqa: E701
                    continue
            response = p.send_messages(
                system, [{"role": "user", "content": prompt}],
                job=self.config.job, tools=tools, thinking_mode=self.thinking_level,
                mp=self,
            )
            if not response.tool_calls: return response.text or ""  # noqa: E701
            dispatcher = ToolDispatcher(self)
            for tc in response.tool_calls:
                if self.cancel_event.is_set(): return ""  # noqa: E701
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

    def get_previous_messages(self) -> str:
        """Assemble the ## Previous Messages block for this channel.

        The read counterpart of ``_compact_history()`` (§4b). Channel-scoped,
        watermark-bounded, and short-circuited for housekeeping loops.

        suppress_history → '' (M6 — housekeeping loops never replay).
        Otherwise: find the channel's latest history-compaction row (the
        watermark = its ``compacted_up_to_id``), read transcript rows with
        ``id > watermark`` for this channel, prepend the compaction summary,
        and render each row.

        Literal format (locked by the north star):
        - input rows  : ``[YYYY-MM-DD HH:MM] <role>: <content>`` — role is
                        rendered lowercase, except ``assistant`` → ``Assistant``.
        - durable     : ``[<tool_name>(<k>="<v>";…)] <result>`` — bare, no
          tool_calls    timestamp, interleaved under their owning transcript row
                        ordered by created_at. Ephemeral (``ephemeral=1``) rows
                        are never emitted; the ``compaction`` / ``thinking``
                        audit pseudo-tools are filtered (Decision 4B).

        Returns '' when the channel has no rows and no compaction summary.

        Spec §4 / §4a / AC-26 / AC-27 / M6-M8.
        """
        if self.config.suppress_history:
            return ""

        from services import compaction_persistence, transcript_service
        from services.tool_call_service import ToolCallService

        compaction = compaction_persistence.get_compaction(self.config.channel)
        watermark = compaction["compacted_up_to_id"] if compaction else 0

        entries = transcript_service.get_recent(
            self.config.channel, since_id=watermark
        )

        if not entries and not (compaction and compaction.get("compacted_text")):
            return ""

        # Batch-load durable (ephemeral=0) tool_calls for all transcript rows.
        all_ids = [e["id"] for e in entries if e.get("id")]
        durable_by_id: dict[int, list] = {}
        if all_ids:
            durable_by_id = ToolCallService().get_by_transcript_ids(
                all_ids, include_ephemeral=False
            )

        lines: list[str] = []

        if compaction and compaction.get("compacted_text"):
            lines.append(compaction["compacted_text"])

        for entry in entries:
            ts = _format_ts(
                entry.get("created_at"),
                row_kind="transcript",
                row_id=entry.get("id"),
            )
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
                lines.append(
                    _render_tool_call_for_previous(tc_name, tc_params, tc_result)
                )

        return "\n".join(lines)

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
        """True when a non-compaction trail row exists since last compaction.

        Queries tool_calls via ActTrail.fetch_by_transcript_id and slices from
        the last trail_compaction row.  Returns True only when at least one
        non-trail_compaction row exists in that slice.

        Spec §4c / F9.
        """
        if self.uid is None:
            return False
        from services.act_trail import ActTrail  # noqa: PLC0415
        rows = _from_last_compaction(ActTrail().fetch_by_transcript_id(self.uid))
        return any(r["tool_name"] != "trail_compaction" for r in rows)

    def _render_act_trail(self) -> str:
        """Assemble the ACT trail string for the current turn.

        Fetches all tool_calls rows for self.uid ordered by id, slices from the
        last trail_compaction row (inclusive), and renders each via ActTrail.render().
        Returns '' when uid is None or no rows exist.

        Spec §4c / _render_act_trail.
        """
        if self.uid is None:
            return ""
        from services.act_trail import ActTrail  # noqa: PLC0415
        trail = ActTrail()
        rows = _from_last_compaction(trail.fetch_by_transcript_id(self.uid))
        return "\n".join(trail.render(r) for r in rows)

    def _compact_trail(self) -> bool:
        """Trail compaction (>90%): summarise the trail-so-far into ONE
        'trail_compaction' tool_calls row.

        Compaction is itself a tool call. We assemble the current trail
        (everything since the last 'trail_compaction' row — §4c), summarise
        it via MessageProcessor.process() with CompactionConfig, and record
        the summary as a new 'trail_compaction' row. Because trail assembly
        always starts at the LATEST 'trail_compaction' row, that one row
        instantly becomes the new head of the trail and every prior row drops
        out of view. No watermark field, no list surgery, no mid-turn deletes.

        Returns True on success (incl. empty-trail no-op), False on failure
        (caller should abort the turn by returning '').

        Spec §4a / D2 / D8 / D9 / D10 / AC-23.
        """
        from configs.channels import CompactionConfig  # noqa: PLC0415
        from services.act_trail import ActTrail  # noqa: PLC0415

        trail_text = self._render_act_trail()
        if not trail_text.strip():
            return True  # D9: nothing to compact — no-op success

        compacted = MessageProcessor.process(
            f"## Tool Results\n{trail_text}", CompactionConfig()
        )
        if not compacted.strip():
            logger.warning(
                "[COMPACTION] %s: trail compaction returned empty summary — aborting turn",
                self.config.channel,
            )
            return False  # D10: empty summary → abort turn

        # Record the summary AS a trail tool call — it becomes the head of the
        # trail on the next assembly (distinct tool_name from history 'compaction').
        ActTrail().record(
            tool_name="trail_compaction",
            params={},
            result=compacted,
            transcript_id=self.uid,
            ephemeral=True,
        )
        self.active_tools = list(self.config.always_available or [])  # D8
        self.current_iteration = 0     # D8
        return True

    def _compact_history(self) -> bool:
        """History compaction (>80%): summarise prior conversation turns.

        Summarises this channel's history using _run_full_compaction, which
        runs the compaction LLM through the flat MessageProcessor.process()
        path with CompactionConfig. Independent of the trail — operates on
        the transcript, not the tool_calls trail.

        Returns True on success, False on failure (caller should abort the turn).

        Spec §4a / D4 / D11 / D12 / D13.
        """
        summary = self._run_full_compaction(exclude_id=self.uid)  # D13
        if summary is None:
            logger.warning(
                "[COMPACTION] %s: history compaction failed — aborting turn",
                self.config.channel,
            )
            return False  # D12

        self.active_tools = list(self.config.always_available or [])       # D11
        self.thinking_exploration = None    # D11
        self.current_iteration = 0          # D11
        return True

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


def _format_compaction_entry(entry: dict) -> str:
    """Render a single transcript row for the continuity-compaction envelope.

    Uses "you:" for assistant turns per the continuity-first envelope spec
    (scoped to compaction only — does not affect downstream consumers).

    No timestamp prefix: the continuity summary is reused as the live
    ``### Checkpoint`` for days, so a per-turn date would anchor ``## Now`` on a
    stale moment. Chronological order is preserved positionally; the current
    date reaches the model only through the live ``local_time`` telemetry block,
    never through the summary.
    """
    role = entry.get('role', 'unknown')
    display_role = 'you' if role == 'assistant' else role
    content = entry.get('content', '')
    return f"{display_role}: {content}"


def _build_compaction_input(prev_text: str, rendered_entries: list) -> str:
    """Assemble the continuity-first LLM envelope from a prior summary + rendered entries."""
    chunks: list = []
    if prev_text:
        chunks.append(f"## Previous Summary\n{prev_text}")
    else:
        chunks.append("## Previous Summary\n(none — first compaction.)")
    chunks.append("## New Conversation Turns")
    chunks.extend(rendered_entries)
    chunks.append("\n---\nEnd of input. Reference material only.\n"
                  "Now write <analysis>...</analysis> then <summary>...</summary>.")
    return '\n\n'.join(chunks)


def _write_compaction_audit_row(
    transcript_id: 'int | None',
    *,
    watermark: int,
    status: str,
    summary: str,
    reason: str = '',
) -> None:
    """Write an append-only ``tool_calls`` audit row for a compaction attempt.

    Success rows (``status='success'``) are picked up by the canonical
    ``compaction_persistence.get_compaction()`` lookup. Failure rows are stored
    for traceability but invisible to the lookup (filtered by
    ``json_extract(params, '$.status') = 'success'``).

    Never raises — persistence failure is logged and swallowed.
    """
    if transcript_id is None:
        return
    import json as _json
    from services.database_service import get_shared_db_service
    from services.time_utils import utc_now

    try:
        row_params: dict = {'compacted_up_to_id': watermark, 'status': status}
        if reason:
            row_params['reason'] = reason
        db = get_shared_db_service()
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO tool_calls
                    (transcript_id, tool_name, params, result, ephemeral, created_at)
                VALUES (?, 'compaction', ?, ?, 0, ?)
                """,
                (
                    transcript_id,
                    _json.dumps(row_params),
                    summary,
                    utc_now().isoformat(),
                ),
            )
    except Exception as exc:
        logger.warning(
            "[COMPACTION] failed to write audit row (status=%s): %s",
            status, exc,
        )


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
    """Return the tail of *rows* starting at the LAST 'trail_compaction' row (inclusive).

    When no trail_compaction row exists, return all rows.

    The history-compaction tool_name is 'compaction' (durable, channel-scoped).
    Only 'trail_compaction' is a trail boundary.  'compaction' rows are
    NOT boundaries and are included as-is in whatever slice they fall in.

    Spec §4c / _from_last_compaction / F5 / F6 / F7.
    """
    last: "int | None" = None
    for i, r in enumerate(rows):
        if r.get("tool_name") == "trail_compaction":
            last = i
    return rows if last is None else rows[last:]


#: Durable tool_call names that must never surface in the ## Previous Messages
#: block.  The history ``compaction`` row is stored ``ephemeral=0`` for audit,
#: but its content is already replayed through the checkpoint prepend at the top
#: of the block — rendering it again would double-inject the summary on every
#: subsequent turn (Decision 4B).
_NEVER_RENDER_IN_PREVIOUS: frozenset[str] = frozenset({'compaction', 'thinking'})


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
