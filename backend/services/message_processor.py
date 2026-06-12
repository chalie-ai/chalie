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

import json
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

#: tool_name under which _record_narration persists mid-loop narration text.
#: Imported by abilities/review_tool_calls.py to exclude these rows — keep the
#: writer and that reader on the same name via this constant.
NARRATION_TOOL = "narration"

#: Parses the document id out of a rendered ``document.upload`` success envelope so
#: the turn-0 attachment seed can build the transcript<->doc link. The upload body
#: is the structured ToolResult JSON ``{"id":"<hex>",...}`` (TKT-893); ``id`` is a
#: ``secrets.token_hex`` value, so the capture is hex-only. A failed upload renders
#: ``status=error`` with no ``"id":`` key and therefore links nothing.
_SEED_UPLOAD_ID_RE = re.compile(r'"id":"([0-9a-f]+)"')


def _is_auto_memory_recall(row: "dict") -> bool:
    """True for the framework's turn-0 automatic memory-recall seed.

    The seed is dispatched as ``memory(action='recall', _auto=True)`` in
    ``_seed_turn_zero``; a model-initiated ``memory`` call carries no ``_auto``
    flag. Identified off the persisted params (stored as a JSON string by
    ``ActTrail.record``) so the canonical design's "excluding internal:
    compaction + automatic memory recall" rule (§2-3) drops ONLY the framework
    seed from the trail-compaction path — genuine agent ``memory`` calls still
    count as real trail and are compacted normally."""
    if row.get("tool_name") != "memory":
        return False
    raw = row.get("params") or "{}"
    try:
        params = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return False
    return bool(isinstance(params, dict) and params.get("_auto"))


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


# ── Traceability spine ────────────────────────────────────────────────────────
#
# The MessageProcessor instance is the "parent" of everything that runs inside a
# turn. Rather than hide it behind a global ContextVar, it is threaded
# explicitly: ``ToolDispatcher(self).dispatch(…)`` binds it onto each per-call
# ability as ``self.mp`` (constructor-injected), and the Providers facade receives it as ``mp=``.
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

    # ── Tool visibility — per channel, no class-level default ──────────────────
    #
    # Tool discovery/blocking is NOT a MessageProcessor class constant.  It is
    # carried per turn by the channel's ``ProcessorConfig``:
    #   * ``mp.config.discoverable`` — names find_tools may surface for this turn
    #   * ``mp.config.blocked``      — names never offered, even via find_tools
    # ``find_tools`` reads both off ``mp.config`` (see FindToolsAbility), so the
    # single source of truth for the default discoverable surface is
    # ``configs.channels._common.DEFAULT_DISCOVERABLE``.  The legacy static
    # ``DISCOVERABLE`` class list and the never-set ``_BLOCKED`` attribute were
    # removed here: they made the per-config ``blocked`` set dead code and leaked
    # delegate-exclusive raw tools (browser/search) onto every channel.
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, raw_input: str, metadata: dict | None = None):
        self._raw_input = raw_input
        self._metadata = metadata or {}
        # Per-turn config — a plain attribute set by `mp.config = X` in
        # process() / the background workers.  None until attached.
        self.config: "ProcessorConfig | None" = None
        # Tracks the current ACT loop iteration for tool-event emission
        # without thread-local indirection.
        self._current_iteration: int = 0
        self._loop_exited_cleanly: bool = False
        self._active_tools: list[str] = []
        self._uid: int | None = None
        self._deliberation_scalar: float | None = None   # raw sigmoid for this turn
        self._deliberation_ema: float | None = None      # EMA after this turn's update
        # User thinking-level override (None = auto, gate decides). When set to
        # 'medium'/'high' it bypasses the deliberation gate and is forced on every
        # send via Providers.send's precedence. process() reads it once per turn;
        # default None keeps the gate in control for old-path / background MPs.
        self.thinking_override: str | None = None
        # Accumulator starts immediately so exploration + compaction tokens count.
        self._metrics: MetricsAccumulator = MetricsAccumulator()
        # The mp owns its provider gateway — mp-free standalone orchestrator.
        from services.providers import Providers  # noqa: PLC0415
        self.providers = Providers()
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
    def raw_input(self) -> str:
        """Public, read-only alias for the turn's raw input text.

        Read by the turn-0 flashback (continuation gate + topic inference) so the
        seed never reaches into the private backing field across class lines.
        """
        return self._raw_input

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

        # User override bypasses the gate entirely (no classifier inference).
        # NOTE: this sets thinking_level only on the user channel, which is also
        # the only channel that fires the turn-0 seed thinking pass. The override
        # still reaches the provider on EVERY channel via resolve_thinking_mode()
        # in Providers.send() — background loops get high-mode reasoning at the
        # request level without injecting an extra seed deliberation pass.
        if self.thinking_override:
            self.thinking_level = self.thinking_override
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
        # Resolve the persisted user override once per turn. None = auto (gate
        # decides); 'medium'/'high' bypass the gate on EVERY channel via the
        # Providers.send precedence.
        from services.thinking_override_service import get_thinking_override
        mp.thinking_override = get_thinking_override()
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

        # a. Memory flashback — fire once when the declarative flag is set, and
        #    only when the continuation gate decides this is a session start or a
        #    topic shift (a continuation like "yes, do that" re-fires nothing).
        #    The flashback runs the seed recall (caller='seed'), renders a curated
        #    block (≤5 facts + ≤3 dated one-liners, supers preferred) instead of
        #    raw recall JSON, and records its own _auto memory(recall) trail row.
        #    The explicit, model-invoked memory.recall keeps its JSON contract.
        if self.config.memory_seed:
            from services.turn_zero_flashback import TurnZeroFlashback  # noqa: PLC0415
            TurnZeroFlashback(self).seed()

        # b. Attachment uploads — presence-gated.  Each file's upload IS the
        #    ingest (no second auto document.view).  Every upload internally runs
        #    its own (now network-bound) vision/OCR extraction, so N attachments
        #    are fanned out across a bounded thread pool and JOINED before this
        #    method returns: the model's first request still sees every uploaded
        #    document, but the extractions overlap instead of serialising.  The
        #    ``with`` block's exit IS the barrier.  This is safe because each task
        #    builds its OWN ToolDispatcher (dispatch binds a fresh per-call
        #    ability) and each thread holds its own thread-local connection:
        #    documents-table writes serialise through the single-threaded write
        #    queue, while data_graph and act-trail writes serialise at the SQLite
        #    WAL layer (single-writer + 15s busy_timeout).  No shared cursor, and
        #    last_insert_rowid is never read cross-connection — doc_id is a
        #    pre-generated random hex, not a rowid.
        attachments = list(self._metadata.get("attachments") or [])
        if attachments:
            from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
            with ThreadPoolExecutor(max_workers=min(len(attachments), 8)) as pool:
                list(pool.map(self._seed_upload_attachment, attachments))

        # c. High-deliberation thinking pass — programmatic, never model-visible.
        #    Fires LAST, after the upload barrier above, so the deliberation
        #    snapshot already carries the uploaded documents' act-trail rows: the
        #    thinking pass is single-pass with tools disabled, so it can only
        #    reason about vision output that is ALREADY in the parent's rendered
        #    body at dispatch time.
        if getattr(self, "thinking_level", "low") == "high":
            dispatcher.dispatch("thinking", {})

    def _seed_upload_attachment(self, path: str) -> None:
        """Dispatch one turn-0 attachment's blocking ``document.upload`` by PATH.

        Runs on a worker thread of ``_seed_turn_zero``'s pool.  Builds its OWN
        ToolDispatcher (dispatch binds a fresh, isolated per-call ability) so
        concurrent uploads never share dispatch state.  An unsafe/missing path is
        logged and skipped WITHOUT aborting the others (the pool keeps every other
        task alive).

        Dispatches the file PATH, never its bytes: the dispatch params land in the
        act-trail verbatim, so a base64 blob would blow the context window.  Only
        paths under the Chalie temp prefix are accepted (path-traversal guard).
        """
        import os  # noqa: PLC0415
        from abilities._dispatcher import ToolDispatcher  # noqa: PLC0415
        from services.tmp_storage import TMP_PATH_PREFIX  # noqa: PLC0415

        real = os.path.realpath(path)
        if not real.startswith(TMP_PATH_PREFIX) or not os.path.isfile(real):
            logger.warning("[SEED] unsafe or missing attachment path: %r", path)
            return
        result = ToolDispatcher(self).dispatch("document", {
            "action": "upload",
            "path": real,
        })
        # Persist the turn<->doc link so the chat can re-render this attachment on
        # page refresh — the live preview is a browser-only blob: URL that dies on
        # reload (api.conversation.get_recent_history reads this link back).
        # The upload's structured success body carries "id":"<doc_id>" ONLY on
        # success, so a failed upload (status=error, no id key) links nothing.
        # Scoped to the user-attachment seed point on purpose: a model-issued
        # document.upload mid-turn must NOT render as a user attachment.
        if self.uid is not None:
            from services.transcript_service import link_transcript_doc  # noqa: PLC0415
            match = _SEED_UPLOAD_ID_RE.search(result)
            if match:
                link_transcript_doc(self.uid, match.group(1))

    def _build_send_dto(self):
        """Build a ProviderApiRequest from this mp's current state.

        Assembles system prompt, user messages (with checkpoint), act-trail,
        tools, and thinking level. Called by both the normal and force-send paths
        in _loop(). The DTO carries _job_name and _usage_class as attributes for
        the telemetry chokepoint in Providers._log_after_call.
        """
        from services.provider_api import ProviderApiRequest, ThinkingLevel, ProviderType  # noqa: PLC0415
        from abilities._registry import AbilityRegistry  # noqa: PLC0415
        from services.providers import resolve_thinking_mode  # noqa: PLC0415

        system = self.config.get_system_prompt(self)
        messages = self._build_send_messages()
        tools = AbilityRegistry.build_tools(self)

        thinking_str = resolve_thinking_mode(
            getattr(self.config, "thinking_mode", None),
            getattr(self, "thinking_override", None),
            self.thinking_level,
        )
        try:
            level = ThinkingLevel(thinking_str) if thinking_str else ThinkingLevel.LOW
        except ValueError:
            level = ThinkingLevel.LOW

        # Derive provider type from the config flag (replaces uses_vision_provider).
        uses_vision = getattr(self.config, "uses_vision_provider", False)
        provider_type = ProviderType.VISION if uses_vision else ProviderType.CHAT

        return ProviderApiRequest(
            system=system,
            messages=messages,
            type=provider_type,
            tools=tools or None,
            thinking_mode=level,
            cache_prefix=True,
            _job_name=self.config.job,
            _usage_class=getattr(self.config, 'usage_class', None) or 'chat',
            _caller=type(self).__name__,
        )

    def _build_send_messages(self) -> list[dict]:
        """Build the single-element user-message list for this mp's turn.

        Attaches the checkpoint wrapper and the config's image when present.
        """
        user = _wrap_with_checkpoint(self.config.channel, self.config.get_user_prompt(self))
        message: dict = {"role": "user", "content": user}
        img = self.config.get_image(self)
        if img is not None:
            message["image"] = img
        return [message]

    def _loop(self) -> str:
        """ACT game loop — compact-first (canonical design).

        Every iteration rebuilds the request from the DB and measures it.
        Providers.send(dto) raises RequestOverCapError when the FULL request
        reaches the cap. On that signal the loop fires both compactors FIRST —
        normal tool calls, so the loop pauses and the act-trail WS events emit
        for free — which advance the watermark and collapse the act-trail. The
        next iteration re-reads a collapsed history from the DB: no partial-view
        turn. If compaction made no progress the request is irreducible, so it is
        sent as-is (force path) and the provider is the source of truth — an
        over-budget request fails loudly rather than looping forever.

        ResponseOverLimitError (server-side 413) triggers the same compact-and-retry
        path: both over-limit types drive the same recovery.
        """
        from abilities._dispatcher import ToolDispatcher  # noqa: PLC0415
        from services.provider_api import RequestOverCapError, ResponseOverLimitError  # noqa: PLC0415
        while True:
            if self._should_stop():
                return ""
            dto = self._build_send_dto()
            try:
                response = self.providers.send(dto)
                self._record_metrics(response, getattr(response, 'latency_ms', None))
            except (RequestOverCapError, ResponseOverLimitError):
                if self._dispatch_compaction():
                    continue                              # rebuild from the collapsed DB
                # Irreducible — rebuild dto and send without the cap check.
                force_dto = self._build_send_dto()
                response = self._force_send(force_dto)
            if not response.tool_calls:
                return self._format_final_response(response.text)
            dispatcher = ToolDispatcher(self)
            for tc in response.tool_calls:
                if self.cancel_event.is_set():
                    return ""
                dispatcher.dispatch(tc["name"], tc["input"])
            self._record_narration(response)
            self.current_iteration += 1

    def _format_final_response(self, text: "str | None") -> str:
        """Final user-visible text leaving the loop, before any other pipeline.

        On the user-facing channel the LLM is asked to emit Chalie's HTML subset,
        but models still occasionally leak the most common markdown markers
        (``**bold**``, ``_under_``, `` `code` ``). Run a best-effort
        markdown→HTML fallback here — the single point the final response leaves
        the loop, ahead of ``_record`` / ``write_assistant_row`` / post-turn
        hooks and the api-layer ``sanitize()``.

        Gated on ``broadcast_to == 'user'``: every other channel emits JSON or
        plain text (DMN summaries, encoders, compaction) that must not have
        markdown markers rewritten into tags.
        """
        if self.config.broadcast_to == "user":
            from services.markup import markdown_to_html  # noqa: PLC0415
            return markdown_to_html(text)
        return text or ""

    def _force_send(self, dto):
        """Send without the pre-flight cap check (irreducible path).

        Bypasses Providers.send() to skip the over-cap check, but still routes
        through the client and telemetry so the call is logged correctly.

        Telemetry (_log_after_call + _record_metrics) fires only on success — a
        failed send writes NO accounting row, matching the pre-refactor success-
        only logging.  Writing a sentinel under the real job_name would corrupt
        get_last_chat_request_tokens (context indicator → 0).  On failure the
        attempt is surfaced via logger.warning with wall-time, then the exception
        propagates to kill the turn.
        """
        import time  # noqa: PLC0415
        client = self.providers._resolve(dto.type)
        t0 = time.monotonic()
        response = None
        try:
            response = client.send(dto)
            return response
        finally:
            wall_ms = int((time.monotonic() - t0) * 1000)
            if response is not None:
                self.providers._log_after_call(dto, response, wall_ms)
                self._record_metrics(response, wall_ms)
            else:
                logger.warning(
                    "[FORCE-SEND] provider send failed after %dms; no telemetry row written",
                    wall_ms,
                )

    def _record_metrics(self, response, wall_ms=None) -> None:
        """Fold per-send telemetry into the turn's MetricsAccumulator.

        Called after every successful send (normal and force paths). Best-effort:
        failures must never break the send path.
        """
        from services.providers import Providers  # noqa: PLC0415
        latency_ms = wall_ms if wall_ms is not None else getattr(response, 'latency_ms', None)
        try:
            if latency_ms is not None:
                self._metrics.record_llm_call(latency_ms)
        except Exception as exc:
            logger.debug("[LLM LOG] record_llm_call failed: %s", exc)
        try:
            self._metrics.accumulate(response)
        except Exception as exc:
            logger.debug("[LLM LOG] accumulate failed: %s", exc)
        try:
            Providers._record_send_counters(self)
        except Exception as exc:
            logger.debug("[LLM LOG] send-counter record failed: %s", exc)

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

        All tool_calls rows are durable; the 7-day retention janitor in
        DecayEngineService handles cleanup. No per-turn purge.

        Spec §4 / M4-M5 / C3-C4.
        """
        from services.transcript_service import write_assistant_row

        if self.cancel_event.is_set():
            self._cleanup_cancelled()
            return

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

    def get_previous_messages(self, drop_oldest: int = 0) -> str:
        """Render the ## Previous Messages block from _previous_rows().

        Renders every watermark-bounded row. There is NO fixed row cap: history is
        bounded by context-window size, not a turn count. ``drop_oldest`` skips the
        oldest N rows — used ONLY by ChatHistoryCompactor's rare bare-request
        fallback (canonical design step 4.2) when even the tool-free compaction
        request overflows; _previous_rows() is id-ASC, so the first ``drop_oldest``
        entries are the OLDEST messages.

        Tool calls are NOT rendered in the previous messages block — the act-trail
        is provided to the model separately for the current turn only.
        """
        if self.config.suppress_history:
            return ""
        entries = self._previous_rows()
        if drop_oldest:
            entries = entries[drop_oldest:]
        if not entries:
            return ""
        lines: list[str] = []
        for entry in entries:
            ts = _format_ts(entry.get("created_at"), row_kind="transcript", row_id=entry.get("id"))
            raw_role = entry.get("role") or "unknown"
            role_label = "Assistant" if raw_role == "assistant" else raw_role
            content = (entry.get("content") or "").replace("\n", " ").strip()
            lines.append(f"[{ts}] {role_label}: {content}")
        return "\n".join(lines)

    # ── Trail API (T4: act-trail-as-a-query) ─────────────────────────────────

    def _has_trail(self) -> bool:
        """True when real (non-internal) tool activity exists since the last
        trail boundary.

        Slices from the last tool_chain_compactor boundary and returns True only
        when at least one row in that slice is genuine agent activity — i.e. NOT
        a framework compactor marker and NOT the automatic turn-0 memory-recall
        seed (canonical design §2: ToolChainCompactor no-ops when there are no
        tool calls "excluding internal ones (compaction + automatic memory
        recall)"). Used by ToolChainCompactor to silently no-op when there is
        nothing to compact.
        """
        if self.uid is None:
            return False
        from services.act_trail import ActTrail  # noqa: PLC0415
        rows = _from_last_compaction(ActTrail().fetch_by_transcript_id(self.uid))
        return any(
            r["tool_name"] not in _COMPACTOR_TOOLS and not _is_auto_memory_recall(r)
            for r in rows
        )

    def _render_act_trail(self, for_compaction: bool = False) -> str:
        """Assemble the ACT trail string for the current turn.

        Fetches all tool_calls rows for self.uid ordered by id and slices from
        the last tool_chain_compactor boundary (inclusive); the boundary row's
        handover renders in place of the pre-compacted calls. Framework markers
        are filtered: chat_history_compactor rows never reach the model (their
        compacted history arrives via the checkpoint prepend), and an empty
        tool_chain_compactor no-op row is dropped. Returns '' when uid is None.

        ``for_compaction=True`` additionally drops the automatic turn-0
        memory-recall seed (canonical design §3: ToolChainCompactor compacts the
        act-trail "excluding internal: compaction + automatic memory recall").
        The default (False) keeps the seed so the per-turn act-trail the model
        sees still shows what was auto-recalled — only the compaction INPUT is
        filtered, matching the spec's narrow exclusion.
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
            if for_compaction and _is_auto_memory_recall(r):
                continue
            out.append(trail.render(r))
        return "\n".join(out)

    def _dispatch_compaction(self) -> bool:
        """Fire both compactors through the normal tool-dispatch chokepoint and
        report whether they made progress.

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

        Returns True when compaction advanced the state — the act-trail was
        collapsed OR the chat-history watermark moved forward. The ACT loop uses a
        False return (nothing left to compact) to detect an irreducible request and
        stop compacting (send as-is, force) instead of looping forever.
        """
        from abilities._dispatcher import ToolDispatcher  # noqa: PLC0415
        from services import compaction_persistence  # noqa: PLC0415

        trail_before = self._has_trail()
        before = compaction_persistence.get_compaction(self.config.channel)
        before_id = before["compacted_up_to_id"] if before else 0

        dispatcher = ToolDispatcher(self)
        dispatcher.dispatch("tool_chain_compactor", {"act_summary": "Compacting tool history"})
        dispatcher.dispatch("chat_history_compactor", {"act_summary": "Compacting conversation"})

        after = compaction_persistence.get_compaction(self.config.channel)
        after_id = after["compacted_up_to_id"] if after else 0
        # Progress = the act-trail was actually COLLAPSED (a non-empty handover
        # landed a new boundary, so _has_trail flips True→False) OR the chat-history
        # watermark advanced. The trail term is measured AFTER dispatch on purpose:
        # ToolChainCompactor records a no-op row (no boundary) when the handover
        # summariser returns an empty string, leaving _has_trail True. Counting that
        # as progress would spin the OVER_CAP loop forever (that branch has no
        # iteration cap). Returning False instead falls through to send(force=True),
        # letting the provider be the source of truth and fail loud on an
        # irreducible request rather than hang.
        trail_collapsed = trail_before and not self._has_trail()
        return trail_collapsed or after_id > before_id

    def _record_narration(self, response: "object") -> None:  # type: ignore[override]
        """Mid-loop: persist LLM text between iterations as a durable trail row.

        Records a tool_calls row with tool_name='narration'.
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
            tool_name=NARRATION_TOOL,
            params={},
            result=text,
            transcript_id=self.uid,
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
