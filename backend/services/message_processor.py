# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MessageProcessor — single flat processor for all LLM message turns.

One instance per turn (no sharing, no subclasses, no singleton accessors).
Every channel calls the static ``process()`` entry point with a per-turn
``ProcessorConfig`` carrying all channel-specific behaviour.
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

# ── Per-channel turn serialisation ─────────────────────────────────────────────
#
# Every turn on a channel runs while holding that channel's lock (see _run), so
# two turns on the same channel can never both advance the channel's turn cursor
# or both fire a compaction at once — e.g. a delegate finishing into the user
# channel while the user's own turn is still synthesising. Turns on different
# channels hold different locks and run in parallel. Ordering between two
# same-channel turns is deliberately NOT guaranteed: which one runs first is
# unimportant; not corrupting the shared channel state is.
_channel_locks: "dict[str, threading.Lock]" = {}
_channel_locks_guard = threading.Lock()


def _channel_lock(channel: str) -> threading.Lock:
    """Return the process-wide lock for ``channel``, creating it once."""
    lock = _channel_locks.get(channel)
    if lock is None:
        with _channel_locks_guard:
            lock = _channel_locks.get(channel)
            if lock is None:
                lock = threading.Lock()
                _channel_locks[channel] = lock
    return lock

# ── Compaction marker ────────────────────────────────────────────────────────
#
# ChatHistoryCompactor is dispatched programmatically (never model-selected) by
# _dispatch_compaction(); it writes the durable transcript watermark. Its
# recorded tool_calls row is a framework marker, not real tool activity, so
# _render_act_trail drops it from the trail the model sees (the compacted output
# reaches the model through the checkpoint prepend instead).

#: Parses the document id out of a rendered ``document.upload`` success envelope so
#: the turn-0 attachment seed can build the transcript<->doc link. The upload body
#: is the structured ToolResult JSON ``{"id":"<hex>",...}`` (TKT-893); ``id`` is a
#: ``secrets.token_hex`` value, so the capture is hex-only. A failed upload renders
#: ``status=error`` with no ``"id":`` key and therefore links nothing.
_SEED_UPLOAD_ID_RE = re.compile(r'"id":"([0-9a-f]+)"')


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
    """Single flat message processor for every channel — one instance per turn."""

    def __init__(self, raw_input: str, metadata: dict | None = None):
        self._raw_input = raw_input
        self._metadata = metadata or {}
        # Per-turn config — a plain attribute set by `mp.config = X` in
        # process() / the background workers.  None until attached.
        self.config: "ProcessorConfig | None" = None
        self._active_tools: list[str] = []
        self._uid: int | None = None
        # Per-channel monotonic turn boundary. None until _setup()
        # opens this turn: writing the input row allocates the next turn_id for
        # the channel atomically, and _setup reads it back. Every call opens its
        # OWN turn — an async tool result or a delegate return is a NEW turn, not
        # a continuation. It is the single boundary the act-trail and the
        # previous-messages history render by (tool_calls derive it via a join on
        # the transcript input row).
        self.turn_id: int | None = None
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
        # chain to stop at the next step boundary. Never raises — _step checks
        # is_set() before each send and between tool dispatches.
        self._cancel_event: threading.Event = threading.Event()
        # ── Recursive-chain state ────────────────────────────────────────────
        # The transcript row this step's tool calls anchor to. Defaults to the
        # input row (uid); a tool-bearing step resets it to its OWN step row in
        # _store_row before dispatching, so each step's tools attach to the
        # assistant row that emitted them.
        self.anchor: int | None = None
        # Every turn_id this chain opened. A mid-turn compaction splits the turn
        # (a fresh turn_id for the continuation), so cancellation cleanup must
        # sweep them all. Shared by reference down the chain via _continue.
        self._chain_turn_ids: set[int] = set()
        # Set on a continuation MP spawned AFTER a mid-turn compaction: the model
        # lost its working context, so the user prompt prepends a recovery banner
        # restating the request and pointing at the checkpoint + review_transcript.
        self.post_compaction_continuation: bool = False
        self.continuation_user_query: str | None = None

    def cancel(self) -> None:
        self._cancel_event.set()

    # ── Public per-turn attribute aliases ─────────────────────────────────────
    # The flat process() lifecycle reads/writes mp.uid / mp.cancel_event /
    # mp.active_tools; these properties bridge to the private backing fields
    # set up in __init__.

    @property
    def raw_input(self) -> str:
        return self._raw_input

    @property
    def uid(self) -> 'int | None':
        return self._uid

    @uid.setter
    def uid(self, value: 'int | None') -> None:
        self._uid = value

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel_event

    @cancel_event.setter
    def cancel_event(self, value: threading.Event) -> None:
        self._cancel_event = value

    @property
    def active_tools(self) -> list:
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

        Every call opens its OWN per-channel turn: _setup() allocates
        the next turn_id when it writes the input row. Fresh user / external
        input, an async tool result and a delegate return are each a new turn —
        none re-enters a prior turn.
        """
        mp = object.__new__(MessageProcessor)
        # Initialise old-path attributes (metrics, cancel, etc.) via old __init__.
        MessageProcessor.__init__(mp, raw_input, metadata)
        # New flat-path attributes (spec §4 field list).
        mp.config = config
        mp.uid: "int | None" = None
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
        """Lifecycle wrapper — run the whole turn under the channel lock.

        Setup, the recursive step chain, and finalisation all run while holding
        this channel's lock, so a second turn on the same channel waits here
        until this one finishes: it cannot open its turn, advance the channel's
        turn cursor, or start a second compaction concurrently. Different
        channels hold different locks and run fully in parallel. On cooperative
        cancellation the chain unwinds to here returning '', and every turn it
        opened is deleted.
        """
        with _channel_lock(self.config.channel):
            self._setup()
            result = self._step()
            if self.cancel_event.is_set():
                self._cleanup_cancelled()
                return ""
            return result

    def _setup(self) -> None:
        """Pre-chain.  Executes once per turn, before the first step.

        1. Seed ACTIVE_TOOLS with this channel's always_available tier.
        2. Open this turn by writing the input row.
        3. Run thinking gate (user channel only).
        4. Seed turn 0 — framework tool calls before the first LLM turn.
        """
        # ACTIVE_TOOLS is live from iteration 0; find_tools appends to it and
        # build_tools resolves it each turn. Empty for compaction / encoder
        # channels whose always_available is empty by design.
        self.active_tools = list(self.config.always_available or [])

        from services.transcript_service import turn_id_of_row, write_input_row

        # Open this turn. Writing the input row allocates the next
        # turn_id for the channel ATOMICALLY (a COALESCE subquery inside the
        # INSERT, under the writer lock) and we read the value back — never
        # pre-compute max+1 in Python, which would race two concurrent
        # same-channel turns (e.g. parallel scheduled tasks on their own threads)
        # onto one turn_id and merge their act-trails. Channels with no input row
        # (skip_input_row / skip_transcript) leave turn_id None: they record no
        # tool calls and their end message allocates its own turn at write time.
        if not self.config.skip_transcript and not self.config.skip_input_row:
            self.uid = write_input_row(
                self.config.channel, self.config.role, self._raw_input,
            )
            self.turn_id = turn_id_of_row(self.uid)
            # The first step's tools anchor to the input row until _store_row
            # advances the anchor to the step's own assistant row. Track this turn
            # so cancellation cleanup sweeps it (and any split-off continuation).
            self.anchor = self.uid
            if self.turn_id is not None:
                self._chain_turn_ids.add(self.turn_id)

        if self.config.channel == "user":
            self._run_thinking_gate()

        self._seed_turn_zero()

    def _seed_turn_zero(self) -> None:
        """Framework-issued tool calls fired once before iteration 0.

        Two declarative behaviours, zero hooks.  Each call goes through
        ToolDispatcher.dispatch() so it BLOCKS, records a tool_calls row, and is
        rendered into the trail exactly like an LLM-issued call.  The model's
        first turn already sees memory matches and uploaded documents.
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
        in _step(). The DTO carries _job_name and _usage_class as attributes for
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

        # Derive provider type from the config flags — precedence
        # vision > delegate > chat. VisionConfig is itself a delegate channel
        # (delegate:vision) but keeps VISION precedence so it always resolves
        # the image-understanding provider.
        uses_vision = getattr(self.config, "uses_vision_provider", False)
        uses_delegate = getattr(self.config, "uses_delegate_provider", False)
        if uses_vision:
            provider_type = ProviderType.VISION
        elif uses_delegate:
            provider_type = ProviderType.DELEGATE
        else:
            provider_type = ProviderType.CHAT

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

    def _step(self) -> str:
        """One link in the recursive turn chain — exactly one LLM API call.

        An over-cap / over-limit signal fires chat-history compaction (a normal
        tool dispatch, so its act-trail WS events emit for free) and the chain
        continues on the fresh post-compaction turn. Otherwise the response's
        assistant row is stored; if it carries tool calls they dispatch and BLOCK
        against that row and the chain recurses for the next API call, else the
        row is the turn's end. No loop, no iteration cap — depth is bounded only
        by when the model stops emitting tool calls (a runaway turn is a
        hard-restart condition; Python's recursion limit is the backstop).
        """
        from services.provider_api import RequestOverCapError, ResponseOverLimitError  # noqa: PLC0415

        if self.cancel_event.is_set():
            return ""
        try:
            response = self.providers.send(self._build_send_dto())
        except (RequestOverCapError, ResponseOverLimitError):
            self._dispatch_compaction()
            return self._continue(post_compaction=True)._step()
        self._record_metrics(response, getattr(response, 'latency_ms', None))

        formatted = self._store_row(response.text)
        if not response.tool_calls:
            return self._end_turn(formatted)
        self._emit_interim(formatted)
        self._dispatch_tools(response.tool_calls)
        return self._continue()._step()

    def _store_row(self, text: "str | None") -> str:
        """Persist this step's assistant row and anchor its tool calls to it.

        Every step writes its OWN assistant transcript row — the row the model
        emitted — so the chat renders assistant prose and tool batches
        interleaved down the turn (the row's tools anchor here, never on the
        input row). The dispatcher reads ``self.anchor`` for the transcript_id it
        records each tool call against, so the anchor advances to this row before
        any tools dispatch. ``skip_transcript`` channels persist nothing and keep
        ``anchor == uid``. A hidden-input continuation whose input row was never
        written carries turn_id None until the first row it writes adopts that
        row's turn, so the rest of the chain shares one cursor. Returns the
        formatted text for the caller to emit and propagate.
        """
        formatted = self._format_response(text or "")
        if self.config.skip_transcript:
            return formatted
        from services.transcript_service import turn_id_of_row, write_assistant_row  # noqa: PLC0415

        row_id = write_assistant_row(self.config.channel, formatted, turn_id=self.turn_id)
        if self.turn_id is None:
            self.turn_id = turn_id_of_row(row_id)
            if self.turn_id is not None:
                self._chain_turn_ids.add(self.turn_id)
        self.anchor = row_id
        return formatted

    def _emit_interim(self, formatted: str) -> None:
        """Push a mid-turn assistant row live on the user channel.

        The turn's final row is emitted by the api layer once the chain returns;
        only mid-turn prose is broadcast here so the surface updates within the
        turn. Empty text and non-user channels are no-ops.
        """
        if self.config.broadcast_to == "user" and formatted.strip():
            from api.chat import _broadcast_interim  # noqa: PLC0415
            _broadcast_interim(self._metadata, formatted)

    def _dispatch_tools(self, tool_calls: list) -> None:
        """Dispatch every tool call of one step through the chokepoint, in order.

        Each dispatch BLOCKS and records its tool_calls row against ``self.anchor``
        (this step's row). Stops early if cancellation fires between calls; the
        chain then unwinds at the next step's cancel guard.
        """
        from abilities._dispatcher import ToolDispatcher  # noqa: PLC0415

        dispatcher = ToolDispatcher(self)
        for tc in tool_calls:
            if self.cancel_event.is_set():
                return
            dispatcher.dispatch(tc["name"], tc["input"])

    def _continue(self, *, post_compaction: bool = False) -> "MessageProcessor":
        """Spawn the next link in the chain — a hidden-input continuation MP.

        Same turn, same channel: the model just emitted tool calls (or the turn
        was just compacted), so a fresh MP rebuilds the request from the collapsed
        DB for the next API call. It inherits the turn's identity (uid, turn_id)
        and SHARES the chain's mutable state by reference — the cancel Event, the
        metrics accumulator, the active-tools list, the chain's turn-id set, and
        the rich-media ordinal counter — so cancellation, cleanup, telemetry, and
        monotonic media ordinals all see the whole chain as one unit. The config
        is flipped to hidden-input (skip_input_row) so the continuation opens no
        new input row, and the anchor resets to the input row until this link's
        own step row (if any) advances it.

        When ``post_compaction`` is set the prior step lost its working context to
        a compaction, so the continuation carries the recovery banner: the latest
        user input is restated and the transcript-reading tool is surfaced.
        """
        child = object.__new__(MessageProcessor)
        MessageProcessor.__init__(child, self._raw_input, self._metadata)
        child.config = self.config.with_hidden_input()
        child.uid = self.uid
        child.turn_id = self.turn_id
        child.anchor = self.uid
        child.active_tools = self.active_tools            # SAME list
        child._metrics = self._metrics                    # SAME accumulator
        child._chain_turn_ids = self._chain_turn_ids      # SAME set
        child._rich_media_ordinals = getattr(self, "_rich_media_ordinals", None)
        child.cancel_event = self.cancel_event            # SAME Event
        # Turn-scoped accumulators/snapshots (pattern touched-ids, save counters,
        # the skill-suggestion request snapshot) span the whole turn, not one link.
        # Carry each forward by reference so a continuation does not reset them and
        # the final step's post-turn hooks see the turn's full state. The configs'
        # own lazy-init guards (if not hasattr) then leave the inherited value be.
        for _name in self.config.turn_scoped_state():
            if hasattr(self, _name):
                setattr(child, _name, getattr(self, _name))
        child.thinking_level = getattr(self, "thinking_level", "low")
        child.thinking_override = self.thinking_override
        child.providers = self.providers
        if post_compaction:
            from services.transcript_service import latest_input_content  # noqa: PLC0415
            child.post_compaction_continuation = True
            child.continuation_user_query = latest_input_content(self.config.channel)
            if "review_transcript" not in child.active_tools:
                child.active_tools.append("review_transcript")
        return child

    def _format_response(self, text: "str | None") -> str:
        """Normalise assistant text before it is persisted or broadcast.

        On the user-facing channel the LLM is asked to emit Chalie's HTML subset,
        but models still occasionally leak the most common markdown markers
        (``**bold**``, ``_under_``, `` `code` ``). Run a best-effort
        markdown→HTML fallback here — the single point every assistant row (each
        mid-turn step row AND the final end message) is normalised, ahead of
        ``write_assistant_row`` / post-turn hooks and the api-layer ``sanitize()``.

        Gated on ``broadcast_to == 'user'``: every other channel emits JSON or
        plain text (DMN summaries, encoders, compaction) that must not have
        markdown markers rewritten into tags.
        """
        if self.config.broadcast_to == "user":
            from services.markup import markdown_to_html  # noqa: PLC0415
            return markdown_to_html(text)
        return text or ""

    def _record_metrics(self, response, wall_ms=None) -> None:
        """Fold per-send telemetry into the turn's MetricsAccumulator.

        Called after every successful send. Best-effort: failures must never
        break the send path.
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

    def _end_turn(self, formatted: str) -> str:
        """End the turn: fan-out the after-turn side-effects, return the text up.

        Reached when a step's response carries no tool calls — that row, already
        stored by ``_store_row``, is the turn's end message. The api layer emits
        the final message once the chain returns, so this only runs the post-turn
        hooks and propagates the text. All tool_calls rows are durable; the 7-day
        retention janitor in DecayEngineService handles cleanup — no per-turn purge.
        """
        if self.cancel_event.is_set():
            return ""
        # After-turn hooks: mutually independent, failure-isolated.  One hook
        # raising is a non-event for the others — the order is undefined and may
        # become concurrent, so each call is isolated (log + continue).
        for hook in self.config.post_turn_hooks:
            try:
                hook.run(self, formatted)
            except Exception as exc:  # noqa: BLE001 — failure isolation contract
                logger.warning(
                    "[post_turn] hook %s failed (isolated): %s",
                    type(hook).__name__,
                    exc,
                )
        return formatted

    def _cleanup_cancelled(self) -> None:
        """Delete every DB row of the cancelled chain — both tables, every turn.

        A chain may span more than one turn_id (a mid-turn compaction splits it),
        so cleanup sweeps every turn the chain opened, not just the latest. For
        each, the tool calls anchored to that turn's transcript rows are deleted
        BEFORE the transcript rows — tool_calls carries no turn column, so its rows
        are scoped through the turn's transcript ids (a subquery), and the FK from
        tool_calls.transcript_id has no ON DELETE CASCADE, so removing the
        transcript rows first would raise FOREIGN KEY constraint. Only the
        turn-owning foreground MP is cancellable.
        """
        turn_ids = {t for t in (self._chain_turn_ids or {self.turn_id}) if t is not None}
        if not turn_ids:
            return
        channel = self.config.channel
        try:
            from services.database_service import get_shared_db_service  # noqa: PLC0415
            db = get_shared_db_service()
            with db.connection() as conn:
                for tid in sorted(turn_ids):
                    conn.execute(
                        "DELETE FROM tool_calls WHERE transcript_id IN "
                        "(SELECT id FROM transcript WHERE channel = ? AND turn_id = ?)",
                        (channel, tid),
                    )
                    conn.execute(
                        "DELETE FROM transcript WHERE channel = ? AND turn_id = ?",
                        (channel, tid),
                    )
            logger.info(
                "[MessageProcessor] %s: cleaned up cancelled chain (turn_ids=%s)",
                channel,
                sorted(turn_ids),
            )
        except Exception as exc:
            logger.warning(
                "[MessageProcessor] %s: failed to clean up cancelled chain (turn_ids=%s): %s",
                channel,
                sorted(turn_ids),
                exc,
            )

    def _previous_rows(self) -> list:
        """Watermark-bounded transcript rows for this channel, EXCLUDING the
        current turn.

        SELECT * FROM transcript WHERE channel=? AND id > <watermark> ORDER BY id ASC
        No LIMIT (fixes the 20-row bug). Rows of the current turn (turn_id ==
        self.turn_id) are dropped: the live input is rendered separately and the
        turn's end message is not written until the loop finishes. Empty for
        suppress_history channels and post-compaction turns (the compaction row's
        own id IS the watermark)."""
        if self.config.suppress_history:
            return []
        from services import compaction_persistence, transcript_service  # noqa: PLC0415
        compaction = compaction_persistence.get_compaction(self.config.channel)
        watermark = compaction["compacted_up_to_id"] if compaction else 0
        rows = transcript_service.get_recent(self.config.channel, since_id=watermark)
        if self.turn_id is None:
            return rows
        return [r for r in rows if (r.get("turn_id") or 0) < self.turn_id]

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

    # ── Trail API: act-trail-as-a-query, keyed on (channel, turn_id) ─────────

    def _render_act_trail(self) -> str:
        """Assemble the current turn's act-trail — its tool calls.

        The turn's tool calls are fetched by joining transcript on (channel,
        turn_id) and rendered in id order — the order the dispatcher recorded
        them. Mid-turn assistant text is no longer persisted (one input, one end
        message per turn), so there is nothing to interleave. The
        chat_history_compactor marker is dropped — its compacted output reaches
        the model through the checkpoint prepend, not the trail. Returns '' when
        this MP has no turn.
        """
        if self.turn_id is None:
            return ""
        from services.act_trail import ActTrail  # noqa: PLC0415

        trail = ActTrail()
        lines: list[str] = []
        for row in trail.fetch_by_turn(self.config.channel, self.turn_id):
            if row.get("tool_name") == "chat_history_compactor":
                continue
            lines.append(trail.render(row))
        return "\n".join(lines)

    def _dispatch_compaction(self) -> None:
        """Fire chat-history compaction through the normal tool-dispatch chokepoint.

        No alternative path: this is the SAME machinery as the turn-0
        memory/thinking seeds. ChatHistoryCompactor goes through
        ToolDispatcher.dispatch so it auto-records its tool_calls row AND
        auto-emits the act-trail WS events (act_tool_start/act_tool_end) — that is
        how compaction shows up in the frontend act-trail without any hand-rolled
        emit. It reads get_previous_messages() off this mp and advances the
        durable transcript watermark. INTERNAL (policy gate bypassed) and
        never-discoverable.

        When the watermark moves forward the compactor wrote a fresh
        role='compaction' row on its OWN new turn; this method adopts that turn so
        the post-compaction continuation MP, and everything it writes, lives on the
        new turn — the mid-turn turn_id split. The pre-compaction tool calls stay
        anchored to the original turn's rows.
        """
        from abilities._dispatcher import ToolDispatcher  # noqa: PLC0415
        from services import compaction_persistence  # noqa: PLC0415
        from services.transcript_service import turn_id_of_row  # noqa: PLC0415

        before = compaction_persistence.get_compaction(self.config.channel)
        before_id = before["compacted_up_to_id"] if before else 0

        ToolDispatcher(self).dispatch(
            "chat_history_compactor", {"act_summary": "Compacting conversation"}
        )

        after = compaction_persistence.get_compaction(self.config.channel)
        after_id = after["compacted_up_to_id"] if after else 0
        if after_id <= before_id:
            return
        new_turn = turn_id_of_row(after_id)
        if new_turn is not None:
            self.turn_id = new_turn
            self._chain_turn_ids.add(new_turn)


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
