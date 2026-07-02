# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MessageProcessor — single flat processor for all LLM message turns."""

import logging
import re
import threading
from typing import TYPE_CHECKING, TypeAlias, cast

from services.time_formatter_service import TimeFormatterService

if TYPE_CHECKING:
    from services.processor_config import ProcessorConfig
    from services.provider_api import ProviderApiRequest, ProviderApiResponse

logger = logging.getLogger(__name__)

_OptStr: TypeAlias = "str | None"

# Total provider-send attempts per ACT step (the original try + resends). The
# provider layer never retries; this is the ONE retry policy. On a provider
# failure the MessageProcessor resends the exact same request, up to this many
# attempts, then terminates the turn.
_MAX_PROVIDER_ATTEMPTS = 3

# Per-channel turn serialisation: every turn runs under its channel's lock (see
# _run), so two same-channel turns can never both advance the turn cursor or fire
# a compaction at once. Different channels hold different locks and run in
# parallel; same-channel ordering is deliberately unspecified.
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


# turn_id → cancel_event for live turns. A turn registers its handle for the life
# of its chain (see _setup/_run) so DELETE /api/thread/<turn_id> — a separate request
# holding no in-memory reference to the turn — can cancel it by turn_id. This is the
# ONLY cross-request state a turn keeps; lifecycle, continuation and broadcasts all
# ride the live MP instance (one instance runs the whole turn's step loop).
_cancellers: "dict[int, threading.Event]" = {}
_cancellers_guard = threading.Lock()


def _register_canceller(turn_id: "int | None", event: threading.Event) -> None:
    """Expose a running turn's cancel_event by turn_id; no-op for an unallocated turn."""
    if turn_id is None:
        return
    with _cancellers_guard:
        _cancellers[turn_id] = event


def _unregister_canceller(turn_id: "int | None") -> None:
    """Drop a turn's cancel handle when its chain ends; a None turn_id never held one."""
    if turn_id is None:
        return
    with _cancellers_guard:
        _cancellers.pop(turn_id, None)


#: Parses the document id out of a rendered ``document.upload`` success envelope
#: (structured ToolResult JSON ``{"id":"<hex>",...}``) so the turn-0 attachment
#: seed can link transcript<->doc. A failed upload renders no ``"id":`` key.
_SEED_UPLOAD_ID_RE = re.compile(r'"id":"([0-9a-f]+)"')


_LLM_SENTINEL_PATTERNS = (
    re.compile(r'<\|[^|<>]*\|>'),
    re.compile(r'<\|[^|<>]*\|'),
)


def _sanitize_llm_args(value: object) -> object:
    """Strip leaked provider sentinel tokens (``<|...|>``) from tool args, recursively."""
    if isinstance(value, str):
        for p in _LLM_SENTINEL_PATTERNS:
            value = p.sub('', value)
        return value.strip()
    if isinstance(value, list):
        return [_sanitize_llm_args(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_llm_args(v) for k, v in value.items()}
    return value


#: Appended to the system prompt on any channel whose config sets
#: ``SUPPORTS_ASYNC`` — the SAME gate that exposes the ``async`` tool parameter —
#: so enabling async on a new ProcessorConfig surfaces this guidance with zero
#: extra wiring. Appended trailing (constant text) to keep the cached system
#: prefix byte-stable across turns.
_ASYNC_GUIDANCE = """

## Background tasks

Some tools accept an `async` flag. Set `async: true` to run a tool in the background: you get an immediate acknowledgement, the current turn ends, and the moment the tool finishes you are automatically invoked again with its result as a new turn — so you can keep talking to the user while the work runs.

Choose `async: true` when the user asks for something to happen "in the background" or "while" they do something else, or when a call is likely to be slow (web research, browsing, lengthy shell or file work) and the user should not have to wait. Call tools normally (synchronously) for quick results the user is actively waiting on."""


class MessageProcessor:
    """Single flat message processor for every channel — one instance per turn."""

    # The entire lean WS vocabulary — the five signals a surface listens on.
    # `broadcast()` is the sole emitter; every byte of turn DATA is pulled over
    # REST. There is no `created` signal — the turn_id is returned in the POST
    # body, so the surface already holds it. working → spin the state up;
    # updated → refetch the block; done → drop the working indicator;
    # tool_called → render the call summary + start its act-trail timer;
    # tool_done → stop that timer.
    _WS_WORKING_STATE = 'working'
    _WS_UPDATED_STATE = 'updated'
    _WS_DONE_STATE = 'done'
    _WS_TOOL_CALLED = 'tool_called'
    _WS_TOOL_DONE = 'tool_done'

    def __init__(
        self,
        config: "ProcessorConfig",
        turn_id: int = -1,
        raw_input: str = "",
        metadata: dict[str, object] | None = None,
    ):
        self.config = config
        self._raw_input = raw_input
        self._metadata = metadata or {}
        self._active_tools: list[str] = []
        self._uid: int | None = None
        # Per-channel monotonic turn boundary — the *thread* id. Resolved once by
        # ``self.ts`` at construction (below): a fresh thread allocates
        # ``MAX(turn_id)+1`` for the channel; a real ``turn_id`` appends to that
        # thread. ``make_row_id`` mirrors it onto ``self.turn_id``; every step of
        # the turn's loop shares it.
        self.turn_id: "int | None" = None
        # User thinking-level override (None = auto/gate decides). 'medium'/'high'
        # bypass the gate and are forced on every send via Providers.send.
        self.thinking_override: str | None = None
        # The mp owns its provider gateway — mp-free standalone orchestrator.
        from services.providers import Providers  # noqa: PLC0415
        self.providers = Providers()
        # Cooperative cancellation flag; _step checks is_set() before each send
        # and between tool dispatches. Never raises.
        self._cancel_event: threading.Event = threading.Event()
        # The transcript row this step's tool calls anchor to. Defaults to the
        # input row; a tool-bearing step advances it to its OWN step row.
        self.anchor: int | None = None
        # Armed for the step AFTER a mid-turn compaction: the model lost its
        # working context, so that step's user prompt prepends a recovery banner.
        self.post_compaction_continuation: bool = False
        self.continuation_user_query: str | None = None
        # Deliberation level: 'low'/'medium'/'high'. Overwritten by process() and
        # by _run_thinking_gate(); default 'low' is safe for non-user channels.
        self.thinking_level: str = "low"
        # Rich media ordinal counter — set by cards/media abilities; None until first use.
        self._rich_media_ordinals: object = None
        # Pre-allocate this turn's anchoring input row SYNCHRONOUSLY (claiming its
        # turn_id) so get_meta_data() can answer before run() touches the LLM. The
        # row id is this MP's anchor for its tool calls and act-trail.
        from services.transcript import TranscriptService  # noqa: PLC0415
        self.ts = TranscriptService(self.config, turn_id)
        # A supplied turn_id must name a real turn — a reply can only fork what
        # exists. -1 (unset) allocates fresh and skips the check.
        if turn_id != TranscriptService._UNSET_TURN_ID and not self.ts.turn_exists():
            raise ValueError("Invalid turn_id specified")
        # View mode — an INTERNAL switch, never supplied by callers. A reply INTO
        # an existing thread carries that thread's turn_id (≠ the unset sentinel −1)
        # → FORK read (the whole turn, no settle0 floor) + a fork-scoped checkpoint;
        # a fresh message (turn_id unset) → MAIN spine.
        self._forked: bool = turn_id != TranscriptService._UNSET_TURN_ID
        self.transcript_row_id: int | None = self.make_row_id()
        self._uid = self.transcript_row_id
        self.anchor = self.transcript_row_id

    def make_row_id(self) -> "int | None":
        """Claim this turn's anchoring input row. ``turn_id`` was resolved once by
        ``self.ts`` at construction (verbatim when supplied, else the channel's next
        turn); the row binds to it. A ``skip_input_row`` config writes no row (returns
        ``None``) and lets its first assistant row open the turn."""
        self.turn_id = self.ts.turn_id
        if self._cfg.skip_input_row:
            return None
        return self.ts.insert_row(self._raw_input)

    def get_meta_data(self) -> dict[str, object]:
        """Synchronous turn metadata — available the instant the constructor
        returns, before run() touches the LLM. Drives the thread API's fast POST
        response: the surface holds turn_id + the pre-allocated row id with no WS
        round-trip."""
        return {
            "turn_id": self.turn_id,
            "type": self._cfg.type(),
            "transcript_row_id": self.transcript_row_id,
        }

    def run(self) -> None:
        """Spawn the turn on its own daemon thread and return immediately — the MP
        owns its whole lifecycle (working → updated → done + the per-tool timers).
        The cancel handle is registered synchronously here so DELETE
        /api/thread/<turn_id> is live the instant the POST hands the surface its
        turn_id (before the worker thread's _setup runs)."""
        _register_canceller(self.turn_id, self.cancel_event)
        threading.Thread(
            target=self._run_guarded, daemon=True, name=f"turn-{self.turn_id}",
        ).start()

    def _run_guarded(self) -> None:
        """Thread entry for the async send path. ``_run`` already fired the terminal
        ``done`` from the live instance and re-raised; surface the channel-wide error
        toast the foreground needs, unless the turn was cancelled (stop button stays
        silent). The synchronous ``process()`` path never enters here — a failed
        background pass propagates to its caller, toastless, as before."""
        try:
            self._run()
        except Exception as exc:
            logger.exception("[MP] turn %s failed: %s", self.turn_id, exc)
            if not self.cancel_event.is_set():
                from services.websocket_broker import WebSocketBroker  # noqa: PLC0415
                WebSocketBroker().broadcast(
                    {"type": "error", "message": "Turn failed unexpectedly", "recoverable": False}
                )

    def cancel(self) -> None:
        self._cancel_event.set()

    # Public per-turn aliases — process() and the abilities read & write mp.uid /
    # mp.cancel_event / mp.active_tools; these bridge to the private backing fields.

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
    def active_tools(self) -> list[str]:
        return self._active_tools

    @active_tools.setter
    def active_tools(self, value: list[str]) -> None:
        self._active_tools = value

    @property
    def _cfg(self) -> "ProcessorConfig":
        """Non-None config — always set before any method except __init__ is called."""
        return self.config

    def broadcast(self, state: str, turn_id: int | None, **extra: object) -> None:
        """The sole WS turn-signal chokepoint. No-op unless this config opts in
        via ``BROADCASTS_STATE`` (the ``user`` and ``scheduled`` types do).
        Carries the lean ``status`` (the turn-lifecycle state) + turn_id + ``type``
        (the FE routes by it — ``user`` → main spine, ``scheduled`` → dock + that
        schedule's thread), plus a tool call's id/name/summary for tool_called and
        tool_done; the surface pulls every byte of turn DATA back over REST. A
        broker fault is swallowed so a dead socket never breaks the turn loop."""
        if not self._cfg.BROADCASTS_STATE:
            return
        from services.websocket_broker import WebSocketBroker  # noqa: PLC0415
        try:
            WebSocketBroker().broadcast(
                {"status": state, "turn_id": turn_id, "type": self._cfg.type(), **extra}
            )
        except Exception as exc:  # noqa: BLE001 — a dead surface must not break the loop
            logger.debug("[MP.broadcast] %s failed: %s", state, exc)

    def _run_thinking_gate(self) -> None:
        """Regression-head deliberation scoring → self.thinking_level (user channel only)."""
        if self._cfg.channel != 'user':
            return

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
                logger.info(
                    "[DELIBERATION] turn=%s scalar=None ema=%s bucket=low fallback=true",
                    self._uid, ema_svc.peek(),
                )
                return

            ema, bucket = ema_svc.update_and_bucket(scalar)
            self.thinking_level = bucket
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

    @staticmethod
    def process(
        raw_input: str,
        config: "ProcessorConfig",  # noqa: F821 — deferred import avoids circular dep
        metadata: "dict[str, object] | None" = None,
        cancel_event: "threading.Event | None" = None,
    ) -> str:
        """Single entry point: build an MP, run the turn under its lane lock, return text."""
        meta = metadata or {}
        turn_id = cast("int | None", meta.get("turn_id"))
        mp = MessageProcessor(
            config, turn_id if turn_id is not None else -1, raw_input, meta,
        )
        # Report the turn this ran on back to the caller (mutates meta): a first-fire
        # scheduler reads the freshly-allocated id to persist its series continuity.
        meta["turn_id"] = mp.turn_id
        if cancel_event is not None:
            mp.cancel_event = cancel_event
        # Default 'low' — the gate must explicitly raise it. A 'medium' default
        # would apply deliberation pressure to every turn the gate didn't run
        # (non-user channels) or that crashed, regressing simple recall/chit-chat.
        mp.thinking_level = "low"
        from services.thinking_override_service import get_thinking_override  # noqa: PLC0415
        mp.thinking_override = get_thinking_override()
        return mp._run()

    @staticmethod
    def interrupt(turn_id: "int | None") -> bool:
        """Cancel the running turn with this id by setting its cancel_event; True if a
        live turn was found (it then deletes its own rows and emits no error)."""
        if turn_id is None:
            return False
        with _cancellers_guard:
            event = _cancellers.get(turn_id)
        if event is None:
            return False
        event.set()
        return True

    def _lock_key(self) -> str:
        """Serialize turns within a surface, not across surfaces: the main spine and
        each fork thread hold independent locks so they execute concurrently; two
        turns on the same surface still serialize (shared turn cursor). Non-user
        channels have no turn_id and collapse to one lock per channel as before."""
        if self._forked:
            return f"{self._cfg.channel}:t{self.turn_id}"
        return f"{self._cfg.channel}:main"

    def _run(self) -> str:
        """Run the whole turn under the surface lock: setup, step loop, finalise.
        ``_setup`` registers this turn's cancel handle by turn_id (the only state
        DELETE /api/thread/<turn_id> can reach it through); the ``finally`` drops it
        once the whole step loop — which runs inline here — ends."""
        with _channel_lock(self._lock_key()):
            try:
                self._setup()
                result = self._step()
                if self.cancel_event.is_set():
                    self._cleanup_cancelled()
                    return ""
                return result
            except Exception:
                # The turn died before _end_turn fired its terminal ``done``. Fire it
                # here from the live instance (it owns turn_id in memory) so the surface
                # drops its working indicator, then re-raise for the caller to log + toast.
                # A cancelled turn took the ``return ""`` path above, so this is genuine
                # failure only; the broadcast is BROADCASTS_STATE-gated, so silent off-user.
                self.broadcast(self._WS_DONE_STATE, self.turn_id)
                raise
            finally:
                _unregister_canceller(self.turn_id)

    def _setup(self) -> None:
        """Pre-chain, once per turn: seed active tools, open the turn, gate, seed turn 0."""
        # ACTIVE_TOOLS is live from iteration 0; find_tools appends, build_tools
        # resolves it each turn. Empty for compaction/encoder channels by design.
        self.active_tools = list(self._cfg.always_available or [])

        # The input row was pre-allocated by the constructor (make_row_id), which
        # set uid/anchor/turn_id. Announce the turn with ``working`` now that work
        # is starting — it fires
        # unconditionally for any channel that owns a row; the surface keys on its
        # own current state. A skip_input_row channel has no row/turn yet (its end
        # message allocates one), so it announces nothing. The cancel handle is
        # re-registered (idempotent) so the synchronous process() path is also
        # interruptible; run() registered it up front for the API path.
        if not self._cfg.skip_input_row and self.turn_id is not None:
            _register_canceller(self.turn_id, self.cancel_event)
            self.broadcast(self._WS_WORKING_STATE, self.turn_id)

        if self._cfg.channel == "user":
            self._run_thinking_gate()

        self._seed_turn_zero()

        # Label a turn once it first grows into a thread — i.e. on the first
        # reply past its settle0 (``_forked``). Fire the gist delegate only when no
        # label exists yet, so a thread is summarized exactly once, at birth. The
        # schedule channel rides this same delegate (§13.1): a recurring fire is a
        # reply past settle0, so its gist generates identically to a user thread's.
        if self._forked and self.turn_id is not None:
            try:
                from services.thread_gist_service import get_thread_gist_service  # noqa: PLC0415
                if not get_thread_gist_service().bulk_get(self._cfg.channel, [self.turn_id]):
                    from services.thread_gist_message_processor import maybe_ingest_gist  # noqa: PLC0415
                    maybe_ingest_gist(self._cfg.channel, self.turn_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[MP] thread gist fire failed (non-fatal): %s", exc)

    def _seed_turn_zero(self) -> None:
        """Framework-issued tool calls fired once before iteration 0."""
        # a. Memory flashback — gated on the declarative flag AND the continuation
        #    gate deciding this is a session start / topic shift (a "yes, do that"
        #    re-fires nothing). Renders a curated block and records its own _auto
        #    memory(recall) row; the model-invoked memory.recall keeps its JSON contract.
        if self._cfg.memory_seed:
            from services.turn_zero_flashback import TurnZeroFlashback  # noqa: PLC0415
            TurnZeroFlashback(self).seed()

        # b. Attachment uploads — presence-gated; each file's upload IS the ingest.
        #    Uploads run their own network-bound vision/OCR, so N attachments fan out
        #    across a bounded pool and JOIN before return (the `with` exit is the
        #    barrier). Safe concurrently: each task builds its OWN ToolDispatcher and
        #    holds its own thread-local connection; writes serialise at the SQLite WAL
        #    layer, and doc_id is a pre-generated random hex, never a cross-connection rowid.
        attachments = list(cast("list[str]", self._metadata.get("attachments") or []))
        if attachments:
            from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
            with ThreadPoolExecutor(max_workers=min(len(attachments), 8)) as pool:
                list(pool.map(self._seed_upload_attachment, attachments))

    def _seed_upload_attachment(self, path: str) -> None:
        """Dispatch one turn-0 attachment's blocking ``document.upload`` by PATH."""
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
        # Persist the turn<->doc link so the chat re-renders this attachment on
        # refresh (the live preview is a browser-only blob: URL). The success body
        # carries "id":"<doc_id>" ONLY on success, so a failed upload links nothing.
        # Scoped to the user-attachment seed: a model-issued upload mid-turn must
        # NOT render as a user attachment.
        if self.uid is not None:
            from services.transcript_service import Transcript  # noqa: PLC0415
            match = _SEED_UPLOAD_ID_RE.search(result)
            if match:
                Transcript.link_transcript_doc(self.uid, match.group(1))

    def _build_send_dto(self) -> "ProviderApiRequest":
        """Build a ProviderApiRequest from this mp's current state."""
        from services.provider_api import ProviderApiRequest, ThinkingLevel, ProviderType  # noqa: PLC0415
        from abilities._registry import AbilityRegistry  # noqa: PLC0415
        from services.providers import resolve_thinking_mode  # noqa: PLC0415

        system = self._cfg.get_system_prompt(self)
        if self._cfg.SUPPORTS_ASYNC:
            system += _ASYNC_GUIDANCE
        messages = self._build_send_messages()
        tools = AbilityRegistry.build_tools(self)

        thinking_str = resolve_thinking_mode(
            cast(_OptStr, getattr(self.config, "thinking_mode", None)),
            cast(_OptStr, getattr(self, "thinking_override", None)),
            self.thinking_level,
        )
        try:
            level = ThinkingLevel(thinking_str or "low")
        except ValueError:
            level = ThinkingLevel.LOW

        # Provider type precedence vision > delegate > chat. VisionConfig is itself
        # a delegate channel but keeps VISION precedence to resolve the image provider.
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
            _job_name=self._cfg.job,
            _usage_class=cast("str", getattr(self.config, 'usage_class', None) or 'chat'),
            _caller=type(self).__name__,
        )

    def _build_send_messages(self) -> list[dict[str, object]]:
        """Build the single-element user-message list, with checkpoint wrapper and config image."""
        user = _wrap_with_checkpoint(
            self._cfg.read_channel or self._cfg.channel, self._cfg.get_user_prompt(self),
            self.turn_id if self._forked else None,
        )
        message: dict[str, object] = {"role": "user", "content": user}
        img = self._cfg.get_image(self)
        if img is not None:
            message["image"] = img
        return [message]

    def _step(self) -> str:
        """The turn's step loop — exactly one LLM API call per iteration. A step
        that returns tool calls dispatches them and loops for the next step; a
        mid-turn over-cap compacts, arms the recovery banner, and loops."""
        from services.provider_api import RequestOverCapError, ResponseOverLimitError  # noqa: PLC0415

        while not self.cancel_event.is_set():
            try:
                response = self._send_with_retry(self._build_send_dto())
            except (RequestOverCapError, ResponseOverLimitError):
                self._dispatch_compaction()
                # Arm the next step's recovery banner: the compaction cost the
                # model its working context, so that step's user prompt restates
                # the user's request and points at the checkpoint +
                # review_transcript (which stays available for the rest of the turn).
                from services.transcript_service import Transcript  # noqa: PLC0415
                self.post_compaction_continuation = True
                self.continuation_user_query = Transcript.latest_input_content(self._cfg.channel)
                if "review_transcript" not in self.active_tools:
                    self.active_tools.append("review_transcript")
                continue
            self.providers._record_send_counters(self)
            # The recovery banner (if armed) rode the send that just landed — one-shot.
            self.post_compaction_continuation = False
            self.continuation_user_query = None

            formatted = self._store_row(response.text)
            if not response.tool_calls:
                return self._end_turn(formatted)
            self._dispatch_tools(response.tool_calls)
        return ""

    def _send_with_retry(self, dto: "ProviderApiRequest") -> "ProviderApiResponse":
        """Send through the provider chokepoint, resending the SAME request on failure.

        The provider layer never retries — any failure (timeout, rate limit,
        transport, bad response) bubbles up here. We resend the exact same dto up
        to _MAX_PROVIDER_ATTEMPTS times. RequestOverCapError/ResponseOverLimitError
        are size signals, not provider failures, so they pass straight through to
        the caller's compaction path. After the final failure the turn is
        terminated with a user-facing ProviderRetriesExhaustedError.
        """
        from services.provider_api import (  # noqa: PLC0415
            ProviderRetriesExhaustedError, RequestOverCapError, ResponseOverLimitError,
        )

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_PROVIDER_ATTEMPTS + 1):
            try:
                return self.providers.send(dto)
            except (RequestOverCapError, ResponseOverLimitError):
                raise
            except Exception as exc:  # noqa: BLE001 — every provider failure is retriable here
                last_exc = exc
                if self.cancel_event.is_set():
                    raise
                logger.warning(
                    "[MP] provider send failed (attempt %d/%d) channel=%s: %s",
                    attempt, _MAX_PROVIDER_ATTEMPTS, self._cfg.channel, exc,
                )
                if attempt < _MAX_PROVIDER_ATTEMPTS:
                    # A transient notice, not a turn state (no ``done`` follows; the
                    # turn is still in flight) — toast a surface that a resend is underway.
                    self.broadcast(
                        "provider_retry", self.turn_id,
                        message="The AI provider had a problem — retrying…",
                        attempt=attempt + 1, max_attempts=_MAX_PROVIDER_ATTEMPTS,
                    )

        logger.critical(
            "[MP] provider send failed after %d attempts channel=%s — terminating turn: %s",
            _MAX_PROVIDER_ATTEMPTS, self._cfg.channel, last_exc,
        )
        raise ProviderRetriesExhaustedError(
            "The AI provider failed to respond after several attempts. "
            "Please try again in a moment.",
            provider=getattr(last_exc, "provider", ""),
        ) from last_exc

    def _store_row(self, text: _OptStr) -> str:
        """Persist this step's assistant row and advance the anchor its tool calls record against."""
        formatted = self._format_response(text or "")
        if self._cfg.skip_transcript:
            return formatted
        self.anchor = self.ts.insert_row(formatted, role='assistant', settled=1)
        # Persistence boundary: every chain step and the final synthesis land here.
        # Signal the surface to refetch the turn block (it coalesces by turn_id).
        self.broadcast(self._WS_UPDATED_STATE, self.turn_id)
        return formatted

    def _dispatch_tools(self, tool_calls: list[dict[str, object]]) -> None:
        """Dispatch one step's tool calls in order through the chokepoint."""
        from abilities._dispatcher import ToolDispatcher  # noqa: PLC0415

        dispatcher = ToolDispatcher(self)
        for tc in tool_calls:
            if self.cancel_event.is_set():
                return
            dispatcher.dispatch(cast("str", tc["name"]), cast("dict[str, object]", tc["input"]))

    def _format_response(self, text: _OptStr) -> str:
        """Normalise assistant text before it is persisted or broadcast. Any
        live-surfaced channel (``broadcast_to`` set — user or schedule) renders
        markdown to HTML for its thread view; silent channels keep raw text."""
        if self._cfg.broadcast_to is not None:
            from services.markup import markdown_to_html  # noqa: PLC0415
            return markdown_to_html(text)
        return text or ""

    def _end_turn(self, formatted: str) -> str:
        """End the turn: fan out the failure-isolated post-turn hooks, return the text up."""
        if self.cancel_event.is_set():
            return ""
        # Hooks are mutually independent; order is undefined and may become
        # concurrent, so each call is isolated (log + continue).
        for hook in self._cfg.post_turn_hooks:
            try:
                hook.run(self, formatted)
            except Exception as exc:  # noqa: BLE001 — failure isolation contract
                logger.warning(
                    "[post_turn] hook %s failed (isolated): %s",
                    type(hook).__name__,
                    exc,
                )
        # Turn ended without a re-run: fold the latest window into episodes when the
        # channel's tail has grown enough. Profile-gated and off-thread inside, so
        # firing once per turn (not per row) is the DB-churn win, never a block.
        from services.database_service import get_shared_db_service  # noqa: PLC0415
        from services.episodic_service import EpisodicService  # noqa: PLC0415
        EpisodicService(get_shared_db_service()).check_and_store(self._cfg)
        # The turn's last assistant message carried no tool calls — it is the
        # terminal. Drop the surface's working indicator (spec §6.4/§7.4).
        self.broadcast(self._WS_DONE_STATE, self.turn_id)
        return formatted

    def _cleanup_cancelled(self) -> None:
        """Delete the cancelled chain's OWN rows from this turn — those at or above
        the chain's first row (``self.uid``). A fork reply shares its turn_id with
        the thread it replies into, so deleting the whole turn would wipe history
        that predates the reply; the id floor confines the purge to what THIS chain
        wrote. ``uid`` None (no input row written yet) ⇒ floor 0 ⇒ the turn is this
        chain's alone (a starter / skip-input continuation) and clears whole."""
        if self.turn_id is None:
            return
        channel, floor = self._cfg.channel, self.uid or 0
        try:
            from services.database_service import get_shared_db_service  # noqa: PLC0415
            db = get_shared_db_service()
            with db.connection() as conn:
                conn.execute(
                    "DELETE FROM tool_calls WHERE transcript_id IN "
                    "(SELECT id FROM transcript WHERE channel = ? AND turn_id = ? AND id >= ?)",
                    (channel, self.turn_id, floor),
                )
                conn.execute(
                    "DELETE FROM transcript WHERE channel = ? AND turn_id = ? AND id >= ?",
                    (channel, self.turn_id, floor),
                )
            logger.info(
                "[MessageProcessor] %s: cleaned up cancelled chain (turn_id=%s, id>=%s)",
                channel, self.turn_id, floor,
            )
        except Exception as exc:
            logger.warning(
                "[MessageProcessor] %s: failed to clean up cancelled chain (turn_id=%s): %s",
                channel, self.turn_id, exc,
            )

    def _previous_rows(self) -> list[dict[str, object]]:
        """The LLM ``previous_messages`` row set for this turn's view — the hub of the
        read+compaction flow, read chronologically (``id ASC``) so it re-assembles the
        transcript. Both views read through ``self.ts`` (the channel/turn-bound
        TranscriptService); no piece below is correct alone — they compose:

          watermark       ``compaction_persistence.get_compaction(channel,
                          for_turn_id)`` — two axes that never collide: MAIN
                          (for_turn_id=None) is a *turn_id*; FORK (for_turn_id=T) is a
                          *transcript.id* within T. Everything at/below it is already
                          carried as prose by the checkpoint summary ``_wrap_with_
                          checkpoint`` prepends; this returns only the live tail ABOVE
                          it. Model context = summary(≤wm) ⊕ these rows(>wm).
          FORK            ``self.ts.get_turn_rows("id ASC")`` — the whole forked turn
                          (no settle0 floor; that cut is MAIN-only) — kept between its
                          id watermark and the live reply's own rows (``id >= self.uid``,
                          which render separately).
          MAIN            every turn above the watermark (``get_turns_since``) read in
                          turn order, each floored at settle0 via ``get_turn_by_id(...,
                          include_post_settled=False)`` (the cut is a SQL WHERE, so a
                          turn with no settled reply drops out entirely). The watermark
                          is the ONLY spine bound — no turn cap (it would silently drop
                          un-summarised turns).
          writer          ``chat_history_compactor.run`` folds these rows into the
                          summary and advances the watermark to their max id/turn_id.
                          That "always advances" guarantee is WHY this keeps rows
                          strictly above the watermark — else the compactor re-folds
                          already-folded rows and livelocks."""
        if self._cfg.suppress_history:
            return []
        from services import compaction_persistence  # noqa: PLC0415
        # The compaction watermark axis follows the history SCOPE: a split-channel
        # config reads cross-turn history from ``read_channel`` and its checkpoint
        # lives there too, so read both off the read channel (``None`` ⇒ the write
        # channel — every non-split config). The row fetches go through ``self.ts``,
        # already routed to the read channel by TranscriptService.
        channel = self._cfg.read_channel or self._cfg.channel
        if self._forked and self.turn_id is not None:
            compaction = compaction_persistence.get_compaction(channel, self.turn_id)
            wm = cast("int", compaction["compacted_up_to"]) if compaction else 0
            upper = self.uid if self.uid is not None else 9223372036854775807
            return [r for r in self.ts.get_turn_rows("id ASC")
                    if wm < int(cast("int", r["id"])) < upper]
        compaction = compaction_persistence.get_compaction(channel, None)
        wm = cast("int", compaction["compacted_up_to"]) if compaction else 0
        rows: list[dict[str, object]] = []
        for tid in self.ts.get_turns_since(wm):
            rows += self.ts.get_turn_by_id(tid, "id ASC", False)
        return rows

    def get_previous_messages(self, drop_oldest: int = 0) -> str:
        """Render the ## Previous Messages block from _previous_rows()."""
        entries = self._previous_rows()[drop_oldest:]
        if not entries:
            return ""
        lines: list[str] = []
        for entry in entries:
            ts = _format_ts(cast(_OptStr, entry.get("created_at")), row_kind="transcript", row_id=cast("int | None", entry.get("id")))
            raw_role = cast("str", entry.get("role") or "unknown")
            role_label = "Assistant" if raw_role == "assistant" else raw_role
            content = cast("str", entry.get("content") or "").replace("\n", " ").strip()
            lines.append(f"[{ts}] {role_label}: {content}")
        return "\n".join(lines)

    def _render_act_trail(self) -> str:
        """Assemble the current turn's act-trail — its tool calls, in record order.

        A mid-turn compaction RESETS the visible trail: only tool calls emitted
        after the turn's most recent ``chat_history_compactor`` marker are
        rendered. The compacted checkpoint already carries prior continuity, so
        on the resume the model re-fires any tools it still needs rather than
        re-carrying the bloated pre-compaction results (which would re-overflow
        the window and loop). With no compaction in the turn there is no marker,
        so the whole turn's calls render — byte-identical to before, preserving
        intra-turn tool-result continuity for ordinary multi-step loops."""
        if self.turn_id is None:
            return ""
        from services.act_trail import ActTrail  # noqa: PLC0415

        trail = ActTrail()
        rows = trail.fetch_by_turn(self._cfg.channel, self.turn_id)
        last_compaction = max(
            (int(cast("int", r["id"])) for r in rows
             if r.get("tool_name") == "chat_history_compactor"),
            default=0,
        )
        lines: list[str] = []
        for row in rows:
            if row.get("tool_name") == "chat_history_compactor":
                continue
            if int(cast("int", row["id"])) <= last_compaction:
                continue
            lines.append(trail.render(row))
        return "\n".join(lines)

    def _dispatch_compaction(self) -> None:
        """Fire chat-history compaction through the normal tool-dispatch chokepoint.

        The compactor scopes its own checkpoint by view (``mp._forked``) and writes
        it into the ``compactions`` table — off the transcript spine, so firing
        never moves ``turn_id`` and the turn boundary survives a mid-turn collapse.
        The post-compaction continuation re-reads ``_previous_rows`` against the
        fresh watermark; nothing here needs to inspect the result."""
        from abilities._dispatcher import ToolDispatcher  # noqa: PLC0415
        ToolDispatcher(self).dispatch(
            "chat_history_compactor", {"act_summary": "Compacting conversation"}
        )


# ── Module-private helpers ────────────────────────────────────────────────────


#: Rendered for a missing/unparseable ``created_at``. Exactly 16 chars so the
#: ``[YYYY-MM-DD HH:MM]`` column width in Previous Messages stays stable.
_MISSING_TS_PLACEHOLDER = '????-??-?? ??:??'


def _wrap_with_checkpoint(channel: str, user_body: str, for_turn_id: "int | None") -> str:
    """Wrap the user-message body with a ### Checkpoint envelope when a compaction
    exists for this view's scope: ``for_turn_id`` is the thread id
    for a FORK reply, ``None`` for the MAIN spine — the same axis the matching
    ``_previous_rows`` read uses, so the envelope and the history never disagree.

    This summary IS the reconstruction of everything at/below the watermark; it
    pairs with the live tail above the watermark that ``_previous_rows`` returns
    (model context = summary(≤wm) ⊕ tail(>wm)). See the flow narrative on
    ``MessageProcessor._previous_rows``."""
    from services import compaction_persistence

    row = compaction_persistence.get_compaction(channel, for_turn_id)
    if not row or not (compacted := cast('str', row.get('compacted_text') or '').strip()):
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
    """Format a UTC SQLite/ISO timestamp into ``YYYY-MM-DD HH:MM`` in the user's local tz."""
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
