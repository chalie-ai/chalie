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
# ride the live MP instance (turn_id is in memory, _continue threads it + the Event).
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

    def __init__(self, raw_input: str, metadata: dict[str, object] | None = None):
        self._raw_input = raw_input
        self._metadata = metadata or {}
        # Per-turn config — set by `mp.config = X` in process() / workers.
        self.config: "ProcessorConfig | None" = None
        self._active_tools: list[str] = []
        self._uid: int | None = None
        # Per-channel monotonic turn boundary — redefined as the *thread* id. A
        # thread-starter has ``thread_id=None`` and _setup allocates a fresh one;
        # a reply has ``thread_id`` set (from metadata) and _setup appends to it.
        # Every step in the chain shares it (propagated via _continue). The
        # act-trail and previous-messages history render by it; tool_calls derive
        # it via a join on the input row.
        self.turn_id: "int | None" = cast("int | None", self._metadata.get("thread_id"))
        # View mode. True ⇒ this is a genuine user reply INTO a
        # thread → FORK read (the turn's settle0-floored continuation) and a fork-
        # scoped checkpoint. Driven by a dedicated metadata key set ONLY by the
        # user-reply path, NEVER raw ``thread_id`` presence — a delegate reply
        # routed into a main turn reuses ``thread_id`` and must stay MAIN.
        self._forked: bool = bool(self._metadata.get("is_thread_reply"))
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
        # Set on a continuation MP spawned AFTER a mid-turn compaction: the model
        # lost its working context, so the user prompt prepends a recovery banner.
        self.post_compaction_continuation: bool = False
        self.continuation_user_query: str | None = None
        # Deliberation level: 'low'/'medium'/'high'. Overwritten by process() and
        # by _run_thinking_gate(); default 'low' is safe for non-user channels.
        self.thinking_level: str = "low"
        # Rich media ordinal counter — set by cards/media abilities; None until first use.
        self._rich_media_ordinals: object = None

    def cancel(self) -> None:
        self._cancel_event.set()

    # Public per-turn aliases — process()/_continue read & write mp.uid /
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
        return cast("ProcessorConfig", self.config)

    def broadcast(self, state: str, turn_id: int | None, **extra: object) -> None:
        """The sole WS turn-signal chokepoint. No-op unless this channel opts in
        via ``BROADCASTS_STATE`` (the ``user`` and ``schedule`` channels do).
        Carries the lean state + turn_id + channel (the FE routes by it — ``user``
        → main spine, ``schedule`` → dock + that schedule's thread), plus a tool
        call's id/name/summary for tool_called and tool_done; the surface pulls
        every byte of turn DATA back over REST. A broker fault is swallowed so a
        dead socket never breaks the turn loop."""
        if not self._cfg.BROADCASTS_STATE:
            return
        from services.websocket_broker import WebSocketBroker  # noqa: PLC0415
        try:
            WebSocketBroker().broadcast(
                {"type": state, "turn_id": turn_id, "channel": self._cfg.channel, **extra}
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
        mp = MessageProcessor(raw_input, metadata)
        mp.config = config
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
        channels have no thread_id and collapse to one lock per channel as before."""
        if self._forked:
            return f"{self._cfg.channel}:t{self._metadata.get('thread_id')}"
        return f"{self._cfg.channel}:main"

    def _run(self) -> str:
        """Run the whole turn under the surface lock: setup, step chain, finalise.
        ``_setup`` registers this turn's cancel handle by turn_id (the only state
        DELETE /api/thread/<turn_id> can reach it through); the ``finally`` drops it
        once the whole chain — every ``_continue`` link runs inline here — ends."""
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

        from services.transcript_service import Transcript  # noqa: PLC0415

        # Writing the input row allocates the next turn_id ATOMICALLY (a COALESCE
        # subquery inside the INSERT, under the writer lock) — never max+1 in
        # Python, which would race two concurrent same-channel turns onto one
        # turn_id and merge their act-trails. The input row is gated on
        # skip_input_row alone: a skip_transcript channel still anchors its
        # act-trail to a uid (the background pattern/geo/summary passes set
        # skip_input_row=False for exactly this — their per-turn budget, dedup and
        # decay are DB-derived and key on the turn_id this row allocates). Only a
        # skip_input_row channel (vision, skill_association) leaves turn_id None
        # and lets its end message allocate its own turn.
        if not self._cfg.skip_input_row:
            # The request thread (api.chat._start_turn) pre-writes the input row so
            # the POST can return the allocated turn_id inline; ADOPT it here rather
            # than writing a second row. Channels without that pre-write (delegate
            # synthesis, background passes) still open their own row.
            pre_uid = self._metadata.get("input_uid")
            if pre_uid is not None:
                self.uid = cast("int", pre_uid)
            else:
                self.uid = Transcript.write_input_row(
                    self._cfg.channel, self._cfg.role, self._raw_input,
                    turn_id=self.turn_id,
                )
            if self.turn_id is None:
                self.turn_id = Transcript.turn_id_of_row(self.uid)
            self.anchor = self.uid
            # Announce the turn with ``working`` (spin the state up). It fires
            # unconditionally — a fresh thread and a reply take the identical path;
            # the surface keys on its own current state. There is no ``created``
            # signal: the turn_id rode the POST response body.
            if self.turn_id is not None:
                # Register the cancel handle BEFORE announcing the turn so DELETE
                # /api/thread/<turn_id> is live the instant ``working`` fires.
                # Idempotent — the pre-write path already registered it.
                _register_canceller(self.turn_id, self.cancel_event)
                self.broadcast(self._WS_WORKING_STATE, self.turn_id)

        if self._cfg.channel == "user":
            self._run_thinking_gate()

        self._seed_turn_zero()

        # Label a turn once it first grows into a thread — i.e. on the first user
        # reply past its settle0 (``_forked``). Fire the gist delegate only when no
        # label exists yet, so a thread is summarized exactly once, at birth. The
        # ``schedule`` channel is exempt: its gist lives on the schedule row, not
        # ``thread_gist`` (§13.3), generated by the scheduler from the prompt.
        if self._forked and self.turn_id is not None and self._cfg.channel != "schedule":
            try:
                from services.thread_gist_service import get_thread_gist_service  # noqa: PLC0415
                if not get_thread_gist_service().bulk_get(self._cfg.channel, [self.turn_id]):
                    from services.thread_gist_message_processor import maybe_ingest_gist  # noqa: PLC0415
                    maybe_ingest_gist(self._cfg.channel, self.turn_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[MP] thread gist fire failed (non-fatal): %s", exc)

    def _seed_turn_zero(self) -> None:
        """Framework-issued tool calls fired once before iteration 0."""
        from abilities._dispatcher import ToolDispatcher  # noqa: PLC0415

        dispatcher = ToolDispatcher(self)

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

        # c. High-deliberation thinking pass — programmatic, never model-visible.
        #    Fires LAST so it reasons over the uploaded documents' act-trail rows
        #    already in the parent body (single-pass, tools disabled).
        if getattr(self, "thinking_level", "low") == "high":
            dispatcher.dispatch("thinking", {})

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
            self._cfg.channel, self._cfg.get_user_prompt(self),
            self.turn_id if self._forked else None,
        )
        message: dict[str, object] = {"role": "user", "content": user}
        img = self._cfg.get_image(self)
        if img is not None:
            message["image"] = img
        return [message]

    def _step(self) -> str:
        """One link in the recursive turn chain — exactly one LLM API call."""
        from services.provider_api import RequestOverCapError, ResponseOverLimitError  # noqa: PLC0415

        if self.cancel_event.is_set():
            return ""
        try:
            response = self._send_with_retry(self._build_send_dto())
        except (RequestOverCapError, ResponseOverLimitError):
            self._dispatch_compaction()
            return self._continue(post_compaction=True)._step()
        self.providers._record_send_counters(self)

        formatted = self._store_row(response.text)
        if not response.tool_calls:
            return self._end_turn(formatted)
        self._dispatch_tools(response.tool_calls)
        return self._continue()._step()

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
        from services.transcript_service import Transcript  # noqa: PLC0415

        row_id = Transcript.write_assistant_row(self._cfg.channel, formatted, turn_id=self.turn_id)
        if self.turn_id is None:
            self.turn_id = Transcript.turn_id_of_row(row_id)
        self.anchor = row_id
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

    def _continue(self, *, post_compaction: bool = False) -> "MessageProcessor":
        """Spawn the next link in the chain — a hidden-input continuation MP."""
        child = MessageProcessor(self._raw_input, self._metadata)
        child.config = self._cfg.with_hidden_input()
        child.uid = self.uid
        child.turn_id = self.turn_id
        child.anchor = self.uid
        child.active_tools = self.active_tools            # SAME list
        child._rich_media_ordinals = getattr(self, "_rich_media_ordinals", None)
        child.cancel_event = self.cancel_event            # SAME Event
        child.thinking_level = getattr(self, "thinking_level", "low")
        child.thinking_override = self.thinking_override
        child.providers = self.providers
        if post_compaction:
            from services.transcript_service import Transcript  # noqa: PLC0415
            child.post_compaction_continuation = True
            child.continuation_user_query = Transcript.latest_input_content(self._cfg.channel)
            if "review_transcript" not in child.active_tools:
                child.active_tools.append("review_transcript")
        return child

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
        """The LLM ``previous_messages`` row set for this turn's view — and the hub
        of the read+compaction flow. Read this whole docstring before adding any
        new read path; no single function here is correct in isolation — they only
        make sense composed:

          one getter      ``Transcript.by_turn(channel, turn_id=None)`` —
                          turn_id → that turn's rows (a thread); None → the whole
                          channel spine in ONE query. No filtering; raw rows.
          consumers       mutate the rows HERE, in Python (never in SQL): keep rows
                          ABOVE the watermark; MAIN also drops each turn's rows past
                          settle0 via ``Transcript.drop_post_settle0``.
          watermark       ``compaction_persistence.get_compaction(channel,
                          for_turn_id)`` — two axes that never collide: MAIN
                          (for_turn_id=None) is a *turn_id*; FORK (for_turn_id=T) is
                          a *transcript.id* within T.
          summary ⊕ tail  the rows returned here are the live tail ABOVE the
                          watermark; everything BELOW it is already carried as prose
                          by the checkpoint summary that ``_wrap_with_checkpoint``
                          prepends. Model context = summary(≤wm) ⊕ these rows(>wm).
          writer          ``chat_history_compactor.run`` folds *these* rows into the
                          summary and advances the watermark to their max id/turn_id.
                          That "always advances" guarantee is WHY this read keeps
                          id>wm and never id≤wm: keep id≤wm and the compactor re-folds
                          already-folded rows and livelocks.

        THREAD (``self._forked`` + turn_id): that turn's rows above its transcript-id
        watermark, minus the live reply's own rows (``id >= self.uid``, which render
        separately). MAIN: the whole channel spine, dropping turns at/below the main
        watermark (a turn_id) and every row past each turn's settle0. The watermark
        is the ONLY spine bound — no turn-count cap (it would silently drop
        un-summarised turns).

        Forbidden — alternate paths tried and rejected; do NOT reintroduce them:
        a second getter (e.g. assembling the spine from ``recent_threads`` + a
        per-turn ``by_turn`` loop — ``recent_threads`` is for the threads feed
        only); pushing the watermark/settle0 filter into SQL or getter params;
        capping the turn count (the watermark is the sole spine bound; a cap
        silently drops un-summarised turns); or inverting the comparator to keep
        rows at/below the watermark."""
        if self._cfg.suppress_history:
            return []
        from services import compaction_persistence  # noqa: PLC0415
        from services.transcript_service import Transcript  # noqa: PLC0415
        channel = self._cfg.channel
        if self._forked and self.turn_id is not None:
            compaction = compaction_persistence.get_compaction(channel, self.turn_id)
            wm = cast("int", compaction["compacted_up_to"]) if compaction else 0
            upper = self.uid if self.uid is not None else 9223372036854775807
            return [r for r in Transcript.by_turn(channel, self.turn_id) if wm < int(cast("int", r["id"])) < upper]
        compaction = compaction_persistence.get_compaction(channel, None)
        wm = cast("int", compaction["compacted_up_to"]) if compaction else 0
        rows = [r for r in Transcript.by_turn(channel)
                if r["turn_id"] is not None and int(cast("int", r["turn_id"])) > wm]
        return Transcript.drop_post_settle0(rows, channel)

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
        """Assemble the current turn's act-trail — its tool calls, in record order."""
        if self.turn_id is None:
            return ""
        from services.act_trail import ActTrail  # noqa: PLC0415

        trail = ActTrail()
        lines: list[str] = []
        for row in trail.fetch_by_turn(self._cfg.channel, self.turn_id):
            if row.get("tool_name") == "chat_history_compactor":
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
