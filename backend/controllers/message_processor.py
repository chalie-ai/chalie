# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MessageProcessor — the single MP-spine orchestrator (§1.1).

The one class that drives an LLM turn end-to-end. Every rule the spine is built
on lands here:

* **Rule 1** — this is the *sole* orchestrator; nothing else sequences a turn.
* **Rule 2** — the constructor only wires instances and flags (Critical 2 / I2:
  zero DB, zero WS, zero side-effects at construction — the ctor is inert).
* **Rules 3–4** — every LLM-tracking coordinating service is constructed with
  ``self`` and only ever reaches another service through ``self.mp.<service>``;
  this controller is the hub, the services are spokes, and no spoke talks to
  another spoke directly.
* **Rules 5–9** — every DB-stored, LLM-touching object is a model reached through
  a service (SQL only in Model/Query), every external effect (WS emit) fires from
  those services, and this controller never emits WS itself. The terminal-state
  frame and (on a crash) the user-facing crash toast are both broadcast by
  :meth:`TurnExecutionService.finish`, not from here.

**The process/begin/_step contract (Dylan's ruling, §4.3):**

``process()`` (the single public entrypoint, Critical 1) constructs the inert
instance, calls ``begin()``, then returns the instance. ``begin()`` runs the
synchronous side-effects — atomic per-channel turn-id allocation + input row +
execution-row open, all inside one single-writer ``BEGIN IMMEDIATE`` transaction
(§6.8) — then fires a daemon thread running ``_drive`` and returns immediately
(it does **not** join). Because the execution row is opened synchronously before
the thread spawns, ``mp.execution`` is populated the moment ``process()``
returns: a POST handler reads the WORKING execution handle instantly and live
output then streams over WS, while fire-and-forget callers (scheduler) read the
allocated id off ``mp.metadata``.

The thinking override is resolved per-turn in ``_setup`` (after ``begin()`` has
allocated ``turn_id`` and written the input row) by reading the transcript's
``thinking_level`` column for the current (channel, turn_id); ``medium``/``high``
becomes the override, ``auto``/NULL leaves it unset so the deliberation gate
decides normally.

``_step()`` is the **recursive** step loop: send → (compact-and-continue on
over-cap) → store any prose → dispatch tool calls → recurse; it bottoms out when
the model returns a turn with no tool calls. The first ``_step()`` is isolated onto
the daemon thread by ``begin()``; every recursive call is synchronous within that
thread. Synchronous text-consumers (delegate abilities, mcp_server,
skill_association) call ``process()`` then ``result()`` to join the thread and
read the final text.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from threading import Thread
from typing import TYPE_CHECKING, Protocol, cast

from configs.channels.user import UserConfig
from configs.enums.channels import Channel
from configs.enums.provider_type import ProviderType
from configs.enums.thinking_level import ThinkingLevel
from exceptions import (
    ContextLimit,
    EmptyCompletionLoop,
    ProviderResponseError,
    ProviderRetriesExhaustedError,
    RunAwayLoop,
)
from models.provider_request import ProviderRequest
from models.turn_execution import TurnExecution
from models.transcript import Transcript
from models.transcript_thinking import TranscriptThinking
from services.behavioral_pattern_service import BehavioralPatternService
from services.compaction_service import CompactionService
from services.database import Database
from services.dispatch_service import DispatchService
from services.gist_service import GistService
from services.llm_log_service import LlmLogService
from services.llm_service import estimate_tokens
from services.prompt_service import PromptService
from services.provider_service import ProviderService
from services.tool_call_service import ToolCallService
from services.transcript_service import TranscriptService
from services.turn_execution_service import TurnExecutionService
from services.user_synthesis import UserSynthesis
from services.websocket import Websocket

if TYPE_CHECKING:
    from contracts.json_serializable import JsonSerializable
    from models.provider_response import ProviderResponse
    from services.processor_config import ProcessorConfig

logger = logging.getLogger(__name__)

#: Provider resend budget — three attempts before a turn surfaces as failed.
_MAX_PROVIDER_ATTEMPTS = 3
#: Real tool-calls a ``user`` turn must make before a proactive-skill scan fires.
_PROACTIVE_SUGGESTION_MIN_CALLS = 4

#: Runaway-loop thresholds. A turn has no iteration cap, so a model stuck
#: re-emitting the same tool call or the same prose would loop until the context
#: caps out. The same (tool, params) invoked this many times, or the same
#: non-empty response text emitted this many times, within one turn_execution
#: trips a loud ``RunAwayLoop`` (a CRASHED turn).
_RUNAWAY_TOOL_CALL_LIMIT = 5
_RUNAWAY_TEXT_LIMIT = 3

#: Empty-completion steers granted before the turn trips ``EmptyCompletionLoop``
#: (a CRASHED turn): a completion with no tool calls and no text, on a turn with
#: no tool activity at all, is steered into a retry — the steer text enters the
#: next request body, since without it the re-send would be byte-identical.
_EMPTY_COMPLETION_STEER_LIMIT = 2
_EMPTY_COMPLETION_STEER = (
    "Your previous reply was empty — no message and no tool calls. An empty "
    "reply is never a valid answer. Address the user's request now: call the "
    "tools you need, or write your reply."
)

#: Consecutive ``ContextLimit`` hits a single turn may recover from before the
#: error is allowed out. Compaction is the only lever, so a hit that survives it
#: will survive the next one too — recursing past this is an infinite loop, not
#: persistence.
_CONTEXT_LIMIT_RECOVERY_LIMIT = 3


class _TurnCancelled(Exception):
    """Raised internally when a cancel request is observed mid-turn; caught by
    ``_drive`` to stamp the execution CANCELLED (never leaves this module)."""


class _ExternalAgentConfig(Protocol):
    """The external-agent config surface ``_disclose_to_human`` reads — declared
    structurally so the loop-in-human branch stays typed without importing the
    concrete config (Rule 4: the controller depends on shape, not identity)."""

    _agent_name: str
    _project: str
    _loop_in_human: bool


class MessageProcessor:
    """Drives one LLM turn (§1.1). Constructed inert; run via :meth:`process`."""

    def __init__(
        self,
        config: ProcessorConfig,
        turn_id: int = -1,
        raw_input: str = "",
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Wire instances and flags only — no DB, no WS, no side-effects (I2).

        ``turn_id`` is the caller's request: ``-1`` opens a fresh MAIN turn;
        a real id (``>= 0``) forks a reply into that existing turn. The
        allocated id is resolved later, synchronously, inside :meth:`begin`."""
        self.config = config
        self.raw_input = raw_input
        self.metadata: dict[str, object] = metadata if metadata is not None else {}
        self.channel: str = config.channel
        self.turn_id: int = turn_id
        self._forked: bool = turn_id != -1

        # Per-turn state, populated synchronously in begin() / on the drive thread.
        self.uid: int | None = None
        self.current_transcript_id: int | None = None
        self.execution: TurnExecution | None = None
        self.active_tools: list[str] = []
        self.thinking_level: str = "low"
        self.thinking_override: str | None = None
        self.turn_handover: str = ""
        self.seeding_turn_zero: bool = False
        self._placed_attachments: list[str] = []
        self._trigger_channel: str | None = cast("str | None", self.metadata.get("trigger_channel"))
        self._trigger_turn_id: int | None = cast("int | None", self.metadata.get("trigger_turn_id"))
        self._thread: Thread | None = None
        self._result_text: str = ""

        # Runaway-loop guard tallies — scoped to this turn_execution (this
        # instance persists across the whole recursive _step chain). Keyed by
        # identical tool call ``(name, canonical-params)`` and by identical
        # (stripped, non-empty) response text. ``_invocation_key`` is the shared
        # key constructor; ``repeat_call_count`` is the read path used by
        # DispatchService to steer from the second identical call;
        # ``_guard_runaway`` is the only writer.
        self._tool_invocations: Counter[tuple[str, str]] = Counter()
        self._text_emissions: Counter[str] = Counter()

        # Empty-completion guard — scoped to this turn_execution like the tallies
        # above: counts completions with no tool calls and no text on a turn with
        # no tool activity at all; the flag arms a one-shot steer for the next
        # _build_messages.
        self._empty_completions: int = 0
        self._empty_completion_steer: bool = False
        # Reset by every send that gets through, so it counts consecutive
        # failures to fit rather than a turn's lifetime total — a long turn may
        # legitimately outgrow the window more than once as tools add output.
        self._context_limit_hits: int = 0

        # Infrastructure handles.
        self.db = Database()

        # Coordinating services (Rule 3: each holds this mp; Rule 4: they reach
        # one another only through self.mp.<service>).
        self.transcript_service = TranscriptService(self)
        self.tool_call_service = ToolCallService(self)
        self.turn_execution_service = TurnExecutionService(self)
        self.provider_service = ProviderService(self)
        self.compaction_service = CompactionService(self)
        self.behavioral_pattern_service = BehavioralPatternService(self)
        self.dispatch_service = DispatchService(self)
        self.llm_log_service = LlmLogService(self)
        self.prompt_service = PromptService(self)
        self.gist_service = GistService(self)

    # ── websocket ────────────────────────────────────────────────────────────────

    def push_websocket(self, instance: "JsonSerializable | None") -> None:
        """Emit ``instance`` to the frontend, gated by this turn's config — the one
        pre-flight every spine service shares, so no emit site re-inlines it. A
        turn broadcasts only when its config opts in (``BROADCASTS_STATE`` — the
        ``user`` and ``scheduled`` types) AND carries an addressable routing type;
        an internal channel (``type_value() is None``) never reaches the wire. Past
        the gate this is a thin pass to ``Websocket.broadcast`` (itself a no-op on
        ``None`` and when nobody is listening)."""
        if not self.config.BROADCASTS_STATE or self.config.type_value() is None:
            return
        Websocket.broadcast(instance)

    # ── entrypoint ─────────────────────────────────────────────────────────────

    @classmethod
    def process(
        cls,
        config: ProcessorConfig,
        raw_input: str = "",
        metadata: dict[str, object] | None = None,
        turn_id: int = -1,
    ) -> MessageProcessor:
        """The single public entrypoint (Critical 1): construct inert, resolve
        the thinking override per-turn in ``_setup`` (read from the transcript
        row written by ``begin``), kick off ``begin()`` (which spawns the drive
        thread), and return the live instance."""
        mp = cls(config, turn_id=turn_id, raw_input=raw_input, metadata=metadata)
        mp.begin()
        return mp

    def begin(self) -> None:
        """Synchronous turn setup, then hand off to the drive thread. The
        turn-id allocation, fork guard and input row land in one single-writer
        transaction (§6.8). Attachments land *after* that transaction commits
        and *before* the execution row opens: after, because copying a file of
        arbitrary size inside ``BEGIN IMMEDIATE`` would hold SQLite's write
        lock for the length of the copy; before, because the ``working`` frame
        ``open()`` emits is the cue clients refetch on — a turn announced
        before its ``transcript_files`` rows exist renders without its
        attachments and nothing signals them again (turn-zero seeding is WS-
        silent). Then the execution row opens so ``mp.execution`` is set before
        the thread starts. ``begin()`` never joins — it returns the instant the
        daemon is running."""
        with self.db.transaction():
            self.turn_id = self.transcript_service.allocate_turn()
            if self.config.external_turn_id:
                # turn_id came from a key space the caller owns (schedule id). Its first
                # use opens a MAIN turn; a repeat appends as FORK. Derive forked-ness from
                # existence, never reject — a fresh external key is legitimately new.
                self._forked = self.transcript_service.turn_exists()
            elif self._forked and not self.transcript_service.turn_exists():
                raise ValueError("Invalid turn_id specified")
            self.uid = self._open_input_row()
            self.current_transcript_id = self.uid
        self._land_attachments()
        self.turn_execution_service.open()
        self.metadata["turn_id"] = self.turn_id
        self._thread = Thread(target=self._drive, daemon=True, name=f"turn-{self.turn_id}")
        self._thread.start()

    def _open_input_row(self) -> int | None:
        """Write this turn's anchoring input row and return its id — unless the
        config skips it (channels whose input is not a transcript utterance)."""
        if self.config.skip_input_row:
            return None
        raw_level = self.metadata.get("thinking_level")
        return self.transcript_service.append_input(
            self.raw_input, thinking_level=raw_level if isinstance(raw_level, str) else None,
        )

    def result(self) -> str:
        """Join the drive thread and return the turn's final text — the
        synchronous read for text-consuming callers (delegate abilities,
        skill-association). Fire-and-forget callers simply ignore it."""
        if self._thread is not None:
            self._thread.join()
        return self._result_text

    # ── the drive thread ───────────────────────────────────────────────────────

    def _drive(self) -> None:
        """The daemon-thread body: run setup + the recursive step loop, then
        stamp the terminal execution state. This is the ONLY place a turn's
        terminal state is written — COMPLETED on a clean return, CANCELLED on a
        mid-turn stop, CRASHED (with the reason) on any other exception. The WS
        lifecycle frame for that terminal state — and, for CRASHED, the
        user-facing crash toast — both fire inside ``finish`` (Rule 7)."""
        try:
            self._setup()
            self._result_text = self._step()
            self.turn_execution_service.finish(TurnExecution.COMPLETED)
        except _TurnCancelled:
            self.turn_execution_service.finish(TurnExecution.CANCELLED)
        except Exception as exc:  # noqa: BLE001 — the drive thread is the last line of defence
            logger.exception("[MessageProcessor] turn %s crashed", self.turn_id)
            self.turn_execution_service.finish(TurnExecution.CRASHED, str(exc))

    def _step(self) -> str:
        """One provider step, recursing until the model stops calling tools.

        Send → on ``ContextLimit``, compact and continue (re-enter with the
        transcript reviewer armed; a payload that will not shrink is let out
        after ``_CONTEXT_LIMIT_RECOVERY_LIMIT`` attempts rather than spun on)
        → store any prose the model emitted → if it made no tool
        calls the turn is done (end); otherwise dispatch the calls and recurse.
        A cancel observed at the top of any step aborts the whole turn. Every
        provider client is a single blocking, non-streaming call (§ llm_clients/*)
        with no mid-flight abort hook, so a cancel requested while that call is
        in flight can only be observed once it returns — the checkpoint right
        after it, BEFORE the response is stored, is what makes that observation
        count: without it a cancel that lands mid-generation is silently
        ignored and the full response is persisted and rendered as if the turn
        had completed normally."""
        if self.turn_execution_service.should_stop():
            raise _TurnCancelled()
        try:
            response = self._send_with_retry(self._build_request())
        except ContextLimit as limit:
            # Compaction is the only lever here, so a hit that survives it is
            # not a retry — it is a request that cannot be made to fit (one
            # oversized turn, a window smaller than the prompt itself). Let it
            # out rather than recursing forever on an unshrinkable payload.
            self._context_limit_hits += 1
            if self._context_limit_hits > _CONTEXT_LIMIT_RECOVERY_LIMIT:
                raise
            limit.recover()
            return self._step()
        self._context_limit_hits = 0
        if self.turn_execution_service.should_stop():
            raise _TurnCancelled()
        tool_calls = response.tool_calls
        if not tool_calls:
            if not (response.text or "").strip() and not self._tool_invocations:
                # The model did NOTHING this turn: no text, no tool calls now,
                # and no tool activity on any earlier step (the tally is
                # populated by every tool-bearing step). Settling would store
                # an empty assistant row and render silence as an answered
                # turn. A turn that DID run tools may still finish silently —
                # background channels end that way by design.
                # A thinking-only response still carries evidence: persist its
                # trace (before the loop guard can raise) so the steered retry
                # re-reads it via the act trail instead of re-deriving blind.
                self._capture_thinking_trace(response)
                self._empty_completions += 1
                if self._empty_completions > _EMPTY_COMPLETION_STEER_LIMIT:
                    raise EmptyCompletionLoop(
                        f"turn {self.turn_id}: model returned "
                        f"{self._empty_completions} empty completions (no text, "
                        f"no tool calls) — refusing to settle an unanswered turn",
                    )
                logger.warning(
                    "[MessageProcessor] turn %s: empty completion %s/%s — steering a retry",
                    self.turn_id, self._empty_completions, _EMPTY_COMPLETION_STEER_LIMIT,
                )
                self._empty_completion_steer = True
                return self._step()
            formatted = self._store(response.text)
            self._capture_thinking_trace(response)
            self._end(response.text)
            return formatted
        self._guard_runaway(response.text, tool_calls)
        if response.text:
            self._store(response.text)
        self._capture_thinking_trace(response)
        self._dispatch_tools(tool_calls)
        return self._step()

    # ── provider send ──────────────────────────────────────────────────────────

    def _send_with_retry(self, request: ProviderRequest) -> ProviderResponse:
        """Send with the turn's resend budget. ``ContextLimit`` is re-raised
        untouched — resending an oversized request unchanged just fails again,
        so the step loop compacts and continues instead; every other
        provider failure is retried up to ``_MAX_PROVIDER_ATTEMPTS``, emitting a
        user-facing retry notice between attempts, and a cancel observed
        mid-retry aborts the turn. Exhausting the budget raises a clean,
        surfaceable ``ProviderRetriesExhaustedError``."""
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_PROVIDER_ATTEMPTS + 1):
            try:
                return self.provider_service.send(request)
            except ContextLimit:
                raise
            except Exception as exc:  # noqa: BLE001 — resend policy lives here, not in the provider
                last_exc = exc
                if self.turn_execution_service.should_stop():
                    raise _TurnCancelled() from exc
                logger.warning(
                    "[MessageProcessor] provider attempt %s/%s failed: %s",
                    attempt, _MAX_PROVIDER_ATTEMPTS, exc,
                )
                if attempt < _MAX_PROVIDER_ATTEMPTS:
                    self.provider_service.emit_retry(
                        attempt + 1, _MAX_PROVIDER_ATTEMPTS,
                        "The AI provider had a problem — retrying…",
                    )
        provider = last_exc.provider if isinstance(last_exc, ProviderResponseError) else ""
        raise ProviderRetriesExhaustedError(
            "The AI provider failed to respond after several attempts. Please try again in a moment.",
            provider=provider,
        ) from last_exc

    def _build_request(self) -> ProviderRequest:
        """Assemble this step's provider-neutral request off the prompt and tool
        services (§4). The system prompt already carries the async guidance the
        config warrants — this controller does not append it."""
        from abilities._registry import AbilityRegistry  # noqa: PLC0415
        tools = AbilityRegistry.build_tools(self)
        thinking_str = self.provider_service.resolve_thinking_mode()
        try:
            level = ThinkingLevel(thinking_str or "low")
        except ValueError:
            level = ThinkingLevel.LOW
        return ProviderRequest(
            system=self.prompt_service.system_prompt(),
            messages=self._build_messages(),
            type=self._provider_type(),
            tools=tools or None,
            thinking_mode=level,
            cache_prefix=True,
        )

    def _provider_type(self) -> ProviderType:
        """Route to the vision/delegate/chat provider lane by config (precedence
        vision > delegate > chat)."""
        if self.config.uses_vision_provider:
            return ProviderType.VISION
        if self.config.uses_delegate_provider:
            return ProviderType.DELEGATE
        return ProviderType.CHAT

    def _build_messages(self) -> list[dict[str, object]]:
        """The single user message for this step: the assembled user prompt
        (history + tool-result re-feed live inside ``prompt_service``) wrapped by
        the compaction checkpoint, plus this turn's image when one is attached."""
        body = self.compaction_service.checkpoint(self.prompt_service.user_prompt())
        if self._empty_completion_steer:
            self._empty_completion_steer = False
            body += "\n\n" + _EMPTY_COMPLETION_STEER
        message: dict[str, object] = {"role": "user", "content": body}
        image = self.prompt_service.image()
        if image is not None:
            message["image"] = image
        return [message]

    # ── transcript writes ──────────────────────────────────────────────────────

    def _store(self, text: str) -> str:
        """Format the model's prose for its channel and, unless the config skips
        the transcript, append it as this turn's assistant row (which pokes open
        surfaces itself). Returns the formatted text."""
        formatted = self._format(text or "")
        if self.config.skip_transcript:
            return formatted
        self.current_transcript_id = self.transcript_service.append_assistant(formatted)
        return formatted

    def _capture_thinking_trace(self, response: "ProviderResponse") -> None:
        """Persist one ``transcript_thinking`` row when the provider returned a
        non-empty ``thinking_block``. Skips entirely when ``skip_transcript`` is
        set (same gate as ``_store`` — those channels have no transcript anchor).
        The trace is captured after the cancel checkpoint and after any prose
        storage for this response, so the anchor is fresh:
        ``current_transcript_id if set else uid`` — exactly the rule
        ``ToolCallService._transcript_id`` uses. A settled response anchors to
        its own stored row; a tool-calls-only response (no prose) anchors to the
        prior anchor, same as its tool calls."""
        if self.config.skip_transcript:
            return
        trace = response.thinking_block
        if not trace:
            return
        transcript_id = (
            self.current_transcript_id
            if self.current_transcript_id is not None
            else self.uid
        )
        if transcript_id is None:
            return
        duration_ms = response.latency_ms or 0
        tokens = (
            response.tokens_thinking
            if response.tokens_thinking
            else estimate_tokens(trace)
        )
        TranscriptThinking.insert(transcript_id, trace, duration_ms, tokens)

    def _format(self, text: str) -> str:
        """Render markdown to HTML for surface-broadcasting channels; pass raw
        text through for background/silent channels (``RENDERS_HTML`` False).
        HTML branch is sanitized at the persist-time boundary so both the live
        WS send and the GET/refresh read paths inherit it."""
        if self.config.RENDERS_HTML:
            from services.markup import markdown_to_html, sanitize  # noqa: PLC0415
            return sanitize(markdown_to_html(text))
        return text or ""

    @staticmethod
    def _invocation_key(name: str, canonical: dict[str, object]) -> tuple[str, str]:
        """The guard's per-tool tally key: ``(name, canonical-params-json)``.

        Keyed on the CANONICAL (sanitised + key-healed) params dispatch will
        actually execute, not the raw provider args: a model cycling synonym
        keys (city/loc/place/region → location) would otherwise mint a fresh
        key each step while running byte-identical calls, evading the tally.
        ``canonical`` must therefore be the output of
        ``DispatchService.canonical_params`` — this is the sole key constructor,
        shared by the writer (``_guard_runaway``) and the reader
        (``DispatchService._repeat_call_steer`` via ``repeat_call_count``)."""
        return (name, json.dumps(canonical, sort_keys=True, default=str))

    def repeat_call_count(self, tool_name: str, canonical_params: dict[str, object]) -> int:
        """How many times this tool has been invoked with these exact canonical
        params this turn.

        Read by :meth:`~services.dispatch_service.DispatchService._repeat_call_steer`
        to steer the model from the second identical call. The count was tallied
        by :meth:`_guard_runaway` BEFORE dispatch, so a call that has not yet
        dispatched has not yet been counted. Dispatches bypassing ``_step``
        (compactor, document upload, async delegate's dedicated mp) have zero
        tally and never steer here."""
        return self._tool_invocations[self._invocation_key(tool_name, canonical_params)]

    def _guard_runaway(self, text: str, tool_calls: list[dict[str, object]]) -> None:
        """Trip a loud ``RunAwayLoop`` when this turn's step chain is repeating
        itself instead of converging — the hard backstop for the uncapped loop.

        Called on every tool-bearing (recursing) step BEFORE the calls are stored
        or dispatched, and NEVER on the terminal (no-tool) step, so a clean final
        answer that echoes earlier prose is not mistaken for a loop. Two
        independent tallies, both scoped to this turn_execution (this instance
        persists across the whole recursive ``_step`` chain, including inline
        post-compaction continuations): the same ``(tool, canonical params)``
        invoked ``_RUNAWAY_TOOL_CALL_LIMIT`` times, or the same non-empty response
        text emitted ``_RUNAWAY_TEXT_LIMIT`` times. The tool tally keys on the
        canonical params ``DispatchService`` will execute (sanitised + key-healed),
        so cycling synonym keys that heal to one identical call cannot evade it.
        ``_drive`` catches the raise and stamps the turn CRASHED (Rule 7)."""
        stripped = (text or "").strip()
        if stripped:
            self._text_emissions[stripped] += 1
            if self._text_emissions[stripped] >= _RUNAWAY_TEXT_LIMIT:
                raise RunAwayLoop(
                    f"turn {self.turn_id}: identical response text emitted "
                    f"{self._text_emissions[stripped]} times — the model is looping",
                )
        for call in tool_calls:
            name = cast("str", call["name"])
            canonical = self.dispatch_service.canonical_params(
                name, cast("dict[str, object]", call.get("input") or {}),
            )
            key = self._invocation_key(name, canonical)
            self._tool_invocations[key] += 1
            if self._tool_invocations[key] >= _RUNAWAY_TOOL_CALL_LIMIT:
                raise RunAwayLoop(
                    f"turn {self.turn_id}: tool {name!r} invoked with identical "
                    f"parameters {self._tool_invocations[key]} times — the model is looping",
                )

    def _dispatch_tools(self, tool_calls: list[dict[str, object]]) -> None:
        """Run each tool call in order, checking for a cancel before every one so
        a stop lands between tools rather than mid-turn."""
        for call in tool_calls:
            if self.turn_execution_service.should_stop():
                raise _TurnCancelled()
            self.dispatch_service.dispatch(
                cast("str", call["name"]), cast("dict[str, object]", call["input"]),
            )

    # ── turn end ───────────────────────────────────────────────────────────────

    def _end(self, response_text: str) -> str:
        """The terminal (no-tool) step: run the channel's post-turn work and the
        episodic check, then return the raw response text. A cancel observed here
        still aborts before any post-turn side-effect."""
        if self.turn_execution_service.should_stop():
            raise _TurnCancelled()
        self._post_turn(response_text)
        from services.episodic_service import EpisodicService  # noqa: PLC0415
        EpisodicService().check_and_store(self.config)
        return response_text

    def _post_turn(self, response_text: str) -> None:
        """Dispatch this config's post-turn handler (§4.2), keyed on its stable
        transcript ``role``. The proactive-suggestion handler additionally
        requires the genuine ``user`` channel: DiscoveryConfig and
        ScheduledConfig also carry ``role='user'`` but write to their own
        channels, and neither takes the proactive-suggestion path (discovery is
        a silent loop — ``RENDERS_HTML`` is False, so a suggestion would have
        nowhere to surface; scheduled self-surfaces in its own thread and the
        scheduler dock, with no user-channel relay, §13.9). Every handler is
        isolated — a failure is logged, never propagated, so post-turn work can
        never fail an otherwise-complete turn."""
        role = self.config.role
        try:
            if role == "user" and self.channel == Channel.USER:
                self._voice_presynthesis()
                self._proactive_suggestion()
            elif role == "user_summary":
                UserSynthesis.persist_user_summary(response_text)
            elif role == "pattern_match":
                self._pattern_skill_sync()
            elif role == "external_agent":
                self._disclose_to_human(response_text)
        except Exception as exc:  # noqa: BLE001 — post-turn work must never fail the turn
            logger.warning("[post_turn] %s handler failed (isolated): %s", role, exc)

    def _voice_presynthesis(self) -> None:
        """Kick background speech pre-synthesis for this turn's settled row on a
        fire-and-forget daemon thread — the pipeline owns every gate and terminal
        state, and running it inline would delay the turn-complete frame the
        frontend waits on (``finish(COMPLETED)`` stamps after post-turn work)."""
        settle_id = Transcript.settle0(self.channel, self.turn_id)
        if settle_id is None:
            return
        from services.voice_transcript_service import VoiceTranscriptService  # noqa: PLC0415
        Thread(
            target=VoiceTranscriptService.instance().synthesize_settled,
            args=(settle_id,), daemon=True,
        ).start()

    def _proactive_suggestion(self) -> None:
        """After a ``user`` turn that made enough real tool calls, hand the act
        trail to the skill-suggestion pass. Compaction calls do not count toward
        the threshold."""
        count = sum(1 for c in self.tool_call_service.by_turn() if c.tool_name != "chat_history_compactor")
        if count < _PROACTIVE_SUGGESTION_MIN_CALLS:
            return
        rendered = self.prompt_service.act_trail()
        act_trail = rendered.split("\n") if rendered else []
        from services.skill_suggestion_message_processor import maybe_suggest_skill  # noqa: PLC0415
        maybe_suggest_skill(act_trail, self.raw_input, self.channel, self.turn_id)

    def _pattern_skill_sync(self) -> None:
        """Decay untouched patterns, then run the skill-association pass over the
        patterns written this turn. An empty touched set is NOT a no-op for the
        decay half: nothing is exempt, so every live pattern row decays — the
        intended sweep for a turn that wrote no patterns, not an edge case. The
        association pass does skip on empty."""
        touched = self._touched_pattern_names()
        self.behavioral_pattern_service.decay_untouched(touched)
        from services.skill_association_service import SkillAssociationService  # noqa: PLC0415
        SkillAssociationService().run_pass(self.behavioral_pattern_service.ids_for_touched(touched))

    def _touched_pattern_names(self) -> set[str]:
        """Names of the patterns saved this turn — the ``save_pattern`` calls'
        ``name`` params, skipping any malformed row."""
        names: set[str] = set()
        for call in self.tool_call_service.by_turn():
            if call.tool_name != "save_pattern":
                continue
            try:
                params = json.loads(call.params)
            except (json.JSONDecodeError, TypeError):
                continue
            name = params.get("name") if isinstance(params, dict) else None
            if isinstance(name, str) and name:
                names.add(name)
        return names

    def _disclose_to_human(self, response_text: str) -> None:
        """When an external-agent config opts into looping in the human, open a
        fresh hidden ``user`` turn narrating the exchange. A no-op for agents
        that do not."""
        config = cast("_ExternalAgentConfig", self.config)
        if not config._loop_in_human:
            return
        disclosure_input = (
            f"An external agent called '{config._agent_name}' just contacted you "
            f"about '{config._project}'. "
            f"Here's what they said:\n\n\"{self.raw_input}\"\n\n"
            f"You replied:\n\n\"{response_text}\"\n\n"
            "Let the user know about this exchange in your own words."
        )
        MessageProcessor.process(
            UserConfig({"hidden_input": True}),
            raw_input=disclosure_input,
            metadata={"hidden_input": True},
        )

    # ── setup (on the drive thread, before the first send) ─────────────────────

    def _setup(self) -> None:
        """Pre-loop setup: seed the always-available tools, gate thinking on the
        ``user`` channel, seed the turn-zero flashback + any attachments, and
        ingest the reply's parent gist on a fork."""
        self.active_tools = list(self.config.always_available or [])
        # Resolve thinking override from the transcript row written by begin()
        # (the current turn's input row). medium/high becomes the override,
        # auto/NULL leaves it unset so the deliberation gate decides normally.
        level = Transcript.latest_thinking_level(self.channel, self.turn_id)
        self.thinking_override = level if level in ("medium", "high") else None
        if self.channel == Channel.USER:
            self._run_thinking_gate()
        self._seed_turn_zero()
        self._maybe_fire_gist()

    def _run_thinking_gate(self) -> None:
        """Pick this turn's thinking level: an explicit override wins outright;
        otherwise classify the input, fold it into the deliberation EMA to pick a
        bucket, and persist the raw score. Any classifier failure falls back to
        ``low`` — thinking selection can never fail a turn."""
        if self.thinking_override:
            self.thinking_level = self.thinking_override
            return
        try:
            from services.deliberation_score_service import DeliberationScoreService  # noqa: PLC0415
            from services.deliberation_ema_service import DeliberationEmaService  # noqa: PLC0415
            scalar = DeliberationScoreService().classify(self.raw_input)
            if scalar is None:
                self.thinking_level = "low"
                return
            _, bucket = DeliberationEmaService().update_and_bucket(scalar)
            self.thinking_level = bucket
            self.transcript_service.set_deliberation_score(scalar)
        except Exception:  # noqa: BLE001 — thinking selection must never fail the turn
            logger.exception("[MessageProcessor] thinking gate failed — defaulting to low")
            self.thinking_level = "low"

    def _seed_turn_zero(self) -> None:
        """Fire the turn-0 memory recall (when the config opts in) and upload any
        attachments in parallel before the first send. The recall dispatches like a
        normal tool call — ``_auto`` marks it the framework seed (``caller='seed'``
        in ``handle_recall``); the recorded row grounds the model's first request
        and is turn-scoped, so it drops at the next turn. ``PromptService.act_trail``
        renders it as a ``[background_memory]`` block, not a tool-call row.

        ``seeding_turn_zero`` is raised for the duration: seed dispatches are
        recorded on the trail but never surface a live pill (§6.10 — the WS
        silence lives in ``ToolCallService._emit``)."""
        self.seeding_turn_zero = True
        try:
            if self.config.memory_seed:
                self.dispatch_service.dispatch(
                    "recall", {"query": self.raw_input, "_auto": True}
                )
            attachments = list(self._placed_attachments)
            if attachments:
                from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
                with ThreadPoolExecutor(max_workers=min(len(attachments), 8)) as pool:
                    list(pool.map(self._seed_upload_attachment, attachments))
        finally:
            self.seeding_turn_zero = False

    def _land_attachments(self) -> None:
        """For each pending attachment: copy into uploads/, write the
        transcript_files link row, and remove the tmp staging file.

        Runs synchronously inside ``begin`` — after its write transaction has
        committed (a file copy must never hold ``BEGIN IMMEDIATE``'s write
        lock) and before ``turn_execution_service.open()`` emits ``working``,
        so a client refetching on that cue already sees the linked rows. Each
        ``link()`` commits on its own; the connection autocommits outside a
        transaction. Extraction and the MIME dispatch stay on the drive thread
        via ``_seed_upload_attachment`` — a per-image vision call would
        otherwise block the request for seconds.

        Sandbox-checked: only real paths inside ``TMP_PATH_PREFIX`` that
        exist on disk are accepted; everything else is refused with a warn.
        Links are skipped for ``skip_input_row`` configs (uid is None).
        """
        import os

        from services.tmp_storage import TMP_PATH_PREFIX
        from services.file_mapper_service import FileMapperService
        from services.file_parser_service import FileParserService
        from models.transcript_file import TranscriptFile

        raw_paths = cast("list[str]", self.metadata.get("attachments") or [])
        for path in raw_paths:
            real = os.path.realpath(path)
            if not real.startswith(TMP_PATH_PREFIX) or not os.path.isfile(real):
                logger.warning("[MessageProcessor] refusing attachment outside tmp sandbox: %s", path)
                continue
            basename = re.sub(r"^chalie_([0-9a-f]{8}_)?", "", os.path.basename(real)) or os.path.basename(real)
            saved_path = FileParserService().place(real, name=basename, subdir="uploads")
            # Copy succeeded (FileParserService.place copies the file, never moves it).
            # Remove the tmp staging file.
            try:
                os.unlink(real)
            except OSError:
                logger.warning("[MessageProcessor] staged attachment not cleaned up: %s", real)
            self._placed_attachments.append(saved_path)
            if self.uid is not None:
                relpath = os.path.relpath(saved_path, FileMapperService.get_documents_path())
                TranscriptFile(transcript_id=self.uid, path=relpath).link()

    def _seed_upload_attachment(self, saved_path: str) -> None:
        """Index the previously-placed attachment and dispatch by MIME.

        The file is assumed to already live in the documents store and the
        transcript_files link row is already written (performed synchronously
        in ``_land_attachments`` before the execution row opens). This method
        only extracts content and dispatches ``vision`` (images) or ``read``
        (anything else) on the SAVED absolute path.

        An extraction failure raises ``ValueError`` from ``FileParserService.index``
        and logs a warning — the dispatched ``read`` below still surfaces the
        failure loudly on the act trail.
        """
        import mimetypes

        from services.file_parser_service import FileParserService

        try:
            FileParserService().index(saved_path)
        except ValueError as exc:
            logger.warning(
                "[MessageProcessor] indexing %s failed for turn-zero dispatch: %s",
                saved_path, exc,
            )

        mime = mimetypes.guess_type(saved_path)[0] or ""
        if mime.startswith("image/"):
            self.dispatch_service.dispatch("vision", {"image": saved_path, "instructions": "Describe this image"})
        else:
            self.dispatch_service.dispatch("read", {"source": saved_path})

    def _maybe_fire_gist(self) -> None:
        """Ingest this thread's gist when it has none stored yet, so the reply
        prompt carries the thread's condensed context.

        Two threads qualify. A fork, whose parent turn is the thread being
        replied into. And a schedule fire, which is always a MAIN turn — its
        ``turn_id`` IS the schedule's id (§13.1), so the gist generated here is
        the label the scheduler surfaces; without this the schedule channel
        would never produce one. Other MAIN turns skip this. The ``bulk_get``
        emptiness check keeps it once-only per thread on both paths; any ingest
        hiccup is swallowed (best-effort context)."""
        if not self._forked and self.channel != Channel.SCHEDULE:
            return
        try:
            if not self.gist_service.bulk_get(self.channel, [self.turn_id]):
                from services.thread_gist_message_processor import maybe_ingest_gist  # noqa: PLC0415
                maybe_ingest_gist(self.channel, self.turn_id, self.config.type_value())
        except Exception as exc:  # noqa: BLE001 — gist ingest is best-effort context
            logger.debug("[MessageProcessor] gist ingest skipped for turn %s: %s", self.turn_id, exc)
