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
instance, resolves the thinking override, and calls ``begin()``, then returns the
instance. ``begin()`` runs the synchronous side-effects — atomic per-channel
turn-id allocation + input row + execution-row open, all inside one
single-writer ``BEGIN IMMEDIATE`` transaction (§6.8) — then fires a daemon thread
running ``_drive`` and returns immediately (it does **not** join). Because the
execution row is opened synchronously before the thread spawns, ``mp.execution``
is populated the moment ``process()`` returns: a POST handler reads the WORKING
execution handle instantly and live output then streams over WS, while
fire-and-forget callers (scheduler) read the allocated id off ``mp.metadata``.

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
from threading import Thread
from typing import TYPE_CHECKING, Protocol, cast

from configs.enums.provider_type import ProviderType
from configs.enums.thinking_level import ThinkingLevel
from models.provider_errors import (
    ProviderResponseError,
    ProviderRetriesExhaustedError,
    RequestOverCapError,
    ResponseOverLimitError,
)
from models.provider_request import ProviderRequest
from models.turn_execution import TurnExecution
from services.behavioral_pattern_service import BehavioralPatternService
from services.compaction_service import CompactionService
from services.data_graph_service import DataGraphService
from services.database import Database
from services.dispatch_service import DispatchService
from services.gist_service import GistService
from services.llm_log_service import LlmLogService
from services.prompt_service import PromptService
from services.provider_service import ProviderService
from services.thinking_override_service import get_thinking_override
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
#: Pulls the document id out of the upload ability's JSON result for doc-linking.
_SEED_UPLOAD_ID_RE = re.compile(r'"id":"([0-9a-f]+)"')


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
        self.post_compaction_continuation: bool = False
        self._trigger_channel: str | None = cast("str | None", self.metadata.get("trigger_channel"))
        self._trigger_turn_id: int | None = cast("int | None", self.metadata.get("trigger_turn_id"))
        self._thread: Thread | None = None
        self._result_text: str = ""

        # Infrastructure handles.
        self.db = Database()

        # Coordinating services (Rule 3: each holds this mp; Rule 4: they reach
        # one another only through self.mp.<service>).
        self.transcript_service = TranscriptService(self)
        self.tool_call_service = ToolCallService(self)
        self.turn_execution_service = TurnExecutionService(self)
        self.provider_service = ProviderService(self)
        self.compaction_service = CompactionService(self)
        self.data_graph_service = DataGraphService(self)
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
        the thinking override, kick off ``begin()`` (which spawns the drive
        thread), and return the live instance."""
        mp = cls(config, turn_id=turn_id, raw_input=raw_input, metadata=metadata)
        mp.thinking_override = get_thinking_override()
        mp.begin()
        return mp

    def begin(self) -> None:
        """Synchronous turn setup, then hand off to the drive thread. The
        turn-id allocation, fork guard and input row land in one single-writer
        transaction (§6.8); the execution row opens right after so
        ``mp.execution`` is set before the thread starts. ``begin()`` never
        joins — it returns the instant the daemon is running."""
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
        self.turn_execution_service.open()
        self.metadata["turn_id"] = self.turn_id
        self._thread = Thread(target=self._drive, daemon=True, name=f"turn-{self.turn_id}")
        self._thread.start()

    def _open_input_row(self) -> int | None:
        """Write this turn's anchoring input row and return its id — unless the
        config skips it (channels whose input is not a transcript utterance)."""
        if self.config.skip_input_row:
            return None
        return self.transcript_service.append_input(self.raw_input)

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

        Send → on over-cap, compact and continue (re-enter with the transcript
        reviewer armed) → store any prose the model emitted → if it made no tool
        calls the turn is done (end); otherwise dispatch the calls and recurse.
        A cancel observed at the top of any step aborts the whole turn."""
        if self.turn_execution_service.should_stop():
            raise _TurnCancelled()
        try:
            response = self._send_with_retry(self._build_request())
        except (RequestOverCapError, ResponseOverLimitError):
            self.dispatch_service.dispatch("chat_history_compactor", {"act_summary": "Compacting conversation"})
            self.post_compaction_continuation = True
            if "review_transcript" not in self.active_tools:
                self.active_tools.append("review_transcript")
            return self._step()
        self.post_compaction_continuation = False
        tool_calls = response.tool_calls
        if not tool_calls:
            self._store(response.text)
            return self._end(response.text)
        if response.text:
            self._store(response.text)
        self._dispatch_tools(tool_calls)
        return self._step()

    # ── provider send ──────────────────────────────────────────────────────────

    def _send_with_retry(self, request: ProviderRequest) -> ProviderResponse:
        """Send with the turn's resend budget. Over-cap/over-limit errors are
        re-raised untouched (``run`` compacts and continues); every other
        provider failure is retried up to ``_MAX_PROVIDER_ATTEMPTS``, emitting a
        user-facing retry notice between attempts, and a cancel observed
        mid-retry aborts the turn. Exhausting the budget raises a clean,
        surfaceable ``ProviderRetriesExhaustedError``."""
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_PROVIDER_ATTEMPTS + 1):
            try:
                return self.provider_service.send(request)
            except (RequestOverCapError, ResponseOverLimitError):
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
            _job_name=self.config.job,
            _usage_class=self.config.usage_class,
            _caller=type(self).__name__,
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

    def _format(self, text: str) -> str:
        """Render markdown to HTML for surface-broadcasting channels; pass raw
        text through for background/silent channels (``broadcast_to`` is None)."""
        if self.config.broadcast_to is not None:
            from services.markup import markdown_to_html  # noqa: PLC0415
            return markdown_to_html(text)
        return text or ""

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
        transcript ``role`` — the identity OLD composed ``post_turn_hooks`` per.
        The proactive-suggestion handler additionally requires the genuine
        ``user`` channel: DiscoveryConfig and ScheduledConfig also carry
        ``role='user'`` but write to their own channels, and OLD never gave
        either the ProactiveSuggestionHook (discovery ran PersistDiscoveryRunHook;
        scheduled hands its result to a separate UserConfig turn that reaches the
        UI). Every handler is isolated — a failure is logged, never propagated, so
        post-turn work can never fail an otherwise-complete turn."""
        role = self.config.role
        try:
            if role == "user" and self.channel == "user":
                self._proactive_suggestion()
            elif role == "user_summary":
                UserSynthesis.persist_user_summary(response_text)
            elif role == "pattern_match":
                self._pattern_skill_sync()
            elif role == "external_agent":
                self._disclose_to_human(response_text)
        except Exception as exc:  # noqa: BLE001 — post-turn work must never fail the turn
            logger.warning("[post_turn] %s handler failed (isolated): %s", role, exc)

    def _proactive_suggestion(self) -> None:
        """After a ``user`` turn that made enough real tool calls, hand the act
        trail to the skill-suggestion pass (legacy ``ProactiveSuggestionHook``).
        Compaction calls do not count toward the threshold."""
        count = sum(1 for c in self.tool_call_service.by_turn() if c.tool_name != "chat_history_compactor")
        if count < _PROACTIVE_SUGGESTION_MIN_CALLS:
            return
        rendered = self.prompt_service.act_trail()
        act_trail = rendered.split("\n") if rendered else []
        from services.skill_suggestion_message_processor import maybe_suggest_skill  # noqa: PLC0415
        maybe_suggest_skill(act_trail, self.raw_input, self.channel, self.turn_id)

    def _pattern_skill_sync(self) -> None:
        """Decay untouched patterns, then run the skill-association pass over the
        patterns written this turn (legacy ``PatternDecayHook`` +
        ``PatternSkillSyncHook``, both self-guarding on an empty touched set)."""
        touched = self._touched_pattern_names()
        self.behavioral_pattern_service.decay(touched)
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
        fresh hidden ``user`` turn narrating the exchange (legacy
        ``DiscloseToHumanHook``). A no-op for agents that do not."""
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
        from configs.channels.user import UserConfig  # noqa: PLC0415
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
        if self.channel == "user":
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
        and is turn-scoped, so it drops at the next turn."""
        if self.config.memory_seed:
            self.dispatch_service.dispatch(
                "memory", {"action": "recall", "query": self.raw_input, "_auto": True}
            )
        attachments = list(cast("list[str]", self.metadata.get("attachments") or []))
        if attachments:
            from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
            with ThreadPoolExecutor(max_workers=min(len(attachments), 8)) as pool:
                list(pool.map(self._seed_upload_attachment, attachments))

    def _seed_upload_attachment(self, path: str) -> None:
        """Upload one attachment through the document ability and link the stored
        doc to this turn's input row. Only files inside the tmp-upload sandbox are
        accepted; anything else is refused loudly."""
        import os  # noqa: PLC0415
        from services.tmp_storage import TMP_PATH_PREFIX  # noqa: PLC0415
        real = os.path.realpath(path)
        if not real.startswith(TMP_PATH_PREFIX) or not os.path.isfile(real):
            logger.warning("[MessageProcessor] refusing attachment outside tmp sandbox: %s", path)
            return
        result = self.dispatch_service.dispatch("document", {"action": "upload", "path": real})
        if self.uid is None:
            return
        match = _SEED_UPLOAD_ID_RE.search(result)
        if match:
            self.transcript_service.link_doc(match.group(1))

    def _maybe_fire_gist(self) -> None:
        """On a fork whose parent turn has no stored gist yet, ingest it so the
        reply prompt carries the thread's condensed context. MAIN turns skip
        this; any ingest hiccup is swallowed (best-effort context)."""
        if not self._forked:
            return
        try:
            if not self.gist_service.bulk_get(self.channel, [self.turn_id]):
                from services.thread_gist_message_processor import maybe_ingest_gist  # noqa: PLC0415
                maybe_ingest_gist(self.channel, self.turn_id)
        except Exception as exc:  # noqa: BLE001 — gist ingest is best-effort context
            logger.debug("[MessageProcessor] gist ingest skipped for turn %s: %s", self.turn_id, exc)
