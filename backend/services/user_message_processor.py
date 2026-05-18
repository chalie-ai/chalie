# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
UserMessageProcessor — user-channel subclass of MessageProcessor v2.

Lifecycle: one instance per user turn. Constructed by websocket.py,
called as:
    proc = UserMessageProcessor(raw_input=text, metadata=metadata,
                                on_narration=_on_narration)
    response = proc.send(request_id=request_id)

Implements the full post_turn() four-step fan-out, memory seeding, narration
callback, and get_system_prompt() override.
"""

import logging
import os
from collections.abc import Callable

from services.message_processor import MessageProcessor
from services.system_message_prompt import UnifiedSystemMessagePrompt
from services.world_state import world_state

logger = logging.getLogger(__name__)


class UserMessageProcessor(MessageProcessor):
    """User-channel MessageProcessor subclass.

    Hardcodes CHANNEL='user', ROLE='user', uses UnifiedSystemMessagePrompt.
    Adds on_narration callback for real-time ACT narration streaming.
    Overrides post_turn() with the four-step fan-out.
    """

    CHANNEL = 'user'
    ROLE = 'user'
    LOG_LABEL = 'chat'
    SYSTEM_PROMPT_CLASS = UnifiedSystemMessagePrompt

    # 9 innate abilities — pre-injected on every ACT iteration. The 10 in
    # DISCOVERABLE are surfaced at runtime via find_tools and never
    # pre-injected. `subagent` lives in DISCOVERABLE so it competes with
    # weather/search/news/browser via find_tools rather than appearing as
    # an always-on fallback that biases routing for single-tool lookups.
    ALWAYS_AVAILABLE: list[str] = [
        "find_tools",
        "document",
        "list",
        "memory",
        "read",
        "review_tool_calls",
        "review_transcript",
        "schedule",
        "timer",
    ]
    DISCOVERABLE: list[str] = [
        "browser",
        "calendar",
        "code_eval",
        "contacts",
        "email",
        "home",
        "news",
        "programming_docs_search",
        "search",
        "subagent",
        "weather",
    ]

    # ── Constructor ───────────────────────────────────────────────────────────

    def __init__(
        self,
        raw_input: str,
        metadata: dict | None = None,
        on_narration: Callable[[str, int], None] | None = None,
        on_tool_event: Callable[[dict], None] | None = None,
    ):
        super().__init__(raw_input, metadata)
        if self._metadata.get('hidden_input'):
            self.SKIP_INPUT_ROW = True
        self._on_narration = on_narration
        self._on_tool_event = on_tool_event
        # Numeric radius used for the pre-act seed recall (stored for drift calc).
        self._memory_seed_radius: float | None = None
        # Set by store() — the final LLM response text, needed by post_turn()
        # for interaction logging and phase updates (call AFTER store()).
        self._last_response: str = ''
        # Cached user synthesis string. get_system_prompt() runs every ACT iteration;
        # without this cache each iteration would re-read user_summary from data_graph.
        # The user summary is stable for the duration of a single turn.
        self._user_definition_cached: str | None = None
        # Per-turn ModeGate instance + cached state vector. Populated by
        # _get_mode_gate() on first access; the gate is ticked exactly once
        # per turn (classify + state update + persist), and the resulting
        # state dict is cached so converse-driven branching in
        # get_user_definition() does not re-read MemoryStore on every ACT
        # iteration. The instance is reused for get_system_prompt_additions()
        # in get_system_prompt().
        self._mode_gate_cached = None  # ModeGateService | None — lazy import
        self._mode_state_cached: dict[str, float] | None = None

    # ── Abstract overrides ────────────────────────────────────────────────────

    def get_user_definition(self) -> str:
        """One-sentence synthesis of the real human user for the system prompt.

        Reads the user_summary record (kind='system', key='user_summary') from data_graph
        and returns its value. This is a human-readable sentence that describes the user
        (e.g. "Dylan is a software engineer based in Malta").

        Falls back to a static peer-to-peer framing on empty or missing record, or on
        any exception. When the row is missing, the static fallback is returned immediately.
        UserSummaryProcessor is driven exclusively by SubconsciousWorker._step_synthesis()
        on each idle tick — no lazy fallback here.

        Per-turn cached: get_system_prompt() runs on every ACT iteration; without this
        cache each iteration would re-query the knowledge table.
        """
        _FALLBACK = "The user is a real human. Treat this conversation as peer-to-peer dialogue."

        if self._user_definition_cached is not None:
            return self._user_definition_cached

        # Pick which row to read: when the converse mode is strongly active
        # (state >= ModeGateService.STEER_THRESHOLD) prefer
        # ``user_summary_long`` for a richer identity anchor; otherwise stay
        # on the short ``user_summary``. If the long row is missing fall back
        # to the short one before the static peer-to-peer fallback.
        from services.mode_gate_service import STEER_THRESHOLD
        prefer_long = (
            self._get_mode_state().get('converse', 0.0) >= STEER_THRESHOLD
        )

        try:
            from services.data_graph_service import get_data_graph_service

            dgs = get_data_graph_service()
            rows = dgs.fetch(kinds=['system'], order_by='retrieval_weight DESC')
            by_key = {r.get('key'): r for r in rows if r.get('key')}

            preferred_key = 'user_summary_long' if prefer_long else 'user_summary'
            entry = by_key.get(preferred_key)
            if (not entry or not entry.get('value')) and prefer_long:
                # Long row missing — fall back to short before the static fallback.
                entry = by_key.get('user_summary')
            if entry and entry.get('value'):
                self._user_definition_cached = entry['value']
                return self._user_definition_cached

        except Exception as e:
            logger.warning(f"[USER MSG] get_user_definition failed: {e}")

        self._user_definition_cached = _FALLBACK
        return self._user_definition_cached

    def get_user_prompt(self) -> str:
        """Build the user-message body for one ACT iteration.

        Section order:
          1. User definition (user_summary short) — placed at the top of the
             user prompt so the model sees identity in the same recency window
             as the current turn line. The base-class system-prompt slot for
             user_definition is suppressed in get_system_prompt() below.
          2. World State block
          3. System Awareness block
          4. ## Previous Messages block (via get_previous_messages())
          (blank line separator)
          5. Memory seed block (canonical tag block set by pre_act(), injected verbatim)
          6. Current turn line: user: <raw_input> [nudge_tag]
          7. ACT loop trail (empty string on iteration 1)

        Note: The ## Checkpoint / ## Current State envelope is NOT emitted
        here — send() wraps this output with it.
        """
        parts = []

        # 1. User definition (identity anchor)
        user_def = self.get_user_definition()
        if user_def:
            parts.append(user_def)

        # 2. World State — injected verbatim (already contains its own header)
        rendered_world_state = world_state.render()
        if rendered_world_state:
            logger.info(
                "[WorldState] injected rendered block into user prompt (%d chars)",
                len(rendered_world_state),
            )
            parts.append(rendered_world_state)

        # 2. System Awareness (degradation signals)
        self_awareness = self._get_self_awareness()
        if self_awareness:
            parts.append(f"## System Awareness\n{self_awareness}")

        # 3. Previous Messages
        prev = self.get_previous_messages()
        if prev:
            parts.append(f"## Previous Messages\n{prev}")

        # Blank separator before current turn content
        parts.append('')

        # 4. Memory seed (set by pre_act() — canonical tag block, injected verbatim)
        if self._memory_seed:
            parts.append(self._memory_seed)

        # 5. Current turn line with optional nudge
        turn_line = f"user: {self._raw_input}"
        nudge_tag = self._metadata.get('nudge_tag')
        if nudge_tag:
            turn_line += ' ' + nudge_tag
        parts.append(turn_line)

        # 6. ACT loop trail (empty on iteration 1)
        trail = self.get_act_loop_trail()
        if trail:
            parts.append(trail)

        return '\n'.join(parts)

    # ── Overridable hooks ─────────────────────────────────────────────────────

    def get_system_prompt(self) -> str:
        """Build the final system prompt for this turn.

        Assembly order:
          1. Voice line — ``"When responding; <personality voice paragraph>"``
             drawn fresh from PersonalityService (O(1) dict lookup + one
             SQLite SELECT).
          2. Template — UnifiedSystemMessagePrompt body.

        The user_definition (user_summary short) is intentionally NOT emitted
        here — it is prepended at the top of get_user_prompt() instead so it
        sits in the same recency window as the current turn line. This
        overrides the base-class behaviour which would otherwise prepend the
        user_definition to the system prompt body.

        The voice line sits at the very top so the LLM sees it first.  The
        stable Identity/Boundaries/Principles prefix follows, keeping the bulk
        of the prompt hot in the provider's prompt cache.
        """
        from services.personality.personality_service import get_current_voice

        template = self.SYSTEM_PROMPT_CLASS().get_prompt()

        voice_line = f"When responding; {get_current_voice()}"
        prompt = f"{voice_line}\n\n{template}"

        # Mode-state-driven steering directives. The mode gate owns the
        # mapping from active modes → directive text — UMP just appends the
        # rendered string. ``_get_mode_state()`` ensures ``tick()`` has fired
        # before the additions are read.
        self._get_mode_state()
        additions = self._get_mode_gate().get_system_prompt_additions()
        if additions:
            prompt = f"{prompt}\n\n{additions}"
        return prompt

    def pre_act(self) -> None:
        """Memory auto-seed via canonical tool dispatch path.

        Runs once at turn start (after self._uid is populated by write_input_row).
        Calls handle_memory directly so the result is a canonical tag block,
        records the row via ToolRenderAndRecordService (ephemeral=False) — same
        storage path as any other durable tool call — and stores the block on
        self._memory_seed for get_user_prompt() to inject verbatim.
        """
        from abilities._registry import AbilityRegistry
        from abilities.memory import MemoryAbility
        from services.tool_render_and_record_service import ToolRenderAndRecordService

        radius = MemoryAbility.SEED_RADIUS_BASELINE
        query = self._raw_input

        # Expose query separately so recall_episodes() can embed the raw text
        # for drift calculation without embedding the tag block string.
        self._memory_seed_query = query
        self._memory_seed_radius = radius

        block = AbilityRegistry.get('memory').execute(self.CHANNEL, {
            'action': 'recall',
            'query': query,
            '_auto': True,
        }, None).get('text', '')

        # Row is recorded every turn — the seed dispatch is part of the ACT
        # trail whether or not it returned matches. Inject into the prompt
        # only when the recall produced real content; empty (`results=0`) and
        # error (`error=...`) header args yield blocks that add noise without
        # value. Inspect only the opener line so a body containing the literal
        # substring `results=0` or `error=` cannot suppress a valid seed.
        header = block.split('\n', 1)[0] if block else ''
        if block and 'results=0' not in header and 'error=' not in header:
            self._memory_seed = block

        if self._uid is None:
            logger.warning(
                "[UMP] pre_act skipped seed-row write: _uid is None "
                "(SKIP_TRANSCRIPT_WRITE subclass?)"
            )
            return

        ToolRenderAndRecordService(
            tool_name='memory',
            params={'action': 'recall', 'query': query, 'radius': radius},
            result=block,
            ephemeral=False,
            transcript_id=self._uid,
        ).render_and_record()

    def _emit_narration(self, text: str, iteration: int) -> None:
        """Push mid-loop narration text to the per-request SSE channel.

        Lifted directly from old UserMessageProcessor._on_narration inner
        function. Fires self._on_narration callback if set. Exceptions from the
        callback are logged and swallowed — never kill the ACT loop.
        """
        if not self._on_narration or not text:
            return
        try:
            self._on_narration(text, iteration)
        except Exception as e:
            logger.debug(f"[USER MSG] Narration callback failed: {e}")

    def _emit_tool_event(self, event: dict) -> None:
        """Push tool start/end events to the per-request SSE channel.

        Mirrors _emit_narration: fires self._on_tool_event if set, swallows
        callback exceptions, never kills the ACT loop.
        """
        if not self._on_tool_event or not event:
            return
        try:
            self._on_tool_event(event)
        except Exception as e:
            logger.debug(f"[USER MSG] Tool event callback failed: {e}")


    def store(self, llm_response: str) -> None:
        """Persist the turn atomically, then capture _last_response for post_turn().

        Calls base store() (which sets self._uid and writes all rows), then
        captures the response text so post_turn() can reference it without
        send() needing to pass it explicitly.
        """
        super().store(llm_response)
        self._last_response = llm_response

    def post_turn(self) -> None:
        """Two-step fan-out, each individually error-isolated.

        Order is load-bearing (see plan § "Ordering constraints"):
          1. ConversationPhaseService — two calls
          2. MetricsService — last ("turn closed" signal)

        DMN is no longer a service with an idle timer — it runs as step 5 of
        SubconsciousWorker, gated by the worker's own user-idle check.
        The on_turn() hook is therefore not needed here.

        Compaction is intentionally NOT here: per the north star
        (message-processing.md § "What does NOT go in post_turn()"),
        compaction is a send()-loop responsibility driven by context
        pressure, not a post-turn consequence. Mid-ACT Stage 1/2 in
        send() owns it.
        """
        channel = self.CHANNEL   # 'user'
        text = self._raw_input
        response = self._last_response

        # 1. Conversation phase — TWO calls (user text + assistant response).
        # Skip the assistant-side update on empty response (cap-exit turns):
        # feeding '' would drift the phase model toward "silence" wrongly
        # (Commit 8 critic P1-3).
        try:
            from services.conversation_phase_service import get_conversation_phase_service
            phase = get_conversation_phase_service()
            phase.update(channel, text, is_user=True, topic=channel)
            if response:
                phase.update(channel, response, is_user=False, topic=channel)
        except Exception as e:
            logger.debug(f"[POSTTURN] Phase update failed: {e}", exc_info=True)

        # 2. Metrics (sync) — last: requests_total is the "turn closed" signal.
        # WARNING level — observability hole if it silently fails
        # (Commit 8 critic P1-2).
        try:
            from services.metrics_service import MetricsService
            m = MetricsService()
            m.record_counter('requests_total')
            m.record_counter('user_messages_total')
        except Exception as e:
            logger.warning(f"[POSTTURN] Metrics failed: {e}", exc_info=True)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _process_file_attachments(self) -> None:
        """Dispatch document.upload + document.view for each attachment.

        Handles the ``attachments`` list of ``/tmp/chalie_*`` paths produced
        by the POST /upload endpoint. For each path:
          1. Path-traversal guard — rejects any path not under /tmp/chalie_.
          2. Read the file from /tmp and base64-encode it for document.upload.
          3. Dispatch document.upload then document.view via handle_tool() so
             each call produces audit rows, WS events, and policy enforcement.
          4. Delete the /tmp file after a successful upload (permanent storage
             now owns it; /tmp is ephemeral and the cleanup worker removes
             stragglers after 24 h anyway).

        Called after pre_act(). No-op when metadata has no ``attachments`` key.
        """
        import base64
        import mimetypes

        attachments = self._metadata.get('attachments') or []
        if not attachments:
            return

        for tmp_path in attachments:
            if not self._validate_tmp_path(tmp_path):
                logger.warning('[UMP] Rejected attachment with unsafe tmp_path: %s', tmp_path)
                continue

            try:
                with open(tmp_path, 'rb') as fh:
                    file_bytes = fh.read()
            except OSError as exc:
                logger.warning('[UMP] Could not read attachment %s: %s', tmp_path, exc)
                continue

            filename = os.path.basename(tmp_path)
            content_type, _ = mimetypes.guess_type(tmp_path)
            content_type = content_type or 'application/octet-stream'
            data_b64 = base64.b64encode(file_bytes).decode()

            upload_result = self.handle_tool({
                'name': 'document',
                'input': {
                    'action': 'upload',
                    'name': filename,
                    'content': data_b64,
                    'content_type': content_type,
                },
            })

            # Remove the /tmp file — permanent storage now owns the content.
            try:
                os.unlink(tmp_path)
            except OSError as exc:
                logger.debug('[UMP] Could not remove tmp file %s: %s', tmp_path, exc)

            doc_id = self._extract_doc_id_from_upload(upload_result)
            if doc_id:
                self.handle_tool({
                    'name': 'document',
                    'input': {'action': 'view', 'id': doc_id},
                })

    def _validate_tmp_path(self, path: str) -> bool:
        """Return True only when path is a real file under /tmp/chalie_.

        Uses os.path.realpath to resolve symlinks before the prefix check,
        preventing directory-traversal attacks via symlink chains.

        Args:
            path: Candidate /tmp path from the client ``attachments`` array.

        Returns:
            True if the resolved path starts with '/tmp/chalie_' and the
            file exists; False otherwise.
        """
        if not path:
            return False
        real = os.path.realpath(path)
        if not real.startswith('/tmp/chalie_'):
            return False
        return os.path.isfile(real)

    def _extract_doc_id_from_upload(self, result_text: str) -> str | None:
        """Parse doc_id from upload result like '(id=abc123ef)'."""
        import re
        match = re.search(r'\bid=([0-9a-f]{8})\b', result_text or '')
        return match.group(1) if match else None

    def _get_self_awareness(self) -> str:
        """Get system health degradation signals from SelfModelService.

        Returns empty string when the system is healthy — get_user_prompt()
        skips the section. Only populates when degradation is detected.
        """
        try:
            from services.self_model_service import SelfModelService
            return SelfModelService().format_for_prompt()
        except Exception as e:
            logger.debug(f"[USER MSG] Self-awareness unavailable: {e}")
            return ''

    def _get_mode_gate(self):
        """Return a ticked ModeGateService instance, cached per turn.

        First call constructs the service and fires ``tick()`` (classify +
        state update + persist) against the current raw input. Subsequent
        calls within the same turn return the same instance so consumers
        share one classification result.
        """
        if self._mode_gate_cached is not None:
            return self._mode_gate_cached

        from services.mode_gate_service import ModeGateService
        gate = ModeGateService()
        try:
            gate.tick(self._raw_input, turn_id=self._uid)
        except Exception as exc:
            logger.warning("[MODE-GATE] tick failed: %s", exc)
        self._mode_gate_cached = gate
        return gate

    def _get_mode_state(self) -> dict[str, float]:
        """Return the per-mode activation state for this turn (cached).

        Wraps ``_get_mode_gate().get_state()`` with a per-turn cache so
        get_user_definition() does not re-read MemoryStore on every ACT
        iteration. On any failure an empty dict is cached so callers see a
        deterministic miss without retry storms.
        """
        if self._mode_state_cached is not None:
            return self._mode_state_cached
        try:
            self._mode_state_cached = self._get_mode_gate().get_state()
        except Exception as exc:
            logger.warning("[MODE-GATE] get_state failed: %s", exc)
            self._mode_state_cached = {}
        return self._mode_state_cached


