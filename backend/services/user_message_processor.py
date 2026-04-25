# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
UserMessageProcessor — user-channel subclass of MessageProcessor v2.

North star: /Volumes/llm/chalie-plans/message-processing.md
Refactor plan: /Users/dylangrech/.claude/plans/joyful-cooking-riddle.md (Commit 8)

Lifecycle: one instance per user turn. Constructed by websocket.py,
called as:
    proc = UserMessageProcessor(raw_input=text, metadata=metadata,
                                on_narration=_on_narration)
    response = proc.send(request_id=request_id)

Implements the full postTurn() seven-service fan-out, memory seeding, narration
callback, and getSystemPrompt() override.
"""

import logging
import threading
from collections.abc import Callable

from services.message_processor import MessageProcessor
from services.system_message_prompt import UnifiedSystemMessagePrompt
from services.innate_skills.registry import ALL_SKILL_NAMES, COGNITIVE_PRIMITIVES_ORDERED
from services.world_state import world_state

logger = logging.getLogger(__name__)

# ── Lazy-synthesis concurrency guard ─────────────────────────────────────────
# Prevents multiple concurrent getUserDefinition() calls from each spawning a
# synthesis daemon when the user_summary row is missing.  The flag is cleared
# in a ``finally`` block so the next call — whether prior synthesis succeeded,
# failed, or raised — re-arms the guard cleanly.
_lazy_fire_lock = threading.Lock()
_lazy_fire_in_flight = False


def _fire_lazy_synthesis() -> None:
    """Spawn a one-shot daemon thread to synthesise the user_summary row.

    Guards against concurrent calls with a module-level flag + lock.
    If synthesis is already in flight the call is a no-op.
    The flag is cleared in a ``finally`` block on every daemon exit path
    (success, exception, or early return) so the guard re-arms for the
    next call regardless of outcome.
    """
    global _lazy_fire_in_flight

    with _lazy_fire_lock:
        if _lazy_fire_in_flight:
            return
        _lazy_fire_in_flight = True

    def _run():
        global _lazy_fire_in_flight
        try:
            from services.user_summary_processor import UserSummaryProcessor

            UserSummaryProcessor().send()
            logger.info("[USER MSG] Lazy synthesis complete")
        except Exception as exc:
            logger.warning("[USER MSG] Lazy synthesis failed: %s", exc)
        finally:
            with _lazy_fire_lock:
                _lazy_fire_in_flight = False

    threading.Thread(target=_run, daemon=True, name="user-summary-lazy").start()
    logger.info("[USER MSG] Lazy synthesis daemon spawned")


class UserMessageProcessor(MessageProcessor):
    """User-channel MessageProcessor subclass.

    Hardcodes CHANNEL='user', ROLE='user', uses UnifiedSystemMessagePrompt.
    Adds on_narration callback for real-time ACT narration streaming.
    Overrides postTurn() with the eight-service fan-out.
    """

    CHANNEL = 'user'
    ROLE = 'user'
    JOB = 'frontal-cortex-unified'
    SYSTEM_PROMPT_CLASS = UnifiedSystemMessagePrompt

    # Narrowed to cognitive primitives only — the unconditional tier.
    # Non-primitive innate skills are promoted via getConditionalTools()
    # (mode gate) rather than injected on every turn. This reduces prompt
    # bloat on conversational / chit-chat turns where only primitives are needed.
    # COGNITIVE_PRIMITIVES_ORDERED preserves insertion order for determinism.
    NATIVE_TOOLS: list[str] = list(COGNITIVE_PRIMITIVES_ORDERED)

    # ── Constructor ───────────────────────────────────────────────────────────

    def __init__(
        self,
        raw_input: str,
        metadata: dict | None = None,
        on_narration: Callable[[str, int], None] | None = None,
    ):
        super().__init__(raw_input, metadata)
        self._on_narration = on_narration
        # Set by _run_memory_seed() — stores the numeric radius used so
        # getUserPrompt() can render `[memory(radius=X)] ...` correctly.
        self._memory_seed_radius: float | None = None
        # Set by store() — the final LLM response text, needed by postTurn()
        # for interaction logging and phase updates (call AFTER store()).
        self._last_response: str = ''
        # Cached user synthesis string. getSystemPrompt() runs every ACT iteration;
        # without this cache each iteration would re-read user_summary from data_graph.
        # The user summary is stable for the duration of a single turn.
        self._user_definition_cached: str | None = None
        # Per-turn cache for getConditionalTools(). Set on first call and reused
        # on every subsequent ACT iteration — the mode gate runs once per turn.
        # Stage 2 ACT restart clears _discovered_tools but NOT this cache —
        # the mode classification was correct for this turn (spec §3.2).
        self._conditional_tools_cached: list[dict] | None = None

    # ── Abstract overrides ────────────────────────────────────────────────────

    def getUserDefinition(self) -> str:
        """One-sentence synthesis of the real human user for the system prompt.

        Reads the user_summary record (kind='system', key='user_summary') from data_graph.
        from DataGraphService and returns its value. This is a human-readable sentence
        that describes the user (e.g. "Dylan is a software engineer based in Malta").

        Falls back to a static peer-to-peer framing on empty or missing record, or on
        any exception.  When the row is missing but ``user_specific`` traits exist a
        one-shot background synthesis is fired via the lazy-fallback path below so
        that future turns find the row populated.

        Writer path: ``UserSummaryProcessor`` (driven by SubconsciousWorker
        idle tick, plus ``getUserDefinition()`` lazy fallback).  Traits are
        written continuously by
        the LLM-native memory skill (``memory_skill._handle_store`` →
        ``DataGraphService.store(kind='user_specific', …)``) whenever the user
        discloses a personal fact.

        Per-turn cached: getSystemPrompt() runs on every ACT iteration; without this
        cache each iteration would re-query the knowledge table.
        """
        _FALLBACK = "The user is a real human. Treat this conversation as peer-to-peer dialogue."

        if self._user_definition_cached is not None:
            return self._user_definition_cached
        try:
            from services.data_graph_service import get_data_graph_service

            dgs = get_data_graph_service()
            rows = dgs.fetch(kinds=['system'], order_by='retrieval_weight DESC')
            entry = next((r for r in rows if r.get('key') == 'user_summary'), None)
            if entry and entry.get('value'):
                self._user_definition_cached = entry['value']
                return self._user_definition_cached

            # user_summary row is missing — check whether any traits exist so we
            # know if synthesis is worthwhile.
            trait_rows = dgs.fetch(kinds=['user_specific'], limit=1)
            if trait_rows:
                _fire_lazy_synthesis()

        except Exception as e:
            logger.warning(f"[USER MSG] getUserDefinition failed: {e}")

        self._user_definition_cached = _FALLBACK
        return self._user_definition_cached

    def getConditionalTools(self) -> list[dict]:  # NOSONAR — overrides MessageProcessor hook (camelCase by contract)
        """Return mode-gated tool schemas for this turn (cached after first call).

        Consults ModeGateService once per turn. The gate returns names for tools
        whose modes are active, which are resolved to schemas here.

        Exceptions are caught: any failure logs at WARN and caches [] so
        subsequent ACT iterations also get the empty list (consistent state).
        """
        if self._conditional_tools_cached is not None:
            return self._conditional_tools_cached

        from services.mode_gate_service import ModeGateService
        from services.tool_schema_service import get_skill_schemas, get_external_tool_schemas

        try:
            gate = ModeGateService()
            names = gate.get_promoted_tool_names(self._raw_input, turn_id=self._uid)
        except Exception as exc:
            logger.warning("[MODE-GATE] get_promoted_tool_names failed, returning []: %s", exc)
            self._conditional_tools_cached = []
            return []

        innate_names = [n for n in names if n in ALL_SKILL_NAMES]
        external_names = [n for n in names if n not in ALL_SKILL_NAMES]

        schemas: list[dict] = []
        if innate_names:
            schemas.extend(get_skill_schemas(innate_names))
        if external_names:
            schemas.extend(get_external_tool_schemas(external_names))

        self._conditional_tools_cached = schemas
        return schemas

    def getUserPrompt(self) -> str:
        """Build the user-message body for one ACT iteration.

        Section order:
          1. User definition (user_summary short) — placed at the top of the
             user prompt so the model sees identity in the same recency window
             as the current turn line. The base-class system-prompt slot for
             user_definition is suppressed in getSystemPrompt() below.
          2. World State block
          3. System Awareness block
          4. ## Previous Messages block (via getPreviousMessages())
          (blank line separator)
          5. Memory seed line: [memory(radius=X)] ...
          6. Current turn line: user: <raw_input> [file_tags] [nudge_tag]
          7. ACT loop trail (empty string on iteration 1)

        Note: The ## Checkpoint / ## Current State envelope is NOT emitted
        here — send() wraps this output with it.
        """
        parts = []

        # 1. User definition (identity anchor)
        user_def = self.getUserDefinition()
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
        prev = self.getPreviousMessages()
        if prev:
            parts.append(f"## Previous Messages\n{prev}")

        # Blank separator before current turn content
        parts.append('')

        # 4. Memory seed (set by _run_memory_seed at turn start)
        if self._memory_seed and self._memory_seed_radius is not None:
            parts.append(f"[memory(radius={self._memory_seed_radius})] {self._memory_seed}")

        # 5. Current turn line with optional file tags and nudge
        turn_line = f"user: {self._raw_input}"
        file_tags = self._metadata.get('file_tags', [])
        nudge_tag = self._metadata.get('nudge_tag')
        if file_tags:
            kinds = [t.split(' ', 1)[0].lstrip('[') for t in file_tags]
            logger.info(
                f"[UMP] file_tags present uuid={self._uid} "
                f"count={len(file_tags)} kinds={kinds}"
            )
            turn_line += ' ' + ' '.join(file_tags)
        if nudge_tag:
            turn_line += ' ' + nudge_tag
        parts.append(turn_line)

        # 6. ACT loop trail (empty on iteration 1)
        trail = self.getActLoopTrail()
        if trail:
            parts.append(trail)

        return '\n'.join(parts)

    # ── Overridable hooks ─────────────────────────────────────────────────────

    def getSystemPrompt(self) -> str:
        """Build the final system prompt for this turn.

        Assembly order:
          1. Voice line — ``"When responding; <personality voice paragraph>"``
             drawn fresh from PersonalityService (O(1) dict lookup + one
             SQLite SELECT).
          2. Template — UnifiedSystemMessagePrompt body.

        The user_definition (user_summary short) is intentionally NOT emitted
        here — it is prepended at the top of getUserPrompt() instead so it
        sits in the same recency window as the current turn line. This
        overrides the base-class behaviour which would otherwise prepend the
        user_definition to the system prompt body.

        The voice line sits at the very top so the LLM sees it first.  The
        stable Identity/Boundaries/Principles prefix follows, keeping the bulk
        of the prompt hot in the provider's prompt cache.
        """
        from services.personality.personality_service import get_current_voice

        template = self.SYSTEM_PROMPT_CLASS().getPrompt()

        voice_line = f"When responding; {get_current_voice()}"
        return f"{voice_line}\n\n{template}"

    def _run_memory_seed(self) -> None:
        """Auto-seed memory once at turn start. Runs before getUserPrompt()."""
        from services.innate_skills.memory_skill import (
            recall_episodes, SEED_RADIUS_BASELINE, _format_results,
        )
        from services.tool_render_and_record_service import ToolRenderAndRecordService

        radius = SEED_RADIUS_BASELINE

        try:
            hits, _status = recall_episodes(
                channel=self.CHANNEL,
                query=self._raw_input,
                caller='seed',
                baseline_radius=radius,
                return_raw=False,
            )
            seed_text = _format_results(hits) if hits else ''
        except Exception as exc:
            logger.warning(f"[USER MSG] Memory auto-seed failed: {exc}")
            seed_text = ''

        if seed_text:
            self._memory_seed = seed_text
            self._memory_seed_radius = radius
            ToolRenderAndRecordService(
                tool_name='memory',
                params={'action': 'recall', 'radius': radius},
                result=seed_text,
                ephemeral=False,
                transcript_id=self._uid,
            ).renderAndRecord()
        else:
            logger.debug("[USER MSG] Memory auto-seed: no result returned")

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

    def _drain_steering(self, request_id: str | None) -> list[str]:
        """Drain mid-loop user steering messages from MemoryStore.

        WebSocket /steer endpoint pushes user feedback to ``steer:{request_id}``
        via ``rpush``. Without this override, the base no-op returns [] and
        every steer is silently dropped — Commit 8 critic P0.

        Returns one string per queued steer (preserving rpush insertion order).
        Deletes the key after draining. Errors are logged at DEBUG and return
        [] — a missing or unreadable queue must not abort the turn.
        """
        if not request_id:
            return []
        try:
            from services.memory_client import MemoryClientService
            _store = MemoryClientService.create_connection()
            key = f"steer:{request_id}"
            steers = _store.lrange(key, 0, -1)
            if not steers:
                return []
            _store.delete(key)
            return [s.decode() if isinstance(s, bytes) else s for s in steers]
        except Exception as exc:
            logger.debug(f"[USER MSG] Steer drain failed: {exc}")
            return []

    def store(self, llm_response: str) -> None:
        """Persist the turn atomically, then capture _last_response for postTurn().

        Calls base store() (which sets self._uid and writes all rows), then
        captures the response text so postTurn() can reference it without
        send() needing to pass it explicitly.
        """
        super().store(llm_response)
        self._last_response = llm_response

    def postTurn(self) -> None:
        """Five-service fan-out, each individually error-isolated.

        Order is load-bearing (see plan § "Ordering constraints"):
          1. ConversationPhaseService — two calls
          2. SaveSuggestionService — user-side trigger THEN response-side detect
          3. DMNService.on_turn() — R10 critical
          4. MetricsService — last ("turn closed" signal)
          5. compaction_service.check_and_compact — end-turn backstop
        """
        channel = self.CHANNEL   # 'user'
        text = self._raw_input
        response = self._last_response
        metadata = self._metadata

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

        # 2. Save suggestion scan — TWO paths in correct order:
        #    2a: user-side trigger must fire BEFORE 2b detect (ordering constraint)
        try:
            from services.save_suggestion_service import SaveSuggestionService
            save_svc = SaveSuggestionService()

            # 2a: User-side completion/deferral trigger — clears existing flag
            save_flag = save_svc.get_saveable_flag()
            if save_flag:
                trigger = save_svc.detect_save_trigger(text)
                if trigger:
                    save_svc.emit_save_card(save_flag['content_type'])
                    save_svc.clear_flag()

            # 2b: Response-side saveable content detection — sets new flag.
            # NOTE: flag_saveable expects exchange_id (UUID string) for window
            # correlation, not the integer transcript row id (self._uid). Prefer
            # the UUID from metadata; fall back to a stringified row id if absent.
            saveable = save_svc.detect_saveable_content(response, channel)
            if saveable:
                exchange_id_for_flag = (
                    metadata.get('exchange_id')
                    or metadata.get('uuid')
                    or str(self._uid)
                )
                save_svc.flag_saveable(
                    channel, saveable['content_type'], exchange_id_for_flag
                )
        except Exception as e:
            logger.debug(f"[POSTTURN] Save suggestion failed: {e}", exc_info=True)

        # 3. DMN idle reset — CRITICAL (R10): must fire on every user turn
        # so the DMN idle timer is deferred while the user is active.
        # WARNING level — failure here means DMN can fire mid-conversation
        # (Commit 8 critic P1-2).
        try:
            from services.dmn_service import get_dmn_service
            get_dmn_service().on_turn()
        except Exception as e:
            logger.warning(f"[POSTTURN] DMN on_turn failed: {e}", exc_info=True)

        # 4. Metrics (sync) — last: requests_total is the "turn closed" signal.
        # WARNING level — observability hole if it silently fails
        # (Commit 8 critic P1-2).
        try:
            from services.metrics_service import MetricsService
            m = MetricsService()
            m.record_counter('requests_total')
            m.record_counter('user_messages_total')
        except Exception as e:
            logger.warning(f"[POSTTURN] Metrics failed: {e}", exc_info=True)

        # 5. End-turn compaction backstop (safety net; mid-loop compaction in send()
        # should handle most cases).
        # WARNING level — silent compaction failure leads to context overflow
        # (Commit 8 critic P1-2).
        try:
            from services import compaction_service
            compaction_service.check_and_compact(channel, self._context_budget())
        except Exception as e:
            logger.warning(f"[POSTTURN] End-turn compaction failed: {e}", exc_info=True)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _context_budget(self) -> int:
        """Estimate the context budget for end-turn compaction check.

        Reads the context limit from the provider for JOB, caps at 60% of that
        limit or 150,000 tokens, falls back to 32,000 on error.
        """
        try:
            from services.providers import Providers
            ctx_limit = Providers.instance().get_context_limit(job=self.JOB)
            return min(int(ctx_limit * 0.6), 150_000)
        except Exception:
            return 32_000

    def _get_self_awareness(self) -> str:
        """Get system health degradation signals from SelfModelService.

        Returns empty string when the system is healthy — getUserPrompt()
        skips the section. Only populates when degradation is detected.
        """
        try:
            from services.self_model_service import SelfModelService
            return SelfModelService().format_for_prompt()
        except Exception as e:
            logger.debug(f"[USER MSG] Self-awareness unavailable: {e}")
            return ''

