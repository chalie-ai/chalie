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

COMMIT 8 STATUS: Amber.
    - Removes UserMessageProcessor.instance() singleton
    - Removes UserMessageProcessor.process()
    - Wires full postTurn() eight-service fan-out
    - Wires getUserPrompt() absorbing UserPromptAssemblyService
    - Wires getUserDefinition() absorbing identity modulation + adaptive directives
    - Wires _run_memory_seed() with ephemeral=0 DTO
    - Wires _emit_narration() callback
    - getSystemPrompt() override weaves {{adaptive_directives}} into the
      UnifiedSystemMessagePrompt template

LEAVE IN PLACE until Commit 11:
    - backend/services/user_prompt_assembly_service.py
    - backend/services/system_prompt_assembly_service.py
"""

import logging
from collections.abc import Callable

from services.message_processor import MessageProcessor
from services.system_message_prompt import UnifiedSystemMessagePrompt
from services.innate_skills.registry import ALL_SKILL_NAMES

logger = logging.getLogger(__name__)


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

    # All innate skills available for user turns; voice mode may narrow this
    # at runtime via metadata['source']=='voice' inside getTools().
    # Sorted for deterministic ordering — ALL_SKILL_NAMES is a frozenset.
    NATIVE_TOOLS: list[str] = sorted(ALL_SKILL_NAMES)

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
        # without this cache each iteration would re-read KnowledgeService DB rows.
        # The user summary is stable for the duration of a single turn.
        self._user_definition_cached: str | None = None
        # Cached voice modulation string. Per-turn cache — same reasoning as above.
        # Consumed only by getSystemPrompt() during {{voice_modulation}} weave.
        self._voice_modulation_cached: str | None = None

    # ── Abstract overrides ────────────────────────────────────────────────────

    def getUserDefinition(self) -> str:
        """One-sentence synthesis of the real human user for the system prompt.

        Reads the user_summary record (kind='system', key='user_summary') from data_graph.
        from DataGraphService and returns its value. This is a human-readable sentence
        that describes the user (e.g. "Dylan is a software engineer based in Malta").

        Falls back to a static peer-to-peer framing on empty or missing record, or on
        any exception.

        Writer path: populated at boot by ``run.py`` which synthesises a one-sentence
        summary from existing ``kind='user_specific'`` rows in data_graph. Traits
        themselves are written continuously by the LLM-native memory skill
        (``memory_skill._handle_store`` → ``DataGraphService.store(kind='user_specific', …)``)
        whenever the user discloses a personal fact. The background trait-extraction
        pipeline that used to produce user_summary mid-session was removed on
        2026-04-11 (trait-extraction RIP) — continuous re-synthesis now happens at
        the next boot, not mid-session. The fallback covers first-session (no traits
        yet) and any read error.

        Per-turn cached: getSystemPrompt() runs on every ACT iteration; without this
        cache each iteration would re-query the knowledge table.
        """
        if self._user_definition_cached is not None:
            return self._user_definition_cached
        try:
            from services.data_graph_service import get_data_graph_service

            rows = get_data_graph_service().fetch(kinds=['system'], order_by='retrieval_weight DESC')
            entry = next((r for r in rows if r.get('key') == 'user_summary'), None)
            if entry and entry.get('value'):
                self._user_definition_cached = entry['value']
            else:
                self._user_definition_cached = "The user is a real human. Treat this conversation as peer-to-peer dialogue."
        except Exception as e:
            logger.warning(f"[USER MSG] getUserDefinition failed: {e}")
            self._user_definition_cached = "The user is a real human. Treat this conversation as peer-to-peer dialogue."
        return self._user_definition_cached

    def getUserPrompt(self) -> str:
        """Build the user-message body for one ACT iteration.

        Section order (north star §"Body structure of getUserPrompt()"):
          1. World State block
          (1b. Voice-mode instruction if source == 'voice')
          2. System Awareness block
          3. ## Previous Messages block (via getPreviousMessages())
          (blank line separator)
          4. Memory seed line: [memory(radius=X)] ...
          5. Current turn line: user: <raw_input> [file_tags] [nudge_tag]
          6. ACT loop trail (empty string on iteration 1)

        Note: The ## Checkpoint / ## Current State envelope is NOT emitted
        here — send() wraps this output with it.
        """
        parts = []

        # 1. World State
        world_state = self._get_world_state()
        if world_state:
            parts.append(f"## World State\n{world_state}")

        # 1b. Voice mode instruction (per-turn — user may switch mode)
        if self._metadata.get('source') == 'voice':
            parts.append(
                'IMPORTANT: The user is in voice mode. Your response will be spoken aloud via TTS. '
                'Respond in plain conversational text only. No markdown formatting, code blocks, '
                'tables, bullet lists, links, or structured formatting. Write as you would speak.'
            )

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
        """Override to weave both dynamic placeholders into the UNIFIED prompt body.

        UnifiedSystemMessagePrompt.getPrompt() returns _UNIFIED_PROMPT, an
        inlined Python constant with two placeholders:
          - {{voice_modulation}} — mid-prompt in the Voice section; per-turn
            cached; supplied by _get_voice_modulation().
          - {{adaptive_directives}} — bottom of prompt; per-turn; supplied by
            _get_adaptive_directives().

        After both are filled, getUserDefinition() (the user synthesis line) is
        prepended as the first line of the final system prompt.

        Weave order: voice first (mid-prompt), adaptive second (bottom), then
        prepend user definition. This preserves the stable Identity/Boundaries/
        Principles prefix for provider-side prompt caching.
        """
        template = self.SYSTEM_PROMPT_CLASS().getPrompt()

        thread_id = self._metadata.get('thread_id')
        template = template.replace('{{voice_modulation}}', self._get_voice_modulation())
        template = template.replace('{{adaptive_directives}}', self._get_adaptive_directives(thread_id=thread_id))

        return f"{self.getUserDefinition()}\n\n{template}"

    def getTools(self) -> list[dict]:
        """Narrow native tools for voice mode (exclude rich_render).

        Voice responses are spoken aloud via TTS — rich_render output is not
        speakable, so it is excluded for voice-source turns. All other native
        tools are kept unchanged.

        Out-of-scope §11 preservation: keeps the voice filter logic that was
        in the old UserMessageProcessor.process() at lines 69-73.
        """
        if self._metadata.get('source') == 'voice':
            from services.tool_schema_service import get_skill_schemas

            voice_tools = get_skill_schemas(
                [s for s in self.NATIVE_TOOLS if s != 'rich_render']
            )
            dynamic = self.getDynamicTools()

            seen: set[str] = set()
            result: list[dict] = []
            for schema in voice_tools + dynamic:
                name = schema.get('name')
                if name and name not in seen:
                    seen.add(name)
                    result.append(schema)
            return result

        # Non-voice: standard base resolution
        return super().getTools()

    def _run_memory_seed(self) -> None:
        """Auto-seed memory once at turn start. Runs before getUserPrompt().

        Calls recall_episodes(caller='seed') directly so the telemetry row in
        memory_recall_log carries caller='seed'. This distinguishes the pre-turn
        seed from LLM-driven recall calls (caller='llm_recall') and allows the
        meta-harness to tune SEED_RADIUS_BASELINE independently.

        The result is stored in self._memory_seed and self._memory_seed_radius,
        and a durable (ephemeral=0) DTO is appended to self._pending_tool_calls
        so store() can write it linked to self._uid.

        Does NOT re-run on ACT iterations (send() calls this once, before the
        loop). Does NOT re-run on steers.

        Errors are logged and swallowed — a failed seed does not abort the turn.
        """
        from services.innate_skills.memory_skill import (
            recall_episodes, SEED_RADIUS_BASELINE, _format_results,
        )
        from services.time_utils import utc_now

        radius = SEED_RADIUS_BASELINE  # 0.2 per memory_skill.py module constant

        try:
            hits, _status = recall_episodes(
                channel=self.CHANNEL,
                query=self._raw_input,
                caller='seed',
                baseline_radius=radius,
                return_raw=False,
            )
            seed_text = _format_results(hits, self._raw_input) if hits else ''
        except Exception as exc:
            logger.warning(f"[USER MSG] Memory auto-seed failed: {exc}")
            seed_text = ''

        if seed_text:
            self._memory_seed = seed_text
            self._memory_seed_radius = radius
            self._pending_tool_calls.append({
                'name': 'memory',
                'params': {'action': 'recall', 'radius': radius},
                'result': seed_text,
                'ephemeral': 0,        # durable — replays in Previous Messages
                'invoked_by': 'system',
                'timestamp': utc_now().isoformat(),
            })
        else:
            # Seed returned nothing — still valid, just no injection in getUserPrompt()
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
        """Seven-service fan-out, each individually error-isolated.

        Order is load-bearing (see plan § "Ordering constraints"):
          1. ConversationPhaseService — two calls
          2. SituationModelService
          3. SaveSuggestionService — user-side trigger THEN response-side detect
          4. _detect_fork_response + _store_adaptive_signals
          5. DMNService.on_turn() — R10 critical
          6. MetricsService — last ("turn closed" signal)
          7. compaction_service.check_and_compact — end-turn backstop
        """
        channel = self.CHANNEL   # 'user'
        text = self._raw_input
        response = self._last_response
        metadata = self._metadata
        source = metadata.get('source', 'unknown')

        # thread_id: websocket.py does NOT inject thread_id into metadata.
        # Resolve it from MemoryStore active_channel key, exactly as
        # digest_worker.py:717 does. Fall back to metadata dict if caller
        # (e.g. tests) pre-injects it.
        thread_id = metadata.get('thread_id')
        if not thread_id:
            try:
                from services.memory_client import MemoryClientService
                _store = MemoryClientService.create_connection()
                _raw = _store.get('active_channel:default')
                if _raw:
                    thread_id = _raw.decode() if isinstance(_raw, bytes) else _raw
            except Exception as _tid_e:
                logger.debug(f"[POSTTURN] thread_id resolution failed: {_tid_e}")

        # 1. Conversation phase — TWO calls (user text + assistant response).
        # Skip the assistant-side update on empty response (cap-exit turns):
        # feeding '' would drift the phase model toward "silence" wrongly
        # (Commit 8 critic P1-3).
        try:
            from services.conversation_phase_service import get_conversation_phase_service
            phase = get_conversation_phase_service()
            phase.update(thread_id, text, is_user=True, topic=channel)
            if response:
                phase.update(thread_id, response, is_user=False, topic=channel)
        except Exception as e:
            logger.debug(f"[POSTTURN] Phase update failed: {e}", exc_info=True)

        # 2. Situation model (sync, MemoryStore-only write)
        try:
            from services.situation_model_service import get_situation_model_service
            get_situation_model_service().update_on_message(thread_id)
        except Exception as e:
            logger.debug(f"[POSTTURN] Situation update failed: {e}", exc_info=True)

        # 3. Save suggestion scan — TWO paths in correct order:
        #    3a: user-side trigger must fire BEFORE 3b detect (ordering constraint #3)
        try:
            from services.save_suggestion_service import SaveSuggestionService
            save_svc = SaveSuggestionService()

            # 3a: User-side completion/deferral trigger — clears existing flag
            save_flag = save_svc.get_saveable_flag(thread_id)
            if save_flag:
                trigger = save_svc.detect_save_trigger(text)
                if trigger:
                    save_svc.emit_save_card(
                        thread_id,
                        save_flag.get('topic', channel),
                        save_flag['content_type'],
                    )
                    save_svc.clear_flag(thread_id)

            # 3b: Response-side saveable content detection — sets new flag.
            # NOTE: flag_saveable expects exchange_id (UUID string) for window
            # correlation, not the integer transcript row id (self._uid). Prefer
            # the UUID from metadata; fall back to a stringified row id if absent.
            saveable = save_svc.detect_saveable_content(response, channel, thread_id)
            if saveable:
                exchange_id_for_flag = (
                    metadata.get('exchange_id')
                    or metadata.get('uuid')
                    or str(self._uid)
                )
                save_svc.flag_saveable(
                    thread_id, channel, saveable['content_type'], exchange_id_for_flag
                )
        except Exception as e:
            logger.debug(f"[POSTTURN] Save suggestion failed: {e}", exc_info=True)

        # 4. Adaptive layer — fork detection + signal write (sync, MemoryStore)
        try:
            from workers.post_exchange_hooks import _store_adaptive_signals, _detect_fork_response
            _detect_fork_response(text, thread_id)
            _store_adaptive_signals(thread_id, text)
        except Exception as e:
            logger.debug(f"[POSTTURN] Adaptive signals failed: {e}", exc_info=True)

        # 5. DMN idle reset — CRITICAL (R10): must fire on every user turn
        # so the DMN idle timer is deferred while the user is active.
        # WARNING level — failure here means DMN can fire mid-conversation
        # (Commit 8 critic P1-2).
        try:
            from services.dmn_service import get_dmn_service
            get_dmn_service().on_turn()
        except Exception as e:
            logger.warning(f"[POSTTURN] DMN on_turn failed: {e}", exc_info=True)

        # 6. Metrics (sync) — last: requests_total is the "turn closed" signal.
        # WARNING level — observability hole if it silently fails
        # (Commit 8 critic P1-2).
        try:
            from services.metrics_service import MetricsService
            m = MetricsService()
            m.record_counter('requests_total')
            m.record_counter('user_messages_total')
        except Exception as e:
            logger.warning(f"[POSTTURN] Metrics failed: {e}", exc_info=True)

        # 7. End-turn compaction backstop (safety net; mid-loop compaction in send()
        # should handle most cases — this mirrors digest_worker behaviour and
        # can be deleted in a follow-up once mid-loop compaction is confirmed).
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

        Mirrors digest_worker logic: reads the context limit from the provider
        for JOB, caps at 60% of that limit or 150,000 tokens, falls back to
        32,000 on error.
        """
        try:
            from services.providers import Providers
            ctx_limit = Providers.instance().get_context_limit(job=self.JOB)
            return min(int(ctx_limit * 0.6), 150_000)
        except Exception:
            return 32_000

    def _get_world_state(self) -> str:
        """Get world state string from WorldStateService.

        Returns empty string on error — getUserPrompt() skips the section.
        thread_id is extracted from metadata so WorldStateService can
        scope calendar/location data to the active thread if needed.
        """
        try:
            from services.world_state_service import WorldStateService
            thread_id = self._metadata.get('thread_id')
            return WorldStateService().get_world_state(self.CHANNEL, thread_id=thread_id)
        except Exception as e:
            logger.debug(f"[USER MSG] World state unavailable: {e}")
            return ''

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

    def _get_voice_modulation(self) -> str:
        """Per-turn cached voice modulation string for the Voice section of the system prompt.

        Calls IdentityService + VoiceMapperService to generate a behavioural style string
        (e.g. "Speak warmly with a tone of patient curiosity."). This string is injected into
        the {{voice_modulation}} placeholder in _UNIFIED_PROMPT by getSystemPrompt().

        Consumed only by getSystemPrompt(). Not called from getUserDefinition().

        Read-only: does NOT call identity.check_coherence() — that is a DB-writing
        path and belongs in postTurn() or a background service, not the per-turn
        prompt build which runs on every ACT iteration.

        Falls back to 'Engage naturally as a peer.' on any error.
        """
        if self._voice_modulation_cached is not None:
            return self._voice_modulation_cached
        try:
            from services.identity_service import IdentityService
            from services.voice_mapper_service import VoiceMapperService
            from services.database_service import get_shared_db_service

            db = get_shared_db_service()
            identity = IdentityService(db)
            mapper = VoiceMapperService()

            vectors = identity.get_vectors()
            modulation = mapper.generate_modulation(vectors)
            self._voice_modulation_cached = modulation if modulation else "Engage naturally as a peer."
        except Exception as e:
            logger.warning(f"[USER MSG] _get_voice_modulation failed: {e}")
            self._voice_modulation_cached = "Engage naturally as a peer."
        return self._voice_modulation_cached

    def _get_adaptive_directives(self, thread_id: str | None = None) -> str:
        """Get adaptive response directives for system prompt placeholder injection.

        The {{adaptive_directives}} placeholder in the UNIFIED template receives
        this value. Reads adaptive signals from MemoryStore keyed by thread_id.

        NOTE: This method does NOT use WorkingMemoryService (deprecated per
        project_memory_reduction.md and cleanup audit). The adaptive signals
        snapshot from MemoryStore is sufficient — WorkingMemoryService's
        get_recent_turns() was the only remaining consumer here.
        """
        try:
            from services.adaptive_layer_service import AdaptiveLayerService
            from services.database_service import get_shared_db_service

            db = get_shared_db_service()
            service = AdaptiveLayerService(db)

            current_signals = {
                'prompt_token_count': len(self._raw_input.split()),
            }
            if thread_id:
                try:
                    from services.memory_client import MemoryClientService
                    import json as _json
                    store = MemoryClientService.create_connection()
                    snapshot_raw = store.get(f"adaptive_signals:{thread_id}")
                    if snapshot_raw:
                        snapshot = _json.loads(snapshot_raw)
                        current_signals.update(snapshot)
                except Exception:
                    pass

            return service.generate_directives(
                thread_id=thread_id,
                working_memory_turns=[],  # WorkingMemoryService dropped
                current_signals=current_signals,
                current_message=self._raw_input,
            ) or ''
        except Exception as e:
            logger.warning(f"[USER MSG] Adaptive directives unavailable: {e}")
            return ''
