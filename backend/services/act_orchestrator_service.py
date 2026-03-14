"""
ACT Orchestrator Service — Single, parameterized ACT loop implementation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: This is the SOLE ACT loop implementation.

Do NOT copy this loop into workers, services, or anywhere else.
All ACT loop execution MUST go through ACTOrchestrator.run().
If you need different behavior, add a parameter to the constructor.

Historical context: Before this service existed, the ACT loop was
duplicated in tool_worker, digest_worker, and persistent_task_worker.
Features silently diverged across copies (critic only in tool_worker,
budget safety net only in digest_worker, etc.) causing reliability
gaps. This unification was done to prevent that class of bug.

If you believe you need a separate ACT loop, discuss with the team
first. The cost of duplication is always higher than parameterization.

Termination model (fatigue-free):
  - Hard iteration cap: max_iterations (default 30)
  - Cumulative timeout: safety net for runaway loops
  - Semantic repetition: embedding-based (>0.85 cosine similarity)
  - Type-based repetition: same action 3x in a row
  - No actions returned by LLM: natural completion signal
  - Soft nudge: at iteration 10, inject a prompt hint encouraging
    the LLM to conclude if it has enough information.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable

from services.act_loop_service import ActLoopService
from services.innate_skills.registry import COGNITIVE_PRIMITIVES

logger = logging.getLogger(__name__)
LOG_PREFIX = "[ACT ORCHESTRATOR]"


@dataclass
class ACTResult:
    """Outcome of a complete ACT orchestrator run."""
    act_history: list = field(default_factory=list)
    iteration_logs: list = field(default_factory=list)
    termination_reason: str = ''
    loop_id: Optional[str] = None
    iterations_used: int = 0
    critic_telemetry: dict = field(default_factory=dict)
    loop_telemetry: dict = field(default_factory=dict)
    reflection: Optional[dict] = None


class ACTOrchestrator:
    """
    Unified ACT loop runner with parameterized behavior.

    Replaces the three divergent loops in tool_worker, digest_worker,
    and persistent_task_worker. Each caller configures the behavior it
    needs via constructor parameters.
    """

    def __init__(
        self,
        config: dict,
        max_iterations: int = 30,
        cumulative_timeout: float = 60.0,
        per_action_timeout: float = 10.0,
        critic_enabled: bool = False,
        smart_repetition: bool = True,
        escalation_hints: bool = False,
        persistent_task_exit: bool = False,
        deferred_card_context: bool = False,
    ):
        """
        Args:
            config: Cortex configuration dict (model, timeouts, etc.)
            max_iterations: Hard iteration cap (default 30)
            cumulative_timeout: Maximum total loop time (seconds)
            per_action_timeout: Maximum time per individual action (seconds)
            critic_enabled: Deprecated — accepted for backward compatibility but
                ignored. Post-loop reflection always runs via _post_loop_reflection().
            smart_repetition: Embedding-based semantic repetition detection
            escalation_hints: Inject pivot hints on type-based repetition
            persistent_task_exit: Exit loop when a persistent_task is dispatched
            deferred_card_context: Inject deferred card offers into act_history
        """
        self.config = config
        self.max_iterations = max_iterations
        self.cumulative_timeout = cumulative_timeout
        self.per_action_timeout = per_action_timeout
        self.critic_enabled = critic_enabled
        self.smart_repetition = smart_repetition
        self.escalation_hints = escalation_hints
        self.persistent_task_exit = persistent_task_exit
        self.deferred_card_context = deferred_card_context

        # Repetition similarity threshold (configurable)
        self.repetition_sim_threshold = config.get(
            'act_repetition_similarity_threshold', 0.85
        )

    def run(
        self,
        topic: str,
        text: str,
        cortex_service,
        act_prompt: str,
        classification: dict,
        chat_history: list,
        relevant_tools=None,
        selected_skills=None,
        selected_tools=None,
        assembled_context=None,
        inclusion_map=None,
        on_iteration_complete: Optional[Callable] = None,
        on_narration: Optional[Callable] = None,
        context_extras: Optional[dict] = None,
        session_id: str = 'orchestrator',
        exchange_id: str = 'unknown',
        request_id: str = '',
    ) -> ACTResult:
        """
        Execute the unified ACT loop.

        Args:
            topic: Conversation topic
            text: Original user prompt
            cortex_service: FrontalCortexService for LLM calls
            act_prompt: ACT mode prompt template
            classification: Topic classification dict
            chat_history: Conversation history for context
            relevant_tools: Tools scored by embedding relevance
            selected_skills: Triage-selected innate skills
            selected_tools: Triage-selected tools
            assembled_context: Pre-assembled context from ContextAssemblyService
            inclusion_map: Context relevance inclusion map
            on_iteration_complete: Optional callback(act_loop, iteration_start, actions_executed,
                termination_reason) -> Optional[str]. Return a termination reason string
                to abort the loop, or None to continue. Use for heartbeat, cancellation,
                custom termination logic.
            on_narration: Optional callback(narration_text: str, step: int) -> None.
                Called when the LLM emits a narration line during a narrated ACT loop.
                Used by digest_worker to stream progress to the user via WebSocket.
            context_extras: Extra params merged into every action dispatch
            session_id: Session identifier for iteration logging
            exchange_id: Exchange correlation ID for iteration logging
            request_id: Per-request UUID for user steering (steer:{request_id} in MemoryStore)

        Returns:
            ACTResult with full loop outcome
        """
        # ── Build the ACT loop service ──────────────────────────────────
        act_loop = ActLoopService(
            config=self.config,
            cumulative_timeout=self.cumulative_timeout,
            per_action_timeout=self.per_action_timeout,
            max_iterations=self.max_iterations,
        )

        if context_extras:
            act_loop.context_extras = context_extras

        # ── Iteration logging ───────────────────────────────────────────
        iteration_service = None
        loop_id = None
        try:
            from services.database_service import get_shared_db_service
            from services.cortex_iteration_service import CortexIterationService
            db_service = get_shared_db_service()
            iteration_service = CortexIterationService(db_service)
            loop_id = iteration_service.create_loop_id()
            act_loop.loop_id = loop_id
            act_loop.context_extras['loop_id'] = loop_id
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Iteration logging init failed (will retry at write): {e}")

        # ── Narration state ───────────────────────────────────────────────
        self._narrated = False  # Set on iteration 0 by LLM decision
        self._request_id = request_id

        # ── Append mode: build system prompt once, grow message array ────
        append_mode = self.config.get('append_mode', False)
        _system_prompt = None   # Set below when append_mode is True
        _messages = None        # Growing message array for append mode

        if append_mode:
            _system_prompt = cortex_service.build_system_prompt(
                system_prompt_template=act_prompt,
                original_prompt=text,
                classification=classification,
                chat_history=chat_history,
                assembled_context=assembled_context,
                relevant_tools=relevant_tools,
                selected_tools=selected_tools,
                selected_skills=selected_skills,
                thread_id=session_id,
                returning_from_silence=False,
                inclusion_map=inclusion_map,
            )
            _messages = [{"role": "user", "content": text}]
            logger.debug(f"{LOG_PREFIX} Append mode: system prompt built once ({len(_system_prompt)} chars)")

        # ── Repetition detection state ──────────────────────────────────
        consecutive_same_action = 0
        last_action_type = None
        pivot_hint_injected = False
        recent_action_entries = []  # (fingerprint, types_set) for smart repetition

        # ── Tool health tracking (cross-loop via MemoryStore) ─────────
        from services.tool_health_service import (
            get_potential, record_outcome as _record_health,
            classify_result as _classify_health, format_health_hint,
        )

        termination_reason = None

        # ── Main loop ───────────────────────────────────────────────────
        while True:
            iteration_start = time.time()

            # ── Collect tool names for health hints ───────────────────
            _tool_names = set()
            if selected_tools:
                _tool_names.update(selected_tools)
            if relevant_tools:
                _tool_names.update(
                    item['name'] for item in relevant_tools
                    if isinstance(item, dict) and item.get('type') == 'tool'
                )

            if append_mode:
                # ── Append mode: grow message array ──────────────────────
                # Collect per-iteration context updates into a single user message
                # so the system prompt (and its cache) stay untouched each iteration.
                context_updates = []

                # Steering from the user mid-loop
                if self._request_id:
                    steer_text = self._get_steering_text()
                    if steer_text:
                        context_updates.append(steer_text)

                # Tool health signals (degraded tools only)
                if _tool_names:
                    _potentials = {t: get_potential(t) for t in _tool_names}
                    _health_hint = format_health_hint(_potentials)
                    if _health_hint:
                        context_updates.append(f"[Tool Health]\n{_health_hint}")

                # Cautionary lessons from procedural memory (after first iteration)
                if act_loop.act_history:
                    _lessons_hint = self._get_cautionary_lessons(act_loop.act_history)
                    if _lessons_hint:
                        context_updates.append(f"[Cautionary Lessons]\n{_lessons_hint}")

                # ACT history delta (results from the last iteration's actions)
                act_history_str = act_loop.get_history_context()
                if self.deferred_card_context:
                    act_history_str = self._inject_deferred_card_context(
                        act_history_str, topic
                    )
                if act_history_str and act_history_str != "(none)":
                    context_updates.append(act_history_str)

                if context_updates:
                    _messages.append({
                        "role": "user",
                        "content": "\n\n".join(context_updates),
                    })

                # Token budget guard — prune oldest message pairs when approaching limit
                context_budget = self.config.get('context_budget_tokens', 32000)
                _messages = self._prune_messages(_messages, context_budget)

                response_data = cortex_service.generate_response_appended(
                    system_prompt=_system_prompt,
                    messages=_messages,
                    cache_prefix=True,
                )

                # Append assistant turn for the next iteration
                raw_response = response_data.get('raw_response', response_data.get('response', ''))
                if raw_response:
                    _messages.append({"role": "assistant", "content": raw_response})

            else:
                # ── Legacy mode: rebuild full prompt each iteration ────────
                act_history_str = act_loop.get_history_context()
                if self.deferred_card_context:
                    act_history_str = self._inject_deferred_card_context(
                        act_history_str, topic
                    )

                # Inject user steering input (if any)
                if self._request_id:
                    act_history_str = self._inject_steering(act_history_str)

                # Inject tool health signals (degraded tools only)
                if _tool_names:
                    _potentials = {t: get_potential(t) for t in _tool_names}
                    _health_hint = format_health_hint(_potentials)
                    if _health_hint:
                        act_history_str += f"\n\n[Tool Health]\n{_health_hint}"

                # Inject cautionary lessons (after first iteration)
                if act_loop.act_history:
                    _lessons_hint = self._get_cautionary_lessons(act_loop.act_history)
                    if _lessons_hint:
                        act_history_str += f"\n\n[Cautionary Lessons]\n{_lessons_hint}"

                response_data = cortex_service.generate_response(
                    system_prompt_template=act_prompt,
                    original_prompt=text,
                    classification=classification,
                    chat_history=chat_history,
                    act_history=act_history_str,
                    relevant_tools=relevant_tools,
                    selected_skills=selected_skills,
                    selected_tools=selected_tools,
                    assembled_context=assembled_context,
                    inclusion_map=inclusion_map,
                )

            actions = response_data.get('actions', [])

            # ── Narration gate (iteration 0 only) + emission ─────────────
            if act_loop.iteration_number == 0:
                self._narrated = bool(response_data.get('narrated', False))
                if self._narrated:
                    logger.info(f"{LOG_PREFIX} Narrated ACT loop enabled")

            if self._narrated:
                narration_text = response_data.get('narration', '')
                logger.info(
                    f"{LOG_PREFIX} Narration check: has_callback={on_narration is not None}, "
                    f"has_text={bool(narration_text)}, text={narration_text[:50] if narration_text else '(none)'}"
                )
                if on_narration and narration_text:
                    try:
                        on_narration(narration_text, act_loop.iteration_number)
                    except Exception as e:
                        logger.error(f"{LOG_PREFIX} Narration callback error: {e}", exc_info=True)

            # ── Repetition detection (type-based) ───────────────────────
            if actions and len(actions) == 1:
                current_type = actions[0].get('type', '')
                if current_type == last_action_type:
                    consecutive_same_action += 1
                else:
                    consecutive_same_action = 1
                last_action_type = current_type
            elif actions:
                consecutive_same_action = 0
                last_action_type = None

            if consecutive_same_action >= 3:
                if self.escalation_hints and not pivot_hint_injected:
                    # Digest-worker style: inject pivot hint, give one more chance
                    logger.info(
                        f"{LOG_PREFIX} Repetition '{last_action_type}' x{consecutive_same_action} "
                        f"— injecting pivot hint"
                    )
                    act_loop.append_results([{
                        'action_type': 'system',
                        'status': 'info',
                        'execution_time': 0.0,
                        'result': (
                            f"SYSTEM: You have called '{last_action_type}' "
                            f"{consecutive_same_action} times with similar queries. "
                            "Try a DIFFERENT action type that builds on existing results, "
                            "or return empty actions to finish."
                        ),
                    }])
                    pivot_hint_injected = True
                    consecutive_same_action = 0
                    can_continue, termination_reason = act_loop.can_continue()
                else:
                    # Tool-worker style: hard exit on repetition
                    logger.warning(
                        f"{LOG_PREFIX} Repetition detected: '{last_action_type}' "
                        f"x{consecutive_same_action} — forcing exit"
                    )
                    termination_reason = 'repetition_detected'
                    can_continue = False
            else:
                can_continue, termination_reason = act_loop.can_continue()

            # ── Soft nudge at iteration 10 ───────────────────────────────
            if (
                can_continue
                and actions
                and act_loop.iteration_number >= 10
                and not act_loop.soft_nudge_injected
            ):
                logger.info(
                    f"{LOG_PREFIX} Soft nudge at iteration {act_loop.iteration_number} "
                    f"— hinting LLM to conclude if sufficient information gathered"
                )
                act_loop.append_results([{
                    'action_type': 'system',
                    'status': 'info',
                    'execution_time': 0.0,
                    'result': (
                        "SYSTEM: You've been working on this for a while. "
                        "If you have enough information to respond, do so now by returning "
                        "empty actions. If not, continue exploring."
                    ),
                }])
                act_loop.soft_nudge_injected = True

            # ── No actions → exit ───────────────────────────────────────
            if not actions:
                logger.info(f"{LOG_PREFIX} No actions, exiting ACT loop")
                termination_reason = 'no_actions'
                act_loop.log_iteration(
                    started_at=iteration_start,
                    completed_at=time.time(),
                    chosen_mode='ACT',
                    chosen_confidence=response_data.get('confidence', 0.5),
                    actions_executed=[],
                    frontal_cortex_response=response_data,
                    termination_reason=termination_reason,
                    decision_data={'net_value': 0.0},
                )
                act_loop.iteration_number += 1
                break

            if not can_continue:
                # Log the skipped iteration before breaking
                act_loop.log_iteration(
                    started_at=iteration_start,
                    completed_at=time.time(),
                    chosen_mode='ACT',
                    chosen_confidence=response_data.get('confidence', 0.5),
                    actions_executed=[],
                    frontal_cortex_response=response_data,
                    termination_reason=termination_reason,
                    decision_data={'net_value': 0.0},
                )
                act_loop.iteration_number += 1
                break

            # ── Execute actions ─────────────────────────────────────────
            actions_executed = act_loop.execute_actions(
                topic=topic,
                actions=actions,
            )

            # ── Tool health: record outcomes + check exhaustion ────────
            for _exec_r in actions_executed:
                _atype = _exec_r.get('action_type', '')
                if _atype in COGNITIVE_PRIMITIVES or _atype == 'system':
                    continue  # Only track external tools
                _outcome = _classify_health(_exec_r)
                _new_potential = _record_health(_atype, _outcome)
                if _new_potential < 0.15 and not termination_reason:
                    logger.warning(
                        f"{LOG_PREFIX} Tool '{_atype}' exhausted "
                        f"(potential={_new_potential:.2f}) — forcing exit"
                    )
                    termination_reason = 'tool_exhausted'

            act_loop.append_results(actions_executed)

            # ── Smart repetition detection (embedding-based) ────────────
            if self.smart_repetition and not termination_reason:
                current_fingerprint = _action_fingerprint(actions)
                current_types = _action_types(actions)
                recent_action_entries.append((current_fingerprint, current_types))

                if len(recent_action_entries) > 1:
                    smart_reason = self._check_smart_repetition(
                        current_fingerprint, current_types, recent_action_entries
                    )
                    if smart_reason:
                        termination_reason = smart_reason

            # ── Persistent task exit ────────────────────────────────────
            if self.persistent_task_exit and not termination_reason:
                if any(
                    r.get('action_type') == 'persistent_task'
                    and r.get('status') == 'success'
                    for r in actions_executed
                ):
                    logger.info(
                        f"{LOG_PREFIX} persistent_task dispatched — exiting loop"
                    )
                    termination_reason = 'persistent_task_dispatched'

            # ── Record skill outcomes ───────────────────────────────────
            try:
                from services.skill_outcome_recorder import record_skill_outcomes
                record_skill_outcomes(actions_executed, topic)
            except Exception:
                pass

            # ── Check timeout/max_iterations if no reason yet ───────────
            if not termination_reason:
                can_continue, exit_reason = act_loop.can_continue()
                if not can_continue:
                    termination_reason = exit_reason

            # ── Log iteration ───────────────────────────────────────────
            iteration_end = time.time()
            act_loop.log_iteration(
                started_at=iteration_start,
                completed_at=iteration_end,
                chosen_mode='ACT',
                chosen_confidence=response_data.get('confidence', 0.5),
                actions_executed=actions_executed,
                frontal_cortex_response=response_data,
                termination_reason=termination_reason if termination_reason else None,
                decision_data={'net_value': 0.0},
            )

            act_loop.iteration_number += 1

            # ── Caller callback (heartbeat, cancellation, etc.) ─────────
            if on_iteration_complete:
                try:
                    callback_reason = on_iteration_complete(
                        act_loop, iteration_start, actions_executed, termination_reason
                    )
                    if callback_reason and not termination_reason:
                        termination_reason = callback_reason
                except Exception as e:
                    logger.warning(f"{LOG_PREFIX} on_iteration_complete error: {e}")

            if termination_reason:
                break

        # ── Post-loop: batch write iterations ───────────────────────────
        if act_loop.iteration_logs:
            # If init failed earlier, retry now — DB is guaranteed ready post-loop
            if iteration_service is None:
                try:
                    from services.database_service import get_shared_db_service
                    from services.cortex_iteration_service import CortexIterationService
                    db_service = get_shared_db_service()
                    iteration_service = CortexIterationService(db_service)
                    loop_id = iteration_service.create_loop_id()
                except Exception as e:
                    logger.error(
                        f"{LOG_PREFIX} Iteration logging unavailable — "
                        f"{len(act_loop.iteration_logs)} iterations lost: {e}"
                    )
            if iteration_service:
                try:
                    iteration_service.log_iterations_batch(
                        loop_id=loop_id,
                        topic=topic,
                        exchange_id=exchange_id,
                        session_id=session_id,
                        iterations=act_loop.iteration_logs,
                    )
                except Exception as e:
                    logger.error(f"{LOG_PREFIX} Failed to log iterations: {e}")

        # ── Post-loop: loop telemetry ────────────────────────────────────
        loop_telemetry = act_loop.get_loop_telemetry()
        loop_telemetry['termination_reason'] = termination_reason
        logger.info(f"{LOG_PREFIX} Loop telemetry: {loop_telemetry}")

        try:
            from services.database_service import get_shared_db_service
            from services.interaction_log_service import InteractionLogService
            _tel_db = get_shared_db_service()
            _tel_log = InteractionLogService(_tel_db)
            _tel_log.log_event(
                event_type='act_loop_telemetry',
                payload=loop_telemetry,
                topic=topic,
                source='act_loop',
            )
        except Exception:
            pass

        # ── Post-loop: automatic reflection (fire-and-forget) ────────
        _maybe_auto_reflect(
            topic=topic,
            iteration_logs=act_loop.iteration_logs,
            termination_reason=termination_reason,
            iterations_used=act_loop.iteration_number,
        )

        # ── Post-loop: critic reflection → procedural memory ─────────
        reflection = self._post_loop_reflection(
            act_history=act_loop.act_history,
            original_goal=text,
            iterations_used=act_loop.iteration_number,
            termination_reason=termination_reason or '',
            topic=topic,
        )

        return ACTResult(
            act_history=act_loop.act_history,
            iteration_logs=act_loop.iteration_logs,
            termination_reason=termination_reason or '',
            loop_id=loop_id,
            iterations_used=act_loop.iteration_number,
            critic_telemetry={},
            loop_telemetry=loop_telemetry,
            reflection=reflection,
        )

    # ── Private helpers ─────────────────────────────────────────────────

    def _inject_steering(self, act_history_str: str) -> str:
        """Check MemoryStore for user steering input and inject into act_history.

        Used by legacy (non-append) mode only.  Append mode uses
        :meth:`_get_steering_text` to obtain the steering text as a plain
        string that is appended to the context-update message instead.
        """
        try:
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            steer_key = f"steer:{self._request_id}"
            steers = store.lrange(steer_key, 0, -1)
            if steers:
                store.delete(steer_key)
                for steer in steers:
                    steer_text = steer if isinstance(steer, str) else steer.decode()
                    act_history_str += f"\n\n⚡ [User interrupted]: {steer_text}"
                    logger.info(f"{LOG_PREFIX} Injected user steer: {steer_text[:80]}")
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Steering check failed: {e}")
        return act_history_str

    def _get_steering_text(self) -> str:
        """Drain the MemoryStore steering queue and return formatted text.

        Used by append mode so that steering content can be included in a
        discrete user message rather than appended to the act_history string.

        Returns:
            Formatted steering lines joined by newlines, or an empty string
            when there is no pending steering input or the store is unavailable.
        """
        try:
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            steer_key = f"steer:{self._request_id}"
            steers = store.lrange(steer_key, 0, -1)
            if steers:
                store.delete(steer_key)
                parts = []
                for steer in steers:
                    steer_text = steer if isinstance(steer, str) else steer.decode()
                    parts.append(f"⚡ [User interrupted]: {steer_text}")
                    logger.info(f"{LOG_PREFIX} Injected user steer: {steer_text[:80]}")
                return '\n'.join(parts)
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Steering check failed: {e}")
        return ''

    def _prune_messages(self, messages: list, budget_tokens: int) -> list:
        """Prune the oldest user/assistant pairs when the message array nears the token budget.

        The first message (the original user prompt) is always kept.  When the
        estimated token count exceeds ``budget_tokens``, the oldest non-first
        messages are dropped in pairs until the array fits the budget or only
        the minimum tail (first message + 2 most-recent messages) remains.

        Token count is estimated as ``word_count * 1.3`` — the same heuristic
        used elsewhere in the ACT loop.

        Args:
            messages: Current message array (mutated copy is returned; original
                is not modified)
            budget_tokens: Token budget threshold.  No pruning occurs when the
                estimated count is at or below this value.

        Returns:
            Pruned (or unchanged) message list.
        """
        if not messages:
            return messages

        total_text = ' '.join(m.get('content', '') for m in messages)
        estimated_tokens = int(len(total_text.split()) * 1.3)

        if estimated_tokens <= budget_tokens or len(messages) <= 3:
            return messages

        # Keep the first message (original user prompt) plus the most-recent
        # tail.  Start with the last 4 messages and expand backward; if that
        # still exceeds the budget, remove pairs from position 1 onward.
        keep_tail = min(4, len(messages) - 1)
        pruned = [messages[0]] + messages[-keep_tail:]

        while len(pruned) > 3:
            total = ' '.join(m.get('content', '') for m in pruned)
            if int(len(total.split()) * 1.3) <= budget_tokens:
                break
            # Remove the oldest non-first message
            pruned.pop(1)

        logger.debug(
            f"{LOG_PREFIX} _prune_messages: {len(messages)} → {len(pruned)} messages "
            f"(est. {estimated_tokens} tokens > budget {budget_tokens})"
        )
        return pruned

    def _post_loop_reflection(
        self,
        act_history: list,
        original_goal: str,
        iterations_used: int,
        termination_reason: str,
        topic: str,
    ) -> Optional[dict]:
        """Run post-loop critic reflection and store the lesson in procedural memory.

        This is the only critic call in the ACT loop. It runs once, after the loop
        exits, and feeds the result into procedural memory. It never blocks the
        response — failures are caught and logged.

        Returns:
            Reflection dict {outcome_quality, what_worked, what_failed, lesson,
            confidence} or None if reflection failed or was skipped.
        """
        # Skip trivial single-action loops — not enough signal
        if iterations_used < 2:
            return None

        try:
            from services.critic_service import CriticService

            # Extract actions and results from act_history
            actions_taken = []
            results = []
            for entry in act_history:
                if isinstance(entry, dict):
                    atype = entry.get('action_type', '')
                    if atype and atype != 'system':
                        actions_taken.append({'type': atype})
                        results.append(entry)

            if not results:
                return None

            critic = CriticService()
            reflection = critic.reflect_on_execution(
                actions_taken=actions_taken,
                results=results,
                original_goal=original_goal,
                iterations=iterations_used,
                termination_reason=termination_reason,
            )

            if reflection is None:
                return None

            # Store lesson in procedural memory for each unique action type used
            lesson = reflection.get('lesson')
            outcome_quality = reflection.get('outcome_quality', 0.5)
            if lesson:
                try:
                    from services.database_service import get_shared_db_service
                    from services.procedural_memory_service import ProceduralMemoryService
                    db = get_shared_db_service()
                    proc_mem = ProceduralMemoryService(db)

                    # Record outcome for each action type used in the loop
                    seen_types = set()
                    for entry in results:
                        atype = entry.get('action_type', '')
                        if not atype or atype in ('system', 'critic_escalation') or atype in seen_types:
                            continue
                        seen_types.add(atype)
                        success = outcome_quality >= 0.5
                        reward = (outcome_quality - 0.5) * 2.0  # map [0,1] → [-1,1]
                        proc_mem.record_action_outcome(
                            action_name=atype,
                            success=success,
                            reward=reward,
                            topic=topic,
                        )
                except Exception as e:
                    logger.debug(f"{LOG_PREFIX} Procedural memory write failed (non-fatal): {e}")

            # Record failure lessons when outcome is poor
            if outcome_quality < 0.4 and reflection.get('what_failed'):
                for atype in seen_types:
                    self._record_failure_lesson(
                        action_type=atype,
                        failure_context={
                            'original_request': original_goal,
                            'action_type': atype,
                            'action_intent': {},
                            'action_result': {'status': 'poor_outcome', 'quality': outcome_quality},
                            'error_signals': {
                                'what_failed': reflection['what_failed'],
                                'termination_reason': termination_reason,
                            },
                        },
                        severity='minor',
                    )

            return reflection

        except Exception as e:
            logger.debug(f"{LOG_PREFIX} _post_loop_reflection failed (non-fatal): {e}")
            return None

    def _get_cautionary_lessons(self, recent_history: list) -> str:
        """Retrieve failure lessons relevant to recently executed action types."""
        try:
            from services.failure_analysis_service import FailureAnalysisService
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            fas = FailureAnalysisService(db)
            action_types = {
                r.get('action_type', '')
                for r in recent_history
                if r.get('action_type')
            }
            all_lessons = []
            for at in action_types:
                all_lessons.extend(fas.get_relevant_lessons(at))
            if not all_lessons:
                return ''
            all_lessons.sort(key=lambda l: l.get('times_seen', 1), reverse=True)
            lines = [
                f"- [{l['blame']}] {l['lesson']} (seen {l.get('times_seen', 1)}x)"
                for l in all_lessons[:3]
            ]
            return '\n'.join(lines)
        except Exception:
            return ''

    def _record_failure_lesson(
        self, action_type: str, failure_context: dict, severity: str = 'minor'
    ) -> None:
        """Analyse a failed action and store a lesson. Major = sync, minor = async."""
        def _do_record():
            try:
                from services.failure_analysis_service import FailureAnalysisService
                from services.database_service import get_shared_db_service
                db = get_shared_db_service()
                fas = FailureAnalysisService(db)
                analysis = fas.analyze(failure_context)
                if analysis:
                    fas.store_lesson(analysis, action_type)
            except Exception as exc:
                logger.warning(f"{LOG_PREFIX} Failure lesson recording failed: {exc}")

        if severity == 'major':
            _do_record()
        else:
            import threading
            t = threading.Thread(
                target=_do_record,
                daemon=True,
                name=f"failure-lesson-{action_type[:20]}",
            )
            t.start()

    def _escalate_and_wait(
        self,
        act_loop: ActLoopService,
        escalation_text: str,
        exchange_id: str,
        poll_interval: float = 1.0,
        max_wait: float = 30.0,
    ) -> str | None:
        """Send critic escalation to user and block until they respond or timeout."""
        import time as _time
        topic = act_loop.context_extras.get('topic', '')
        try:
            from services.output_service import OutputService
            OutputService().enqueue_text(
                topic=topic, response=escalation_text, mode='ACT',
                confidence=0.0, generation_time=0.0,
                original_metadata={'source': 'critic_escalation', 'exchange_id': exchange_id},
            )
        except Exception as _esc_err:
            logger.warning(f"{LOG_PREFIX} Failed to send escalation: {_esc_err}")
            return None

        if not self._request_id:
            logger.warning(f"{LOG_PREFIX} No request_id — cannot wait for user response")
            return None

        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()
        steer_key = f"steer:{self._request_id}"
        deadline = _time.monotonic() + max_wait
        logger.info(f"{LOG_PREFIX} Waiting up to {max_wait}s for user response on {steer_key}")

        while _time.monotonic() < deadline:
            _time.sleep(poll_interval)
            steers = store.lrange(steer_key, 0, -1)
            if steers:
                store.delete(steer_key)
                response = steers[0] if isinstance(steers[0], str) else steers[0].decode()
                logger.info(f"{LOG_PREFIX} User responded to escalation: {response[:80]}")
                return response

        logger.info(f"{LOG_PREFIX} Escalation timed out after {max_wait}s")
        return None

    def _check_smart_repetition(
        self,
        current_fingerprint: str,
        current_types: set,
        recent_entries: list,
    ) -> Optional[str]:
        """Embedding-based semantic repetition check (same-type only).

        Requires 2+ consecutive similar iterations to trigger — a single
        similar search is "exploring a topic from different angles", not
        being stuck.
        """
        try:
            from services.embedding_service import get_embedding_service
            import numpy as np

            emb_service = get_embedding_service()
            current_vec = emb_service.generate_embedding_np(current_fingerprint)

            consecutive_hits = 0
            # Check most recent entries (newest first)
            for prev_fingerprint, prev_types in reversed(recent_entries[:-1]):
                if not current_types & prev_types:
                    break  # Type mismatch breaks the consecutive streak
                prev_vec = emb_service.generate_embedding_np(prev_fingerprint)
                sim = float(np.dot(current_vec, prev_vec))
                if sim > self.repetition_sim_threshold:
                    consecutive_hits += 1
                else:
                    break  # Below threshold breaks the streak

            if consecutive_hits >= 2:
                logger.warning(
                    f"{LOG_PREFIX} Smart repetition (same-type): "
                    f"{consecutive_hits} consecutive similar iterations "
                    f"(threshold={self.repetition_sim_threshold})"
                )
                return 'smart_repetition'
        except Exception:
            pass
        return None

    @staticmethod
    def _inject_deferred_card_context(
        act_history_str: str, topic: str
    ) -> str:
        """Append deferred card offers to act_history context string."""
        try:
            from services.memory_client import MemoryClientService as _RCS
            _store_dc = _RCS.create_connection()
            _deferred_items = _store_dc.lrange(
                f"deferred_cards:{topic}", 0, -1
            )
            if _deferred_items:
                _cards = [json.loads(i) for i in _deferred_items]
                _lines = ["\n## Available Card Offers"]
                for _card in _cards:
                    _media = []
                    if _card.get("has_images"):
                        _media.append("images")
                    _media_str = (
                        f" ({', '.join(_media)})" if _media else ""
                    )
                    _lines.append(
                        f"- {_card['tool_name']} "
                        f"(id: {_card['invocation_id']}, "
                        f"{_card['source_count']} sources, "
                        f"{_card['unique_domains']} domains{_media_str})"
                    )
                _lines.append(
                    "\nUse emit_card with the invocation_id above to "
                    "display a visual card, or return empty actions to "
                    "respond with text."
                )
                act_history_str += "\n".join(_lines)
        except Exception as _dc_err:
            logger.debug(
                f"{LOG_PREFIX} Deferred card context injection failed: "
                f"{_dc_err}"
            )
        return act_history_str


# ── Auto-reflection (post-loop, fire-and-forget) ────────────────────

# Thresholds for triggering automatic reflection
_AUTO_REFLECT_HIGH_VALUE = 3.0    # Total net value above this → "what worked"
_AUTO_REFLECT_LOW_VALUE = -1.0    # Total net value below this → "what didn't"
_AUTO_REFLECT_COOLDOWN_S = 1800   # 30 min cooldown per topic
_AUTO_REFLECT_MIN_ITERATIONS = 2  # Skip trivial 1-iteration loops

# Termination reasons that indicate degraded exits worth reflecting on
_DEGRADED_EXITS = frozenset({
    'repetition_detected', 'smart_repetition', 'tool_exhausted',
})


def _maybe_auto_reflect(
    topic: str,
    iteration_logs: list,
    termination_reason: str | None,
    iterations_used: int,
) -> None:
    """
    Fire background reflection after significant ACT loops.

    Triggers on: high-value loops, negative-value loops, or degraded exits.
    Uses MemoryStore cooldown to prevent spam (1 per topic per 30 min).
    Never blocks — runs in a daemon thread.
    """
    import threading

    if iterations_used < _AUTO_REFLECT_MIN_ITERATIONS:
        return

    # Aggregate net value from iteration logs
    total_net_value = sum(
        log.get('net_value', 0.0) for log in iteration_logs
    )

    should_reflect = (
        total_net_value >= _AUTO_REFLECT_HIGH_VALUE
        or total_net_value <= _AUTO_REFLECT_LOW_VALUE
        or (termination_reason or '') in _DEGRADED_EXITS
    )

    if not should_reflect:
        return

    # Check cooldown
    try:
        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()
        cooldown_key = f"auto_reflect_cooldown:{topic}"
        if store.get(cooldown_key):
            logger.debug(f"{LOG_PREFIX} Auto-reflect cooldown active for {topic}")
            return
        store.setex(cooldown_key, _AUTO_REFLECT_COOLDOWN_S, '1')
    except Exception:
        pass  # If cooldown check fails, proceed anyway

    reason = (
        f"high_value({total_net_value:.1f})" if total_net_value >= _AUTO_REFLECT_HIGH_VALUE
        else f"low_value({total_net_value:.1f})" if total_net_value <= _AUTO_REFLECT_LOW_VALUE
        else f"degraded_exit({termination_reason})"
    )
    logger.info(f"{LOG_PREFIX} Triggering auto-reflect: {reason}")

    def _run_reflect():
        try:
            from services.innate_skills.reflect_skill import handle_reflect
            handle_reflect(topic, {
                'query': f'automatic reflection triggered by: {reason}',
                'scope': 'recent',
                'store': True,
            })
            logger.info(f"{LOG_PREFIX} Auto-reflect completed for {topic}")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Auto-reflect failed: {e}")

    t = threading.Thread(target=_run_reflect, daemon=True, name=f"auto-reflect-{topic[:20]}")
    t.start()


# ── Fingerprinting utilities (shared across all loop callers) ───────

def _action_fingerprint(actions: list) -> str:
    """Create a text fingerprint from action specs for embedding comparison."""
    parts = []
    for a in actions:
        atype = a.get('type', '')
        query = a.get('query', a.get('description', a.get('text', '')))
        parts.append(f"{atype}:{query}")
    return ' | '.join(parts)


def _action_types(actions: list) -> set:
    """Extract the set of action types from a list of action specs."""
    return {a.get('type', '') for a in actions}
