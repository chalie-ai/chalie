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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable

from services.act_loop_service import ActLoopService
from services.act_action_categories import ACTION_FATIGUE_COSTS
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
    fatigue: float = 0.0
    iterations_used: int = 0
    critic_telemetry: dict = field(default_factory=dict)
    fatigue_telemetry: dict = field(default_factory=dict)


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
        max_iterations: int = 7,
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
            config: Cortex configuration dict (fatigue budget, model, etc.)
            max_iterations: Hard iteration cap
            cumulative_timeout: Maximum total loop time (seconds)
            per_action_timeout: Maximum time per individual action (seconds)
            critic_enabled: Post-action verification via CriticService
            smart_repetition: Embedding-based semantic repetition detection
            escalation_hints: Budget safety net warnings for the LLM
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
        context_extras: Optional[dict] = None,
        session_id: str = 'orchestrator',
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
            context_extras: Extra params merged into every action dispatch
            session_id: Session identifier for iteration logging

        Returns:
            ACTResult with full loop outcome
        """
        # ── Build the ACT loop service ──────────────────────────────────
        critic = None
        if self.critic_enabled:
            from services.critic_service import CriticService
            critic = CriticService()

        act_loop = ActLoopService(
            config=self.config,
            cumulative_timeout=self.cumulative_timeout,
            per_action_timeout=self.per_action_timeout,
            max_iterations=self.max_iterations,
            critic=critic,
        )

        if context_extras:
            act_loop.context_extras = context_extras

        # ── Iteration logging ───────────────────────────────────────────
        iteration_service = None
        loop_id = None
        exchange_id = 'unknown'
        try:
            from services.database_service import get_shared_db_service
            from services.cortex_iteration_service import CortexIterationService
            db_service = get_shared_db_service()
            iteration_service = CortexIterationService(db_service)
            loop_id = iteration_service.create_loop_id()
            act_loop.loop_id = loop_id
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Iteration logging init failed: {e}")

        # ── Repetition detection state ──────────────────────────────────
        consecutive_same_action = 0
        last_action_type = None
        pivot_hint_injected = False
        recent_action_entries = []  # (fingerprint, types_set) for smart repetition

        termination_reason = None

        # ── Main loop ───────────────────────────────────────────────────
        while True:
            iteration_start = time.time()

            # Build act_history string (with optional deferred card context)
            act_history_str = act_loop.get_history_context()
            if self.deferred_card_context:
                act_history_str = self._inject_deferred_card_context(
                    act_history_str, topic
                )

            # Generate action plan via LLM
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

            # ── Budget safety net (escalation hints) ────────────────────
            if (
                self.escalation_hints
                and can_continue
                and actions
                and not act_loop._escalation_hint_injected
            ):
                self._maybe_inject_budget_warning(act_loop, actions)

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
                    decision_data={
                        'net_value': 0.0,
                        'total_cost': act_loop.fatigue,
                        'iteration_cost': 0.0,
                    },
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
                    decision_data={
                        'net_value': 0.0,
                        'total_cost': act_loop.fatigue,
                        'iteration_cost': 0.0,
                    },
                )
                act_loop.iteration_number += 1
                break

            # ── Execute actions ─────────────────────────────────────────
            actions_executed = act_loop.execute_actions(
                topic=topic,
                actions=actions,
            )

            # ── Critic verification (if enabled) ────────────────────────
            if self.critic_enabled and critic:
                actions_executed = self._run_critic(
                    act_loop, critic, text, actions, actions_executed, exchange_id
                )

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

            # ── Fatigue + net value ─────────────────────────────────────
            fatigue_added = act_loop.accumulate_fatigue(
                actions_executed, act_loop.iteration_number
            )
            iteration_net_value = ActLoopService.estimate_net_value(
                actions_executed, act_loop.iteration_number
            )

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

            # ── Check fatigue/timeout/max_iterations if no reason yet ───
            if not termination_reason:
                can_continue, fatigue_reason = act_loop.can_continue()
                if not can_continue:
                    termination_reason = fatigue_reason

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
                decision_data={
                    'net_value': iteration_net_value,
                    'total_cost': act_loop.fatigue,
                    'iteration_cost': fatigue_added,
                },
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
        if iteration_service and act_loop.iteration_logs:
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

        # ── Post-loop: fatigue telemetry ────────────────────────────────
        fatigue_telemetry = act_loop.get_fatigue_telemetry()
        fatigue_telemetry['termination_reason'] = termination_reason
        if self.critic_enabled:
            fatigue_telemetry.update(act_loop.get_critic_telemetry())
        logger.info(f"{LOG_PREFIX} Fatigue telemetry: {fatigue_telemetry}")

        try:
            from services.database_service import get_shared_db_service
            from services.interaction_log_service import InteractionLogService
            _tel_db = get_shared_db_service()
            _tel_log = InteractionLogService(_tel_db)
            _tel_log.log_event(
                event_type='act_loop_telemetry',
                payload=fatigue_telemetry,
                topic=topic,
                source='act_loop',
            )
        except Exception:
            pass

        return ACTResult(
            act_history=act_loop.act_history,
            iteration_logs=act_loop.iteration_logs,
            termination_reason=termination_reason or '',
            loop_id=loop_id,
            fatigue=act_loop.fatigue,
            iterations_used=act_loop.iteration_number,
            critic_telemetry=act_loop.get_critic_telemetry(),
            fatigue_telemetry=fatigue_telemetry,
        )

    # ── Private helpers ─────────────────────────────────────────────────

    def _run_critic(
        self,
        act_loop: ActLoopService,
        critic,
        original_text: str,
        actions: list,
        actions_executed: list,
        exchange_id: str,
    ) -> list:
        """Post-action critic verification with retry for safe actions."""
        from services.critic_service import MAX_CRITIC_RETRIES, CRITIC_FATIGUE_COST

        critic_corrected = []
        for idx, result in enumerate(actions_executed):
            action_spec = actions[idx] if idx < len(actions) else {}
            action_type = result.get('action_type', 'unknown')

            if critic.should_skip(action_type, result):
                critic_corrected.append(result)
                continue

            retries = 0
            current_result = result
            while retries < MAX_CRITIC_RETRIES:
                verdict = critic.evaluate(
                    original_request=original_text,
                    action_type=action_type,
                    action_intent=action_spec,
                    action_result=current_result,
                )
                act_loop.charge_critic_fatigue(CRITIC_FATIGUE_COST)

                if verdict.get('verified', True):
                    break

                correction = verdict.get('correction')
                if not correction:
                    if not critic.is_safe_action(action_type):
                        logger.info(
                            f"{LOG_PREFIX} Critic escalation for {action_type}: "
                            f"{verdict.get('issue', 'unknown')}"
                        )
                        # Notify user about the paused action
                        issue = verdict.get('issue', 'something unexpected')
                        action_desc = action_spec.get('description', action_type)
                        escalation_text = (
                            f"I was about to {action_desc}, but I paused — "
                            f"{issue}. Should I go ahead?"
                        )
                        try:
                            from services.output_service import OutputService
                            OutputService().enqueue_text(
                                topic=act_loop.context_extras.get('topic', ''),
                                response=escalation_text,
                                mode='ACT',
                                confidence=0.0,
                                generation_time=0.0,
                                original_metadata={
                                    'source': 'critic_escalation',
                                    'exchange_id': exchange_id,
                                },
                            )
                        except Exception as _esc_err:
                            logger.warning(
                                f"{LOG_PREFIX} Failed to send escalation: {_esc_err}"
                            )
                    break

                # Log correction entry
                correction_entry = {
                    'action_type': action_type,
                    'status': 'critic_correction',
                    'result': critic.format_correction_entry(
                        action_type=action_type,
                        original_result=str(current_result.get('result', '')),
                        correction=correction,
                        final_result=correction,
                    ),
                    'execution_time': 0.0,
                    'confidence': 0.0,
                    'notes': f"critic correction attempt {retries + 1}",
                }
                act_loop.append_results([correction_entry])

                if critic.is_safe_action(action_type):
                    from services.act_dispatcher_service import ActDispatcherService
                    retry_dispatcher = ActDispatcherService(
                        timeout=act_loop.per_action_timeout
                    )
                    corrected_action = {
                        **action_spec,
                        '_critic_correction': correction,
                    }
                    current_result = retry_dispatcher.dispatch_action(
                        act_loop.context_extras.get('topic', ''),
                        corrected_action,
                    )
                else:
                    logger.info(
                        f"{LOG_PREFIX} Critic correction for consequential "
                        f"{action_type}, not retrying"
                    )
                    break

                retries += 1
                if retries >= MAX_CRITIC_RETRIES:
                    critic.oscillation_events += 1
                    logger.warning(
                        f"{LOG_PREFIX} Critic MAX_RETRIES reached for {action_type}"
                    )

            critic_corrected.append(current_result)

        return critic_corrected

    def _maybe_inject_budget_warning(
        self, act_loop: ActLoopService, actions: list
    ) -> None:
        """Inject budget exhaustion warning when utilization approaches 85%."""
        _tool_action_count = sum(
            1 for r in act_loop.act_history
            if (
                r.get('action_type') not in COGNITIVE_PRIMITIVES
                and r.get('action_type') not in ('system', None)
            )
        )
        if _tool_action_count < 4 or act_loop.fatigue_budget <= 0:
            return

        _predicted_cost = sum(
            ACTION_FATIGUE_COSTS.get(a.get('type', ''), 1.0)
            * (1.0 + act_loop.fatigue_growth_rate * act_loop.iteration_number)
            for a in actions
        )
        _predicted_fatigue = act_loop.fatigue + _predicted_cost
        _predicted_util = _predicted_fatigue / act_loop.fatigue_budget

        if _predicted_util >= 0.85:
            logger.info(
                f"{LOG_PREFIX} Budget safety net: predicted "
                f"{_predicted_util:.0%} after this iteration "
                f"({_tool_action_count} tool actions)"
            )
            act_loop.append_results([{
                'action_type': 'system',
                'status': 'info',
                'execution_time': 0.0,
                'result': (
                    f"SYSTEM: Action budget nearly exhausted "
                    f"(~{_predicted_util:.0%}). This is your last iteration. "
                    "If significant work remains, create a persistent_task now. "
                    "Otherwise return empty actions to finish."
                ),
            }])
            act_loop._escalation_hint_injected = True

    def _check_smart_repetition(
        self,
        current_fingerprint: str,
        current_types: set,
        recent_entries: list,
    ) -> Optional[str]:
        """Embedding-based semantic repetition check (same-type only)."""
        try:
            from services.embedding_service import get_embedding_service
            import numpy as np

            emb_service = get_embedding_service()
            current_vec = emb_service.generate_embedding_np(current_fingerprint)

            for prev_fingerprint, prev_types in recent_entries[-4:-1]:
                if not current_types & prev_types:
                    continue
                prev_vec = emb_service.generate_embedding_np(prev_fingerprint)
                sim = float(np.dot(current_vec, prev_vec))
                if sim > self.repetition_sim_threshold:
                    logger.warning(
                        f"{LOG_PREFIX} Smart repetition (same-type): "
                        f"sim={sim:.3f} > {self.repetition_sim_threshold}"
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
            from services.redis_client import RedisClientService as _RCS
            _redis_dc = _RCS.create_connection()
            _deferred_items = _redis_dc.lrange(
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
