"""
ACT Orchestrator Service — Single, parameterized ACT loop implementation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: This is the SOLE ACT loop implementation.

Do NOT copy this loop into workers, services, or anywhere else.
All ACT loop execution MUST go through ACTOrchestrator.run().
If you need different behavior, add a parameter to the constructor.

Historical context: Before this service existed, the ACT loop was
duplicated across multiple workers. Features silently diverged across
copies causing reliability gaps. This unification prevents that class
of bug.

If you believe you need a separate ACT loop, discuss with the team
first. The cost of duplication is always higher than parameterization.

Termination model (fatigue-free):
  - Hard iteration cap: max_iterations (default 30)
  - Cumulative timeout: safety net for runaway loops
  - Semantic repetition: embedding-based (>0.85 cosine similarity)
  - No actions returned by LLM: natural completion signal
  - Soft nudge: at iteration 10, inject a prompt hint encouraging
    the LLM to conclude if it has enough information.
  - Forced-exit synthesis: when loop exits without a final response,
    one last tool-free LLM call synthesizes from accumulated results.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable

from services.act_loop_service import ActLoopService
from services.innate_skills.registry import COGNITIVE_PRIMITIVES
from services.tool_schema_service import get_skill_schemas, get_external_tool_schemas
from services.llm_service import estimate_tokens

try:
    from services.telemetry_service import (
        get_telemetry_collector,
        ACT_LOOP_ITERATION,
        ACT_LOOP_COMPLETE,
    )
    _TELEMETRY_AVAILABLE = True
except Exception as e:  # pragma: no cover
    _TELEMETRY_AVAILABLE = False
    logging.debug(f"Telemetry service import unavailable: {e}")

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
    final_response: str = ''  # Model's text when it stops calling tools


class ACTOrchestrator:
    """
    Unified ACT loop runner with parameterized behavior.

    Single parameterized ACT loop. Each caller configures the behavior
    it needs via constructor parameters.
    """

    def __init__(
        self,
        config: dict,
        max_iterations: int = 30,
        cumulative_timeout: float = 60.0,
        per_action_timeout: float = 10.0,
        critic_enabled: bool = False,
        smart_repetition: bool = True,
        escalation_hints: bool = False,  # Deprecated — type-based repetition removed
        persistent_task_exit: bool = False,
        execution_gate: bool = True,
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
            escalation_hints: Deprecated — accepted for backward compatibility but
                ignored. Type-based repetition was removed; smart (embedding-based)
                repetition detection is the sole mechanism.
            persistent_task_exit: Exit loop when a persistent_task is dispatched
            execution_gate: Whether to apply the autonomous execution gate to
                non-safe actions. False for user-initiated ACT loops (the user
                already asked for this), True for autonomous/background execution.
        """
        self.config = config
        self.max_iterations = max_iterations
        self.cumulative_timeout = cumulative_timeout
        self.per_action_timeout = per_action_timeout
        self.critic_enabled = critic_enabled
        self.smart_repetition = smart_repetition
        self.escalation_hints = escalation_hints
        self.persistent_task_exit = persistent_task_exit
        self.execution_gate = execution_gate

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
            execution_gate=self.execution_gate,
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

        # ── Build system prompt once, grow message array ─────────────────
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

        # Build native tool schemas for all innate skills
        _native_tools = get_skill_schemas(selected_skills)

        # Auto-calculate context budget from provider limits
        try:
            _context_limit = cortex_service.get_context_limit()
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} get_context_limit failed, using fallback 32000: {e}")
            _context_limit = 32000  # Safe fallback
        _context_budget = min(int(_context_limit * 0.6), 150_000)
        logger.info(
            f"{LOG_PREFIX} System prompt built ({len(_system_prompt)} chars), "
            f"{len(_native_tools)} native tools, "
            f"context budget: {_context_budget} tokens (limit: {_context_limit})"
        )

        # ── Repetition detection state ──────────────────────────────────
        recent_action_entries = []  # (fingerprint, types_set) for smart repetition

        # ── Tool health tracking (cross-loop via MemoryStore) ─────────
        from services.tool_health_service import (
            get_potential, record_outcome as _record_health,
            classify_result as _classify_health, format_health_hint,
        )

        termination_reason = None
        _final_response = ''  # Model's text when it stops calling tools

        # ── Main loop ───────────────────────────────────────────────────
        while True:
            iteration_start = time.time()

            # ── Telemetry: iteration start ─────────────────────────────
            if _TELEMETRY_AVAILABLE:
                try:
                    get_telemetry_collector().record(ACT_LOOP_ITERATION, {
                        "iteration": act_loop.iteration_number,
                        "elapsed_seconds": round(
                            iteration_start - act_loop.start_time, 2
                        ),
                    })
                except Exception as e:
                    logger.debug(f"{LOG_PREFIX} ACT_LOOP_ITERATION telemetry emit failed: {e}")

            # ── Collect tool names for health hints ───────────────────
            _tool_names = set()
            if selected_tools:
                _tool_names.update(selected_tools)
            if relevant_tools:
                _tool_names.update(
                    item['name'] for item in relevant_tools
                    if isinstance(item, dict) and item.get('type') == 'tool'
                )

            # ── Grow message array ─────────────────────────────────────
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
            if act_history_str and act_history_str != "(none)":
                context_updates.append(act_history_str)

            if context_updates:
                _messages.append({
                    "role": "user",
                    "content": "\n\n".join(context_updates),
                })

            # Token budget guard — prune oldest message pairs when approaching limit
            _messages = self._prune_messages(_messages, _context_budget)

            try:
                response_data = cortex_service.generate_response_appended(
                    system_prompt=_system_prompt,
                    messages=_messages,
                    cache_prefix=True,
                    tools=_native_tools,
                )
            except Exception as _gen_err:
                logger.error(
                    f"{LOG_PREFIX} LLM call failed at iteration "
                    f"{act_loop.iteration_number}: {_gen_err}", exc_info=True
                )
                termination_reason = 'generation_error'
                act_loop.log_iteration(
                    started_at=iteration_start, completed_at=time.time(),
                    chosen_mode='ACT', chosen_confidence=0.0,
                    actions_executed=[], frontal_cortex_response={'error': str(_gen_err)},
                    termination_reason=termination_reason, decision_data={'net_value': 0.0},
                )
                act_loop.iteration_number += 1
                break

            # Append assistant turn for the next iteration
            if response_data.get('tool_calls'):
                # Native tool calling: append assistant message with tool_calls
                _messages.append({
                    "role": "assistant",
                    "content": response_data.get('narration', ''),
                    "tool_calls": response_data['tool_calls'],
                })
            else:
                # Text-only response (no tools called)
                raw_response = response_data.get('raw_response', response_data.get('response', ''))
                if raw_response:
                    _messages.append({"role": "assistant", "content": raw_response})

            actions = response_data.get('actions') or []

            # ── Narration emission ─────────────────────────────────────
            if on_narration and actions:
                narration_text = response_data.get('narration', '')
                if not narration_text:
                    # Auto-narrate from action types (OpenAI models don't
                    # return text alongside tool_calls)
                    action_names = [a.get('type', '') for a in actions if a.get('type')]
                    if action_names:
                        narration_text = f"Using {', '.join(action_names)}..."
                if narration_text:
                    try:
                        on_narration(narration_text, act_loop.iteration_number)
                    except Exception as e:
                        logger.error(f"{LOG_PREFIX} Narration callback error: {e}", exc_info=True)

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
                _final_response = response_data.get('response', '') or response_data.get('narration', '')
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

            # ── Dynamic tool injection (find_tools → native tools) ─────
            for _exec_r in actions_executed:
                if _exec_r.get('action_type') != 'find_tools':
                    continue
                # _discovered_tools is preserved at top level by the dispatcher
                _discovered = _exec_r.get('_discovered_tools', [])
                if not _discovered:
                    continue
                try:
                    _new_schemas = get_external_tool_schemas(_discovered)
                    _existing = {t['name'] for t in _native_tools}
                    _injected = []
                    for _schema in _new_schemas:
                        if _schema['name'] not in _existing:
                            _native_tools.append(_schema)
                            _existing.add(_schema['name'])
                            _injected.append(_schema['name'])
                    if _injected:
                        logger.info(
                            f"{LOG_PREFIX} Dynamically injected {len(_injected)} "
                            f"tool schema(s): {_injected}"
                        )
                except Exception as _inj_err:
                    logger.warning(
                        f"{LOG_PREFIX} Dynamic tool injection failed: {_inj_err}"
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

            # ── Native tool calling: append tool_result messages ─────────
            if _native_tools and response_data.get('tool_calls'):
                for tc, exec_r in zip(response_data['tool_calls'], actions_executed):
                    result_text = exec_r.get('result', '')
                    if isinstance(result_text, dict):
                        # Skills returning {"text": ..., "_meta": ...} —
                        # send the text to the LLM, strip internal metadata.
                        result_text = result_text.get('text') or json.dumps(result_text, default=str)
                    elif not isinstance(result_text, str):
                        result_text = str(result_text)
                    _messages.append({
                        "role": "tool",
                        "tool_call_id": tc['id'],
                        "name": tc['name'],
                        "content": result_text[:8000],  # Prevent context overflow
                    })

            # ── Persist to transcript (fire-and-forget) ──────────────
            try:
                from services import transcript_service
                # Record tool results
                for exec_r in actions_executed:
                    _atype = exec_r.get('action_type', '')
                    _result = exec_r.get('result', '')
                    if isinstance(_result, dict):
                        _result = _result.get('text', str(_result))
                    elif not isinstance(_result, str):
                        _result = str(_result)
                    if _result:
                        transcript_service.append(
                            topic, 'tool', _result[:80000],
                            tool_name=_atype,
                        )
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} Transcript append failed (non-fatal): {e}")

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

        # ── Post-loop: synthesis call for forced exits ────────────────
        # When the loop was forced to exit (timeout, repetition, etc.)
        # the LLM never got a chance to produce a final text response.
        # Make one last call WITHOUT tools so it can synthesize from the
        # accumulated tool results in _messages.
        if not _final_response and termination_reason and termination_reason != 'generation_error':
            logger.info(
                f"{LOG_PREFIX} Forced exit ({termination_reason}) with no final response "
                f"— running synthesis call"
            )
            _messages.append({
                "role": "user",
                "content": (
                    "SYSTEM: The action loop has ended. Based on the results above, "
                    "respond to the user now. Do NOT call any tools."
                ),
            })
            try:
                _synth = cortex_service.generate_response_appended(
                    system_prompt=_system_prompt,
                    messages=self._prune_messages(_messages, _context_budget),
                    cache_prefix=True,
                )
                _final_response = (
                    _synth.get('response', '') or _synth.get('narration', '')
                )
                if _final_response:
                    logger.info(
                        f"{LOG_PREFIX} Synthesis produced {len(_final_response)} chars"
                    )
            except Exception as _synth_err:
                logger.warning(
                    f"{LOG_PREFIX} Synthesis call failed: {_synth_err}"
                )

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

        # ── Telemetry: loop complete ──────────────────────────────────
        if _TELEMETRY_AVAILABLE:
            try:
                get_telemetry_collector().record(ACT_LOOP_COMPLETE, {
                    "iterations_used": loop_telemetry.get("iterations_used"),
                    "termination_reason": termination_reason,
                    "elapsed_seconds": loop_telemetry.get("elapsed_seconds"),
                    "actions_total": loop_telemetry.get("actions_total"),
                })
            except Exception as e:
                logger.debug(f"{LOG_PREFIX} ACT_LOOP_COMPLETE telemetry emit failed: {e}")

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
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Loop telemetry write failed (non-fatal): {e}")

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
            final_response=_final_response,
        )

    # ── Private helpers ─────────────────────────────────────────────────

    def _get_steering_text(self) -> str:
        """Drain the MemoryStore steering queue and return formatted text.

        Steering content is included in a discrete user message within the
        message array each iteration.

        Returns:
            Formatted steering lines joined by newlines, or an empty string
            when there is no pending steering input or the store is unavailable.
        """
        try:
            from services.memory_store import get_shared_store
            store = get_shared_store()
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

        total_text = ' '.join(m.get('content', '') or '' for m in messages)
        estimated_tokens = estimate_tokens(total_text)

        if estimated_tokens <= budget_tokens or len(messages) <= 3:
            return messages

        # Keep the first message (original user prompt) plus the most-recent
        # tail.  Start with the last 4 messages and expand backward; if that
        # still exceeds the budget, remove pairs from position 1 onward.
        keep_tail = min(4, len(messages) - 1)
        pruned = [messages[0]] + messages[-keep_tail:]

        while len(pruned) > 3:
            total = ' '.join(m.get('content', '') or '' for m in pruned)
            if estimate_tokens(total) <= budget_tokens:
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
                    from services.knowledge_service import KnowledgeService
                    db = get_shared_db_service()
                    ks = KnowledgeService(db)

                    # Record outcome for each action type used in the loop
                    seen_types = set()
                    for entry in results:
                        atype = entry.get('action_type', '')
                        if not atype or atype in ('system', 'critic_escalation') or atype in seen_types:
                            continue
                        seen_types.add(atype)
                        success = outcome_quality >= 0.5
                        reward = (outcome_quality - 0.5) * 2.0  # map [0,1] → [-1,1]
                        ks.record_procedure_outcome(
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
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} _get_cautionary_lessons failed (non-fatal): {e}")
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

        from services.memory_store import get_shared_store
        store = get_shared_store()
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
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} _check_smart_repetition failed (non-fatal): {e}")
        return None



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
        from services.memory_store import get_shared_store
        store = get_shared_store()
        cooldown_key = f"auto_reflect_cooldown:{topic}"
        if store.get(cooldown_key):
            logger.debug(f"{LOG_PREFIX} Auto-reflect cooldown active for {topic}")
            return
        store.setex(cooldown_key, _AUTO_REFLECT_COOLDOWN_S, '1')
    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Auto-reflect cooldown check failed, proceeding: {e}")

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
