"""
Tool Worker — Background ACT loop processing.

Picks up tool work from tool-queue, runs the ACT reasoning loop,
and enqueues a follow-up on prompt-queue when done.

This decouples heavy tool work from the fast response path.
"""

import json
import time
import logging

logger = logging.getLogger(__name__)

INNATE_SKILLS = {'recall', 'memorize', 'introspect', 'associate', 'schedule'}


def _is_ephemeral_tool(tool_name: str) -> bool:
    """Return True if the tool declares output.ephemeral=true in its manifest."""
    try:
        from services.tool_registry_service import ToolRegistryService
        manifest = ToolRegistryService().get_tool_full_description(tool_name)
        if manifest:
            return manifest.get('output', {}).get('ephemeral', False)
    except Exception:
        pass
    return False


def _enqueue_tool_reflection(act_history: list, topic: str, user_prompt: str):
    """Push tool outputs to Redis for background experience assimilation.

    Applies novelty gate layers 1 (ephemeral tool type) and 2 (output size).
    Layer 3 (content hash dedup) runs in the assimilation service.
    """
    try:
        tool_outputs = []
        for action in act_history:
            if action.get('status') != 'success':
                continue
            action_type = action.get('action_type', '')
            if action_type in INNATE_SKILLS:
                continue
            if _is_ephemeral_tool(action_type):
                continue
            result_str = str(action.get('result', ''))
            if len(result_str) < 50:
                continue
            tool_outputs.append({
                'tool': action_type,
                'result': result_str[:2000],
            })

        if not tool_outputs:
            return

        from services.redis_client import RedisClientService
        redis_conn = RedisClientService.create_connection()
        payload = json.dumps({
            'topic': topic,
            'user_prompt': user_prompt,
            'tool_outputs': tool_outputs,
            'timestamp': time.time(),
        })
        redis_conn.rpush('tool_reflection:pending', payload)
        redis_conn.expire('tool_reflection:pending', 86400)
        logger.debug(
            f"[TOOL WORKER] Enqueued reflection for topic '{topic}' "
            f"({len(tool_outputs)} tool output(s))"
        )
    except Exception as e:
        logger.debug(f"[TOOL WORKER] Reflection enqueue failed: {e}")


def tool_worker(job_data: dict) -> str:
    """
    Background ACT loop worker.

    Processes tool work spawned by the fast path in digest_worker.
    Runs the full ACT reasoning loop, then enqueues a follow-up
    on prompt-queue for response generation.

    Args:
        job_data: Dict with:
            - cycle_id: Current tool_work cycle ID
            - parent_cycle_id: Fast response cycle ID
            - root_cycle_id: Original user input cycle ID
            - topic: Conversation topic
            - text: Original user prompt
            - intent: Classified intent metadata
            - context_snapshot: Context state at time of spawn
            - metadata: Original request metadata
            - tool_hints: Suggested tools (from CognitiveTriageService)

    Returns:
        str: Status message
    """
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Tool worker job exceeded hard timeout")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)  # 5 min hard timeout

    try:
        cycle_id = job_data.get('cycle_id', '')
        parent_cycle_id = job_data.get('parent_cycle_id', '')
        root_cycle_id = job_data.get('root_cycle_id', '')
        topic = job_data['topic']
        text = job_data['text']
        intent = job_data.get('intent', {})
        context_snapshot = job_data.get('context_snapshot', {})
        metadata = job_data.get('metadata', {})

        logger.info(
            f"[TOOL WORKER] Starting ACT loop for cycle {cycle_id[:8] if cycle_id else '?'} "
            f"(topic={topic})"
        )

        # Update cycle status to processing
        cycle_service = _get_cycle_service()
        if cycle_service and cycle_id:
            cycle_service.update_cycle_status(cycle_id, 'processing')

        # Check for cancellation before starting
        if _is_cancelled(cycle_id):
            logger.info(f"[TOOL WORKER] Cycle {cycle_id[:8]} cancelled before start")
            return f"Topic '{topic}' | CANCELLED before start"

        # Load configs
        from services import ConfigService, FrontalCortexService
        from services.thread_conversation_service import ThreadConversationService
        from services.act_loop_service import ActLoopService
        from services.cortex_iteration_service import CortexIterationService

        cortex_config = ConfigService.resolve_agent_config("frontal-cortex")
        act_prompt = ConfigService.get_agent_prompt("frontal-cortex-act")

        # Load ACT-specific config
        try:
            act_config = ConfigService.resolve_agent_config("frontal-cortex-act")
        except Exception:
            act_config = cortex_config

        cortex_service = FrontalCortexService(act_config)
        thread_id = metadata.get('thread_id')
        conversation_service = ThreadConversationService()
        chat_history = conversation_service.get_conversation_history(thread_id) if thread_id else []

        # Classification stub
        classification = {
            'topic': topic,
            'confidence': 10,
            'similar_topic': '',
            'topic_update': '',
        }

        # Initialize ACT loop
        act_cumulative_timeout = cortex_config.get('act_cumulative_timeout', 60.0)
        act_per_action_timeout = cortex_config.get('act_per_action_timeout', 10.0)
        max_act_iterations = cortex_config.get('max_act_iterations', 7)

        act_loop = ActLoopService(
            config=cortex_config,
            cumulative_timeout=act_cumulative_timeout,
            per_action_timeout=act_per_action_timeout,
            max_iterations=max_act_iterations,
        )
        act_loop.context_warmth = context_snapshot.get('context_warmth', 0.0)

        # Relevant tools from embedding-based scoring (passed from digest_worker)
        relevant_tools = context_snapshot.get('relevant_tools', None) or None

        # Repetition similarity threshold for embedding-based dedup
        repetition_sim_threshold = cortex_config.get('act_repetition_similarity_threshold', 0.85)

        # Initialize iteration logging
        iteration_service = None
        loop_id = None
        try:
            from services.database_service import get_shared_db_service
            db_service = get_shared_db_service()
            iteration_service = CortexIterationService(db_service)
            loop_id = iteration_service.create_loop_id()
            act_loop.loop_id = loop_id
        except Exception as e:
            logger.warning(f"[TOOL WORKER] Iteration logging init failed: {e}")

        try:
            exchange_id = conversation_service.get_latest_exchange_id(topic)
        except Exception:
            exchange_id = "unknown"

        # ACT loop
        consecutive_same_action = 0
        last_action_type = None
        recent_action_texts = []  # For embedding-based smart repetition

        while True:
            # Check cancellation each iteration
            if _is_cancelled(cycle_id):
                logger.info(f"[TOOL WORKER] Cycle {cycle_id[:8]} cancelled by user")
                if cycle_service and cycle_id:
                    cycle_service.complete_cycle(cycle_id, 'cancelled')
                return f"Topic '{topic}' | CANCELLED by user"

            iteration_start = time.time()

            # Generate action plan
            response_data = cortex_service.generate_response(
                system_prompt_template=act_prompt,
                original_prompt=text,
                classification=classification,
                chat_history=chat_history,
                act_history=act_loop.get_history_context(),
                relevant_tools=relevant_tools,
            )

            actions = response_data.get('actions', [])

            # Repetition detection (type-based)
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
                logger.warning(f"[TOOL WORKER] Repetition detected: '{last_action_type}' x{consecutive_same_action}")
                termination_reason = 'repetition_detected'
                can_continue = False
            else:
                # Embedding-based smart repetition check
                smart_repeat = False
                if actions and recent_action_texts:
                    try:
                        from services.embedding_service import get_embedding_service
                        import numpy as np
                        emb_service = get_embedding_service()
                        current_action_text = _action_fingerprint(actions)
                        current_vec = emb_service.generate_embedding_np(current_action_text)
                        for prev_text in recent_action_texts[-3:]:
                            prev_vec = emb_service.generate_embedding_np(prev_text)
                            sim = float(np.dot(current_vec, prev_vec))
                            if sim > repetition_sim_threshold:
                                logger.warning(f"[TOOL WORKER] Smart repetition: sim={sim:.3f} > {repetition_sim_threshold}")
                                smart_repeat = True
                                break
                    except Exception:
                        pass

                if smart_repeat:
                    termination_reason = 'smart_repetition'
                    can_continue = False
                else:
                    can_continue, termination_reason = act_loop.can_continue()

            # Execute actions
            actions_executed = []
            fatigue_added = 0.0
            iteration_net_value = 0.0
            if can_continue and actions:
                actions_executed = act_loop.execute_actions(
                    topic=topic,
                    actions=actions
                )
                act_loop.append_results(actions_executed)

                # Track action fingerprint for smart repetition
                recent_action_texts.append(_action_fingerprint(actions))

                # Accumulate fatigue
                fatigue_added = act_loop.accumulate_fatigue(actions_executed, act_loop.iteration_number)

                # Estimate net value (for cortex_iterations logging + strategy analysis)
                iteration_net_value = ActLoopService.estimate_net_value(actions_executed, act_loop.iteration_number)

                # Record per-skill outcomes to procedural memory
                from services.skill_outcome_recorder import record_skill_outcomes
                record_skill_outcomes(actions_executed, topic)

            elif not actions:
                logger.info("[TOOL WORKER] No actions, exiting ACT loop")
                termination_reason = 'no_actions'
                can_continue = False

            # Log iteration
            iteration_end = time.time()
            act_loop.log_iteration(
                started_at=iteration_start,
                completed_at=iteration_end,
                chosen_mode='ACT',
                chosen_confidence=response_data.get('confidence', 0.5),
                actions_executed=actions_executed,
                frontal_cortex_response=response_data,
                termination_reason=termination_reason if not can_continue else None,
                decision_data={
                    'net_value': iteration_net_value,
                    'total_cost': act_loop.fatigue,
                    'iteration_cost': fatigue_added,
                },
            )

            act_loop.iteration_number += 1

            if not can_continue:
                break

        # Batch write iterations
        if iteration_service and act_loop.iteration_logs:
            try:
                iteration_service.log_iterations_batch(
                    loop_id=loop_id,
                    topic=topic,
                    exchange_id=exchange_id,
                    session_id="tool_worker",
                    iterations=act_loop.iteration_logs
                )
            except Exception as e:
                logger.error(f"[TOOL WORKER] Failed to log iterations: {e}")

        # Log fatigue telemetry
        telemetry = act_loop.get_fatigue_telemetry()
        telemetry['termination_reason'] = termination_reason
        logger.info(f"[TOOL WORKER] Fatigue telemetry: {telemetry}")
        try:
            from services.interaction_log_service import InteractionLogService
            _tel_log = InteractionLogService(db_service)
            _tel_log.log_event(
                event_type='act_loop_telemetry',
                payload=telemetry,
                topic=topic,
                source='act_loop',
            )
        except Exception:
            pass

        # Build tool results summary
        act_history_context = act_loop.get_history_context()

        # Action-completion verification: if action-oriented tools were expected
        # but none were successfully invoked, inject a [NO_ACTION_TAKEN] signal
        # so the followup prompt knows the action failed.
        relevant_tools_list = context_snapshot.get('relevant_tools', []) or []
        expected_action_tools = [
            t['name'] for t in relevant_tools_list
            if t.get('type') == 'tool' and not _is_ephemeral_tool(t['name'])
        ]
        if expected_action_tools:
            action_tool_used = any(
                not _is_ephemeral_tool(r.get('action_type', ''))
                and r.get('action_type', '') not in INNATE_SKILLS
                and r.get('status') == 'success'
                for r in act_loop.act_history
            )
            if not action_tool_used:
                failed_tools = ', '.join(expected_action_tools)
                act_history_context = (
                    f"[NO_ACTION_TAKEN] The requested action could not be completed. "
                    f"Expected tool(s) [{failed_tools}] were not successfully invoked. "
                    f"Do NOT claim the action was performed.\n\n"
                    + act_history_context
                )

        # Complete cycle
        if cycle_service and cycle_id:
            cycle_service.complete_cycle(cycle_id, 'completed')

        # Enqueue tool outputs for background experience assimilation
        _enqueue_tool_reflection(act_loop.act_history, topic, text)

        # Render and deliver cards; gate text follow-up on replaces_response
        card_replaces = _enqueue_tool_cards(act_loop.act_history, topic, metadata)
        if not card_replaces:
            _enqueue_followup(
                topic=topic,
                text=text,
                act_history_context=act_history_context,
                cycle_id=cycle_id,
                parent_cycle_id=parent_cycle_id,
                root_cycle_id=root_cycle_id,
                metadata=metadata,
                original_created_at=job_data.get('created_at', time.time()),
            )

        total_time = time.time() - act_loop.start_time
        logger.info(
            f"[TOOL WORKER] ACT loop complete: {act_loop.iteration_number} iterations, "
            f"{len(act_loop.act_history)} actions, {total_time:.1f}s total"
        )

        # Record performance metrics for each tool invocation
        try:
            from services.tool_performance_service import ToolPerformanceService
            perf_service = ToolPerformanceService()
            for action in act_loop.get_history_context() if hasattr(act_loop, 'get_history_context') else []:
                action_type = action.get('action_type', '') if isinstance(action, dict) else ''
                if action_type and action_type not in ('recall', 'memorize', 'introspect', 'associate', 'schedule', 'goal', 'focus', 'list', 'autobiography', 'introspect'):
                    action_success = action.get('status') == 'success' if isinstance(action, dict) else False
                    action_latency = float(action.get('latency_ms', 0)) if isinstance(action, dict) else 0.0
                    perf_service.record_invocation(
                        tool_name=action_type,
                        exchange_id=exchange_id,
                        success=action_success,
                        latency_ms=action_latency,
                    )
        except Exception as _perf_err:
            logger.debug(f"[TOOL WORKER] Performance recording failed: {_perf_err}")

        return (
            f"Topic '{topic}' | Tool work complete: "
            f"{act_loop.iteration_number} iterations in {total_time:.1f}s"
        )

    except TimeoutError:
        logger.error(f"[TOOL WORKER] Hard timeout for topic '{topic}'")
        if cycle_service and cycle_id:
            cycle_service.complete_cycle(cycle_id, 'failed')
        return f"Topic '{topic}' | TIMEOUT"
    except Exception as e:
        logger.error(f"[TOOL WORKER] Failed: {e}", exc_info=True)
        if cycle_service and cycle_id:
            cycle_service.complete_cycle(cycle_id, 'failed')
        return f"Topic '{topic}' | ERROR: {e}"
    finally:
        signal.alarm(0)


def _enqueue_tool_cards(act_history: list, topic: str, metadata: dict) -> bool:
    """Render and enqueue cards for card-enabled tools. Returns True if any card replaces the text response."""
    any_replaces = False
    try:
        from services.redis_client import RedisClientService
        from services.tool_registry_service import ToolRegistryService
        from services.card_renderer_service import CardRendererService
        from services.output_service import OutputService

        redis = RedisClientService.create_connection()
        raw_items = redis.lrange(f"tool_raw_cache:{topic}", 0, -1)
        redis.delete(f"tool_raw_cache:{topic}")

        # Build {tool_name: raw_result} map (last result per tool wins)
        raw_map = {}
        for item in raw_items:
            entry = json.loads(item)
            raw_map[entry['tool']] = entry['data']

        registry = ToolRegistryService()
        renderer = CardRendererService()
        output_svc = OutputService()

        for action in act_history:
            if action.get('status') != 'success':
                continue
            tool_name = action.get('action_type', '')
            tool = registry.tools.get(tool_name)
            if not tool:
                continue
            output_config = tool['manifest'].get('output', {})
            card_config = output_config.get('card', {})
            if not card_config or not card_config.get('enabled'):
                continue

            # If synthesize is false, invoke() already rendered the card inline.
            # Just suppress the follow-up text response without re-rendering.
            if not output_config.get('synthesize', True):
                any_replaces = True
                continue

            raw = raw_map.get(tool_name)
            if not raw:
                continue

            card_data = renderer.render(tool_name, raw, card_config, tool['dir'])
            if card_data:
                output_svc.enqueue_card(topic, card_data, metadata)
                if card_config.get('replaces_response'):
                    any_replaces = True

    except Exception as e:
        logger.warning(f"[TOOL WORKER] Card enqueue failed: {e}")

    return any_replaces


def _action_fingerprint(actions: list) -> str:
    """Build a text fingerprint from action list for embedding-based repetition check."""
    parts = []
    for a in actions:
        action_type = a.get('type', 'unknown')
        query = a.get('query', a.get('text', a.get('input', '')))
        parts.append(f"{action_type}: {query}")
    return ' | '.join(parts)


def _get_cycle_service():
    """Get CycleService instance (lazy init, tolerant of failures)."""
    try:
        from services.cycle_service import CycleService
        from services.database_service import get_shared_db_service
        db_service = get_shared_db_service()
        return CycleService(db_service)
    except Exception:
        return None


def _is_cancelled(cycle_id: str) -> bool:
    """Check if this cycle has been cancelled via Redis flag."""
    if not cycle_id:
        return False
    try:
        from services.redis_client import RedisClientService
        redis_conn = RedisClientService.create_connection()
        return bool(redis_conn.get(f"cancel:{cycle_id}"))
    except Exception:
        return False


def _enqueue_followup(
    topic: str,
    text: str,
    act_history_context: str,
    cycle_id: str,
    parent_cycle_id: str,
    root_cycle_id: str,
    metadata: dict,
    original_created_at: float,
):
    """Enqueue a follow-up message on prompt-queue for response generation."""
    try:
        from workers.digest_worker import digest_worker
        from services.prompt_queue import PromptQueue

        followup_metadata = {
            'type': 'tool_result',
            'topic': topic,
            'original_prompt': text,
            'act_history_context': act_history_context,
            'tool_cycle_id': cycle_id,
            'parent_cycle_id': parent_cycle_id,
            'root_cycle_id': root_cycle_id,
            'original_created_at': original_created_at,
            'destination': metadata.get('destination', 'web'),
            'thread_id': metadata.get('thread_id'),
            'source': 'tool_followup',
        }

        followup_queue = PromptQueue(queue_name="prompt-queue", worker_func=digest_worker)
        followup_queue.enqueue(
            f"[TOOL RESULT] Background research completed for: {text[:100]}",
            followup_metadata,
        )

        logger.info(f"[TOOL WORKER] Enqueued follow-up on prompt-queue for topic '{topic}'")

    except Exception as e:
        logger.error(f"[TOOL WORKER] Failed to enqueue follow-up: {e}")
