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

from services.innate_skills.registry import REFLECTION_FILTER_SKILLS as INNATE_SKILLS


from services.act_reflection_service import enqueue_tool_reflection as _enqueue_tool_reflection
from services.tool_card_enqueue_service import enqueue_tool_cards as _enqueue_tool_cards
from services.act_completion_service import inject_no_action_signal as _inject_no_action_signal


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

        # Heartbeat tracking for SSE health monitoring (Phase 3)
        _heartbeat_redis = None
        _heartbeat_job_id = None
        _last_heartbeat = 0.0
        try:
            from services.redis_client import RedisClientService
            _heartbeat_redis = RedisClientService.create_connection()
            # Try to get the RQ job ID from the current job context
            import rq
            current_job = rq.get_current_job()
            _heartbeat_job_id = current_job.id if current_job else cycle_id
        except Exception:
            _heartbeat_job_id = cycle_id

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

        # Relevant tools from embedding-based scoring (passed from digest_worker)
        relevant_tools = context_snapshot.get('relevant_tools', None) or None

        # Triage-selected innate skills (passed from digest_worker via context_snapshot)
        selected_skills = context_snapshot.get('triage_selected_skills', None) or None

        # Always inject emit_card when external tools are active so Chalie can
        # decide whether to render a deferred visual card. Generic — no tool names.
        triage_selected_tools = context_snapshot.get('triage_selected_tools', [])
        if relevant_tools or triage_selected_tools:
            selected_skills = list(selected_skills or []) + ['emit_card']

        return _tool_worker_orchestrator(
            cortex_config=cortex_config,
            act_config=act_config,
            act_prompt=act_prompt,
            cortex_service=cortex_service,
            classification=classification,
            chat_history=chat_history,
            relevant_tools=relevant_tools,
            selected_skills=selected_skills,
            topic=topic,
            text=text,
            context_snapshot=context_snapshot,
            metadata=metadata,
            job_data=job_data,
            cycle_id=cycle_id,
            parent_cycle_id=parent_cycle_id,
            root_cycle_id=root_cycle_id,
            cycle_service=cycle_service,
            _heartbeat_redis=_heartbeat_redis,
            _heartbeat_job_id=_heartbeat_job_id,
            _last_heartbeat=_last_heartbeat,
        )

    except TimeoutError:
        logger.error(f"[TOOL WORKER] Hard timeout for topic '{topic}'")
        if cycle_service and cycle_id:
            cycle_service.complete_cycle(cycle_id, 'failed')
        # Notify SSE loop of failure so it doesn't hang for 600s
        _notify_sse_error(metadata, "Tool execution timed out")
        return f"Topic '{topic}' | TIMEOUT"
    except Exception as e:
        logger.error(f"[TOOL WORKER] Failed: {e}", exc_info=True)
        if cycle_service and cycle_id:
            cycle_service.complete_cycle(cycle_id, 'failed')
        # Notify SSE loop of failure so it doesn't hang for 600s
        _notify_sse_error(metadata, f"Tool execution failed: {str(e)[:200]}")
        return f"Topic '{topic}' | ERROR: {e}"
    finally:
        signal.alarm(0)


def _tool_worker_orchestrator(
    cortex_config, act_config, act_prompt, cortex_service, classification,
    chat_history, relevant_tools, selected_skills,
    topic, text, context_snapshot, metadata, job_data,
    cycle_id, parent_cycle_id, root_cycle_id, cycle_service,
    _heartbeat_redis, _heartbeat_job_id, _last_heartbeat,
):
    """
    Unified ACT orchestrator path for tool_worker.

    Replaces the legacy inline loop with ACTOrchestrator.run().
    """
    from services.act_orchestrator_service import ACTOrchestrator

    act_cumulative_timeout = cortex_config.get('act_cumulative_timeout', 60.0)
    act_per_action_timeout = cortex_config.get('act_per_action_timeout', 10.0)
    max_act_iterations = cortex_config.get('max_act_iterations', 7)

    orchestrator = ACTOrchestrator(
        config=cortex_config,
        max_iterations=max_act_iterations,
        cumulative_timeout=act_cumulative_timeout,
        per_action_timeout=act_per_action_timeout,
        critic_enabled=True,
        smart_repetition=True,
        escalation_hints=False,
        persistent_task_exit=False,
        deferred_card_context=True,
    )

    # Heartbeat + cancellation callback
    heartbeat_state = {'last': _last_heartbeat}

    def on_iteration_complete(act_loop, iteration_start, actions_executed, termination_reason):
        # Cancellation check
        if _is_cancelled(cycle_id):
            logger.info(f"[TOOL WORKER] Cycle {cycle_id[:8]} cancelled by user")
            if cycle_service and cycle_id:
                cycle_service.complete_cycle(cycle_id, 'cancelled')
            return 'cancelled'

        # Heartbeat
        if _heartbeat_redis and _heartbeat_job_id:
            _now = time.time()
            if _now - heartbeat_state['last'] >= 10.0:
                try:
                    _heartbeat_redis.setex(f"heartbeat:{_heartbeat_job_id}", 30, "1")
                    heartbeat_state['last'] = _now
                except Exception:
                    pass
        return None

    try:
        from services.thread_conversation_service import ThreadConversationService
        exchange_id = ThreadConversationService().get_latest_exchange_id(topic)
    except Exception:
        exchange_id = "unknown"

    result = orchestrator.run(
        topic=topic,
        text=text,
        cortex_service=cortex_service,
        act_prompt=act_prompt,
        classification=classification,
        chat_history=chat_history,
        relevant_tools=relevant_tools,
        selected_skills=selected_skills,
        assembled_context=_build_assembled_context(text, topic, metadata, classification),
        inclusion_map=_build_inclusion_map(classification),
        on_iteration_complete=on_iteration_complete,
        context_extras={
            'triage_tools': context_snapshot.get('triage_selected_tools', []),
        },
        session_id='tool_worker',
    )

    # Complete cycle
    if cycle_service and cycle_id:
        cycle_service.complete_cycle(cycle_id, 'completed')

    # Build act_history_context for followup
    from services.act_loop_service import ActLoopService
    # Use a temporary ActLoopService just for get_history_context formatting
    _tmp = ActLoopService.__new__(ActLoopService)
    _tmp.act_history = result.act_history
    act_history_context = _tmp.get_history_context()

    # Action-completion verification
    relevant_tools_list = context_snapshot.get('relevant_tools', []) or []
    act_history_context = _inject_no_action_signal(
        result.act_history, act_history_context, relevant_tools_list
    )

    # Enqueue tool reflection
    _enqueue_tool_reflection(result.act_history, topic, text)

    # Render and deliver cards
    card_replaces = _enqueue_tool_cards(result.act_history, topic, metadata, cycle_id=cycle_id)

    # Suppress follow-up when emit_card was called
    if not card_replaces:
        card_replaces = any(
            isinstance(r.get('result'), dict) and r['result'].get('card_emitted') is True
            for r in result.act_history
        )

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

    total_time = time.time() - (time.time() - result.fatigue_telemetry.get('elapsed_seconds', 0))
    logger.info(
        f"[TOOL WORKER] ACT loop complete (orchestrator): {result.iterations_used} iterations, "
        f"{len(result.act_history)} actions, termination={result.termination_reason}"
    )

    # Record performance metrics
    try:
        from services.tool_performance_service import ToolPerformanceService
        perf_service = ToolPerformanceService()
        for action in result.act_history:
            action_type = action.get('action_type', '')
            if action_type and action_type not in INNATE_SKILLS:
                perf_service.record_invocation(
                    tool_name=action_type,
                    exchange_id=exchange_id,
                    success=action.get('status') in ('success', 'completed', True),
                    latency_ms=float(action.get('execution_time') or 0) * 1000,
                )
    except Exception as _perf_err:
        logger.debug(f"[TOOL WORKER] Performance recording failed: {_perf_err}")

    return (
        f"Topic '{topic}' | Tool work complete (orchestrator): "
        f"{result.iterations_used} iterations"
    )


def _build_inclusion_map(classification):
    """Build context inclusion map for ACT mode."""
    try:
        from services.context_relevance_service import ContextRelevanceService
        return ContextRelevanceService().compute_inclusion_map(
            mode='ACT', signals={}, classification=classification, returning_from_silence=False,
        )
    except Exception:
        return None


def _build_assembled_context(text, topic, metadata, classification):
    """Build assembled context for ACT mode."""
    try:
        from services.context_assembly_service import ContextAssemblyService
        thread_id = metadata.get('thread_id')
        return ContextAssemblyService({}).assemble(
            prompt=text, topic=topic, thread_id=thread_id,
        )
    except Exception:
        return None


def _notify_sse_error(metadata: dict, error_message: str):
    """Publish error to SSE channel and clean up sse_pending flag so the SSE loop is released."""
    try:
        sse_uuid = metadata.get('uuid')
        if not sse_uuid:
            return
        from services.redis_client import RedisClientService
        redis_conn = RedisClientService.create_connection()
        # Publish error to SSE channel
        redis_conn.publish(f"sse:{sse_uuid}", json.dumps({"error": error_message}))
        # Delete sse_pending flag so SSE loop breaks immediately
        redis_conn.delete(f"sse_pending:{sse_uuid}")
    except Exception as e:
        logger.warning(f"[TOOL WORKER] Failed to notify SSE of error: {e}")



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
            'uuid': metadata.get('uuid'),   # routes tool result to the waiting SSE connection
        }

        followup_queue = PromptQueue(queue_name="prompt-queue", worker_func=digest_worker)
        followup_queue.enqueue(
            f"[TOOL RESULT] Background research completed for: {text[:100]}",
            followup_metadata,
        )

        logger.info(f"[TOOL WORKER] Enqueued follow-up on prompt-queue for topic '{topic}'")

    except Exception as e:
        logger.error(f"[TOOL WORKER] Failed to enqueue follow-up: {e}")
