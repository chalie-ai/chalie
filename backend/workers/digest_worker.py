# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
import time
import logging
from services import ConfigService, FrontalCortexService, OrchestratorService, PromptQueue, SessionService
from services.llm_service import create_llm_service
from services.prompt_queue import enqueue_episodic_memory
from services.recent_topic_service import RecentTopicService
from services.gist_storage_service import GistStorageService
from services.world_state_service import WorldStateService
from services.working_memory_service import WorkingMemoryService
from services.interaction_log_service import InteractionLogService
from services.event_bus_service import EventBusService, ENCODE_EVENT
from services.metrics_service import MetricsService
from services.topic_classifier_service import TopicClassifierService
from services.fact_store_service import FactStoreService
from services.mode_router_service import ModeRouterService, collect_routing_signals
from services.intent_classifier_service import IntentClassifierService
from services.thread_service import get_thread_service
from services.thread_conversation_service import ThreadConversationService
from .memory_chunker_worker import memory_chunker_worker

# Global session service instance (shared across worker invocations)
_session_service = None

# Global topic classifier instance (cached model per worker)
_topic_classifier = None

# Global mode router instance (shared across invocations)
_mode_router = None

# Global intent classifier instance
_intent_classifier = None

# Global orchestrator instance
_orchestrator = None

# Global thread conversation service
_thread_conv_service = None


def get_orchestrator():
    """Get or create global OrchestratorService instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorService()
    return _orchestrator


def get_thread_conv_service() -> ThreadConversationService:
    """Get or create global ThreadConversationService instance."""
    global _thread_conv_service
    if _thread_conv_service is None:
        _thread_conv_service = ThreadConversationService()
    return _thread_conv_service


def get_session_service():
    """Get or create global session service instance."""
    global _session_service
    if _session_service is None:
        episodic_config = ConfigService.resolve_agent_config("episodic-memory")
        inactivity_timeout = episodic_config.get('inactivity_timeout', 600)
        _session_service = SessionService(inactivity_timeout=inactivity_timeout)
    return _session_service


def get_topic_classifier():
    """Get or create global topic classifier instance (caches embedding model)."""
    global _topic_classifier
    if _topic_classifier is None:
        _topic_classifier = TopicClassifierService()
    return _topic_classifier


def get_intent_classifier():
    """Get or create global intent classifier instance."""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifierService()
    return _intent_classifier


def get_mode_router():
    """Get or create global mode router instance."""
    global _mode_router
    if _mode_router is None:
        import os
        # Prefer generated config (from stability regulator) over base config
        generated_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs", "generated", "mode_router_config.json"
        )
        if os.path.exists(generated_path):
            try:
                with open(generated_path, 'r') as f:
                    router_config = json.load(f)
                logging.info("[DIGEST] Loaded generated mode router config")
            except Exception:
                router_config = ConfigService.get_agent_config("mode-router")
        else:
            router_config = ConfigService.get_agent_config("mode-router")
        _mode_router = ModeRouterService(router_config)
    return _mode_router


def enqueue_memory_chunker(topic: str, exchange_id: str, prompt_message: str, response_message: str, metadata: dict = None, thread_id: str = None):
    """Enqueue a memory chunking job for the completed exchange."""
    memory_queue = PromptQueue(queue_name="memory-chunker-queue", worker_func=memory_chunker_worker)

    job_payload = {
        'topic': topic,
        'exchange_id': exchange_id,
        'prompt_message': prompt_message,
        'response_message': response_message,
        'metadata': metadata or {},
        'thread_id': thread_id or '',
    }

    memory_queue.enqueue(job_payload)


def load_configs():
    """Load frontal cortex mode-specific prompts and configurations."""
    soul_prompt = ConfigService.get_agent_prompt("soul")
    identity_prompt = ConfigService.get_agent_prompt("identity-core")
    cortex_config = ConfigService.resolve_agent_config("frontal-cortex")

    # Mode-specific prompts: soul → identity → mode prompt (instincts + context + contract)
    # Ordering: values first, then voice, then behavioral nudges closest to generation
    respond_prompt = soul_prompt + "\n\n" + identity_prompt + "\n\n" + ConfigService.get_agent_prompt("frontal-cortex-respond")
    clarify_prompt = soul_prompt + "\n\n" + identity_prompt + "\n\n" + ConfigService.get_agent_prompt("frontal-cortex-clarify")
    acknowledge_prompt = identity_prompt + "\n\n" + ConfigService.get_agent_prompt("frontal-cortex-acknowledge")
    # ACT does NOT get identity — reasoning stays pure
    act_prompt = ConfigService.get_agent_prompt("frontal-cortex-act")

    return {
        'cortex': {
            'config': cortex_config,
            'prompt_map': {
                'RESPOND': respond_prompt,
                'CLARIFY': clarify_prompt,
                'ACKNOWLEDGE': acknowledge_prompt,
                'ACT': act_prompt,
            }
        },
        'memory_chunker': {
            'config': ConfigService.resolve_agent_config("memory-chunker")
        }
    }


def get_existing_topics_from_db() -> list:
    """Retrieve existing topics from the topics PostgreSQL table."""
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM topics ORDER BY created_at DESC LIMIT 20")
            rows = cursor.fetchall()
            cursor.close()
            return [row[0] for row in rows if row[0]]
    except Exception as e:
        logging.debug(f"[DIGEST] Could not load existing topics from DB: {e}")
        return []


def calculate_context_warmth(working_memory_len: int, gists: list, world_state_nonempty: bool) -> float:
    """
    Calculate context warmth signal (0.0-1.0) for scaling uncertainty cost.

    Gists with type='cold_start' are excluded so cold-start booster gists
    don't inflate the warmth signal.
    """
    wm_score = min(working_memory_len / 4, 1.0)
    real_gist_count = sum(1 for g in gists if g.get('type') != 'cold_start')
    gist_score = min(real_gist_count / 5, 1.0)
    world_score = 1.0 if world_state_nonempty else 0.0
    warmth = (wm_score + gist_score + world_score) / 3
    return warmth


def classify_prompt(text, existing_topics, recent_topic, gist_context, world_state, classifier_config, classifier_prompt):
    """Classify the prompt and return classification result with timing."""
    parts = []
    if gist_context and gist_context != "No previous conversation context available":
        parts.append(f"## Context\n{gist_context}")
    if world_state:
        parts.append(f"## World State\n{world_state}")
    parts.append(f"## Prompt\n{text}")
    if existing_topics:
        topics_list = '\n'.join([f"- {topic}" for topic in existing_topics])
        parts.append(f"## Existing Topics\n{topics_list}")

    user_message = "\n\n".join(parts)
    classifier = create_llm_service(classifier_config)
    start_time = time.time()
    response = classifier.send_message(classifier_prompt, user_message).text
    classification_time = time.time() - start_time

    return json.loads(response), classification_time


def generate_for_mode(topic, text, mode, classification, thread_conv_service, cortex_config, cortex_prompt_map, metadata=None, act_history_context=None, thread_id=None):
    """
    Generate response for a terminal mode (RESPOND, CLARIFY, ACKNOWLEDGE).

    Single LLM call with mode-specific prompt. No decision gate, no alternative paths.

    Args:
        act_history_context: Optional act_history string from a preceding ACT loop.
            When present, the LLM can reference tool results in its response.
        thread_id: Thread ID for working memory + world state context.

    Returns:
        dict: {mode, modifiers, response, generation_time, actions, confidence}
    """
    from services.config_service import ConfigService

    prompt = cortex_prompt_map.get(mode, cortex_prompt_map['RESPOND'])

    # Load mode-specific config
    config_name = f"frontal-cortex-{mode.lower()}"
    try:
        config = ConfigService.resolve_agent_config(config_name)
        logging.info(f"[Mode:{mode}] Using config {config_name}.json with model {config.get('model')}")
    except Exception as e:
        logging.warning(f"[Mode:{mode}] Config {config_name}.json not found, using base config: {e}")
        config = cortex_config

    cortex_service = FrontalCortexService(config)
    chat_history = thread_conv_service.get_conversation_history(thread_id) if thread_id else []

    response_data = cortex_service.generate_response(
        system_prompt_template=prompt,
        original_prompt=text,
        classification=classification,
        chat_history=chat_history,
        act_history=act_history_context or "(none)",
        thread_id=thread_id,
    )

    # Router decided the mode, not the LLM
    response_data['mode'] = mode

    # Ensure non-empty response for terminal modes
    if mode != 'IGNORE' and not response_data.get('response', '').strip():
        if mode == 'ACKNOWLEDGE':
            response_data['response'] = "Got it."
        elif mode == 'CLARIFY':
            response_data['response'] = "Could you tell me more about what you mean?"
        else:
            response_data['response'] = "I understand. Let me think about that."

    return response_data


def generate_with_act_loop(topic, text, classification, thread_conv_service, cortex_config, cortex_prompt_map, mode_router, signals, metadata=None, context_warmth=1.0, relevant_tools=None, thread_id=None):
    """
    Run ACT loop: execute actions, then re-route to terminal mode for response.

    Simplified ACT loop — no evaluate_paths(), no decision gate override.
    After actions complete or budget exhausted, re-routes through mode_router
    (excluding ACT) for terminal response.

    Returns:
        dict: {mode, modifiers, response, generation_time, actions, confidence}
    """
    from services.act_loop_service import ActLoopService
    from services.cortex_iteration_service import CortexIterationService
    from services.database_service import get_shared_db_service

    act_prompt = cortex_prompt_map['ACT']

    # Load ACT-specific config
    from services.config_service import ConfigService
    try:
        config = ConfigService.resolve_agent_config("frontal-cortex-act")
        logging.info(f"[ACT Loop] Using config frontal-cortex-act.json with model {config.get('model')}")
    except Exception as e:
        logging.warning(f"[ACT Loop] Config frontal-cortex-act.json not found, using base config: {e}")
        config = cortex_config

    cortex_service = FrontalCortexService(config)
    chat_history = thread_conv_service.get_conversation_history(thread_id) if thread_id else []

    # Initialize ACT loop
    act_cumulative_timeout = cortex_config.get('act_cumulative_timeout', 60.0)
    act_per_action_timeout = cortex_config.get('act_per_action_timeout', 10.0)
    max_act_iterations = cortex_config.get('max_act_iterations', 5)

    act_loop = ActLoopService(
        config=cortex_config,
        cumulative_timeout=act_cumulative_timeout,
        per_action_timeout=act_per_action_timeout,
        max_iterations=max_act_iterations,
    )
    act_loop.context_warmth = context_warmth

    # Generate loop ID for iteration tracking
    iteration_service = None
    loop_id = None
    try:
        db_service = get_shared_db_service()
        iteration_service = CortexIterationService(db_service)
        loop_id = iteration_service.create_loop_id()
        act_loop.loop_id = loop_id
    except Exception as e:
        logging.warning(f"Failed to initialize iteration logging: {e}")

    try:
        exchange_id = thread_conv_service.get_latest_exchange_id(thread_id) if thread_id else "unknown"
    except Exception:
        exchange_id = "unknown"

    session_id = "session_placeholder"

    # ACT loop — simplified, safety-capped at max_act_iterations
    # Repetition detection: if same action type called 3+ times consecutively, force exit
    consecutive_same_action = 0
    last_action_type = None

    while True:
        iteration_start = time.time()

        # Generate action plan via ACT-specific prompt
        response_data = cortex_service.generate_response(
            system_prompt_template=act_prompt,
            original_prompt=text,
            classification=classification,
            chat_history=chat_history,
            act_history=act_loop.get_history_context(),
            relevant_tools=relevant_tools,
        )

        actions = response_data.get('actions', [])

        # Repetition detection: detect model stuck in a loop
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
            logging.warning(f"[MODE:ACT] [ACT LOOP] Repetition detected: '{last_action_type}' called {consecutive_same_action} times consecutively, forcing exit")
            termination_reason = 'repetition_detected'
            can_continue = False
        else:
            # Check if loop can continue
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

            # Accumulate fatigue
            fatigue_added = act_loop.accumulate_fatigue(actions_executed, act_loop.iteration_number)

            # Estimate net value (for cortex_iterations logging + strategy analysis)
            iteration_net_value = ActLoopService.estimate_net_value(actions_executed, act_loop.iteration_number)

            # Record skill outcomes to procedural memory
            from services.skill_outcome_recorder import record_skill_outcomes
            record_skill_outcomes(actions_executed, topic)
        elif not actions:
            logging.warning("[MODE:ACT] No actions specified, exiting ACT loop")
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

    # Batch write iterations to database
    if iteration_service and act_loop.iteration_logs:
        try:
            iteration_service.log_iterations_batch(
                loop_id=loop_id,
                topic=topic,
                exchange_id=exchange_id,
                session_id=session_id,
                iterations=act_loop.iteration_logs
            )
        except Exception as e:
            logging.error(f"[ACT LOOP] Failed to log iterations: {e}")

    # Log fatigue telemetry
    telemetry = act_loop.get_fatigue_telemetry()
    telemetry['termination_reason'] = termination_reason
    logging.info(f"[ACT LOOP] Fatigue telemetry: {telemetry}")
    try:
        from services.database_service import get_shared_db_service
        _tel_db = get_shared_db_service()
        _tel_log = InteractionLogService(_tel_db)
        _tel_log.log_event(
            event_type='act_loop_telemetry',
            payload=telemetry,
            topic=topic,
            source='act_loop',
        )
    except Exception:
        pass

    # Re-route through mode router (excluding ACT to prevent oscillation)
    # Update signals with enriched context from actions
    terminal_mode = 'RESPOND'  # Default fallback
    try:
        # Force ACT suppression via previous_mode; skip tie-breaker since terminal mode is implicit
        re_route_result = mode_router.route(signals, text, previous_mode='ACT', skip_tiebreaker=True)
        terminal_mode = re_route_result['mode']
        if terminal_mode == 'ACT':
            terminal_mode = 'RESPOND'  # Safety: never re-ACT
        logging.info(f"[MODE:ACT→{terminal_mode}] Re-routed after ACT loop")
    except Exception as e:
        logging.warning(f"[MODE:ACT→RESPOND] Re-routing failed: {e}")

    # Generate terminal response (pass act_history so LLM can reference tool results)
    act_history_for_respond = act_loop.get_history_context()
    logging.info(f"[MODE:ACT→{terminal_mode}] Passing act_history to terminal response ({len(act_history_for_respond)} chars, {len(act_loop.act_history)} actions)")
    terminal_response = generate_for_mode(
        topic, text, terminal_mode, classification,
        thread_conv_service, cortex_config, cortex_prompt_map, metadata,
        act_history_context=act_history_for_respond,
        thread_id=thread_id,
    )

    # Enqueue tool outputs for background experience assimilation
    try:
        from workers.tool_worker import _enqueue_tool_reflection
        _enqueue_tool_reflection(act_loop.act_history, topic, text)
    except Exception as _e:
        logging.debug(f"[MODE:ACT] Reflection enqueue skipped: {_e}")

    # Carry over action history
    terminal_response['actions'] = [
        {'type': r['action_type'], 'status': r['status'], 'result': r['result']}
        for r in act_loop.act_history
    ] if act_loop.act_history else None

    return terminal_response


def route_and_generate(topic, text, classification, thread_conv_service, cortex_config, cortex_prompt_map,
                       mode_router, signals, fact_store, metadata=None, context_warmth=1.0,
                       pre_routing_result=None, relevant_tools=None, thread_id=None):
    """
    Main routing + generation function.

    Phase C: collect signals → route → generate for selected mode.

    Args:
        pre_routing_result: Optional pre-computed routing result from fast-path check.
            When provided, skips the mode_router.route() call to avoid double routing.

    Returns:
        tuple: (response_data dict, routing_result dict)
    """
    # Initialize routing decision service for logging
    routing_decision_service = None
    try:
        from services.routing_decision_service import RoutingDecisionService
        from services.database_service import get_shared_db_service
        db_service = get_shared_db_service()
        routing_decision_service = RoutingDecisionService(db_service)
    except Exception as e:
        logging.warning(f"[DIGEST] Routing decision logging not available: {e}")

    if pre_routing_result:
        # Use pre-computed routing result (avoids double mode routing)
        routing_result = pre_routing_result
        selected_mode = routing_result['mode']
        previous_mode = None
        logging.info(f"[DIGEST] Using pre-routed mode: {selected_mode} (confidence={routing_result['router_confidence']:.3f})")
    else:
        # Get previous mode for anti-oscillation
        previous_mode = None
        if routing_decision_service:
            previous_mode = routing_decision_service.get_previous_mode(topic)

        # Store prompt text in signals for reflection service
        signals['_prompt_text'] = text

        # Route
        routing_result = mode_router.route(signals, text, previous_mode=previous_mode)
        selected_mode = routing_result['mode']

        logging.info(f"[DIGEST] Mode router selected: {selected_mode} (confidence={routing_result['router_confidence']:.3f})")

    # Log routing decision
    try:
        exchange_id = thread_conv_service.get_latest_exchange_id(thread_id) if thread_id else "unknown"
    except Exception:
        exchange_id = "unknown"

    if routing_decision_service:
        try:
            routing_decision_service.log_decision(
                topic=topic,
                exchange_id=exchange_id,
                routing_result=routing_result,
                previous_mode=previous_mode,
            )
        except Exception as e:
            logging.warning(f"[DIGEST] Failed to log routing decision: {e}")

    # Generate based on selected mode
    if selected_mode == 'ACT':
        response_data = generate_with_act_loop(
            topic, text, classification, thread_conv_service,
            cortex_config, cortex_prompt_map, mode_router, signals,
            metadata=metadata, context_warmth=context_warmth,
            relevant_tools=relevant_tools,
            thread_id=thread_id,
        )
    elif selected_mode == 'IGNORE':
        response_data = {
            'mode': 'IGNORE',
            'modifiers': [],
            'response': '',
            'generation_time': 0.0,
            'actions': None,
            'confidence': 1.0,
        }
    else:
        response_data = generate_for_mode(
            topic, text, selected_mode, classification,
            thread_conv_service, cortex_config, cortex_prompt_map, metadata,
            thread_id=thread_id,
        )

    # Store response
    thread_conv_service.add_response(
        thread_id,
        response_data['response'],
        response_data['generation_time']
    )

    # Route through orchestrator
    if metadata:
        try:
            orchestrator = get_orchestrator()

            # Check if there's a temporary ACK to remove (from previous ACKNOWLEDGE mode)
            removes_id = None
            cycle_id = metadata.get('cycle_id') or metadata.get('root_cycle_id')
            if cycle_id:
                try:
                    from services.redis_client import RedisClientService
                    redis = RedisClientService.create_connection()
                    removes_id = redis.get(f"temp_ack:{cycle_id}")
                    if removes_id:
                        if isinstance(removes_id, bytes):
                            removes_id = removes_id.decode()
                        # Clean up the mapping
                        redis.delete(f"temp_ack:{cycle_id}")
                        logging.debug(f"[DIGEST] Retrieved temp_id to remove: {removes_id}")
                except Exception as e:
                    logging.debug(f"[DIGEST] Failed to retrieve temp_id: {e}")

            context = {
                'topic': topic,
                'response': response_data.get('response', ''),
                'confidence': response_data.get('confidence', 0.0),
                'generation_time': response_data.get('generation_time', 0.0),
                'destination': metadata.get('destination', 'web'),
                'metadata': metadata,
                'actions': response_data.get('actions', []),
                'clarification_question': response_data.get('response', '') if response_data.get('mode') == 'CLARIFY' else None,
            }

            # Pass removes if we found a temporary ACK to remove
            if removes_id:
                context['removes'] = removes_id

            mode = response_data.get('mode', 'RESPOND')
            logging.info(f"[FRONTAL CORTEX] Routing through orchestrator: {mode}")

            orchestrator_result = orchestrator.route_path(mode=mode, context=context)

            if orchestrator_result['status'] == 'error':
                logging.error(f"[ORCHESTRATOR] Error: {orchestrator_result['message']}")
            else:
                logging.info(f"[ORCHESTRATOR] Executed {mode}: {orchestrator_result.get('result', {})}")
        except Exception as e:
            logging.error(f"[ORCHESTRATOR] Failed: {e}")

    return response_data, routing_result


def _handle_cron_tool_result(text: str, metadata: dict) -> str:
    """
    Pipeline for scheduled (cron) tool results.

    Goes directly to response generation (no mode routing or user input logging).
    Tool has already formatted the prompt with its data.
    Enqueues to memory-chunker with memory_durability: 'cron_tool' for 3x decay.
    """
    try:
        from services.config_service import ConfigService

        configs = load_configs()
        cortex_config = configs['cortex']['config']

        tool_name = metadata.get('tool_name', 'unknown')
        priority = metadata.get('priority', 'normal')
        destination = metadata.get('destination', 'web')

        # Resolve thread
        thread_service = get_thread_service()
        thread_id = metadata.get('thread_id')
        if not thread_id:
            platform = metadata.get('source', 'cron_tool')
            resolution = thread_service.resolve_thread('default', 'default', platform)
            thread_id = resolution.thread_id

        thread_conv_service = get_thread_conv_service()
        working_memory = WorkingMemoryService(
            max_turns=cortex_config.get('max_working_memory_turns', 10)
        )

        # Get recent chat history for context
        chat_history = thread_conv_service.get_conversation_history(thread_id) if thread_id else []

        # Load scheduled tool prompt template
        scheduled_tool_template = ConfigService.get_agent_prompt("frontal-cortex-scheduled-tool")

        try:
            scheduled_tool_config = ConfigService.resolve_agent_config("frontal-cortex-scheduled-tool")
        except Exception:
            logging.warning("[CRON TOOL] frontal-cortex-scheduled-tool.json not found, using acknowledge config")
            scheduled_tool_config = ConfigService.resolve_agent_config("frontal-cortex-acknowledge")

        cortex_service = FrontalCortexService(scheduled_tool_config)

        # Generate response using the scheduled tool prompt
        response_data = cortex_service.generate_response(
            system_prompt_template=scheduled_tool_template,
            original_prompt=text,
            classification={
                'topic': f'cron_tool:{tool_name}',
                'confidence': 10,
                'similar_topic': '',
                'topic_update': '',
            },
            chat_history=chat_history,
            thread_id=thread_id,
            user_metadata={
                'tool_name': tool_name,
                'priority': priority,
            },
        )

        # Always RESPOND mode for scheduled tools (bypass mode routing)
        response_data['mode'] = 'RESPOND'

        if not response_data.get('response', '').strip():
            logging.info(f"[CRON TOOL] {tool_name}: Empty response generated — skipping delivery")
            return f"Tool '{tool_name}' | Mode: RESPOND | Empty response (no updates)"

        # Append assistant turn to working memory
        working_memory.append_turn(thread_id or f'cron_tool:{tool_name}', 'assistant', response_data['response'])

        # Store response in conversation history
        if thread_id:
            thread_conv_service.add_response(
                thread_id, response_data['response'], response_data.get('generation_time', 0.0)
            )

        # Route through orchestrator for delivery
        try:
            orchestrator = get_orchestrator()

            context = {
                'topic': f'cron_tool:{tool_name}',
                'response': response_data['response'],
                'confidence': response_data.get('confidence', 0.5),
                'generation_time': response_data.get('generation_time', 0.0),
                'destination': destination,
                'metadata': metadata,
                'actions': [],
            }
            orchestrator.route_path(mode='RESPOND', context=context)
        except Exception as e:
            logging.error(f"[CRON TOOL] {tool_name}: Orchestrator failed: {e}")

        # Enqueue to memory chunker with special durability marker for 3x decay
        try:
            enqueue_memory_chunker(
                topic=f'cron_tool:{tool_name}',
                exchange_id=f'cron_{tool_name}_{int(time.time())}',
                prompt_message=text,
                response_message=response_data['response'],
                metadata={
                    'source': f'cron_tool:{tool_name}',
                    'memory_durability': 'cron_tool',
                    'priority': priority,
                },
                thread_id=thread_id,
            )
        except Exception as e:
            logging.warning(f"[CRON TOOL] {tool_name}: Memory chunker enqueue failed: {e}")

        # Log the cron tool execution
        try:
            from services.database_service import get_shared_db_service
            from services.interaction_log_service import InteractionLogService
            db_service = get_shared_db_service()
            log_service = InteractionLogService(db_service)
            log_service.log_event(
                event_type='cron_tool_executed',
                payload={
                    'tool_name': tool_name,
                    'priority': priority,
                    'response': response_data['response'][:500],
                    'generation_time': response_data.get('generation_time', 0),
                },
                topic=f'cron_tool:{tool_name}',
                source='cron_tool',
                metadata=metadata,
            )
        except Exception:
            pass

        logging.info(
            f"[CRON TOOL] {tool_name} delivered: priority={priority} "
            f"({response_data.get('generation_time', 0):.2f}s)"
        )

        return (
            f"Tool '{tool_name}' | Mode: RESPOND | "
            f"Response generated in {response_data.get('generation_time', 0):.2f}s"
        )

    except Exception as e:
        logging.error(f"[CRON TOOL] Failed: {e}")
        tool_name = metadata.get('tool_name', 'unknown')
        return f"Tool '{tool_name}' | ERROR: cron_tool - {e}"


def _handle_proactive_drift(text: str, metadata: dict) -> str:
    """
    Pipeline for proactive drift messages (system-initiated outreach).

    Goes through full mode routing (router as final judge) but skips:
    - User input logging (no user input)
    - Topic classification (topic provided in metadata)
    - Reward evaluation (no previous exchange to evaluate)

    The mode router may select IGNORE for a weak thought — that's a feature.
    """
    configs = load_configs()
    cortex_config = configs['cortex']['config']
    cortex_prompt_map = configs['cortex']['prompt_map']

    topic = metadata.get('related_topic', 'general')
    drift_gist = metadata.get('drift_gist', text)
    drift_type = metadata.get('drift_type', 'reflection')
    proactive_id = metadata.get('proactive_id', '')
    destination = metadata.get('destination', 'web')

    # Resolve thread for proactive drift
    thread_conv_service = get_thread_conv_service()
    thread_id = metadata.get('thread_id')
    if not thread_id:
        platform = metadata.get('source', 'unknown')
        resolution = get_thread_service().resolve_thread('default', 'default', platform)
        thread_id = resolution.thread_id

    working_memory = WorkingMemoryService(
        max_turns=cortex_config.get('max_working_memory_turns', 10)
    )
    gist_storage = GistStorageService(attention_span_minutes=30, min_confidence=5, max_gists=8)
    world_state_service = WorldStateService()
    fact_store = FactStoreService()

    # Build classification stub (topic is pre-determined)
    classification = {
        'topic': topic,
        'confidence': 10,
        'similar_topic': '',
        'topic_update': '',
    }

    # Collect routing signals for the mode router (full router run)
    mode_router = get_mode_router()
    context_warmth = 0.0
    try:
        wm_turns = working_memory.get_recent_turns(thread_id or topic)
        gists = gist_storage.get_latest_gists(topic)
        world_state = world_state_service.get_world_state(topic, thread_id=thread_id)
        context_warmth = calculate_context_warmth(
            working_memory_len=len(wm_turns),
            gists=gists,
            world_state_nonempty=bool(world_state)
        )
    except Exception:
        pass

    # Collect signals for mode routing
    try:
        from services.session_service import SessionService
        session_service = get_session_service()

        # The prompt to the router is the drift thought itself
        from services.topic_classifier_service import TopicClassifierService
        topic_classifier = get_topic_classifier()
        classification_result = topic_classifier.classify(drift_gist, recent_topic=topic)

        signals = collect_routing_signals(
            text=drift_gist,
            topic=topic,
            context_warmth=context_warmth,
            working_memory=working_memory,
            gist_storage=gist_storage,
            fact_store=fact_store,
            world_state_service=world_state_service,
            classification_result=classification_result,
            session_service=session_service,
        )
    except Exception as e:
        logging.warning(f"[PROACTIVE] Signal collection failed: {e}")
        signals = {}

    # Route through mode router (unbiased — doesn't know this is proactive)
    try:
        routing_result = mode_router.route(signals, drift_gist)
        selected_mode = routing_result['mode']
        logging.info(
            f"[PROACTIVE] Router selected: {selected_mode} "
            f"(confidence={routing_result.get('router_confidence', 0):.3f})"
        )
    except Exception as e:
        logging.warning(f"[PROACTIVE] Routing failed, defaulting to RESPOND: {e}")
        selected_mode = 'RESPOND'
        routing_result = {'mode': 'RESPOND', 'router_confidence': 0.5}

    # If router says IGNORE, respect it — the thought wasn't worth sharing
    if selected_mode == 'IGNORE':
        logging.info(f"[PROACTIVE] Router selected IGNORE — thought filtered")
        # Record as router_ignored for circuit breaker
        try:
            from services.autonomous_actions.engagement_tracker import EngagementTracker
            tracker = EngagementTracker()
            tracker._update_engagement_state(proactive_id, 'router_ignored', -0.3)
        except Exception:
            pass
        return f"Topic '{topic}' | Mode: PROACTIVE_IGNORED | Router filtered thought"

    # ACT mode doesn't make sense for proactive thoughts — fall back to RESPOND
    if selected_mode == 'ACT':
        selected_mode = 'RESPOND'

    # Generate proactive outreach using dedicated prompt template
    try:
        from services.config_service import ConfigService

        proactive_template = ConfigService.get_agent_prompt("frontal-cortex-proactive")

        try:
            proactive_config = ConfigService.resolve_agent_config("frontal-cortex-proactive")
        except Exception:
            logging.warning("[PROACTIVE] frontal-cortex-proactive.json not found, falling back to acknowledge config")
            proactive_config = ConfigService.resolve_agent_config("frontal-cortex-acknowledge")

        cortex_service = FrontalCortexService(proactive_config)
        chat_history = thread_conv_service.get_conversation_history(thread_id) if thread_id else []

        response_data = cortex_service.generate_response(
            system_prompt_template=proactive_template,
            original_prompt=drift_gist,
            classification=classification,
            chat_history=chat_history,
            thread_id=thread_id,
        )

        # Proactive messages always deliver as RESPOND
        response_data['mode'] = 'RESPOND'
        selected_mode = 'RESPOND'

        if not response_data.get('response', '').strip():
            logging.info(f"[PROACTIVE] Empty response generated — skipping delivery")
            return f"Topic '{topic}' | Mode: PROACTIVE_EMPTY | No response generated"

        # Append assistant turn to working memory
        working_memory.append_turn(thread_id or topic, 'assistant', response_data['response'])

        # Store response in conversation history
        if thread_id:
            thread_conv_service.add_response(
                thread_id, response_data['response'], response_data['generation_time']
            )

        # Store proactive_id in Redis for engagement correlation
        if proactive_id:
            try:
                from services.redis_client import RedisClientService
                redis_conn = RedisClientService.create_connection()
                redis_conn.setex(
                    f"proactive_response_tag:{topic}",
                    14400,  # 4h TTL
                    proactive_id,
                )
            except Exception:
                pass

        # Route through orchestrator for delivery
        try:
            orchestrator = get_orchestrator()

            context = {
                'topic': topic,
                'response': response_data['response'],
                'confidence': response_data.get('confidence', 0.5),
                'generation_time': response_data.get('generation_time', 0.0),
                'destination': destination,
                'metadata': metadata,
                'actions': [],
            }
            orchestrator.route_path(mode=selected_mode, context=context)
        except Exception as e:
            logging.error(f"[PROACTIVE] Orchestrator failed: {e}")

        # Log the proactive send
        try:
            from services.database_service import get_shared_db_service
            from services.interaction_log_service import InteractionLogService
            db_service = get_shared_db_service()
            log_service = InteractionLogService(db_service)
            log_service.log_event(
                event_type='proactive_sent',
                payload={
                    'response': response_data['response'][:500],
                    'mode': selected_mode,
                    'drift_type': drift_type,
                    'proactive_id': proactive_id,
                    'router_confidence': routing_result.get('router_confidence', 0),
                },
                topic=topic,
                source='proactive_drift',
                metadata=metadata,
            )
        except Exception:
            pass

        logging.info(
            f"[PROACTIVE] Delivered: [{drift_type}] → {selected_mode} "
            f"(proactive_id={proactive_id[:8] if proactive_id else '?'})"
        )

        return (
            f"Topic '{topic}' | Mode: PROACTIVE_{selected_mode} | "
            f"Response generated in {response_data.get('generation_time', 0):.2f}s"
        )

    except Exception as e:
        logging.error(f"[PROACTIVE] Failed: {e}")
        return f"Topic '{topic}' | ERROR: proactive - {e}"


def _handle_tool_result(text: str, metadata: dict) -> str:
    """
    Handle follow-up from tool_worker after background ACT loop completes.

    Similar to _handle_delegate_result — shortened pipeline:
    - No classification, no user turn append, no reward eval
    - Generates follow-up response via RESPOND mode with follow-up prompt
    - Routes through orchestrator for delivery
    - Includes stale suppression and delivery deferral
    """
    configs = load_configs()
    cortex_config = configs['cortex']['config']

    topic = metadata.get('topic', 'general')
    original_prompt = metadata.get('original_prompt', '')
    act_history_context = metadata.get('act_history_context', '(none)')
    root_cycle_id = metadata.get('root_cycle_id', '')
    tool_cycle_id = metadata.get('tool_cycle_id', '')
    destination = metadata.get('destination', 'web')
    original_created_at = metadata.get('original_created_at', 0)
    # Resolve thread for this tool result
    thread_conv_service = get_thread_conv_service()
    thread_id = metadata.get('thread_id')
    if not thread_id:
        platform = metadata.get('source', 'unknown')
        resolution = get_thread_service().resolve_thread('default', 'default', platform)
        thread_id = resolution.thread_id

    working_memory = WorkingMemoryService(
        max_turns=cortex_config.get('max_working_memory_turns', 10)
    )

    # ── Stale follow-up suppression ──────────────────────────
    # If the user changed topic since the original question, suppress
    recent_topic_service = RecentTopicService(ttl_minutes=30, user_id='default', channel_id='default')
    current_topic = recent_topic_service.get_recent_topic()
    if current_topic and current_topic != topic:
        # Check semantic similarity before suppressing
        try:
            from services.embedding_service import EmbeddingService
            import numpy as np

            emb_service = EmbeddingService()
            current_emb = emb_service.generate_embedding(current_topic)
            original_emb = emb_service.generate_embedding(topic)

            similarity = float(np.dot(current_emb, original_emb) / (
                np.linalg.norm(current_emb) * np.linalg.norm(original_emb) + 1e-8
            ))

            if similarity < 0.45:
                # Genuinely different topic — store as gist silently
                gist_storage = GistStorageService(attention_span_minutes=30, min_confidence=5, max_gists=8)
                gist_storage.store_gists(
                    topic=topic,
                    gists=[{'content': f"[Background research] {act_history_context[:300]}", 'type': 'tool_result', 'confidence': 7}],
                    prompt=original_prompt,
                    response='(suppressed follow-up)'
                )
                _log_cycle_event('followup_suppressed', {'reason': 'stale', 'similarity': similarity}, topic)
                logging.info(f"[TOOL RESULT] Suppressed stale follow-up (topic drift, similarity={similarity:.2f})")
                return f"Topic '{topic}' | SUPPRESSED: topic changed to '{current_topic}'"
        except Exception as e:
            logging.debug(f"[TOOL RESULT] Stale check failed: {e}")

    # ── Delivery deferral ────────────────────────────────────
    # If user is mid-conversation, defer
    defer_result = _should_deliver_followup('default', tool_cycle_id)
    if defer_result == 'suppress':
        gist_storage = GistStorageService(attention_span_minutes=30, min_confidence=5, max_gists=8)
        gist_storage.store_gists(
            topic=topic,
            gists=[{'content': f"[Background research] {act_history_context[:300]}", 'type': 'tool_result', 'confidence': 7}],
            prompt=original_prompt,
            response='(deferred → suppressed)'
        )
        _log_cycle_event('followup_suppressed', {'reason': 'deferred_max'}, topic)
        return f"Topic '{topic}' | SUPPRESSED: max deferrals reached"

    # ── Generate follow-up response ──────────────────────────
    # Use the followup prompt template
    try:
        from services.config_service import ConfigService

        soul_prompt = ConfigService.get_agent_prompt("soul")
        identity_prompt = ConfigService.get_agent_prompt("identity-core")
        followup_template = ConfigService.get_agent_prompt("frontal-cortex-followup")
        followup_prompt = soul_prompt + "\n\n" + identity_prompt + "\n\n" + followup_template

        # Calculate latency tone
        elapsed = time.time() - original_created_at if original_created_at else 0
        if elapsed < 5:
            latency_tone = ""
        elif elapsed < 30:
            latency_tone = "Brief wait acknowledged — be direct with findings."
        else:
            latency_tone = "Significant delay — lead with 'I dug deeper into this' framing."

        followup_prompt = followup_prompt.replace('{{latency_tone}}', latency_tone)
        followup_prompt = followup_prompt.replace('{{original_prompt}}', original_prompt)

        classification = {
            'topic': topic,
            'confidence': 10,
            'similar_topic': '',
            'topic_update': '',
        }

        # Load mode-specific config
        try:
            config = ConfigService.resolve_agent_config("frontal-cortex-respond")
        except Exception:
            config = cortex_config

        cortex_service = FrontalCortexService(config)
        chat_history = thread_conv_service.get_conversation_history(thread_id) if thread_id else []

        response_data = cortex_service.generate_response(
            system_prompt_template=followup_prompt,
            original_prompt=original_prompt,
            classification=classification,
            chat_history=chat_history,
            act_history=act_history_context,
            thread_id=thread_id,
        )

        if not response_data.get('response', '').strip():
            response_data['response'] = "I looked into that, but couldn't find anything conclusive."

        # Append assistant turn
        working_memory.append_turn(thread_id or topic, 'assistant', response_data['response'])
        thread_conv_service.add_response(
            thread_id, response_data['response'], response_data['generation_time']
        )

        # Route through orchestrator
        try:
            from services.redis_client import RedisClientService

            orchestrator = get_orchestrator()

            # Retrieve temporary ID if this tool result has an associated ACK
            removes_id = None
            if tool_cycle_id:
                try:
                    redis = RedisClientService.create_connection()
                    removes_id = redis.get(f"temp_ack:{tool_cycle_id}")
                    if removes_id:
                        if isinstance(removes_id, bytes):
                            removes_id = removes_id.decode()
                        # Clean up the mapping
                        redis.delete(f"temp_ack:{tool_cycle_id}")
                        logging.debug(f"[TOOL RESULT] Retrieved temp_id to remove: {removes_id}")
                except Exception as e:
                    logging.debug(f"[TOOL RESULT] Failed to retrieve temp_id: {e}")

            # Create metadata copy to pass through
            result_metadata = dict(metadata)
            if removes_id:
                result_metadata['removes'] = removes_id

            context = {
                'topic': topic,
                'response': response_data['response'],
                'confidence': response_data.get('confidence', 0.7),
                'generation_time': response_data.get('generation_time', 0.0),
                'destination': destination,
                'metadata': result_metadata,
                'actions': [],
                'removes': removes_id,  # Pass directly for handler access
            }
            orchestrator.route_path(mode='RESPOND', context=context)
        except Exception as e:
            logging.error(f"[TOOL RESULT] Orchestrator failed: {e}")

        # Emit encode event for memory chunker
        try:
            event_bus = EventBusService()
            event_bus.emit_and_handle(ENCODE_EVENT, {
                'topic': topic,
                'exchange_id': f"tool_followup_{tool_cycle_id[:8] if tool_cycle_id else 'unknown'}",
                'prompt_message': original_prompt,
                'response_message': response_data['response'],
                'metadata': {
                    **metadata,
                    'cycle_id': tool_cycle_id,
                    'root_cycle_id': root_cycle_id,
                    'is_followup': True,
                },
                'thread_id': thread_id,
            })
        except Exception as e:
            logging.debug(f"[TOOL RESULT] Encode event failed: {e}")

        _log_cycle_event('followup_delivered', {
            'root_cycle_id': root_cycle_id,
            'latency_ms': int(elapsed * 1000) if elapsed else 0,
        }, topic)

        logging.info(
            f"[TOOL RESULT] Follow-up delivered for topic '{topic}': "
            f"'{response_data['response'][:80]}...'"
        )

        return (
            f"Topic '{topic}' | Mode: TOOL_RESULT_FOLLOWUP | "
            f"Response generated in {response_data.get('generation_time', 0):.2f}s"
        )

    except Exception as e:
        logging.error(f"[TOOL RESULT] Failed: {e}", exc_info=True)
        return f"Topic '{topic}' | ERROR: tool result - {e}"


def _should_deliver_followup(user_id: str, cycle_id: str) -> str:
    """
    Check if a follow-up should be delivered now.

    Returns 'deliver', or 'suppress' (after max deferrals).
    Note: 'defer' with re-enqueue is complex with RQ so we simplify
    to a quick check — if user sent a message very recently, we still deliver
    since the follow-up anchoring makes it coherent.
    """
    try:
        from services.redis_client import RedisClientService
        redis_conn = RedisClientService.create_connection()

        # Check defer count
        defer_key = f"followup_defer:{cycle_id}"
        defer_count = int(redis_conn.get(defer_key) or 0)

        if defer_count >= 3:
            return 'suppress'

        return 'deliver'
    except Exception:
        return 'deliver'


def _check_active_tool_work(text: str, topic: str) -> str:
    """
    If a similar question already has active tool work, respond with progress
    instead of spawning new work.

    Returns progress message string, or None if no match.
    """
    try:
        from services.cycle_service import CycleService
        from services.database_service import get_shared_db_service
        from services.embedding_service import EmbeddingService
        import numpy as np

        db_service = get_shared_db_service()
        cycle_service = CycleService(db_service)
        active_cycles = cycle_service.get_active_cycles(
            topic=topic, cycle_type='tool_work', status='processing'
        )
        if not active_cycles:
            return None

        emb_service = EmbeddingService()
        prompt_embedding = emb_service.generate_embedding(text)

        for cycle in active_cycles:
            root = cycle_service.get_cycle(cycle['root_cycle_id'])
            if not root or not root.get('content'):
                continue
            root_embedding = emb_service.generate_embedding(root['content'])
            similarity = float(np.dot(prompt_embedding, root_embedding) / (
                np.linalg.norm(prompt_embedding) * np.linalg.norm(root_embedding) + 1e-8
            ))

            if similarity > 0.65:
                elapsed = time.time() - cycle['created_at'].timestamp()
                if elapsed < 10:
                    return "Just started looking into that — I'll update you shortly."
                elif elapsed < 30:
                    return "Still working on it — pulling the latest info now."
                else:
                    return "Digging deeper into this — I'll share what I find soon."

        return None
    except Exception as e:
        logging.debug(f"[DIGEST] Active tool work check failed: {e}")
        return None


def _cancel_active_tool_work(topic: str):
    """Cancel all active tool-work cycles for this topic."""
    try:
        from services.cycle_service import CycleService
        from services.database_service import get_shared_db_service
        from services.redis_client import RedisClientService

        db_service = get_shared_db_service()
        redis_conn = RedisClientService.create_connection()
        cycle_service = CycleService(db_service)
        active = cycle_service.get_active_cycles(
            topic=topic, cycle_type='tool_work', status='processing'
        )
        for cycle in active:
            cycle_service.complete_cycle(cycle['cycle_id'], status='cancelled')
            redis_conn.set(f"cancel:{cycle['cycle_id']}", "1", ex=300)
        if active:
            logging.info(f"[DIGEST] Cancelled {len(active)} active tool-work cycles")
    except Exception as e:
        logging.debug(f"[DIGEST] Cancel tool work failed: {e}")


def _log_cycle_event(event_type: str, payload: dict, topic: str):
    """Log a cycle-related event to interaction_log."""
    try:
        from services.database_service import get_shared_db_service
        db_service = get_shared_db_service()
        log_service = InteractionLogService(db_service)
        log_service.log_event(
            event_type=event_type,
            payload=payload,
            topic=topic,
            source='cycle_service',
        )
    except Exception:
        pass


def _try_proactive_engagement_correlation(text: str, topic: str):
    """
    Check if the user is responding to a proactive message and score engagement.

    Called during Phase A of the normal digest pipeline.
    """
    try:
        from services.autonomous_actions.engagement_tracker import EngagementTracker
        tracker = EngagementTracker()
        result = tracker.check_and_score(user_message=text, topic=topic)
        if result:
            logging.info(
                f"[PROACTIVE ENGAGEMENT] Scored response to proactive message: "
                f"{result['outcome']} (similarity={result['similarity']:.3f})"
            )
    except Exception as e:
        logging.debug(f"[PROACTIVE ENGAGEMENT] Correlation failed: {e}")


def digest_worker(text: str, metadata: dict = None) -> str:
    """
    Main worker function that processes prompts through classification and response generation.

    Pipeline: Phase A (immediate commit) → Phase B (retrieval) → Phase C (route + generate)
              → Phase D (post-response commit) → Phase E (async follow-up)

    Proactive drift messages go through full routing but skip user input logging.
    """
    metadata = metadata or {}

    # Tool result shortcut: follow-up from background tool_worker
    if metadata.get('type') == 'tool_result':
        return _handle_tool_result(text, metadata)

    # Proactive drift shortcut: system-initiated outreach from cognitive drift engine
    if metadata.get('type') == 'proactive_drift':
        return _handle_proactive_drift(text, metadata)

    # Cron tool shortcut: background scheduled tool result (not a conversational turn)
    if metadata.get('source', '').startswith('cron_tool:'):
        return _handle_cron_tool_result(text, metadata)

    # Step 1: Load configurations
    configs = load_configs()
    cortex_config = configs['cortex']['config']
    cortex_prompt_map = configs['cortex']['prompt_map']
    memory_chunker_config = configs['memory_chunker']['config']

    # Step 2: Resolve thread
    thread_service = get_thread_service()
    platform = metadata.get('source', 'unknown')
    thread_resolution = thread_service.resolve_thread('default', 'default', platform)
    thread_id = thread_resolution.thread_id
    metadata['thread_id'] = thread_id

    # Step 2a: Initialize services
    thread_conv_service = ThreadConversationService()
    recent_topic_service = RecentTopicService(ttl_minutes=30, user_id='default', channel_id='default')
    gist_storage = GistStorageService(
        attention_span_minutes=30,
        min_confidence=memory_chunker_config.get('min_gist_confidence', 7),
        max_gists=8
    )
    world_state_service = WorldStateService()
    fact_store = FactStoreService()

    # Initialize working memory (keyed by thread_id)
    max_working_memory_turns = cortex_config.get('max_working_memory_turns', 10)
    working_memory = WorkingMemoryService(max_turns=max_working_memory_turns)

    # Initialize interaction log
    interaction_log = None
    try:
        from services.database_service import get_shared_db_service
        interaction_db = get_shared_db_service()
        interaction_log = InteractionLogService(interaction_db)
    except Exception as e:
        logging.warning(f"[DIGEST] Interaction log not available: {e}")

    # Initialize event bus with encode_event handler
    event_bus = EventBusService()

    def _handle_encode_event(event_type, payload):
        """Translate encode_event into memory-chunker-queue enqueue."""
        enqueue_memory_chunker(
            payload['topic'],
            payload['exchange_id'],
            payload['prompt_message'],
            payload['response_message'],
            metadata=payload.get('metadata'),
            thread_id=payload.get('thread_id'),
        )

    event_bus.subscribe(ENCODE_EVENT, _handle_encode_event)

    # Initialize metrics
    metrics = MetricsService()
    trace_id = metrics.start_trace()
    metrics.record_counter('requests_total')
    request_start_time = time.time()

    # Initialize mode router
    mode_router = get_mode_router()

    # ═══════════════════════════════════════════════════════════
    # PHASE A: IMMEDIATE COMMIT (before any LLM call)
    # ═══════════════════════════════════════════════════════════

    # Step 3: Get existing topics and determine context topic
    existing_topics = get_existing_topics_from_db()
    recent_topic = recent_topic_service.get_recent_topic()
    context_topic = recent_topic or (existing_topics[0] if existing_topics else None)
    source = metadata.get('source', 'unknown') if metadata else 'unknown'

    # Step 3a: Immediate commit - append user turn to working memory (keyed by thread_id)
    working_memory.append_turn(thread_id, 'user', text)

    # Step 3b: Immediate commit - log user input event (pre-classification)
    if interaction_log:
        interaction_log.log_event(
            event_type='user_input',
            payload={'message': text},
            topic=context_topic or 'unknown',
            source=source,
            metadata=metadata,
            thread_id=thread_id,
        )

    # Step 3c: Evaluate reward from previous exchange
    try:
        from services.reward_evaluator_service import RewardEvaluatorService
        reward_eval = RewardEvaluatorService()
        prev_turns = working_memory.get_recent_turns(thread_id, n=2)
        previous_input = None
        for turn in prev_turns:
            if turn.get('role') == 'user' and turn.get('content') != text:
                previous_input = turn.get('content')
                break

        if previous_input and context_topic:
            behavior_reward = reward_eval.evaluate_user_behavior(
                current_input=text,
                previous_input=previous_input,
                previous_topic=context_topic,
                current_topic=context_topic
            )
            if behavior_reward != 0.0:
                logging.debug(f"[DIGEST] Previous exchange reward: {behavior_reward:.2f}")
                # Cache reward in Redis for identity reinforcement (read by memory chunker)
                try:
                    from services.redis_client import RedisClientService
                    reward_redis = RedisClientService.create_connection()
                    reward_redis.setex(f"identity_reward:{context_topic}", 1800, str(behavior_reward))
                except Exception as re:
                    logging.debug(f"[DIGEST] Failed to cache identity reward: {re}")

    except Exception as e:
        logging.debug(f"[DIGEST] Reward evaluation skipped: {e}")

    # Step 3d: Proactive engagement correlation
    if context_topic:
        _try_proactive_engagement_correlation(text, context_topic)

    # Step 3e: Record user interaction timestamp (embedding stored after classification in Phase C)
    try:
        from services.autonomous_actions.communicate_action import CommunicateAction
        communicate = CommunicateAction()
        communicate.record_user_interaction()
    except Exception:
        pass

    # ═══════════════════════════════════════════════════════════
    # PHASE B: RETRIEVAL (context assembly)
    # ═══════════════════════════════════════════════════════════

    # Step 4: Inject cold-start gists if topic has none
    if context_topic:
        gist_storage.store_cold_start_gists(context_topic)

    # Step 4a: Get context for frontal cortex
    gists = []
    if context_topic:
        gists = gist_storage.get_latest_gists(context_topic)
        if gists:
            gist_lines = ["Recent conversation context:"]
            for gist in gists:
                gist_lines.append(f"- [{gist['type']}] {gist['content']} (confidence: {gist['confidence']})")
            gist_context = "\n".join(gist_lines)
        else:
            last_msg = gist_storage.get_last_message(context_topic)
            if last_msg:
                gist_context = f"Last exchange:\nUser: {last_msg['prompt']}\nAssistant: {last_msg['response']}"
            else:
                gist_context = "No previous conversation context available"
    else:
        gist_context = "No previous conversation context available"

    world_state = world_state_service.get_world_state(context_topic, thread_id=thread_id) if context_topic else ""

    # Step 4b: Calculate context warmth for cost scaling
    wm_turns = working_memory.get_recent_turns(thread_id)
    context_warmth = calculate_context_warmth(
        working_memory_len=len(wm_turns),
        gists=gists,
        world_state_nonempty=bool(world_state)
    )
    logging.info(f"[DIGEST] Context warmth for '{context_topic}': {context_warmth:.2f}")

    # ═══════════════════════════════════════════════════════════
    # PHASE C: CLASSIFICATION + ROUTING + RESPONSE
    # ═══════════════════════════════════════════════════════════

    # Step 6: Classify the prompt with deterministic embedding-based classifier
    topic_classifier = get_topic_classifier()
    classification_result = topic_classifier.classify(text, recent_topic=recent_topic)

    # Extract for compatibility with handle_classification
    classification = {
        'topic': classification_result['topic'],
        'confidence': int(classification_result['confidence'] * 10),
        'similar_topic': '',
        'topic_update': ''
    }
    classification_time = classification_result['classification_time']

    # Store user message embedding for proactive relevance scoring (256-dim, matches drift engine)
    try:
        from services.embedding_service import EmbeddingService
        emb_service = EmbeddingService()
        msg_embedding = emb_service.generate_embedding(text)
        from services.autonomous_actions.communicate_action import CommunicateAction
        communicate = CommunicateAction()
        communicate.record_user_interaction(message_embedding=msg_embedding)
    except Exception as e:
        logging.debug(f"[DIGEST] Failed to store message embedding for proactive: {e}")

    metrics.record_timing(trace_id, 'classification', classification_time * 1000)
    metrics.record_counter('classifications_total')

    # Step 7: Add exchange to thread conversation
    topic = classification_result['topic']
    exchange_id = thread_conv_service.add_exchange(thread_id, topic, {
        "message": text,
        "classification_time": classification_time,
    })

    # Update thread with current topic
    thread_service.update_topic(thread_id, topic)
    thread_service.increment_exchange_count(thread_id)

    # Step 7a: Log classification event (with resolved topic)
    if interaction_log:
        interaction_log.log_event(
            event_type='classification',
            payload=classification,
            topic=topic,
            exchange_id=exchange_id,
            source=source,
            metadata={'classification_time': classification_time}
        )

    # Step 7b: Encode user message immediately (per-message encoding — Phase A)
    # Always encode the user message, even for fast-path acks (user said something meaningful)
    event_bus.emit_and_handle(ENCODE_EVENT, {
        'topic': topic,
        'exchange_id': exchange_id,
        'prompt_message': text,
        'response_message': '',
        'metadata': metadata,
        'thread_id': thread_id,
    })

    # Step 7c: If topic changed from context_topic, inject cold-start gists for new topic
    if context_topic and topic != context_topic:
        gist_storage.store_cold_start_gists(topic)

    # Step 8: Cache this topic as the most recent
    recent_topic_service.set_recent_topic(topic)

    # Step 9: Track session and check for episode generation
    session_service = get_session_service()
    session_service.set_thread(thread_id)
    is_new_topic = classification.get('is_new_topic', False)
    session_service.track_classification(topic, is_new_topic, time.time())

    exchange_data = {
        'exchange_id': exchange_id,
        'prompt': {'message': text},
        'timestamp': time.time()
    }

    if session_service.check_topic_switch(topic):
        should_generate, reason = session_service.should_generate_episode()
        if should_generate:
            logging.info(f"Episode generation triggered: {reason}")
            session_data = session_service.get_session_data()
            enqueue_episodic_memory(session_data)
            session_service.reset_session()
        session_service.mark_topic_switch(topic)

    # Step 9b: Tool relevance scoring (embedding-based, replaces regex tool hints)
    tool_relevance = None
    try:
        from services.tool_relevance_service import ToolRelevanceService
        tool_relevance = ToolRelevanceService().score_relevance(text, top_k=5)
        logging.info(
            f"[DIGEST] Tool relevance: max_score={tool_relevance['max_relevance_score']:.3f}, "
            f"tools={[t['name'] for t in tool_relevance.get('relevant_tools', [])]}"
        )
    except Exception as e:
        logging.debug(f"[DIGEST] Tool relevance scoring failed: {e}")

    # Step 9c: Intent classification (~5ms, deterministic, no LLM)
    intent_classifier = get_intent_classifier()
    intent = intent_classifier.classify(
        text=text,
        topic=topic,
        context_warmth=context_warmth,
        tool_relevance=tool_relevance,
    )
    logging.info(
        f"[DIGEST] Intent: type={intent['intent_type']}, needs_tools={intent['needs_tools']}, "
        f"complexity={intent['complexity']}, confidence={intent['confidence']:.2f}"
    )

    # ── Cancel intent handling ──
    if intent.get('is_cancel'):
        _cancel_active_tool_work(topic)

    # ── Self-resolved intent handling ──
    if intent.get('is_self_resolved'):
        _cancel_active_tool_work(topic)

    # ── Active tool work dedup ──
    if not intent.get('is_cancel') and not intent.get('is_self_resolved'):
        dedup_response = _check_active_tool_work(text, topic)
        if dedup_response:
            orchestrator = get_orchestrator()
            orchestrator.route_path('RESPOND', {
                'response': dedup_response,
                'confidence': 0.8,
                'topic': topic,
                'destination': metadata.get('destination', 'web'),
                'metadata': metadata,
            })
            working_memory.append_turn(thread_id, 'assistant', dedup_response)
            _log_cycle_event('duplicate_detected', {'response': dedup_response}, topic)
            return f"Topic '{topic}' | DEDUP: active tool work in progress"

    # Step 9d: Collect routing signals (all Redis reads, ~5ms)
    signals = collect_routing_signals(
        text=text,
        topic=topic,
        context_warmth=context_warmth,
        working_memory=working_memory,
        gist_storage=gist_storage,
        fact_store=fact_store,
        world_state_service=world_state_service,
        classification_result=classification_result,
        session_service=session_service,
        intent=intent,
        tool_relevance=tool_relevance,
    )

    # Step 10: Route — check for fast-path before LLM generation
    is_fast_path_ack = False
    try:
        # ── Fast path check: embedding relevance → template ack + background work ──
        tool_rel_score = tool_relevance.get('max_relevance_score', 0.0) if tool_relevance else 0.0
        # Only use fast-path if context warmth is sufficient (not the very first message in a section)
        # This prevents showing temporary ack animations for initial user messages
        min_context_warmth_for_fast_path = 0.1
        if (tool_rel_score > 0.35
                and not intent.get('is_cancel')
                and not intent.get('is_self_resolved')
                and context_warmth >= min_context_warmth_for_fast_path):

            # Pre-route to check if mode router agrees with ACT
            _previous_mode = None
            try:
                from services.routing_decision_service import RoutingDecisionService
                from services.database_service import get_shared_db_service
                _db_service = get_shared_db_service()
                _rds = RoutingDecisionService(_db_service)
                _previous_mode = _rds.get_previous_mode(topic)
            except Exception:
                pass

            signals['_prompt_text'] = text
            routing_result = mode_router.route(signals, text, previous_mode=_previous_mode)
            selected_mode = routing_result['mode']

            if selected_mode == 'ACT':
                # ── FAST PATH: template ack + background tool work ──
                topic_phrase = intent_classifier.extract_topic_phrase(text)

                # Use reflective language when innate skills dominate.
                # Check the #1 ranked tool: if it's an innate skill, we're in reflective mode
                # regardless of what other external tools appear lower in the list.
                _top_tools = tool_relevance.get('relevant_tools', []) if tool_relevance else []
                _top_scorer = _top_tools[0] if _top_tools else None
                _is_reflective = (not _top_scorer) or (_top_scorer.get('type') == 'skill')

                from services.redis_client import RedisClientService
                _redis_for_ack = RedisClientService.create_connection()
                ack_text = intent_classifier.select_template(
                    intent_type=intent['intent_type'],
                    complexity=intent['complexity'],
                    register=intent.get('register', 'neutral'),
                    user_id='default',
                    topic_phrase=topic_phrase,
                    is_reflective=_is_reflective,
                    redis_conn=_redis_for_ack,
                )

                # Deliver ack via TOOL_SPAWN orchestrator path
                orchestrator = get_orchestrator()
                orchestrator.route_path('TOOL_SPAWN', {
                    'response': ack_text,
                    'topic': topic,
                    'destination': metadata.get('destination', 'web'),
                    'confidence': 0.5,
                    'metadata': metadata,
                })

                # Store ack in conversation history
                thread_conv_service.add_response(thread_id, ack_text, 0.0)

                # Create cycle records and spawn tool work
                try:
                    from services.cycle_service import CycleService
                    from services.database_service import get_shared_db_service

                    _db_service2 = get_shared_db_service()
                    cycle_service = CycleService(_db_service2)
                    user_cycle_id = cycle_service.create_cycle(
                        content=text, topic=topic,
                        cycle_type='user_input', source='user',
                    )
                    ack_cycle_id = cycle_service.create_cycle(
                        content=ack_text, topic=topic,
                        cycle_type='fast_response', source='system',
                        parent_cycle_id=user_cycle_id,
                    )

                    from workers.tool_worker import tool_worker
                    tool_queue = PromptQueue(queue_name="tool-queue", worker_func=tool_worker)
                    tool_queue.enqueue({
                        'parent_cycle_id': ack_cycle_id,
                        'root_cycle_id': user_cycle_id,
                        'topic': topic,
                        'text': text,
                        'intent': intent,
                        'metadata': {**metadata, 'thread_id': thread_id},
                        'context_snapshot': {
                            'context_warmth': context_warmth,
                            'tool_hints': intent.get('tool_hints', []),
                            'relevant_tools': tool_relevance.get('relevant_tools', []) if tool_relevance else [],
                        },
                    })

                    logging.info(
                        f"[DIGEST] Fast path: ack delivered, tool work spawned "
                        f"(user_cycle={user_cycle_id}, ack_cycle={ack_cycle_id})"
                    )
                except Exception as e:
                    logging.error(f"[DIGEST] Fast path cycle/spawn failed: {e}")

                response_data = {
                    'response': ack_text,
                    'mode': 'TOOL_SPAWN',
                    'confidence': 0.5,
                    'generation_time': 0.0,
                }
                is_fast_path_ack = True

            else:
                # Router didn't select ACT — intent mismatch, use normal path
                _log_cycle_event('intent_mismatch', {
                    'predicted_needs_tools': True,
                    'actual_mode': selected_mode,
                }, topic)
                # Pass pre-computed routing_result to avoid double mode routing
                _relevant = tool_relevance.get('relevant_tools', []) if tool_relevance else None
                response_data, routing_result = route_and_generate(
                    topic, text, classification, thread_conv_service,
                    cortex_config, cortex_prompt_map, mode_router, signals, fact_store,
                    metadata=metadata, context_warmth=context_warmth,
                    pre_routing_result=routing_result,
                    relevant_tools=_relevant,
                    thread_id=thread_id,
                )
        else:
            # ── Normal path (no tools needed, or low intent confidence) ──
            _relevant = tool_relevance.get('relevant_tools', []) if tool_relevance else None
            response_data, routing_result = route_and_generate(
                topic, text, classification, thread_conv_service,
                cortex_config, cortex_prompt_map, mode_router, signals, fact_store,
                metadata=metadata, context_warmth=context_warmth,
                relevant_tools=_relevant,
                thread_id=thread_id,
            )

        # Add response to exchange data
        exchange_data['response'] = {'message': response_data['response']}
        if response_data.get('actions'):
            exchange_data['steps'] = response_data['actions']

        # Add complete exchange to session
        session_service.add_exchange(exchange_data)

        # ═══════════════════════════════════════════════════════════
        # PHASE D: POST-RESPONSE COMMIT
        # ═══════════════════════════════════════════════════════════

        # Step 11a: Append assistant turn to working memory (keyed by thread_id)
        working_memory.append_turn(thread_id, 'assistant', response_data['response'])

        # Step 11b: Log system response event
        if interaction_log:
            interaction_log.log_event(
                event_type='system_response',
                payload={
                    'message': response_data['response'],
                    'mode': response_data.get('mode', 'RESPOND'),
                    'confidence': response_data.get('confidence', 0.0),
                    'generation_time': response_data.get('generation_time', 0.0)
                },
                topic=topic,
                exchange_id=exchange_id,
                source=source,
                metadata=metadata,
                thread_id=thread_id,
            )

        # Step 11c: Encode assistant response (per-message encoding — Phase D)
        # Skip for fast-path template acks — template has no semantic content worth encoding
        if not is_fast_path_ack:
            event_bus.emit_and_handle(ENCODE_EVENT, {
                'topic': topic,
                'exchange_id': exchange_id,
                'prompt_message': '',
                'response_message': response_data['response'],
                'metadata': metadata,
                'thread_id': thread_id,
            })

        # ═══════════════════════════════════════════════════════════
        # PHASE E: ASYNC FOLLOW-UP
        # ═══════════════════════════════════════════════════════════

        # Step 12: Check for inactivity-based episode generation
        should_generate, reason = session_service.should_generate_episode()
        if should_generate:
            logging.info(f"Episode generation triggered: {reason}")
            session_data = session_service.get_session_data()
            enqueue_episodic_memory(session_data)
            session_service.reset_session()

        # Print the actual response to stdout for the user
        logging.info(f"\n{'='*60}")
        logging.info(f"Topic: {topic}")
        logging.info(f"Mode: {response_data['mode']} (router confidence: {routing_result['router_confidence']:.3f})")
        logging.info(f"{'='*60}")
        logging.info(response_data['response'])
        logging.info(f"{'='*60}\n")

        # Record metrics
        metrics.record_timing(trace_id, 'response_generation', response_data['generation_time'] * 1000)
        metrics.record_timing(trace_id, 'total_request', (time.time() - request_start_time) * 1000)
        metrics.record_counter('responses_total')

        return f"Topic '{topic}' | Mode: {response_data['mode']} | Response generated in {response_data['generation_time']:.2f}s"

    except TimeoutError as e:
        thread_conv_service.add_response_error(thread_id, f"Timeout: {str(e)}")
        metrics.record_counter('errors_total')
        logging.error(f"\n{'='*60}")
        logging.error(f"ERROR: Timeout")
        logging.error(f"{'='*60}")
        logging.error(f"{str(e)}")
        logging.error(f"{'='*60}\n")
        return f"Topic '{topic}' | ERROR: Timeout - {str(e)}"
    except Exception as e:
        thread_conv_service.add_response_error(thread_id, str(e))
        metrics.record_counter('errors_total')
        logging.error(f"\n{'='*60}")
        logging.error(f"ERROR: Response Generation Failed")
        logging.error(f"{'='*60}")
        logging.error(f"{str(e)}")
        logging.error(f"{'='*60}\n")
        return f"Topic '{topic}' | ERROR: {str(e)}"
