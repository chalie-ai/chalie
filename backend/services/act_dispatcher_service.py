"""
ACT Dispatcher Service

Dispatches internal cognitive actions with timeout enforcement.
Uses the innate skills system for all cognitive operations.

Returns structured results with confidence and notes for downstream
critic evaluation.
"""

import contextvars
import time
from typing import Dict, Any
from threading import Thread
import logging

from services.act_action_categories import DETERMINISTIC_ACTIONS as _DETERMINISTIC_ACTIONS, READ_ACTIONS as _READ_ACTIONS


def _estimate_confidence(action_type: str, raw_result: Any) -> float:
    """Estimate confidence based on action type and result richness.

    Deterministic actions always return 0.92.  Read actions are scored by the
    length of the result string.  All other action types default to 0.50.

    Args:
        action_type: The action category string (e.g. ``"recall"``, ``"memorize"``).
        raw_result: Raw result value returned by the action handler.

    Returns:
        Confidence score in the range [0.0, 1.0].
    """
    if action_type in _DETERMINISTIC_ACTIONS:
        return 0.92
    if action_type in _READ_ACTIONS:
        result_str = str(raw_result) if raw_result else ''
        if len(result_str) > 100:
            return 0.75
        if len(result_str) > 20:
            return 0.60
        return 0.40
    return 0.50


def _extract_notes(action_type: str, action: Dict[str, Any], raw_result: Any) -> str:
    """Extract contextual notes from an action result for critic review.

    Currently produces notes for ``schedule`` (parsed date, recurrence) and
    ``recall`` (query string) action types.

    Args:
        action_type: The action category string.
        action: Full action specification dict.
        raw_result: Raw result value returned by the action handler.

    Returns:
        Semicolon-delimited notes string, or empty string if none apply.
    """
    notes_parts = []
    if action_type == 'schedule':
        if isinstance(raw_result, dict):
            if 'parsed_date' in raw_result:
                notes_parts.append(f"parsed date: {raw_result['parsed_date']}")
            if 'recurrence' in raw_result:
                notes_parts.append(f"recurrence: {raw_result['recurrence']}")
        elif isinstance(raw_result, str) and 'scheduled' in raw_result.lower():
            notes_parts.append(f"schedule result: {raw_result[:200]}")
    if action_type == 'recall':
        query = action.get('query', '')
        if query:
            notes_parts.append(f"query: {query}")
    return '; '.join(notes_parts) if notes_parts else ''


class ActDispatcherService:
    """Dispatches internal cognitive actions with timeout enforcement."""

    def __init__(self, timeout: float = 10.0):
        """Initialize dispatcher with ability handlers.

        Args:
            timeout: Maximum execution time per action (seconds)
        """
        self.timeout = timeout
        self.handlers = {}

        # Per-turn ordinal counter: maps tool name → call count this turn.
        # Reset by reset_turn_ordinals() at the start of each ACT turn.
        # Exposed to tools via the '_rich_media_ordinal' key injected into params.
        self._turn_ordinals: Dict[str, int] = {}

        # Register every ability from the registry as a handler.
        # Each handler calls ability.execute(channel, params, telemetry=None).
        # The dispatcher unwraps the returned dict's 'text' key automatically.
        from abilities._registry import AbilityRegistry
        for ability in AbilityRegistry.all():
            _ability = ability  # capture for closure
            self.handlers[_ability.NAME] = (
                lambda channel, action, _a=_ability: _a.execute(
                    channel,
                    {k: v for k, v in action.items() if k not in ('type', 'exchange_id')},
                    None,
                )
            )

    def reset_turn_ordinals(self) -> None:
        """Reset per-tool ordinal counters at the start of a new ACT turn.

        Must be called once per MessageProcessor turn before any dispatch_action
        calls so that ordinals are scoped to the turn, not the dispatcher lifetime.
        """
        self._turn_ordinals = {}

    def dispatch_action(self, channel: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute single action with timeout enforcement.

        Args:
            channel: Current conversation channel
            action: Action specification dict

        Returns:
            Action result dict with status, result text, and execution time
        """
        action_type = action.get('type', 'unknown')
        start_time = time.time()

        logging.info(f"[ACT DISPATCH] Executing {action_type}")

        # Rich-media ordinal is injected ONLY for the user-facing channel.
        # Subagents (and any other internal channel) must never see the trailer
        # because their natural-language synthesis is consumed by the parent
        # rather than rendered to the user — a span emitted at that hop has no
        # tool_calls row paired to it and would either be paraphrased away by
        # the parent or leak through as raw markup. Stripping the ordinal here
        # is the single physical chokepoint that prevents the trailer from
        # ever reaching a non-user dispatch path.
        action = dict(action)
        if channel == 'user':
            self._turn_ordinals[action_type] = self._turn_ordinals.get(action_type, 0) + 1
            action['_rich_media_ordinal'] = self._turn_ordinals[action_type]

        # Get handler
        handler = self.handlers.get(action_type)
        if not handler:
            # Try wrapper intent routing before falling through to error
            wrapper_result = self._try_wrapper_intent(action_type, action)
            if wrapper_result:
                return wrapper_result

            logging.error(f"[ACT DISPATCH] No handler for '{action_type}'. Registered: {list(self.handlers.keys())}")
            return {
                'action_type': action_type,
                'status': 'error',
                'result': f"Unknown action type: {action_type}",
                'execution_time': 0.0,
                'confidence': 0.0,
                'notes': '',
            }

        # Determine effective timeout: ability TIMEOUT ClassVar overrides the default.
        # Falls back to self.timeout when the action type is not a registered ability.
        effective_timeout = self.timeout
        from abilities._registry import AbilityRegistry
        try:
            ability_timeout = AbilityRegistry.get(action_type).TIMEOUT
        except KeyError:
            ability_timeout = None
        if ability_timeout and ability_timeout > effective_timeout:
            effective_timeout = float(ability_timeout)

        return self._execute_with_timeout(action_type, channel, action, handler, effective_timeout, start_time)


    def _execute_with_timeout(
        self,
        action_type: str,
        channel: str,
        action: Dict[str, Any],
        handler: Any,
        effective_timeout: float,
        start_time: float,
    ) -> Dict[str, Any]:
        """Run handler in a daemon thread with timeout; return a result dict.

        Copies calling thread's contextvars so abilities can resolve
        current_processor() (and any ContextVar-backed state bound by
        MessageProcessor.send()). Without this copy the spawned Thread starts
        with a fresh context and current_processor() returns None.
        """
        try:
            result_container = {'result': None, 'error': None}
            ctx = contextvars.copy_context()

            def target():
                """Thread target: invoke the action handler and capture the result."""
                try:
                    result_container['result'] = handler(channel, action)
                except Exception as e:
                    result_container['error'] = str(e)

            thread = Thread(target=ctx.run, args=(target,))
            thread.daemon = True
            thread.start()
            thread.join(timeout=effective_timeout)

            execution_time = time.time() - start_time

            if thread.is_alive():
                return {
                    'action_type': action_type,
                    'status': 'timeout',
                    'result': f"Action exceeded {effective_timeout}s timeout",
                    'execution_time': execution_time,
                    'confidence': 0.0,
                    'notes': '',
                }

            if result_container['error']:
                return {
                    'action_type': action_type,
                    'status': 'error',
                    'result': f"Error: {result_container['error']}",
                    'execution_time': execution_time,
                    'confidence': 0.0,
                    'notes': '',
                }

            return self._build_success_result(action_type, action, result_container['result'], execution_time)

        except Exception as e:
            execution_time = time.time() - start_time
            logging.exception(f"[ACT DISPATCH] Unexpected error in {action_type}:")
            return {
                'action_type': action_type,
                'status': 'error',
                'result': f"Unexpected error: {str(e)}",
                'execution_time': execution_time,
                'confidence': 0.0,
                'notes': '',
            }

    def _build_success_result(
        self,
        action_type: str,
        action: Dict[str, Any],
        raw_result: Any,
        execution_time: float,
    ) -> Dict[str, Any]:
        """Build the success result dict from a raw handler return value."""
        reply_actions = None
        _discovered_tools = None
        if isinstance(raw_result, dict) and 'text' in raw_result:
            reply_actions = raw_result.get('reply_actions')
            _discovered_tools = raw_result.get('_discovered_tools')
            raw_result = raw_result['text']

        confidence = _estimate_confidence(action_type, raw_result)
        notes = _extract_notes(action_type, action, raw_result)

        dispatch_result = {
            'action_type': action_type,
            'status': 'success',
            'result': raw_result,
            'execution_time': execution_time,
            'confidence': confidence,
            'notes': notes,
        }
        if reply_actions:
            dispatch_result['reply_actions'] = reply_actions
        if _discovered_tools:
            dispatch_result['_discovered_tools'] = _discovered_tools
        return dispatch_result

    def _try_wrapper_intent(self, action_type: str, action: Dict[str, Any]) -> Dict[str, Any] | None:
        """Check if a connected wrapper declares the action type as a capability.

        If a capable wrapper is found, emit a CognitiveIntent via IntentService
        and return a structured result dict.  Returns None if no wrapper handles
        this action type (caller should fall through to the error path).

        Args:
            action_type: The action category string (e.g. ``"open_pr"``).
            action: Full action specification dict.

        Returns:
            Result dict with ``status: "intent_emitted"`` on success, or None.
        """
        try:
            from services.database_service import get_shared_db_service
            from services.wrapper_auth_service import WrapperAuthService
            from services.intent_service import IntentService, CognitiveIntent

            db = get_shared_db_service()
            wrapper_svc = WrapperAuthService(db)
            wrappers = wrapper_svc.list_wrappers()

            # Find the first wrapper that declares this action type
            target_wrapper = None
            for w in wrappers:
                caps = w.get("capabilities", {})
                declared_intents = caps.get("intents", [])
                if action_type in declared_intents:
                    target_wrapper = w
                    break

            if not target_wrapper:
                return None

            import uuid
            intent_id = str(uuid.uuid4())
            intent = CognitiveIntent(
                intent_id=intent_id,
                intent_type="execute",
                target_wrapper=target_wrapper["wrapper_id"],
                payload={
                    "action_type": action_type,
                    "params": action.get("params", {}),
                    "description": action.get("description", ""),
                },
                confidence=0.7,
            )

            intent_svc = IntentService()
            intent_svc.emit(intent)

            logging.info(
                f"[ACT DISPATCH] Routed {action_type} to wrapper "
                f"{target_wrapper['wrapper_id']} via intent {intent_id}"
            )

            return {
                "action_type": action_type,
                "status": "intent_emitted",
                "result": f"Intent emitted to wrapper {target_wrapper['wrapper_id']}",
                "target_wrapper": target_wrapper["wrapper_id"],
                "intent_id": intent_id,
                "execution_time": 0.0,
                "confidence": 0.7,
                "notes": f"intent_id={intent_id}; target={target_wrapper['wrapper_id']}",
            }
        except Exception as e:
            logging.debug(f"[ACT DISPATCH] Wrapper intent routing failed: {e}")
            return None

