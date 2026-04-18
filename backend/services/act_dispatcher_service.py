"""
ACT Dispatcher Service

Dispatches internal cognitive actions with timeout enforcement.
Uses the innate skills system for all cognitive operations.

Returns structured results with confidence and notes for downstream
critic evaluation.
"""

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

    def __init__(self, timeout: float = 10.0, execution_gate: bool = True):
        """
        Initialize dispatcher with innate skills.

        Args:
            timeout: Maximum execution time per action (seconds)
            execution_gate: Whether to apply the autonomous execution gate.
                Set to False for user-initiated ACT loops (the user already
                asked for this), True for autonomous/background execution.
        """
        self.timeout = timeout
        self.execution_gate = execution_gate
        self.handlers = {}

        # Register innate skills (new system + backward-compat aliases)
        from services.innate_skills import register_innate_skills
        register_innate_skills(self)

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

        _commit_warning = ''

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

        from services.act_action_categories import SAFE_ACTIONS as _SAFE_ACTIONS
        if self.execution_gate and action_type not in _SAFE_ACTIONS:
            try:
                from services.autonomous_execution_gate import get_autonomous_execution_gate
                gate = get_autonomous_execution_gate()
                action_description = action.get('description', action.get('params', {}).get('query', action_type))
                gate_result = gate.evaluate(
                    str(action_description),
                    domain=channel or 'general',
                )
                if not gate_result['auto_execute']:
                    execution_time = time.time() - start_time
                    logging.info(
                        f"[ACT DISPATCH] Blocked by execution gate: "
                        f"tier={gate_result['consequence_tier']}, "
                        f"action={action_description!r:.100}"
                    )
                    return {
                        'action_type': action_type,
                        'status': 'requires_confirmation',
                        'result': (
                            f"This action requires user confirmation. "
                            f"{gate_result['reasoning']}"
                        ),
                        'execution_time': execution_time,
                        'confidence': 0.0,
                        'notes': gate_result['reasoning'],
                        'gate_decision': gate_result,
                    }
            except Exception:
                pass

        # Determine effective timeout: tool metadata can override the default.
        # This is generic — any tool can declare "timeout": <seconds> in TOOL_METADATA.
        effective_timeout = self.timeout
        try:
            from services.tool_library_service import TOOL_METADATA
            tool_timeout = TOOL_METADATA.get(action_type, {}).get('timeout')
            if tool_timeout and tool_timeout > effective_timeout:
                effective_timeout = float(tool_timeout)
        except Exception:
            pass

        # Execute with timeout
        try:
            result_container = {'result': None, 'error': None}

            def target():
                """Thread target: invoke the action handler and capture the result."""
                try:
                    result_container['result'] = handler(channel, action)
                except Exception as e:
                    result_container['error'] = str(e)

            thread = Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=effective_timeout)

            execution_time = time.time() - start_time
            elapsed_ms = int(execution_time * 1000)

            # Check results
            if thread.is_alive():
                self._track_skill(action_type, False, elapsed_ms, channel, failure_class='timeout')
                return {
                    'action_type': action_type,
                    'status': 'timeout',
                    'result': f"Action exceeded {effective_timeout}s timeout",
                    'execution_time': execution_time,
                    'confidence': 0.0,
                    'notes': '',
                }

            if result_container['error']:
                self._track_skill(action_type, False, elapsed_ms, channel, failure_class='internal')
                return {
                    'action_type': action_type,
                    'status': 'error',
                    'result': f"Error: {result_container['error']}",
                    'execution_time': execution_time,
                    'confidence': 0.0,
                    'notes': '',
                }

            raw_result = result_container['result']

            # Handle structured skill results (dict with text + reply_actions)
            reply_actions = None
            _discovered_tools = None
            if isinstance(raw_result, dict) and 'text' in raw_result:
                reply_actions = raw_result.get('reply_actions')
                _discovered_tools = raw_result.get('_discovered_tools')
                raw_result = raw_result['text']

            confidence = _estimate_confidence(action_type, raw_result)
            notes = _extract_notes(action_type, action, raw_result)
            # Append COMMIT-tier warning to notes so the ACT loop prompt sees it
            if _commit_warning:
                notes = (_commit_warning + '; ' + notes) if notes else _commit_warning

            self._track_skill(action_type, True, elapsed_ms, channel, result=str(raw_result)[:500])

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

        except Exception as e:
            execution_time = time.time() - start_time
            self._track_skill(action_type, False, int(execution_time * 1000), channel, failure_class='internal')
            logging.exception(f"[ACT DISPATCH] Unexpected error in {action_type}:")
            return {
                'action_type': action_type,
                'status': 'error',
                'result': f"Unexpected error: {str(e)}",
                'execution_time': execution_time,
                'confidence': 0.0,
                'notes': '',
            }

    def _track_skill(self, action_type: str, success: bool, latency_ms: int,
                     channel: str = '', failure_class: str | None = None, result: str = '') -> None:
        """Track innate skill invocations via the unified tracker.

        Tools are skipped here — they self-track inside ToolRegistryService.invoke().
        This avoids double-counting while keeping a single tracker for all paths.
        """
        from services.innate_skills.registry import ALL_SKILL_NAMES
        if action_type not in ALL_SKILL_NAMES:
            return
        try:
            from services.invocation_tracker import track
            track(
                name=action_type,
                success=success,
                latency_ms=latency_ms,
                topic=channel,
                failure_class=failure_class,
                result=result,
            )
        except Exception as e:
            logging.debug(f"[ACT DISPATCH] Tracking failed for {action_type}: {e}")

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

