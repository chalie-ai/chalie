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

    def __init__(self, timeout: float = 10.0):
        """
        Initialize dispatcher with innate skills.

        Args:
            timeout: Maximum execution time per action (seconds)
        """
        self.timeout = timeout
        self.handlers = {}

        # Register innate skills (new system + backward-compat aliases)
        from services.innate_skills import register_innate_skills
        register_innate_skills(self)

    def dispatch_action(self, topic: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute single action with timeout enforcement.

        Phase 3 — Pre-action reliability check: if the action references a
        memory marked as 'uncertain' or 'contradicted', log a warning and
        annotate the result so the ACT loop / critic can decide whether to
        proceed or route to CLARIFY.

        Args:
            topic: Current conversation topic
            action: Action specification dict

        Returns:
            Action result dict with status, result text, and execution time
        """
        action_type = action.get('type', 'unknown')
        start_time = time.time()

        logging.info(f"[ACT DISPATCH] Executing {action_type}")

        # Pre-action reliability check — non-blocking: only logs/annotates
        _reliability_warning = self._check_source_reliability(action)

        # Get handler
        handler = self.handlers.get(action_type)
        if not handler:
            logging.error(f"[ACT DISPATCH] No handler for '{action_type}'. Registered: {list(self.handlers.keys())}")
            # Log capability gap when no handler exists for requested action
            try:
                from services.self_model_service import SelfModelService
                SelfModelService().log_capability_gap(
                    request_summary=action.get('params', {}).get('query', action_type)[:200],
                    detection_source="act_loop",
                    confidence=0.6,
                )
            except Exception:
                pass
            return {
                'action_type': action_type,
                'status': 'error',
                'result': f"Unknown action type: {action_type}",
                'execution_time': 0.0,
                'confidence': 0.0,
                'notes': '',
            }

        # Execute with timeout
        try:
            result_container = {'result': None, 'error': None}

            def target():
                """Thread target: invoke the action handler and capture the result."""
                try:
                    result_container['result'] = handler(topic, action)
                except Exception as e:
                    result_container['error'] = str(e)

            thread = Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=self.timeout)

            execution_time = time.time() - start_time

            # Check results
            if thread.is_alive():
                return {
                    'action_type': action_type,
                    'status': 'timeout',
                    'result': f"Action exceeded {self.timeout}s timeout",
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

            raw_result = result_container['result']

            # Handle structured skill results (dict with text + reply_actions)
            reply_actions = None
            if isinstance(raw_result, dict) and 'text' in raw_result:
                reply_actions = raw_result.get('reply_actions')
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
            # Annotate with reliability warning if source memory is unreliable
            if _reliability_warning:
                dispatch_result['reliability_warning'] = _reliability_warning
                # Reduce confidence proportionally
                dispatch_result['confidence'] = confidence * 0.6
                logging.warning(
                    f"[ACT DISPATCH] Unreliable source for {action_type}: {_reliability_warning}"
                )
                # Log to interaction_log for constraint learning
                try:
                    from services.database_service import get_shared_db_service
                    from services.interaction_log_service import InteractionLogService
                    source = action.get('source_memory', {})
                    db = get_shared_db_service()
                    InteractionLogService(db).log_event(
                        event_type='reliability_warning',
                        payload={
                            'action_type': action_type,
                            'memory_type': source.get('type', ''),
                            'memory_id': source.get('id', ''),
                            'reliability_state': _reliability_warning[:100],
                            'confidence_reduction': 0.6,
                        },
                        source='act_dispatcher',
                    )
                except Exception:
                    pass
            return dispatch_result

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

    def _check_source_reliability(self, action: Dict[str, Any]) -> str:
        """
        Phase 3 — Pre-action reliability check.

        Looks for a 'source_memory' key in the action dict containing
        {type, id} identifying the memory that motivated this action.
        Returns a warning string if unreliable, empty string otherwise.
        """
        source = action.get('source_memory')
        if not source or not isinstance(source, dict):
            return ''
        memory_type = source.get('type')
        memory_id = source.get('id')
        if not memory_type or not memory_id:
            return ''
        try:
            from services.uncertainty_service import UncertaintyService
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            reliability = UncertaintyService(db).check_memory_reliability(memory_type, memory_id)
            if reliability in ('uncertain', 'contradicted'):
                return (
                    f"Source memory {memory_type}:{memory_id} is '{reliability}'. "
                    f"Consider clarifying before acting."
                )
        except Exception:
            pass
        return ''
