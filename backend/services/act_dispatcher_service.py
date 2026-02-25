"""
ACT Dispatcher Service

Dispatches internal cognitive actions with timeout enforcement.
Uses the innate skills system for all cognitive operations.

Returns structured results with confidence and notes for downstream
critic evaluation.
"""

import time
from typing import Dict, Any, Callable
from threading import Thread
import logging

# Actions that are deterministic with no ambiguity — high default confidence
_DETERMINISTIC_ACTIONS = {'memorize', 'introspect'}

# Actions that are reads/lookups — moderate default confidence
_READ_ACTIONS = {'recall', 'associate', 'autobiography'}


def _estimate_confidence(action_type: str, raw_result: Any) -> float:
    """Estimate confidence based on action type and result richness."""
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
    """Extract contextual notes from an action result for critic review."""
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

        Args:
            topic: Current conversation topic
            action: Action specification dict

        Returns:
            Action result dict with status, result text, and execution time
        """
        action_type = action.get('type', 'unknown')
        start_time = time.time()

        logging.info(f"[ACT DISPATCH] Executing {action_type}")

        # Get handler
        handler = self.handlers.get(action_type)
        if not handler:
            logging.error(f"[ACT DISPATCH] No handler for '{action_type}'. Registered: {list(self.handlers.keys())}")
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
            confidence = _estimate_confidence(action_type, raw_result)
            notes = _extract_notes(action_type, action, raw_result)

            return {
                'action_type': action_type,
                'status': 'success',
                'result': raw_result,
                'execution_time': execution_time,
                'confidence': confidence,
                'notes': notes,
            }

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
