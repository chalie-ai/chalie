"""
ACT Dispatcher Service

Dispatches internal cognitive actions with timeout enforcement.
Uses the innate skills system for all cognitive operations.
"""

import time
from typing import Dict, Any, Callable
from threading import Thread
import logging


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
                'execution_time': 0.0
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
                    'execution_time': execution_time
                }

            if result_container['error']:
                return {
                    'action_type': action_type,
                    'status': 'error',
                    'result': f"Error: {result_container['error']}",
                    'execution_time': execution_time
                }

            return {
                'action_type': action_type,
                'status': 'success',
                'result': result_container['result'],
                'execution_time': execution_time
            }

        except Exception as e:
            execution_time = time.time() - start_time
            logging.exception(f"[ACT DISPATCH] Unexpected error in {action_type}:")
            return {
                'action_type': action_type,
                'status': 'error',
                'result': f"Unexpected error: {str(e)}",
                'execution_time': execution_time
            }
