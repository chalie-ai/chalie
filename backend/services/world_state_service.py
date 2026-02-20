"""
World State Service - Generates world state context from active steps.

Reads from ThreadConversationService (Redis).
"""

import logging


class WorldStateService:
    """Service for generating world state context from conversation data."""

    def __init__(self, **kwargs):
        pass

    def get_world_state(self, topic: str, thread_id: str = None) -> str:
        """
        Generate world state context including active steps.

        Args:
            topic: Current topic (unused, kept for API compat)
            thread_id: Thread ID for Redis-backed lookup

        Returns:
            str: Formatted world state with active steps (if any)
        """
        if thread_id:
            return self._get_world_state_from_thread(thread_id)
        return "\n## World State\n(new conversation)"

    def _get_world_state_from_thread(self, thread_id: str) -> str:
        """Read active steps from ThreadConversationService."""
        try:
            from services.thread_conversation_service import ThreadConversationService
            conv_service = ThreadConversationService()
            active_steps = conv_service.get_active_steps(thread_id)

            lines = ["\n## World State"]

            if active_steps:
                lines.append("\n**Active Steps**:")
                for step in active_steps:
                    step_type = step.get('type', 'task')
                    description = step.get('description', 'Unknown step')
                    status_val = step.get('status', 'pending')
                    lines.append(f"- [{status_val.upper()}] {step_type}: {description}")

            return "\n".join(lines)
        except Exception as e:
            logging.debug(f"[WORLD STATE] Thread-based lookup failed: {e}")
            return "\n## World State\n(new conversation)"
