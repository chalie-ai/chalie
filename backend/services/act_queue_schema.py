from enum import Enum
from typing import Dict, Any, Optional
import time


class OutputType(Enum):
    """Types of outputs that can be enqueued to the act-queue."""
    RESPOND = "respond"
    CLARIFY = "clarify"
    ACKNOWLEDGE = "acknowledge"
    IGNORE = "ignore"
    BACKGROUND_ACT = "background_act"


class ActQueueMessage:
    """Schema for messages in the act-queue."""

    @staticmethod
    def create_output_message(
        output_type: OutputType,
        topic: str,
        destination: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a properly formatted act-queue message.

        Args:
            output_type: Type of output (OutputType enum)
            topic: Conversation topic identifier
            destination: Target channel (e.g., "telegram", "cli")
            content: The actual output content/message
            metadata: Optional additional context (user_id, confidence, etc.)

        Returns:
            Dict containing the serializable message with timestamp
        """
        return {
            "type": output_type.value,
            "topic": topic,
            "destination": destination,
            "content": content,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
